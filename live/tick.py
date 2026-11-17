"""Game-time updater for the pool page, running on GitHub Actions.

Splash Sports data (games, lines, picks, tiebreakers) arrives via
live/snapshot.json, published post-lock from a residential machine — Splash's
CloudFront firewall 403s requests from datacenter IPs, so this script can
never talk to Splash directly. Everything that changes DURING games comes
from ESPN's public scoreboard instead: statuses, scores and the live win
probability. The week settles here too, when the last game goes final:
placements are most-wins-first, ties broken by smallest absolute tiebreaker
distance, tied entries sharing the place (rule validated against the first
10 officially settled weeks, 200 placements, zero mismatches). A residential
machine later diffs the result against Splash's official leaderboard and
corrects the rare drift (score corrections, voided games).

Self-contained on purpose: requests + numpy only, no imports from the
optimizer that generates the snapshot.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "live" / "snapshot.json"
STATE = ROOT / "live" / "state.json"
N_SIMS = 100_000
NFL_MARGIN_SIGMA = 13.45
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
TO_ESPN = {"AZ": "ARI"}  # our alias -> ESPN's, where they differ


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _spread_prob_home(line: float) -> float:
    """P(home) from the picksheet spread (line < 0 => home favored by |line|),
    margin ~ Normal(spread, sigma) — same model the page always used."""
    if not line:
        return 0.5
    p = 0.5 * (1.0 + math.erf(abs(line) / (NFL_MARGIN_SIGMA * math.sqrt(2.0))))
    return p if line < 0 else 1.0 - p


def fetch_espn() -> dict[frozenset, dict]:
    """{frozenset({road, home} our aliases): {status, rs, hs, winner, p_home}}
    for every game on ESPN's scoreboard. Unofficial API: parse defensively,
    omit anything surprising and let the caller fall back to held state."""
    url = SCOREBOARD + os.environ.get("ESPN_QUERY", "")  # test hook: ?week=N replay
    r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    from_espn = {v: k for k, v in TO_ESPN.items()}
    out: dict[frozenset, dict] = {}
    for ev in r.json().get("events", []):
        try:
            comp = ev["competitions"][0]
            teams = {c["homeAway"]: from_espn.get(c["team"]["abbreviation"],
                                                  c["team"]["abbreviation"])
                     for c in comp["competitors"]}
            scores = {c["homeAway"]: int(c.get("score") or 0) for c in comp["competitors"]}
            state = (((ev.get("status") or {}).get("type") or {}).get("state")
                     or ((comp.get("status") or {}).get("type") or {}).get("state") or "")
            status = {"pre": "scheduled", "in": "in_progress", "post": "final"}.get(state)
            if status is None:
                continue
            g = {"status": status, "rs": scores["away"], "hs": scores["home"],
                 "winner": None, "p_home": None}
            if status == "final":
                g["winner"] = (teams["away"] if scores["away"] > scores["home"]
                               else teams["home"] if scores["home"] > scores["away"]
                               else "TIE")
            elif status == "in_progress":
                prob = ((comp.get("situation") or {}).get("lastPlay") or {}).get("probability") or {}
                p = prob.get("homeWinPercentage")
                if p is not None and 0.0 <= float(p) <= 1.0:
                    g["p_home"] = float(p)
            out[frozenset((teams["home"], teams["away"]))] = g
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return out


def load_state(week: int) -> dict:
    try:
        st = json.loads(STATE.read_text())
        if st.get("week") == week:
            return st
    except (OSError, ValueError):
        pass
    return {"week": week, "games": {}}


def merge_games(snap: dict, espn: dict, state: dict) -> list[dict]:
    """Snapshot games + ESPN live state, falling back to the carried state
    (ESPN's scoreboard can roll to the next week before we settle, or drop a
    probability late in a game — never regress to 'scheduled' or pregame odds
    once better information was published)."""
    now_iso = _now().isoformat()
    games = []
    for g in snap["games"]:
        held = state["games"].get(g["gid"], {})
        e = espn.get(frozenset((g["home"], g["road"])))
        cur = dict(held)
        if e:
            if e["status"] == "scheduled" and held.get("status") in ("in_progress", "final"):
                pass  # stale/mismatched ESPN row; keep what we had
            else:
                cur.update({k: v for k, v in e.items() if k != "p_home"})
                if e["status"] == "in_progress":
                    cur["p_home"] = (e["p_home"] if e["p_home"] is not None
                                     else held.get("p_home"))  # hold last live value
                else:
                    cur["p_home"] = None
        status = cur.get("status", "scheduled")
        final = status == "final"
        games.append({"gid": g["gid"], "road": g["road"], "home": g["home"],
                      "line": g["line"], "startsAt": g["startsAt"],
                      "locked": bool(g.get("locked")) or g["lockAt"] <= now_iso,
                      "status": status, "final": final,
                      "winner": cur.get("winner") if final else None,
                      "rs": cur.get("rs", 0), "hs": cur.get("hs", 0),
                      "live_p_home": cur.get("p_home")})
        state["games"][g["gid"]] = {"status": status, "rs": cur.get("rs", 0),
                                    "hs": cur.get("hs", 0), "winner": cur.get("winner"),
                                    "p_home": cur.get("p_home")}
    return games


def _game_prob_home(g: dict) -> float:
    if g["status"] == "in_progress" and g.get("live_p_home") is not None:
        return g["live_p_home"]
    return _spread_prob_home(g["line"])


def _wins_losses(games: list[dict], picks: dict[str, str]) -> tuple[int, int]:
    w = sum(1 for g in games if g["winner"] and picks.get(g["gid"]) == g["winner"])
    l = sum(1 for g in games if g["winner"] and picks.get(g["gid"]) not in (None, g["winner"]))
    return w, l


def simulate(snap: dict, games: list[dict], n_sims: int = N_SIMS) -> dict:
    """Port of the optimizer's live simulation: placement odds with ties
    shared fractionally, expected dollars from the payout schedule. Odds only
    run once every pick is visible (all_locked), else the page just shows
    wins and the locked picks."""
    all_locked = bool(snap.get("all_locked"))
    entries_in = snap["entries"]
    picks = {e["handle"]: e["picks"] for e in entries_in}
    handles = [e["handle"] for e in entries_in]
    sched = snap["payouts"]
    base = np.array([_wins_losses(games, picks[h])[0] for h in handles], dtype=float)

    if all_locked:
        remaining = [g for g in games if not g["final"]]
        probs = np.array([_game_prob_home(g) for g in remaining])
        # two masks, not one: an entry with no pick must score in neither branch
        PH = np.array([[picks[h].get(g["gid"]) == g["home"] for h in handles]
                       for g in remaining], dtype=bool) if remaining else np.zeros((0, len(handles)), bool)
        PR = np.array([[picks[h].get(g["gid"]) == g["road"] for h in handles]
                       for g in remaining], dtype=bool) if remaining else np.zeros((0, len(handles)), bool)
        rng = np.random.default_rng()
        home_wins = rng.random((n_sims, len(probs))) < probs
        scores = base[None, :] + ((home_wins[:, :, None] & PH[None, :, :]) |
                                  (~home_wins[:, :, None] & PR[None, :, :])).sum(axis=1)
        better = (scores[:, :, None] > scores[:, None, :]).sum(axis=1)
        tied = (scores[:, :, None] == scores[:, None, :]).sum(axis=1)
        pay = np.array(list(sched) + [0.0] * 32)
        cum = np.concatenate([[0.0], np.cumsum(pay)])
        ev = (cum[np.minimum(better + tied, len(cum) - 1)] -
              cum[np.minimum(better, len(cum) - 1)]) / tied

    entries = []
    for i, h in enumerate(handles):
        row = {"handle": h, "wins": int(base[i]),
               "tb": entries_in[i]["tb"] if all_locked else None,
               "picks": [picks[h].get(g["gid"]) if g["locked"] else None for g in games],
               "p_place": None, "ev": None}
        if all_locked:
            row["p_place"] = [round(float((((better[:, i] < k) & (better[:, i] + tied[:, i] >= k))
                                           / tied[:, i]).mean()), 4) for k in (1, 2, 3, 4)]
            row["ev"] = round(float(ev[:, i].mean()), 2)
        entries.append(row)
    entries.sort(key=lambda r: (-(r["ev"] if r["ev"] is not None else 0.0), -r["wins"]))

    out_games = [{"name": f"{g['road']}@{g['home']}", "home": g["home"],
                  "status": g["status"] if not g["final"] else "final",
                  "winner": g.get("winner"),
                  "p_home": None if g["final"] else round(_game_prob_home(g), 3),
                  "live": (not g["final"]) and g["status"] == "in_progress"}
                 for g in games]
    return {"week": snap["week"], "sim": all_locked,
            "updated": _now().isoformat(timespec="seconds"),
            "pot": snap["pot"], "payouts": sched,
            "entries": entries, "games": out_games}


def payout_for_finish(rank: int, n_tied: int, schedule: list[float]) -> float:
    places = [schedule[p - 1] if 0 <= p - 1 < len(schedule) else 0.0
              for p in range(rank, rank + n_tied)]
    return sum(places) / n_tied


def settled_week_data(snap: dict, games: list[dict]) -> dict:
    """Final week: placements by the validated rule (most wins, then smallest
    absolute TB distance, equal keys share the place and split the money)."""
    tbg = next((g for g in games if g["gid"] == snap.get("tb_gid")), None)
    total = (tbg["rs"] + tbg["hs"]) if tbg else None
    rows = []
    for e in snap["entries"]:
        w, l = _wins_losses(games, e["picks"])
        # Splash scores a missing TB guess as 0 (observed week 10), so the
        # distance exists for everyone once the total is known
        td = abs((e["tb"] or 0) - total) if total is not None else None
        rows.append({"handle": e["handle"], "wins": w, "losses": l,
                     "tb": e["tb"], "td": td,
                     "picks": [e["picks"].get(g["gid"]) for g in games]})
    key = lambda r: (-r["wins"], r["td"] if r["td"] is not None else 9999)
    rows.sort(key=key)
    ranks = {}
    place, prev = 0, None
    for i, r in enumerate(rows, 1):
        if key(r) != prev:
            place, prev = i, key(r)
        ranks[r["handle"]] = place
    n_at = {p: sum(1 for v in ranks.values() if v == p) for p in set(ranks.values())}
    for r in rows:
        rank = ranks[r["handle"]]
        r["place"] = f"T{rank}" if n_at[rank] > 1 else str(rank)
        r["won"] = round(payout_for_finish(rank, n_at[rank], snap["payouts"]), 2)
    return {"week": snap["week"], "final": True,
            "updated": _now().isoformat(timespec="seconds"),
            "pot": snap["pot"], "payouts": snap["payouts"], "tb_total": total,
            "entries": rows,
            "games": [{"name": f"{g['road']}@{g['home']}", "home": g["home"],
                       "status": "final", "winner": g.get("winner")} for g in games]}


def _round_tick(ts: str) -> str:
    """Snap to the 5-minute mark the run fired on; a mid-interval run keeps
    its real time (same convention as the replay series always used)."""
    t = datetime.fromisoformat(ts)
    off = (t.minute * 60 + t.second) % 300
    if off <= 45:
        t -= timedelta(seconds=off)
    elif off >= 255:
        t += timedelta(seconds=300 - off)
    else:
        return ts
    return t.isoformat(timespec="seconds")


def _append_history(data: dict, games: list[dict]) -> None:
    """One compact row per simulated tick -> history_{week}.json."""
    path = ROOT / f"history_{data['week']}.json"
    handles = sorted((e["handle"] for e in data["entries"]), key=str.lower)
    names = [g["name"] for g in data["games"]]
    hist = None
    if path.exists():
        hist = json.loads(path.read_text())
        if hist.get("entries") != handles or hist.get("games") != names:
            return  # roster/schedule drift mid-week; don't corrupt the series
        if hist["rows"] and hist["rows"][-1]["t"] == _round_tick(data["updated"]):
            return
    if hist is None:
        hist = {"week": data["week"], "payouts": data["payouts"],
                "entries": handles, "games": names, "rows": []}
    by = {e["handle"]: e for e in data["entries"]}
    if "picks" not in hist:
        hist["picks"] = [by[h]["picks"] for h in handles]
    if "tb" not in hist:
        hist["tb"] = [by[h].get("tb") for h in handles]
    hist["rows"].append({
        "t": _round_tick(data["updated"]),
        "w": [by[h]["wins"] for h in handles],
        "ev": [by[h]["ev"] for h in handles],
        "p1": [by[h]["p_place"][0] for h in handles],
        "g": [[g.get("rs"), g.get("hs"), dg["p_home"],
               2 if dg["status"] == "final" else 1 if dg["status"] == "in_progress" else 0]
              for g, dg in zip(games, data["games"])],
    })
    path.write_text(json.dumps(hist))


def _append_final_history(data: dict, games: list[dict]) -> None:
    """Settled week: close the replay series with one row of official truth."""
    path = ROOT / f"history_{data['week']}.json"
    if not path.exists():
        return
    hist = json.loads(path.read_text())
    if hist.get("settled"):
        return
    by = {e["handle"]: e for e in data["entries"]}
    if (set(by) != set(hist["entries"])
            or [g["name"] for g in data["games"]] != hist["games"]):
        return
    won1 = lambda e: str(e.get("place")) in ("1", "T1")
    hist["rows"].append({
        "t": _round_tick(data["updated"]),
        "w": [by[h]["wins"] for h in hist["entries"]],
        "ev": [by[h]["won"] for h in hist["entries"]],
        "p1": [1.0 if won1(by[h]) else 0.0 for h in hist["entries"]],
        "g": [[g.get("rs"), g.get("hs"), None, 2] for g in games],
    })
    hist["settled"] = True
    path.write_text(json.dumps(hist))


def gate(snap: dict) -> str | None:
    """Reason to run, or None to idle (keeps the every-5-minutes schedule
    nearly free outside game windows). Runs when a game is live or <45 min
    out, a started game isn't final in our carried state, the page is still
    on a different week, or the settle hasn't been archived yet."""
    now = _now()
    state = load_state(snap["week"])
    old_week = None
    try:
        old_week = json.loads((ROOT / "data.json").read_text()).get("week")
    except (OSError, ValueError):
        pass
    n_picked = sum(1 for e in snap["entries"] if e["picks"])
    if n_picked <= 1:
        return None  # nothing visible yet; keep showing the settled week
    if (ROOT / f"week_{snap['week']}.json").exists():
        # week archived: never touch it again (a later tick could otherwise
        # regress the settled page off a rolled-over ESPN scoreboard)
        return None
    if old_week != snap["week"]:
        return "page on another week"
    for g in snap["games"]:
        start = _iso(g["startsAt"])
        if timedelta(0) <= start - now <= timedelta(minutes=45):
            return f"kickoff soon: {g['road']}@{g['home']}"
        st = state["games"].get(g["gid"], {}).get("status")
        if start <= now and st != "final":
            return f"started, not final: {g['road']}@{g['home']}"
    if snap.get("all_locked") and not (ROOT / f"week_{snap['week']}.json").exists():
        return "settle pending"
    return None


def main() -> None:
    if os.environ.get("TICK_TOUCH"):
        # diagnostic (manual dispatch only): force a commit to verify the
        # bot-push -> Pages-deploy path end to end
        (ROOT / "live" / "heartbeat.json").write_text(
            json.dumps({"t": _now().isoformat(timespec="seconds")}))
        print("touch: wrote live/heartbeat.json")
        return
    if not SNAP.exists():
        print("no snapshot; idle")
        return
    snap = json.loads(SNAP.read_text())
    reason = gate(snap)
    if reason is None:
        print("idle (no live or imminent games, page current)")
        return
    print(f"running: {reason}")
    state = load_state(snap["week"])
    try:
        espn = fetch_espn()
    except Exception as e:  # noqa: BLE001 - degrade to held state
        print(f"espn fetch failed ({e}); using held state only")
        espn = {}
    games = merge_games(snap, espn, state)

    week = snap["week"]
    week_payload = None
    if snap.get("all_locked") and all(g["final"] for g in games):
        data = settled_week_data(snap, games)
        week_payload = json.dumps(data)
    else:
        data = simulate(snap, games)
    for g in games:
        if g["status"] == "in_progress":
            print(f"  wp {g['road']}@{g['home']} espn={g.get('live_p_home')}")

    if data.get("sim"):
        _append_history(data, games)
    elif data.get("final"):
        _append_final_history(data, games)
    data["past_weeks"] = [n for n in range(1, week)
                          if (ROOT / f"week_{n}.json").exists()]
    data["history_weeks"] = sorted(int(p.stem.split("_")[1])
                                   for p in ROOT.glob("history_*.json")
                                   if p.stem.split("_")[1].isdigit())

    STATE.write_text(json.dumps({"week": week, "games": state["games"]}))
    if (ROOT / "data.json").exists():
        old = json.loads((ROOT / "data.json").read_text())
        if ({k: v for k, v in data.items() if k != "updated"} ==
                {k: v for k, v in old.items() if k != "updated"}):
            print("no substantive change; not writing data.json")
            return
    if week_payload is not None:
        (ROOT / f"week_{week}.json").write_text(week_payload)
    (ROOT / "data.json").write_text(json.dumps(data))
    print(f"wrote data.json: week {week}, {len(data['entries'])} entries, "
          f"{sum(1 for g in data['games'] if g['status'] != 'final')} games open"
          + (", settled" if week_payload else ""))


if __name__ == "__main__":
    main()
