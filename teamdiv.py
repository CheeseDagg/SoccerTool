#!/usr/bin/env python3
"""teamdiv.py -- each team's CURRENT division, from its most recent match.

    python3 teamdiv.py            # writes data/team_division.json
    python3 teamdiv.py --selftest

The cross-division scale is useless without this. A team promoted or relegated
inside the 3-season window carries a rating in BOTH divisions, and averaging
them blends this season's form with an older season's in a league it has left:
Bromley reads idx 38.7 in League One (now) and 141.2 in League Two (then), so
the average flatters it by a wide margin and invented a +5.6 "edge" against MK
Dons that was nothing but the blend. Five of eight priced cup legs were that
artifact.

The ratings carry no dates, but the match rows do. For each team, take the
division of its latest result -- that is the division it plays in now, and the
only rating of it that describes tonight's side.
"""
import json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def current_division(matches):
    """team -> (division, date of its latest match). Ties break on the division
    seen at that latest date; a team with no dated match is absent, not guessed."""
    best = {}
    for m in matches:
        d = m.get("date")
        if not d:
            continue
        for side in ("home", "away"):
            t = m.get(side)
            if not t:
                continue
            cur = best.get(t)
            if cur is None or d > cur[1]:
                best[t] = (m.get("div"), d)
    return best


def main():
    sys.path.insert(0, HERE)
    import soccer_model as M
    matches, _fx, note = M.fetch_all()
    print(note[:400])
    if not matches:
        print("no results fetched — refusing to write a division map from nothing")
        return 1
    rows = [{"div": m["div"], "date": m["date"].isoformat(),
             "home": m["home"], "away": m["away"]} for m in matches]
    cur = current_division(rows)
    os.makedirs(DATA, exist_ok=True)
    out = {t: {"div": d, "latest": dt} for t, (d, dt) in sorted(cur.items())}
    with open(os.path.join(DATA, "team_division.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    per = defaultdict(int)
    for v in out.values():
        per[v["div"]] += 1
    print(f"wrote {len(out)} teams: " + " ".join(f"{k}={v}" for k, v in sorted(per.items())))
    return 0


def selftest():
    bad = []

    def ck(c, m):
        if not c:
            bad.append(m); print("  FAIL", m)

    ms = [
        {"div": "E3", "date": "2025-04-01", "home": "Bromley", "away": "Barnet"},
        {"div": "E2", "date": "2026-11-04", "home": "Bromley", "away": "Wigan"},
        {"div": "E3", "date": "2026-10-01", "home": "Barnet", "away": "Crewe"},
    ]
    cur = current_division(ms)
    ck(cur["Bromley"][0] == "E2",
       f"Bromley's LATEST match is in League One, so that is its division: {cur['Bromley']}")
    ck(cur["Bromley"][1] == "2026-11-04", f"and the date comes with it: {cur['Bromley']}")
    ck(cur["Barnet"][0] == "E3", f"Barnet is E3 on its own latest: {cur['Barnet']}")
    ck(cur["Crewe"][0] == "E3", "an away-only appearance still counts")
    # order must not matter
    ck(current_division(list(reversed(ms)))["Bromley"][0] == "E2",
       "file order must not decide the answer")
    # an undated row is ignored, never guessed
    ck("Ghost" not in current_division([{"div": "E2", "date": None,
                                        "home": "Ghost", "away": "X"}]),
       "a team with no dated match must be ABSENT, not assigned a division")
    ck(current_division([]) == {}, "no matches is an empty map, not a crash")
    # the whole point: an older division must never win
    older = [{"div": "E0", "date": "2024-05-01", "home": "Leicester", "away": "A"},
             {"div": "E1", "date": "2025-05-01", "home": "Leicester", "away": "B"},
             {"div": "E2", "date": "2026-11-01", "home": "Leicester", "away": "C"}]
    ck(current_division(older)["Leicester"][0] == "E2",
       f"three divisions, the newest wins: {current_division(older)['Leicester']}")
    print("teamdiv selftest:", "FAILED" if bad else "ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv[1:] else main())
