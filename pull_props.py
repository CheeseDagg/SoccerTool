#!/usr/bin/env python3
"""
pull_props.py — run AT HOME (understat blocks datacenter IPs, serves you fine).
Fetches player goals/xG/minutes for all four leagues, writes the pin:
    data/player_shares_pin.json
Commit that file; the daily Action's Props tab runs on it until your next pull.
Once a month is plenty offseason; weekly in season.

TWO THINGS THIS SCRIPT USED TO DO WRONG, both of which turn a bad afternoon into
a bad month:

  1. It wrote the pin UNCONDITIONALLY. If every league failed -- which is exactly
     what happens when understat changes layout, i.e. the one time the pin is the
     only thing keeping Props alive -- it overwrote a perfectly good pin with
     {"leagues": {}} and the fallback was gone too.
  2. It exited 0 no matter what. A run where nothing was fetched looked, to the
     shell and to any wrapper, identical to a clean pull.

Now: a league that fails is CARRIED FORWARD from the existing pin rather than
dropped, the file is only replaced when at least one league actually came back,
the write is atomic, and the exit code is non-zero if anything failed.
"""
import json, os, sys, datetime as dt
import soccer_props as PR

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "data", "player_shares_pin.json")

# Season can be forced from the command line: `python3 pull_props.py 2025`.
# Otherwise it is guessed from the date -- and the guess needs a fallback,
# because the August 1 rollover bites BEFORE the new season has any data in it.
# On 2026-08-03 this computed season 2026, understat's 2026/27 pages existed but
# carried an empty player list (zero minutes played league-wide), and all four
# leagues "failed" on a site that was up the whole time. Early-season shares
# have to come from the completed season anyway, so an empty current season is
# not an error -- it is a season-minus-one situation, handled per-league below.
season = (int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit()
          else dt.date.today().year - (1 if dt.date.today().month < 8 else 0))

prev = {}
try:
    prev = json.load(open(PATH))
    print(f"existing pin: asof {prev.get('asof')} season {prev.get('season')} "
          f"({sum(len(v) for v in (prev.get('leagues') or {}).values())} players)")
except FileNotFoundError:
    print(f"no existing pin at {PATH} — this is the first pull")
except Exception as e:
    print(f"existing pin unreadable ({type(e).__name__}) — treating as absent")

prev_leagues = prev.get("leagues") or {}
out = {"asof": dt.date.today().isoformat(), "season": season,
       "leagues": {}, "carried": {}}
ok, failed = [], []
for div in PR.UNDERSTAT:
    try:
        try:
            players = PR.fetch_league_players(div, season)
            got = season
        except Exception as e_cur:
            # The current season being empty or absent is expected in August;
            # last season's shares are the right answer then, not a failure.
            # Only if season-1 ALSO fails do we report the original error.
            print(f"{div}: season {season} unavailable "
                  f"({str(e_cur)[:120]}) — falling back to {season - 1}")
            players = PR.fetch_league_players(div, season - 1)
            got = season - 1
        out["leagues"][div] = players
        ok.append(div)
        if got != season:
            out.setdefault("fallback_season", {})[div] = got
        print(f"{div}: {len(players)} players (season {got}/{got + 1})")
    except Exception as e:
        failed.append(div)
        print(f"{div}: FAILED — {str(e)[:600]}")
        # Keep whatever the last good pull had for this league rather than
        # publishing a hole. Stale shares beat no shares; both beat a silent hole.
        carried = prev_leagues.get(div)
        if carried:
            out["leagues"][div] = carried
            out["carried"][div] = prev.get("asof")
            print(f"{div}: carried {len(carried)} players from the "
                  f"{prev.get('asof')} pin")

if not ok:
    print(f"\nNOTHING FETCHED ({len(failed)} league(s) failed). Refusing to "
          f"overwrite\n  {PATH}\nbecause the old pin is what Props is running on.")
    sys.exit(1)

# A carried league's rows are from an older pull and possibly an older season.
# Record that so soccer_publish can label it instead of presenting it as current.
if out["carried"]:
    out["carried_season"] = prev.get("season")

# If every league that fetched came from the fallback season, the pin IS a
# season-minus-one pin and must say so in its top-level season field --
# soccer_publish compares that field against its own season guess to decide
# whether to label the shares as stale. Leaving the guessed season here would
# dress last season's shares in this season's label, which is the exact
# mislabeling the pin_stale_season check exists to catch.
fb = out.get("fallback_season", {})
if fb and set(fb) >= set(ok) and len(set(fb.values())) == 1:
    out["season"] = next(iter(fb.values()))
    season = out["season"]

os.makedirs(os.path.dirname(PATH), exist_ok=True)
tmp = PATH + ".tmp"
with open(tmp, "w") as fh:
    json.dump(out, fh)
os.replace(tmp, PATH)          # atomic: a crash mid-write cannot truncate the pin

n = sum(len(v) for v in out["leagues"].values())
missing = [d for d in failed if d not in out["leagues"]]
print(f"\npin written: {PATH} ({n} players, season {season}/{season+1})")
print("  fresh: " + (",".join(ok) or "none")
      + (f" | carried from {prev.get('asof')}: {','.join(out['carried'])}"
         if out["carried"] else "")
      + (f" | STILL MISSING: {','.join(missing)}" if missing else ""))
print("-> commit data/player_shares_pin.json and the Props tab lights up next run")
sys.exit(1 if failed else 0)
