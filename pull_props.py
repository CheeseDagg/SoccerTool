#!/usr/bin/env python3
"""
pull_props.py — run AT HOME (understat blocks datacenter IPs, serves you fine).
Fetches player goals/xG/minutes for all four leagues, writes the pin:
    data/player_shares_pin.json
Commit that file; the daily Action's Props tab runs on it until your next pull.
Once a month is plenty offseason; weekly in season.
"""
import json, os, datetime as dt
import soccer_props as PR

HERE = os.path.dirname(os.path.abspath(__file__))
season = dt.date.today().year - (1 if dt.date.today().month < 8 else 0)
out = {"asof": dt.date.today().isoformat(), "season": season, "leagues": {}}
for div in PR.UNDERSTAT:
    try:
        players = PR.fetch_league_players(div, season)
        out["leagues"][div] = players
        print(f"{div}: {len(players)} players")
    except Exception as e:
        print(f"{div}: FAILED — {str(e)[:400]}")
os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
path = os.path.join(HERE, "data", "player_shares_pin.json")
json.dump(out, open(path, "w"))
n = sum(len(v) for v in out["leagues"].values())
print(f"\npin written: {path} ({n} players, season {season}/{season+1})")
print("-> commit data/player_shares_pin.json and the Props tab lights up next run")
