"""
soccer_publish.py — run the whole pipeline, emit data/slate.json.
Per league: time-decayed Dixon-Coles fit -> ratings table + priced upcoming
fixtures (1X2 / O2.5 / BTTS, with devigged market anchors when books have
lines). Predictions logged pre-match; prior logs graded from the same freshly
fetched results. Walk-forward backtest (model vs closing market) recomputed
weekly and cached. NaN-scrubbed, allow_nan=False.
"""
import os, json, math, datetime as dt
import soccer_model as M
import soccer_grade as G
import soccer_props as PR

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
BT_CACHE = os.path.join(DATA, "backtest_cache.json")
BT_MAX_AGE_DAYS = 7      # backtest is expensive; recompute weekly

def _scrub(o):
    if isinstance(o, float) and not math.isfinite(o): return None
    if isinstance(o, dict):  return {k: _scrub(v) for k, v in o.items()}
    if isinstance(o, list):  return [_scrub(v) for v in o]
    return o

def _backtests(matches):
    try:
        if os.path.exists(BT_CACHE):
            c = json.load(open(BT_CACHE))
            age = (dt.date.today() - dt.date.fromisoformat(c.get("asof", "1970-01-01"))).days
            if age < BT_MAX_AGE_DAYS and set(c.get("leagues", {})) == set(M.LEAGUES):
                print(f"   backtest cache: {age}d old, reusing")
                return c["leagues"]
    except Exception:
        pass
    out = {}
    for div in M.LEAGUES:
        ms = [m for m in matches if m["div"] == div]
        print(f"   backtesting {div} ({len(ms)} matches)…")
        try:
            out[div] = M.backtest_league(ms)
        except Exception as e:
            out[div] = None
            print(f"   {div} backtest failed: {type(e).__name__}")
    try:
        json.dump({"asof": dt.date.today().isoformat(), "leagues": out}, open(BT_CACHE, "w"))
    except Exception:
        pass
    return out

def main():
    os.makedirs(DATA, exist_ok=True)
    print("1) fetch results + fixtures (football-data)…")
    matches, fixtures, note = M.fetch_all()
    print(f"   {note}")
    if not matches:
        raise SystemExit("no results fetched — refusing to publish an empty slate")

    print("2) grade prior predictions from fresh results…")
    n_graded, cal = G.grade_all(matches)
    print(f"   settled {n_graded} | panel n={cal.get('n', 0)}")

    print("2b) player shares (understat)…")
    # understat keys seasons by START year; before August the completed
    # season (year-1) is the one that exists — July 2026 -> 2025 (i.e. 25/26)
    season_year = dt.date.today().year - (1 if dt.date.today().month < 8 else 0)
    props_note = []
    shares_by_div = {}
    pin = {}
    try:
        pin = json.load(open(os.path.join(DATA, "player_shares_pin.json")))
    except Exception:
        pass
    pin_age = None
    if pin.get("asof"):
        pin_age = (dt.date.today() - dt.date.fromisoformat(pin["asof"])).days
    for div in M.LEAGUES:
        fd_teams = {m["home"] for m in matches if m["div"] == div} | \
                   {m["away"] for m in matches if m["div"] == div}
        try:
            players = PR.fetch_league_players(div, season_year)
            shares_by_div[div] = PR.team_shares(players, fd_teams)
            props_note.append(f"{div}:{len(players)}p/{len(shares_by_div[div])}t")
        except Exception as e:
            pinned = (pin.get("leagues") or {}).get(div) or []
            if pinned:
                shares_by_div[div] = PR.team_shares(pinned, fd_teams)
                props_note.append(f"{div}:pin({len(pinned)}p,{pin_age}d)")
            else:
                shares_by_div[div] = {}
                props_note.append(f"{div}:off({type(e).__name__})")
                print(f"   {div} props source failed — {e}"[:900])
    print("   " + " · ".join(props_note))

    print("3) fit + price per league…")
    leagues_out = {}
    log_rows = []
    _today_iso = dt.date.today().isoformat()   # don't price/log fixtures that already kicked off
    for div, name in M.LEAGUES.items():
        ms = [m for m in matches if m["div"] == div]
        if len(ms) < 60:
            leagues_out[div] = {"name": name, "error": f"only {len(ms)} matches fetched"}
            continue
        ratings, home_adv, rho, mu = M.fit_league(ms)
        latest = max(m["date"] for m in ms)
        table = sorted(({"team": t, "att": round(a, 3), "dfn": round(d, 3),
                         "idx": round(100 * math.exp(a + d), 1)}     # single-number strength
                        for t, (a, d) in ratings.items()), key=lambda r: -r["idx"])
        fx_out = []
        for f in sorted((f for f in fixtures if f["div"] == div
                         and f["date"].isoformat()[:10] >= _today_iso), key=lambda x: x["date"]):
            if f["home"] not in ratings or f["away"] not in ratings: continue
            p = M.match_probs(ratings, home_adv, rho, mu, f["home"], f["away"])
            row = {"date": f["date"].isoformat(), "home": f["home"], "away": f["away"],
                   "pH": round(p["pH"], 3), "pD": round(p["pD"], 3), "pA": round(p["pA"], 3),
                   "o25": round(p["o25"], 3), "btts": round(p["btts"], 3)}
            if f.get("mh") and f.get("md") and f.get("ma"):
                ih, idd, ia = 1/f["mh"], 1/f["md"], 1/f["ma"]; s = ih + idd + ia
                row.update({"qH": round(ih/s, 3), "qD": round(idd/s, 3), "qA": round(ia/s, 3)})
            sh = shares_by_div.get(div, {})
            if sh.get(f["home"]) or sh.get(f["away"]):
                row["scorers"] = {
                    "home": PR.anytime_probs(p["lh"], sh.get(f["home"], [])),
                    "away": PR.anytime_probs(p["la"], sh.get(f["away"], []))}
            fx_out.append(row)
            log_rows.append({k: v for k, v in row.items() if k != "scorers"} | {"div": div})
        leagues_out[div] = {"name": name, "n_matches": len(ms),
                            "latest_result": latest.isoformat(),
                            "home_adv": round(home_adv, 3), "rho": round(rho, 3),
                            "ratings": table, "fixtures": fx_out}
        print(f"   {div}: {len(ratings)} teams, {len(fx_out)} priced fixtures, "
              f"top {table[0]['team']} {table[0]['idx']}")

    n_logged = G.log_predictions(log_rows) if log_rows else 0
    print(f"4) logged {n_logged} new predictions")

    print("5) walk-forward backtests (model vs closing market)…")
    bts = _backtests(matches)
    for div, bt in bts.items():
        if bt:
            leagues_out.setdefault(div, {})["backtest"] = bt

    props_src = " · ".join(props_note)
    # Real freshness signal = the fixtures' own dates, not wall-clock `generated`. If the
    # build stalls, slate_end falls into the past and the dashboard can warn.
    _fx_dates = sorted(r["date"][:10] for v in leagues_out.values() for r in v.get("fixtures", []))
    out = {"props_src": props_src,
           "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "slate_date": _fx_dates[0] if _fx_dates else None,
           "slate_end": _fx_dates[-1] if _fx_dates else None,
           "leagues": leagues_out, "note": note + " · props " + " ".join(props_note), "cal": cal}
    with open(os.path.join(DATA, "slate.json"), "w") as f:
        json.dump(_scrub(out), f, indent=1, allow_nan=False)
    total_fx = sum(len(v.get("fixtures", [])) for v in leagues_out.values())
    print(f"slate.json written: {len(leagues_out)} leagues, {total_fx} fixtures, cal n={cal.get('n',0)}")

if __name__ == "__main__":
    main()
