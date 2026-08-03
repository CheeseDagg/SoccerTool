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
XG_CACHE = os.path.join(DATA, "understat_xg.json")


# ---------------------------------------------------------------------------
# xG OVERLAY — fit the ratings on expected goals instead of raw goals wherever a
# cached understat xG match joins. Walk-forward validated (soccer_xg_experiment):
# xG-fit beats goals-fit in 3/3 seasons on 3,240 out-of-sample 1X2 predictions
# (holdout Brier +0.0064, log-loss +0.0102). The overlay ONLY touches the copies
# fed to fit_league — grading and the fixture list keep real goals/names. Fail-soft:
# missing/stale cache -> those matches simply fit on goals as before.
def _load_xg_cache():
    if not os.path.exists(XG_CACHE):
        return []
    try:
        with open(XG_CACHE) as f:
            rows = json.load(f)
        return [r for r in rows if r.get("xgh") is not None and r.get("date")]
    except Exception:
        return []


def xg_overlay(matches):
    """-> (new match list for FITTING, human note). Where an understat xG row joins
    (div, mapped home/away, date within 1 day), hg/ag become xG floats."""
    cache = _load_xg_cache()
    if not cache:
        return matches, "xG overlay off (no cache)"
    fd_teams_by_div = {}
    for m in matches:
        fd_teams_by_div.setdefault(m["div"], set()).update((m["home"], m["away"]))
    idx = {}
    for r in cache:
        div = r.get("div")
        teams = fd_teams_by_div.get(div)
        if not teams:
            continue
        h = PR.map_team(r.get("home", ""), teams)
        a = PR.map_team(r.get("away", ""), teams)
        if not h or not a:
            continue
        try:
            d = dt.date.fromisoformat(str(r["date"])[:10])
        except ValueError:
            continue
        idx.setdefault((div, h, a), []).append((d, float(r["xgh"]), float(r["xga"])))
    out, hits, latest = [], 0, None
    for m in matches:
        cands = idx.get((m["div"], m["home"], m["away"]))
        best = None
        if cands:
            for d, xgh, xga in cands:
                delta = abs((d - m["date"]).days)
                if delta <= 1 and (best is None or delta < best[0]):
                    best = (delta, xgh, xga)
        if best is not None:
            m2 = dict(m)
            # Carry the REAL scoreline alongside the xG response. fit_league fits
            # lambda on hg/ag but the Dixon-Coles rho term is only defined on integer
            # scorelines -- without these it has nothing to fit rho on and pins it to
            # 0. Overwriting hg/ag and dropping the goals is what left rho frozen at
            # its -0.05 initialisation in all four leagues.
            m2["g_h"], m2["g_a"] = m["hg"], m["ag"]
            m2["hg"], m2["ag"] = best[1], best[2]
            out.append(m2); hits += 1
            if latest is None or m["date"] > latest:
                latest = m["date"]
        else:
            # No xG join: hg/ag are already the real goals, but g_h/g_a must be present
            # on EVERY row or fit_league's rho step sees a partially-annotated list and
            # falls back to pinning rho at 0 for the whole league.
            m2 = dict(m); m2["g_h"], m2["g_a"] = m["hg"], m["ag"]
            out.append(m2)
    if not hits:
        return matches, "xG overlay off (0 joins — check cache/team mapping)"
    return out, f"xG overlay: {hits}/{len(matches)} matches fit on xG (through {latest.isoformat()})"

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
    # ratings fit on the xG overlay (validated win); grading above and fixtures below
    # keep real goals & the original match list.
    xg_matches, xg_note = xg_overlay(matches)
    print(f"   {xg_note}")

    leagues_out = {}
    log_rows = []
    _today_iso = dt.date.today().isoformat()   # don't price/log fixtures that already kicked off
    for div, name in M.LEAGUES.items():
        ms = [m for m in xg_matches if m["div"] == div]
        if len(ms) < 60:
            leagues_out[div] = {"name": name, "error": f"only {len(ms)} matches fetched"}
            continue
        ratings, home_adv, rho, mu = M.fit_league(ms)
        # Read these IMMEDIATELY -- they are function attributes, so the next fit
        # overwrites them. Both are diagnostics that were previously invisible: a
        # rho pinned at its init and a lambda level denominated in xG rather than
        # goals both look exactly like a normal fit from the outside.
        _rho_src = getattr(M.fit_league, "rho_source", None)
        _lvl = getattr(M.fit_league, "level_scale", (1.0, 1.0))
        latest = max(m["date"] for m in ms)
        table = sorted(({"team": t, "att": round(a, 3), "dfn": round(d, 3),
                         "idx": round(100 * math.exp(a + d), 1)}     # single-number strength
                        for t, (a, d) in ratings.items()), key=lambda r: -r["idx"])
        fx_out = []
        for f in sorted((f for f in fixtures if f["div"] == div
                         and f["date"].isoformat()[:10] >= _today_iso), key=lambda x: x["date"]):
            if f["home"] not in ratings or f["away"] not in ratings: continue
            p = M.match_probs(ratings, home_adv, rho, mu, f["home"], f["away"])
            # lh/la are the whole model. Every derived market -- any total line,
            # any correct score, any Asian handicap, clean sheets, win-to-nil --
            # is score_grid(lh, la, rho) summed over a different region of the
            # same 9x9 matrix (MAX_GOALS=8). Publishing only five pre-chosen aggregates
            # (pH/pD/pA/o25/btts) forced a Python rebuild for every new question;
            # publishing the two rates plus rho makes the browser able to answer
            # questions nobody has thought of yet, with zero new modelling and
            # ~40 lines of JS. Six decimals because exp() is sensitive and these
            # get re-exponentiated into a grid, not printed.
            row = {"date": f["date"].isoformat(), "home": f["home"], "away": f["away"],
                   "pH": round(p["pH"], 3), "pD": round(p["pD"], 3), "pA": round(p["pA"], 3),
                   "o25": round(p["o25"], 3), "btts": round(p["btts"], 3),
                   "lh": round(p["lh"], 6), "la": round(p["la"], 6)}
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
        # mu is the league's baseline log-goal-rate and was the one fitted
        # parameter never published. Without it the browser can read a rating
        # table but cannot reconstruct a rate for a matchup that is not already
        # on the fixture list -- which is exactly what a "what if these two
        # played" tool has to do. att/dfn/home_adv/rho were already here; mu
        # completes the set, and lh = exp(mu + home_adv + att_h - dfn_a) becomes
        # computable client-side for ANY pair of teams in the league.
        leagues_out[div] = {"name": name, "n_matches": len(ms),
                            "latest_result": latest.isoformat(),
                            "home_adv": round(home_adv, 3), "rho": round(rho, 3),
                            "mu": round(mu, 6),
                            "rho_source": _rho_src,
                            "level_scale": [round(_lvl[0], 4), round(_lvl[1], 4)],
                            "ratings": table, "fixtures": fx_out}
        print(f"   {div}: {len(ratings)} teams, {len(fx_out)} priced fixtures, "
              f"top {table[0]['team']} {table[0]['idx']}  "
              f"[rho {rho:+.3f} from {_rho_src} · level x{_lvl[0]:.3f}/{_lvl[1]:.3f}]")

    n_logged = G.log_predictions(log_rows) if log_rows else 0
    print(f"4) logged {n_logged} new predictions")

    print("5) walk-forward backtests (model vs closing market)…")
    # xg_matches, NOT matches. Step 3 fits and publishes on the xG overlay; passing the
    # raw-goals list here measured the GOALS model and printed its accuracy under the
    # published board. Those are different models -- the whole reason the overlay exists
    # is that they score differently out of sample -- so the number on the dashboard was
    # not the record of the number beside it. backtest_league refits internally on
    # whatever it is handed, and xg_matches carries g_h/g_a, so grading still reads real
    # results.
    bts = _backtests(xg_matches)
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
           "leagues": leagues_out,
           "note": note + " · " + xg_note + " · props " + " ".join(props_note), "cal": cal}
    with open(os.path.join(DATA, "slate.json"), "w") as f:
        json.dump(_scrub(out), f, indent=1, allow_nan=False)
    total_fx = sum(len(v.get("fixtures", [])) for v in leagues_out.values())
    print(f"slate.json written: {len(leagues_out)} leagues, {total_fx} fixtures, cal n={cal.get('n',0)}")

def _selftest_overlay():
    """Offline check of the xG overlay join: alias mapping, ±1-day tolerance,
    fail-soft, and no mutation of the original (grading) list."""
    import tempfile
    global XG_CACHE
    _orig = XG_CACHE
    tmp = tempfile.mkdtemp()
    XG_CACHE = os.path.join(tmp, "xg.json")
    matches = [
        {"div": "E0", "date": dt.date(2026, 3, 7), "home": "Man United", "away": "Wolves", "hg": 1, "ag": 0},
        {"div": "E0", "date": dt.date(2026, 3, 8), "home": "Arsenal", "away": "Leeds", "hg": 2, "ag": 2},
        {"div": "SP1", "date": dt.date(2026, 3, 7), "home": "Girona", "away": "Betis", "hg": 0, "ag": 1},
    ]
    # no cache -> unchanged, off
    out, note = xg_overlay(matches)
    assert out == matches and "off" in note, note
    # cache: alias-mapped name + 1-day date drift joins; unrelated league row ignored
    json.dump([
        {"div": "E0", "date": "2026-03-06", "home": "Manchester United", "away": "Wolverhampton Wanderers",
         "g_h": 1, "g_a": 0, "xgh": 1.9, "xga": 0.4},
        {"div": "D1", "date": "2026-03-07", "home": "Bayern Munich", "away": "Dortmund",
         "g_h": 3, "g_a": 1, "xgh": 2.5, "xga": 1.1},
    ], open(XG_CACHE, "w"))
    out, note = xg_overlay(matches)
    assert out[0]["hg"] == 1.9 and out[0]["ag"] == 0.4, out[0]        # joined via alias + ±1 day
    assert out[1]["hg"] == 2 and out[2]["hg"] == 0                    # unjoined stay on goals
    assert matches[0]["hg"] == 1, "original (grading) list must NOT be mutated"
    assert "1/3" in note, note
    # THE RHO GUARD. Every row -- joined or not -- must carry the real integer
    # scoreline as g_h/g_a. fit_league's rho step is only defined on integer
    # scorelines; if ANY row is missing them it pins rho to 0 for the whole league,
    # and if they are missing everywhere (the old behaviour) rho silently stays at
    # its -0.05 initialisation. backtest_league also grades off these, so without
    # them the walk-forward would score "H/D/A" against expected goals.
    assert all(m.get("g_h") is not None and m.get("g_a") is not None for m in out), \
        "every overlay row must carry the real scoreline for the rho step and for grading"
    assert (out[0]["g_h"], out[0]["g_a"]) == (1, 0), out[0]     # joined row: real result kept
    assert (out[1]["g_h"], out[1]["g_a"]) == (2, 2), out[1]     # unjoined row: annotated too
    import soccer_model as _M
    _M.fit_league.rho_source = None
    _fitrows = [dict(m, date=dt.date(2026, 1, 1) + dt.timedelta(days=i % 90))
                for i, m in enumerate(out * 30)]
    _M.fit_league(_fitrows, iters=20)
    assert _M.fit_league.rho_source == "goals", \
        f"annotated xG rows must fit rho on real goals, got {_M.fit_league.rho_source}"
    _M.fit_league([{k: v for k, v in m.items() if k not in ("g_h", "g_a")} for m in _fitrows],
                  iters=20)
    assert _M.fit_league.rho_source == "unidentifiable-pinned-0", \
        "stripping the scorelines must pin rho at 0, never leave it at the init"
    # 2-day drift must NOT join
    json.dump([{"div": "E0", "date": "2026-03-05", "home": "Manchester United",
                "away": "Wolverhampton Wanderers", "g_h": 1, "g_a": 0, "xgh": 1.9, "xga": 0.4}],
              open(XG_CACHE, "w"))
    out, _ = xg_overlay(matches)
    assert out[0]["hg"] == 1, "2-day-away xG row must not join"
    XG_CACHE = _orig
    print("XG-OVERLAY SELFTEST PASS — alias join, ±1-day tolerance, fail-soft, no mutation")
    return 0


if __name__ == "__main__":
    import sys
    if "--selftest-overlay" in sys.argv:
        sys.exit(_selftest_overlay())
    main()
