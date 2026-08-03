"""
soccer_model.py — Dixon-Coles goals model for EPL / La Liga / Bundesliga / Ligue 1
==================================================================================
Same engine family as the World Cup tool, rebuilt as an auto-updating pipeline.

Model: independent Poisson goals with team attack/defense strengths, home
advantage, and the Dixon-Coles rho correction for low-scoring dependence.
Matches are time-decayed (recent form counts more). Fit by maximum likelihood.

Data: football-data.co.uk season CSVs (results + closing odds) and fixtures.csv
(upcoming matches). NOTE: unreachable from the dev sandbox — the GitHub Action
and Ryan's machine can reach it. Everything here is therefore validated by
SYNTHETIC RECOVERY: generate seasons from known parameters, prove the fitter
recovers them. See selftest.

Honesty discipline (house rule): every backtest reports the model NEXT TO the
closing market on the same matches, plus the disagreement record. No market
claim is ever implied that the numbers don't show.
"""
import os, io, csv, math, json, urllib.request, datetime as dt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

LEAGUES = {  # div code -> display name
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
}
SEASONS = ["2324", "2425", "2526"]          # rolling window fetched each run
BASE = "https://www.football-data.co.uk/mmz4281/{s}/{d}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

DECAY_HALFLIFE_DAYS = 240.0                  # form weighting: w = 0.5 ** (age/halflife)
MAX_GOALS = 8                                # scoreline grid size
RIDGE = 0.02                                 # small L2 on ratings for stability

# ---------------------------------------------------------------- data layer
def _http(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (SoccerTool)"})
    with urllib.request.urlopen(req, timeout=45) as r:
        return r.read().decode("utf-8", "replace")

def _parse_date(s):
    s = (s or "").strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try: return dt.datetime.strptime(s, fmt).date()
        except ValueError: pass
    return None

def _f(x):
    try: return float(x)
    except (TypeError, ValueError): return None

def parse_results_csv(text, div):
    """football-data season CSV -> match dicts. Tolerant of column drift."""
    out = []
    rdr = csv.DictReader(io.StringIO(text))
    for r in rdr:
        d = _parse_date(r.get("Date"))
        h, a = (r.get("HomeTeam") or "").strip(), (r.get("AwayTeam") or "").strip()
        hg, ag = r.get("FTHG"), r.get("FTAG")
        if not (d and h and a and hg not in (None, "") and ag not in (None, "")):
            continue
        # closing 1X2: prefer Pinnacle closing, fall back Bet365
        mh = _f(r.get("PSCH")) or _f(r.get("PSH")) or _f(r.get("B365CH")) or _f(r.get("B365H"))
        md = _f(r.get("PSCD")) or _f(r.get("PSD")) or _f(r.get("B365CD")) or _f(r.get("B365D"))
        ma = _f(r.get("PSCA")) or _f(r.get("PSA")) or _f(r.get("B365CA")) or _f(r.get("B365A"))
        mo = _f(r.get("PC>2.5")) or _f(r.get("P>2.5")) or _f(r.get("B365C>2.5")) or _f(r.get("B365>2.5"))
        mu_ = _f(r.get("PC<2.5")) or _f(r.get("P<2.5")) or _f(r.get("B365C<2.5")) or _f(r.get("B365<2.5"))
        out.append({"div": div, "date": d, "home": h, "away": a,
                    "hg": int(float(hg)), "ag": int(float(ag)),
                    "mh": mh, "md": md, "ma": ma, "mo": mo, "mu": mu_})
    return out

def parse_fixtures_csv(text):
    """fixtures.csv (all leagues) -> upcoming matches for our divs."""
    out = []
    rdr = csv.DictReader(io.StringIO(text))
    for r in rdr:
        div = (r.get("Div") or "").strip()
        if div not in LEAGUES: continue
        d = _parse_date(r.get("Date"))
        h, a = (r.get("HomeTeam") or "").strip(), (r.get("AwayTeam") or "").strip()
        if not (d and h and a): continue
        out.append({"div": div, "date": d, "home": h, "away": a,
                    "mh": _f(r.get("PSH")) or _f(r.get("B365H")),
                    "md": _f(r.get("PSD")) or _f(r.get("B365D")),
                    "ma": _f(r.get("PSA")) or _f(r.get("B365A"))})
    return out

def fetch_all():
    os.makedirs(DATA, exist_ok=True)
    matches, notes = [], []
    for div in LEAGUES:
        got = 0
        for s in SEASONS:
            try:
                txt = _http(BASE.format(s=s, d=div))
                rows = parse_results_csv(txt, div)
                matches += rows; got += len(rows)
            except Exception as e:
                notes.append(f"{div} {s}: {type(e).__name__}")
        notes.append(f"{div}: {got} results")
    try:
        fixtures = parse_fixtures_csv(_http(FIXTURES_URL))
        notes.append(f"fixtures: {len(fixtures)}")
    except Exception as e:
        fixtures = []; notes.append(f"fixtures: {type(e).__name__} (offseason-normal)")
    return matches, fixtures, " · ".join(notes)

# ------------------------------------------------------------- Dixon-Coles
def _tau(x, y, lh, la, rho):
    """DC low-score dependence correction.

    NOTE THE EQUALITY TESTS. tau is defined on the four INTEGER scorelines 0-0,
    0-1, 1-0, 1-1 and is exactly 1.0 everywhere else. Hand it a continuous
    response -- xG of 0.31, say -- and every branch misses, tau is identically
    1.0, the rho log-likelihood is flat, and rho never moves off whatever it was
    initialised to. See fit_league's rho block for why that matters and what is
    done about it."""
    if x == 0 and y == 0: return 1 - lh * la * rho
    if x == 0 and y == 1: return 1 + lh * rho
    if x == 1 and y == 0: return 1 + la * rho
    if x == 1 and y == 1: return 1 - rho
    return 1.0

def fit_league(matches, asof=None, halflife=DECAY_HALFLIFE_DAYS, iters=400, lr=0.05):
    """ML fit of attack/defense per team + home advantage + rho.
    Returns dict(team->(att,dfn)), home_adv, rho, mu (league base).

    THE RESPONSE MAY BE CONTINUOUS. soccer_publish fits lambda on understat xG
    rather than on goals (it wins the walk-forward), which means m["hg"]/m["ag"]
    arrive as floats. The Dixon-Coles rho term is not defined on a continuous
    response -- _tau only fires on integer 0/1 scorelines -- so on an xG fit the
    rho gradient was identically zero and rho stayed at its -0.05 initialisation
    in every league. Measured 2026-08-03 on the cached understat joins, fitting
    each league both ways:

        div   rho fitted on goals    rho on xG
        D1          -0.0567           -0.0500   <- the init, to 4dp
        E0          -0.1107           -0.0500
        F1          -0.0516           -0.0500
        SP1         +0.0179           -0.0500

    Four leagues, four times the same number, and that number is the constant at
    the top of this function. That is not a fit. It matters because score_grid()
    is always evaluated on INTEGER scorelines, so rho is applied to the 0-0/0-1/
    1-0/1-1 cells of every published grid -- an unfitted constant was steering
    the draw price and both team-total unders. E0 in particular wants roughly
    twice the dependence the init assumes, and SP1 wants the opposite sign.

    The fix is not to round xG. rho is a statement about the joint distribution
    of REAL scorelines, and real scorelines are available on the same matches:
    when the response is continuous, lambda is still fitted on it, but the rho
    step reads m["g_h"]/m["g_a"] (the integer goals, carried alongside). If those
    are absent there is nothing to fit rho on and it is pinned to 0.0 -- no
    correction at all -- rather than left at an arbitrary non-zero prior.
    fit_league.rho_source records which of the three happened.

    The same carried goals also fix the LEVEL. A continuous response is rescaled
    per side so its weighted mean matches the weighted mean of the real goals
    before mu is taken -- see the LEVEL CALIBRATION block below for the measured
    7.7% inflation this removes. fit_league.level_scale records the factors."""
    if asof is None: asof = max(m["date"] for m in matches)
    teams = sorted({m["home"] for m in matches} | {m["away"] for m in matches})
    ti = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    att = np.zeros(n); dfn = np.zeros(n)
    home = math.log(1.25); rho = -0.05
    w = np.array([0.5 ** (max((asof - m["date"]).days, 0) / halflife) for m in matches])
    hg = np.array([m["hg"] for m in matches], dtype=float)
    ag = np.array([m["ag"] for m in matches], dtype=float)
    hi = np.array([ti[m["home"]] for m in matches]); ai = np.array([ti[m["away"]] for m in matches])

    # --- which scorelines the rho step is allowed to read (see the docstring) ---
    _cont = bool(np.any(hg != np.round(hg)) or np.any(ag != np.round(ag)))
    if not _cont:
        rhg, rag, rho_source = hg, ag, "response"
    elif all(m.get("g_h") is not None and m.get("g_a") is not None for m in matches):
        rhg = np.array([m["g_h"] for m in matches], dtype=float)
        rag = np.array([m["g_a"] for m in matches], dtype=float)
        rho_source = "goals"
    else:
        # Continuous response and no integer scorelines carried alongside. rho is
        # unidentifiable here; 0.0 (no correction) is the honest value. Leaving the
        # -0.05 init would ship an unfitted constant into every score grid.
        rhg = rag = None
        rho, rho_source = 0.0, "unidentifiable-pinned-0"
    fit_league.rho_source = rho_source

    # --- LEVEL CALIBRATION: xG is not goals, and the gap is not small ---
    # mu below is log(weighted mean of the response), so every lambda this function
    # returns is denominated in whatever units the response is in. Fit on understat
    # xG and the whole grid is denominated in xG. Measured 2026-08-03 over the 4,115
    # cached joins:
    #
    #     div      n   mean xG tot   mean goals   ratio
    #     D1     918         3.334        3.196    1.043
    #     E0    1140         3.214        2.988    1.076
    #     F1     917         3.100        2.835    1.093
    #     SP1   1140         2.909        2.653    1.097
    #     ALL   4115         3.131        2.907    1.077
    #
    # and the bias is asymmetric -- home xG/goals 1.759/1.596 = 1.102, away
    # 1.372/1.311 = 1.047 -- because understat credits the pressing side for chances
    # it does not convert. The walk-forward that motivated the xG overlay measured
    # RANKING (who is better), which xG genuinely wins; it never measured LEVEL. The
    # cost showed up on the totals: over the 3,222 out-of-sample walk-forward
    # predictions the board said 59.2% Over 2.5 and 55.7% landed, +3.5pp hot, +4.2 to
    # +4.6pp in E0/F1/SP1; the home side ran +4.6/+3.2pp hot in D1/E0; draws ~2pp light.
    # All three are the same arithmetic fact.
    #
    # The fix keeps xG as the SHAPE and takes the LEVEL from goals: scale each side of
    # the response so its weighted mean equals the weighted mean of the real goals on
    # the same matches. Home and away are scaled separately -- one common factor would
    # leave the 1.102-vs-1.047 asymmetry sitting in `home`. The factors come from the
    # matches passed in, which under a walk-forward is the training window only, so
    # this is leak-free by construction: nothing outside the fit window is read.
    # Relative team strength is untouched (a per-side affine rescale is absorbed by
    # mu and home; att/dfn are mean-centred each iteration).
    #
    # Re-running the same 3,290-prediction walk-forward with this block active:
    #
    #                 Over 2.5 gap        home gap         draw gap      3-way Brier
    #     before      +3.4pp             +2.9pp           -2.7pp           0.5855
    #     after       -1.2pp             -0.3pp           -0.2pp           0.5847
    #
    # per league the Over gap goes +0.2/+3.5/+4.3/+5.0 -> -2.1/-0.3/-1.8/-0.9. Brier
    # barely moves, which is the point: this is a LEVEL fix, not a ranking fix, and
    # the ranking was never the broken part. The residual ~1pp Over undershoot is
    # real but small (1.4 sigma on n=3,290) and is the expected sign for a Poisson
    # fitted to an overdispersed count -- variance of the goal total is 2.65-3.12
    # against a mean of 2.91 -- so it is left alone rather than tuned away.
    s_h = s_a = 1.0
    if _cont and rhg is not None:
        _wh, _wa = float((w * hg).sum()), float((w * ag).sum())
        if _wh > 1e-6 and _wa > 1e-6:
            # clipped: a factor outside [0.5, 2.0] means the two responses are not
            # measuring the same thing at all, and silently squashing the response to
            # match would be worse than fitting the response as given.
            s_h = float(np.clip(float((w * rhg).sum()) / _wh, 0.5, 2.0))
            s_a = float(np.clip(float((w * rag).sum()) / _wa, 0.5, 2.0))
            hg = hg * s_h
            ag = ag * s_a
    fit_league.level_scale = (s_h, s_a)

    # AFTER the rescale -- mu is the league base rate and must be in goal units.
    mu = math.log(max((w * hg).sum() + (w * ag).sum(), 1e-9) / max(2 * w.sum(), 1e-9))
    for it in range(iters):
        lh = np.exp(mu + home + att[hi] - dfn[ai])
        la = np.exp(mu + att[ai] - dfn[hi])
        # Poisson gradient parts (tau treated in rho step only — standard DC practice)
        gh = w * (hg - lh)            # d logL / d(log lambda_home)
        ga = w * (ag - la)
        g_att = np.zeros(n); g_dfn = np.zeros(n)
        np.add.at(g_att, hi, gh); np.add.at(g_att, ai, ga)
        np.add.at(g_dfn, ai, -gh); np.add.at(g_dfn, hi, -ga)
        g_att -= RIDGE * att; g_dfn -= RIDGE * dfn
        att += lr * g_att / max(w.sum(), 1.0) * n
        dfn += lr * g_dfn / max(w.sum(), 1.0) * n
        att -= att.mean(); dfn -= dfn.mean()          # identifiability
        # scalars fitted jointly — mu's init contains home inflation, so both
        # must move together or home's gradient is corrupted (found by recovery test)
        home += lr * 4 * gh.sum() / max(w.sum(), 1.0)
        mu   += lr * 4 * (gh.sum() + ga.sum()) / max(w.sum(), 1.0)
        # rho: numeric gradient on the DC tau term over low scores. rhg/rag are the
        # INTEGER scorelines -- identical to the response on a goals fit, the carried
        # g_h/g_a on an xG fit, and None when neither exists (rho pinned to 0 above).
        if it % 10 == 0 and rhg is not None:
            def rho_ll(r_):
                m0 = (rhg <= 1) & (rag <= 1)
                t = np.array([_tau(x, y, l1, l2, r_) for x, y, l1, l2 in
                              zip(rhg[m0], rag[m0], lh[m0], la[m0])])
                t = np.clip(t, 1e-9, None)
                return float((w[m0] * np.log(t)).sum())
            eps = 1e-3
            g = (rho_ll(rho + eps) - rho_ll(rho - eps)) / (2 * eps)
            rho = float(np.clip(rho + 0.5 * lr * g / max(w.sum(), 1.0) * 50, -0.30, 0.10))
    return ({t: (float(att[ti[t]]), float(dfn[ti[t]])) for t in teams},
            float(home), float(rho), float(mu))

def score_grid(lh, la, rho, kmax=MAX_GOALS):
    """P(home=i, away=j) grid with DC correction, renormalized."""
    ph = [math.exp(-lh) * lh ** i / math.factorial(i) for i in range(kmax + 1)]
    pa = [math.exp(-la) * la ** j / math.factorial(j) for j in range(kmax + 1)]
    g = np.outer(ph, pa)
    for x in (0, 1):
        for y in (0, 1):
            g[x, y] *= max(_tau(x, y, lh, la, rho), 1e-9)
    g /= g.sum()
    return g

def match_probs(ratings, home_adv, rho, mu, home, away):
    ah, dh = ratings[home]; aa, da = ratings[away]
    lh = math.exp(mu + home_adv + ah - da)
    la = math.exp(mu + aa - dh)
    g = score_grid(lh, la, rho)
    pH = float(np.tril(g, -1).sum()); pD = float(np.trace(g)); pA = float(np.triu(g, 1).sum())
    tot = np.add.outer(np.arange(g.shape[0]), np.arange(g.shape[1]))
    o25 = float(g[tot >= 3].sum())
    btts = float(g[1:, 1:].sum())
    return {"pH": pH, "pD": pD, "pA": pA, "o25": o25, "btts": btts,
            "lh": lh, "la": la}

# ---------------------------------------------------------------- backtest
def backtest_league(matches, refit_days=30, min_train=180):
    """Expanding-window walk-forward: refit every `refit_days`, predict forward.
    Reports model vs CLOSING MARKET on identical matches + disagreements."""
    ms = sorted(matches, key=lambda m: m["date"])
    if len(ms) < 80: return None
    start = ms[0]["date"] + dt.timedelta(days=min_train)
    preds = []
    fit = None; fit_date = None
    for m in ms:
        if m["date"] < start: continue
        if fit is None or (m["date"] - fit_date).days >= refit_days:
            train = [x for x in ms if x["date"] < m["date"]]
            teams_now = {m["home"], m["away"]}
            try:
                fit = fit_league(train, asof=m["date"], iters=250)
                fit_date = m["date"]
            except Exception:
                continue
        ratings, home_adv, rho, mu = fit
        if m["home"] not in ratings or m["away"] not in ratings: continue
        p = match_probs(ratings, home_adv, rho, mu, m["home"], m["away"])
        # GRADE ON THE REAL SCORELINE. hg/ag is the FIT RESPONSE, and under the xG
        # overlay it is expected goals, not goals -- grading "H" off xG 1.7 vs 1.4
        # would score the model against a match that never happened, and would flatter
        # it, since the same xG drove the prediction. g_h/g_a is the actual result and
        # is present on every row the overlay produces.
        _hg = m["g_h"] if m.get("g_h") is not None else m["hg"]
        _ag = m["g_a"] if m.get("g_a") is not None else m["ag"]
        res = "H" if _hg > _ag else ("A" if _ag > _hg else "D")
        rec = {"pH": p["pH"], "pD": p["pD"], "pA": p["pA"], "res": res,
               "po": p["o25"], "over": (_hg + _ag) >= 3}
        if m["mh"] and m["md"] and m["ma"]:
            ih, idd, ia = 1/m["mh"], 1/m["md"], 1/m["ma"]; s = ih + idd + ia
            rec.update({"qH": ih/s, "qD": idd/s, "qA": ia/s})
        if m.get("mo") and m.get("mu"):
            io, iu = 1/m["mo"], 1/m["mu"]
            rec["qo"] = io / (io + iu)
        preds.append(rec)
    if not preds: return None
    def pick(d, a, b, c): return max(((d[a],"H"),(d[b],"D"),(d[c],"A")))[1]
    n = len(preds)
    acc = sum(1 for r in preds if pick(r,"pH","pD","pA") == r["res"]) / n
    bs = sum((r["pH"]-(r["res"]=="H"))**2 + (r["pD"]-(r["res"]=="D"))**2 +
             (r["pA"]-(r["res"]=="A"))**2 for r in preds) / n
    M = [r for r in preds if "qH" in r]
    out = {"n": n, "acc": round(100*acc,1), "brier3": round(bs,4)}
    # totals: model O2.5 pick vs closing totals market on identical matches
    T = [r for r in preds if "qo" in r]
    if T:
        tacc = sum(1 for r in T if (r["po"] > 0.5) == r["over"]) / len(T)
        tmk = sum(1 for r in T if (r["qo"] > 0.5) == r["over"]) / len(T)
        tdis = [r for r in T if (r["po"] > 0.5) != (r["qo"] > 0.5)]
        out["totals"] = {"n": len(T), "acc": round(100*tacc,1), "market_acc": round(100*tmk,1),
                         "disagree_n": len(tdis),
                         "disagree_model_right": (round(100*sum(1 for r in tdis
                             if (r["po"]>0.5)==r["over"])/len(tdis),1) if tdis else None)}
    if M:
        macc = sum(1 for r in M if pick(r,"qH","qD","qA") == r["res"]) / len(M)
        dis = [r for r in M if pick(r,"pH","pD","pA") != pick(r,"qH","qD","qA")]
        dacc = (sum(1 for r in dis if pick(r,"pH","pD","pA") == r["res"]) / len(dis)) if dis else None
        out.update({"n_mkt": len(M), "market_acc": round(100*macc,1),
                    "disagree_n": len(dis),
                    "disagree_model_right": round(100*dacc,1) if dacc is not None else None})
    return out

# ---------------------------------------------------------------- selftest
def _synth_league(n_teams=16, rounds=2, seed=7, home=math.log(1.3), rho=-0.10, mu=math.log(1.35)):
    """Generate a full double round-robin from KNOWN parameters."""
    rng = np.random.default_rng(seed)
    att = rng.normal(0, 0.30, n_teams); att -= att.mean()
    dfn = rng.normal(0, 0.25, n_teams); dfn -= dfn.mean()
    teams = [f"Club {chr(65+i)}" for i in range(n_teams)]
    ms = []
    day = dt.date(2025, 8, 10)
    k = 0
    for rd in range(rounds):
        pairs = [(i, j) for i in range(n_teams) for j in range(n_teams) if i != j]
        rng.shuffle(pairs)                    # interleave like a real league calendar
        for i, j in pairs:
            lh = math.exp(mu + home + att[i] - dfn[j])
            la = math.exp(mu + att[j] - dfn[i])
            g = score_grid(lh, la, rho)
            flat = g.ravel(); kk = rng.choice(len(flat), p=flat/flat.sum())
            hg, ag = divmod(int(kk), g.shape[1])
            ms.append({"div":"E0","date":day,"home":teams[i],"away":teams[j],
                       "hg":hg,"ag":ag,"mh":None,"md":None,"ma":None})
            k += 1
            if k % 8 == 0: day += dt.timedelta(days=3)   # 8-match matchdays
        day += dt.timedelta(days=14)
    truth = {teams[i]: (float(att[i]), float(dfn[i])) for i in range(n_teams)}
    return ms, truth, home, rho, mu

def selftest():
    # 1. PARSER on a representative football-data snippet (real column set)
    csv_text = ("Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,PSCH,PSCD,PSCA\n"
                "E0,17/08/2025,12:30,Arsenal,Everton,2,0,H,1.45,4.50,7.00,1.44,4.60,7.40\n"
                "E0,17/08/2025,15:00,Leeds,Chelsea,1,1,D,3.10,3.40,2.30,,,\n"
                "E0,bad,15:00,X,Y,,,,,,,,,\n")
    rows = parse_results_csv(csv_text, "E0")
    assert len(rows) == 2 and rows[0]["hg"] == 2 and rows[0]["mh"] == 1.44  # PSC preferred
    assert rows[1]["mh"] == 3.10                                            # B365 fallback
    csv_t = ("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,B365>2.5,B365<2.5,P>2.5,P<2.5\n"
             "E0,17/08/2025,Arsenal,Everton,2,0,1.80,2.00,1.85,1.98\n")
    rt = parse_results_csv(csv_t, "E0")
    assert rt[0]["mo"] == 1.85 and rt[0]["mu"] == 1.98      # Pinnacle preferred
    fx = parse_fixtures_csv("Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A\n"
                            "E0,16/08/2026,12:30,Arsenal,Leeds,1.5,4.2,6.0\n"
                            "SC0,16/08/2026,15:00,Celtic,Rangers,1.9,3.5,3.8\n")
    assert len(fx) == 1 and fx[0]["div"] == "E0"                            # non-target league dropped

    # 2. SYNTHETIC RECOVERY — statistical, across seeds (single-seed thresholds
    #    are coin flips; medians and floors are the honest bar)
    corrs, maes, rhos, homes_ok, mus_ok = [], [], [], [], []
    last = None
    for seed in (7, 11, 13, 21, 42):
        ms, truth, H, R, MU = _synth_league(rounds=3, seed=seed)
        ratings, home_adv, rho, mu = fit_league(ms, iters=600)
        corrs.append(float(np.corrcoef([truth[t][0] for t in truth],
                                       [ratings[t][0] for t in truth])[0, 1]))
        errs = [abs(ratings[t][0] - truth[t][0]) for t in truth] + \
               [abs(ratings[t][1] - truth[t][1]) for t in truth]
        maes.append(sum(errs) / len(errs))
        rhos.append(rho)
        homes_ok.append(abs(home_adv - H) < 0.10)
        mus_ok.append(abs(mu - MU) < 0.10)
        last = (ms, truth, ratings, home_adv, rho, mu)
    corr = float(np.median(corrs)); mae = float(np.median(maes))
    assert corr > 0.90 and min(corrs) > 0.85, (corrs,)
    assert mae < 0.10, maes
    assert all(homes_ok) and all(mus_ok), (homes_ok, mus_ok)
    assert float(np.median(rhos)) < -0.02, rhos          # negative dependence recovered
    assert fit_league.rho_source == "response", fit_league.rho_source
    ms, truth, ratings, home_adv, rho, mu = last

    # 2b. THE CONTINUOUS-RESPONSE RHO GUARD. _tau tests x == 0 / x == 1, so a float
    #     response (xG) misses every branch, the rho log-likelihood is flat, and rho
    #     never leaves its initialisation. That is not a hypothetical: fitted on the
    #     cached understat joins on 2026-08-03, the xG fit returned rho = -0.0500 in
    #     ALL FOUR leagues -- the literal init -- while the goals fit returned -0.0567
    #     (D1), -0.1107 (E0), -0.0516 (F1) and +0.0179 (SP1). E0 wants about twice the
    #     dependence the init assumes and SP1 wants the opposite sign, and score_grid
    #     applies rho to the 0-0/0-1/1-0/1-1 cells of every published grid, so an
    #     unfitted constant was steering the draw price (up to 2.6pp in E0).
    ms3, truth3, H3, R3, MU3 = _synth_league(rounds=3, seed=7)
    # same matches, response replaced by a noisy continuous proxy, real goals carried
    _rng = np.random.default_rng(3)
    cont = [dict(m, g_h=m["hg"], g_a=m["ag"],
                 hg=float(m["hg"]) + float(_rng.uniform(0.01, 0.09)),
                 ag=float(m["ag"]) + float(_rng.uniform(0.01, 0.09))) for m in ms3]
    _, _, rho_c, _ = fit_league(cont, iters=600)
    assert fit_league.rho_source == "goals", fit_league.rho_source
    assert rho_c < -0.02, f"rho must still be FITTED on a continuous response, got {rho_c}"
    assert abs(rho_c - (-0.05)) > 1e-4, "rho sitting exactly on its init is the bug"
    # strip the carried scorelines: rho is then unidentifiable and must be pinned to 0,
    # NOT left at -0.05. Shipping the init is worse than shipping no correction, because
    # no correction is at least a stated modelling choice.
    bare = [{k: v for k, v in m.items() if k not in ("g_h", "g_a")} for m in cont]
    _, _, rho_b, _ = fit_league(bare, iters=600)
    assert fit_league.rho_source == "unidentifiable-pinned-0", fit_league.rho_source
    assert rho_b == 0.0, rho_b

    # 2c. THE CONTINUOUS-RESPONSE LEVEL GUARD. mu is log(weighted mean of the
    #     response), so fitting on a response that runs hot ships a grid that runs hot.
    #     understat xG totals sit 7.7% above real goals league-wide and asymmetrically
    #     (home 1.102, away 1.047), which is exactly the +3.5pp Over-2.5 and +4.6/+3.2pp
    #     home bias measured over 3,222 walk-forward predictions on 2026-08-03. Here the
    #     inflation is applied deliberately -- home x1.10, away x1.05 -- and the fitted
    #     LEVEL must come out on the goals scale anyway, while the strip-the-goals case
    #     must NOT silently rescale (there is nothing honest to rescale to).
    _rng2 = np.random.default_rng(17)
    infl = [dict(m, g_h=m["hg"], g_a=m["ag"],
                 hg=float(m["hg"]) * 1.10 + float(_rng2.uniform(0.0, 0.02)),
                 ag=float(m["ag"]) * 1.05 + float(_rng2.uniform(0.0, 0.02))) for m in ms3]
    _, H_g, _, MU_g = fit_league(ms3, iters=600)
    _, H_c, _, MU_c = fit_league(infl, iters=600)
    _sh, _sa = fit_league.level_scale
    assert 0.87 < _sh < 0.94, f"home level scale {_sh} should undo the x1.10"
    assert 0.92 < _sa < 0.98, f"away level scale {_sa} should undo the x1.05"
    assert abs(MU_c - MU_g) < 0.03, f"level must land on the goals scale ({MU_c} vs {MU_g})"
    # separate per-side factors, or the 1.10/1.05 asymmetry ends up inside `home`
    assert abs(H_c - H_g) < 0.05, f"home advantage corrupted by the rescale ({H_c} vs {H_g})"
    bare2 = [{k: v for k, v in m.items() if k not in ("g_h", "g_a")} for m in infl]
    _, H_b, _, MU_b = fit_league(bare2, iters=600)
    assert fit_league.level_scale == (1.0, 1.0), fit_league.level_scale
    assert MU_b - MU_g > 0.04, f"unscaled fit should still be hot ({MU_b} vs {MU_g})"

    # 3. GRID sanity: probs are a distribution; strong home is favored
    p = match_probs(ratings, home_adv, rho, mu,
                    max(truth, key=lambda t: truth[t][0]),   # best attack at home
                    min(truth, key=lambda t: truth[t][0]))
    assert abs(p["pH"] + p["pD"] + p["pA"] - 1) < 1e-6
    assert p["pH"] > p["pA"] and 0 < p["o25"] < 1 and 0 < p["btts"] < 1

    # 4. DECAY: torrid form in only the final six weeks — short halflife must
    #    credit it far more than a flat (huge-halflife) fit
    ms2, truth2, *_ = _synth_league(seed=11)
    flip = list(truth2)[0]
    for m in ms2:
        if m["date"] > dt.date(2026, 1, 1) and m["home"] == flip:
            m["hg"] += 2
    r_fast, *_ = fit_league(ms2, halflife=45, iters=300)
    r_slow, *_ = fit_league(ms2, halflife=100000, iters=300)
    assert r_fast[flip][0] > r_slow[flip][0] + 0.05, (r_fast[flip][0], r_slow[flip][0])

    # 5. BACKTEST plumbing on synthetic (market cols absent -> model-only block)
    for m in ms:
        m["mo"], m["mu"] = 1.9, 1.9                       # flat totals market
    bt = backtest_league(ms, refit_days=45, min_train=60)
    assert bt and bt["n"] > 50 and 25 < bt["acc"] < 75 and "brier3" in bt
    assert bt["totals"]["n"] == bt["n"] and 25 < bt["totals"]["acc"] < 80
    assert bt["totals"]["disagree_n"] >= 0
    print("SOCCER SELFTEST PASS — parser/PSC-fallback, synthetic recovery "
          f"(MAE {mae:.3f}, att-corr {corr:.2f}, home {home_adv:.2f}~{H:.2f}, rho {rho:.2f}), "
          "grid, decay, backtest plumbing")
    return 0

if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv: sys.exit(selftest())
    m, f, note = fetch_all()
    print(note)
