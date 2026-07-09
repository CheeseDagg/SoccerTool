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
        out.append({"div": div, "date": d, "home": h, "away": a,
                    "hg": int(float(hg)), "ag": int(float(ag)),
                    "mh": mh, "md": md, "ma": ma})
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
    """DC low-score dependence correction."""
    if x == 0 and y == 0: return 1 - lh * la * rho
    if x == 0 and y == 1: return 1 + lh * rho
    if x == 1 and y == 0: return 1 + la * rho
    if x == 1 and y == 1: return 1 - rho
    return 1.0

def fit_league(matches, asof=None, halflife=DECAY_HALFLIFE_DAYS, iters=400, lr=0.05):
    """ML fit of attack/defense per team + home advantage + rho.
    Returns dict(team->(att,dfn)), home_adv, rho, mu (league base)."""
    if asof is None: asof = max(m["date"] for m in matches)
    teams = sorted({m["home"] for m in matches} | {m["away"] for m in matches})
    ti = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    att = np.zeros(n); dfn = np.zeros(n)
    home = math.log(1.25); rho = -0.05
    w = np.array([0.5 ** (max((asof - m["date"]).days, 0) / halflife) for m in matches])
    hg = np.array([m["hg"] for m in matches]); ag = np.array([m["ag"] for m in matches])
    hi = np.array([ti[m["home"]] for m in matches]); ai = np.array([ti[m["away"]] for m in matches])
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
        # rho: numeric gradient on the DC tau term over low scores
        if it % 10 == 0:
            def rho_ll(r_):
                m0 = (hg <= 1) & (ag <= 1)
                t = np.array([_tau(x, y, l1, l2, r_) for x, y, l1, l2 in
                              zip(hg[m0], ag[m0], lh[m0], la[m0])])
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
        res = "H" if m["hg"] > m["ag"] else ("A" if m["ag"] > m["hg"] else "D")
        rec = {"pH": p["pH"], "pD": p["pD"], "pA": p["pA"], "res": res}
        if m["mh"] and m["md"] and m["ma"]:
            ih, idd, ia = 1/m["mh"], 1/m["md"], 1/m["ma"]; s = ih + idd + ia
            rec.update({"qH": ih/s, "qD": idd/s, "qA": ia/s})
        preds.append(rec)
    if not preds: return None
    def pick(d, a, b, c): return max(((d[a],"H"),(d[b],"D"),(d[c],"A")))[1]
    n = len(preds)
    acc = sum(1 for r in preds if pick(r,"pH","pD","pA") == r["res"]) / n
    bs = sum((r["pH"]-(r["res"]=="H"))**2 + (r["pD"]-(r["res"]=="D"))**2 +
             (r["pA"]-(r["res"]=="A"))**2 for r in preds) / n
    M = [r for r in preds if "qH" in r]
    out = {"n": n, "acc": round(100*acc,1), "brier3": round(bs,4)}
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
    ms, truth, ratings, home_adv, rho, mu = last

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
    bt = backtest_league(ms, refit_days=45, min_train=60)
    assert bt and bt["n"] > 50 and 25 < bt["acc"] < 75 and "brier3" in bt
    print("SOCCER SELFTEST PASS — parser/PSC-fallback, synthetic recovery "
          f"(MAE {mae:.3f}, att-corr {corr:.2f}, home {home_adv:.2f}~{H:.2f}, rho {rho:.2f}), "
          "grid, decay, backtest plumbing")
    return 0

if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv: sys.exit(selftest())
    m, f, note = fetch_all()
    print(note)
