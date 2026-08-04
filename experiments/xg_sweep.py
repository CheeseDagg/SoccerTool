"""
experiments/xg_sweep.py — hyperparameter sweep for the production soccer model
==============================================================================
Question: can any of these beat the shipped configuration on a leak-free
walk-forward over the cached understat matches (2023-24 .. 2025-26)?

  a. xG weight in the ratings response:  hg = w*xG + (1-w)*goals
     (production = 1.0: pure xG response with goals-level calibration)
  b. time-decay half-life (production = 240d)
  c. probability-level blend of the goals-fit and xG-fit models
  d. shrinkage of low-data (newly promoted) teams toward a below-average prior

HARNESS CONTRACT — identical to soccer_model.backtest_league:
  expanding window, refit every 30 days, min_train = 180 days after the first
  cached match, fit_league(iters=250), fit strictly on earlier matches, grade
  on the REAL result. Metrics: 3-way Brier (primary), accuracy, log-loss.
  The disagreement-vs-market record CANNOT be recomputed here: the cached
  understat data carries no odds. It regenerates on Actions from football-data.

LEAKAGE RULE for anything that ships: a global/parameter choice is made on the
BURN-IN (predictions from the 2023-24 season only, i.e. Feb-May 2024) and must
then also win on the EVAL window (2024-25 + 2025-26) to ship. Full-window
numbers are reported for comparability with the printed record.

Run:  python3 experiments/xg_sweep.py            (all configs, ~minutes)
      python3 experiments/xg_sweep.py --quick    (baseline only, smoke test)
"""
import os, sys, json, math, argparse, datetime as dt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
import soccer_model as sm

CACHE = os.path.join(REPO, "data", "understat_xg.json")
OUT = os.path.join(HERE, "xg_sweep_records.json")

REFIT_DAYS = 30
MIN_TRAIN = 180
ITERS = 250
BURN_SEASON = 2023          # tuning window: predictions from the 23/24 season


def load_matches():
    with open(CACHE) as f:
        rows = json.load(f)
    ms = []
    for r in rows:
        ms.append({"div": r["div"], "season": r["season"],
                   "date": dt.date.fromisoformat(r["date"][:10]),
                   "home": r["home"], "away": r["away"],
                   "g_h": int(r["g_h"]), "g_a": int(r["g_a"]),
                   "xgh": float(r["xgh"]), "xga": float(r["xga"])})
    return ms


def fit_rows(ms, w):
    """Response = w*xG + (1-w)*goals; real goals carried for rho/level/grading."""
    out = []
    for m in ms:
        out.append({"date": m["date"], "home": m["home"], "away": m["away"],
                    "hg": w * m["xgh"] + (1 - w) * m["g_h"],
                    "ag": w * m["xga"] + (1 - w) * m["g_a"],
                    "g_h": m["g_h"], "g_a": m["g_a"]})
    return out


def shrink_ratings(ratings, train, asof, halflife, K, prior_frac):
    """Variant d: shrink each team's rating toward a below-average prior in
    proportion to how little (time-decayed) data it has. Everything read is
    inside the training window — leak-free by construction.

    prior = mean rating of the bottom `prior_frac` of ESTABLISHED teams
    (n_eff >= 25% of the median), i.e. 'a new team is probably bad, not average'.
    lam = n_eff / (n_eff + K)."""
    n_eff = {t: 0.0 for t in ratings}
    for m in train:
        wgt = 0.5 ** (max((asof - m["date"]).days, 0) / halflife)
        if m["home"] in n_eff: n_eff[m["home"]] += wgt
        if m["away"] in n_eff: n_eff[m["away"]] += wgt
    med = float(np.median(list(n_eff.values()))) if n_eff else 0.0
    est = [t for t in ratings if n_eff[t] >= 0.25 * med]
    if len(est) < 4:
        return ratings
    est_sorted = sorted(est, key=lambda t: ratings[t][0] + ratings[t][1])  # worst first
    k = max(2, int(round(prior_frac * len(est_sorted))))
    bottom = est_sorted[:k]
    pa = float(np.mean([ratings[t][0] for t in bottom]))
    pd_ = float(np.mean([ratings[t][1] for t in bottom]))
    out = {}
    for t, (a, d) in ratings.items():
        lam = n_eff[t] / (n_eff[t] + K)
        out[t] = (lam * a + (1 - lam) * pa, lam * d + (1 - lam) * pd_)
    return out


def walk_forward(ms, w=1.0, halflife=sm.DECAY_HALFLIFE_DAYS, shrink=None):
    """Mirror of soccer_model.backtest_league's split, parameterised.
    shrink = (K, prior_frac) or None. Returns per-prediction records."""
    ms = sorted(ms, key=lambda m: (m["date"], m["home"], m["away"]))
    if len(ms) < 80:
        return []
    rows = fit_rows(ms, w)
    start = ms[0]["date"] + dt.timedelta(days=MIN_TRAIN)
    preds, fit, fit_date, train_used = [], None, None, None
    for m, fr in zip(ms, rows):
        if m["date"] < start:
            continue
        if fit is None or (m["date"] - fit_date).days >= REFIT_DAYS:
            train = [x for x in rows if x["date"] < m["date"]]
            try:
                fit = sm.fit_league(train, asof=m["date"], iters=ITERS, halflife=halflife)
                fit_date, train_used = m["date"], train
            except Exception:
                continue
        ratings, home_adv, rho, mu = fit
        if m["home"] not in ratings or m["away"] not in ratings:
            continue
        if shrink is not None:
            K, pf = shrink
            ratings = shrink_ratings(ratings, train_used, fit_date, halflife, K, pf)
        p = sm.match_probs(ratings, home_adv, rho, mu, m["home"], m["away"])
        res = "H" if m["g_h"] > m["g_a"] else ("A" if m["g_a"] > m["g_h"] else "D")
        preds.append({"date": m["date"].isoformat(), "season": m["season"], "div": m["div"],
                      "home": m["home"], "away": m["away"],
                      "pH": p["pH"], "pD": p["pD"], "pA": p["pA"], "res": res})
    return preds


def brier(r):
    return ((r["pH"] - (r["res"] == "H")) ** 2 + (r["pD"] - (r["res"] == "D")) ** 2 +
            (r["pA"] - (r["res"] == "A")) ** 2)


def pick(r):
    return max((r["pH"], "H"), (r["pD"], "D"), (r["pA"], "A"))[1]


def metrics(preds):
    if not preds:
        return None
    n = len(preds)
    return {"n": n,
            "brier": sum(brier(r) for r in preds) / n,
            "acc": 100.0 * sum(1 for r in preds if pick(r) == r["res"]) / n,
            "ll": sum(-math.log(max({"H": r["pH"], "D": r["pD"], "A": r["pA"]}[r["res"]], 1e-12))
                      for r in preds) / n}


def report(tag, preds):
    m = metrics(preds)
    per = {}
    for div in sorted({r["div"] for r in preds}):
        per[div] = metrics([r for r in preds if r["div"] == div])
    burn = metrics([r for r in preds if r["season"] == BURN_SEASON])
    ev = metrics([r for r in preds if r["season"] != BURN_SEASON])
    line = f"{tag:<34} n={m['n']:>4} brier={m['brier']:.4f} acc={m['acc']:.1f} ll={m['ll']:.4f}"
    line += f" | burn[{burn['n']}]={burn['brier']:.4f}" if burn else " | burn=NA"
    line += f" eval[{ev['n']}]={ev['brier']:.4f}" if ev else ""
    line += " | " + " ".join(f"{d}:{per[d]['brier']:.4f}/{per[d]['acc']:.1f}" for d in per)
    print(line, flush=True)
    return {"pooled": m, "burn": burn, "eval": ev, "per_league": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    all_ms = load_matches()
    by_div = {}
    for m in all_ms:
        by_div.setdefault(m["div"], []).append(m)
    print(f"loaded {len(all_ms)} matches: " +
          " ".join(f"{d}:{len(v)}" for d, v in sorted(by_div.items())), flush=True)

    records = {}   # config tag -> list of per-prediction records (all leagues pooled)
    summaries = {}

    def run(tag, **kw):
        preds = []
        for div in sorted(by_div):
            preds += walk_forward(by_div[div], **kw)
        records[tag] = preds
        summaries[tag] = report(tag, preds)

    run("baseline (w=1.0, hl=240)")
    if args.quick:
        return

    # a. xG weight sweep
    for w in (0.0, 0.25, 0.5, 0.75):
        run(f"w={w}", w=w)
    # b. half-life sweep (at production w=1.0)
    for hl in (90, 120, 180, 330, 420, 600):
        run(f"hl={hl}", halflife=hl)
    # d. promoted-club shrinkage on the baseline config
    for K, pf in ((5, 0.2), (15, 0.2), (30, 0.2), (15, 0.35)):
        run(f"shrink K={K} pf={pf}", shrink=(K, pf))

    # c. probability blend of goals-fit and xG-fit (paired on identical matches)
    key = lambda r: (r["date"], r["div"], r["home"], r["away"])
    b1 = {key(r): r for r in records["baseline (w=1.0, hl=240)"]}
    b0 = {key(r): r for r in records["w=0.0"]}
    common = [k for k in b1 if k in b0]
    for alpha in (0.25, 0.5, 0.75):
        blended = []
        for k in common:
            r1, r0 = b1[k], b0[k]
            blended.append({**r1,
                            "pH": alpha * r1["pH"] + (1 - alpha) * r0["pH"],
                            "pD": alpha * r1["pD"] + (1 - alpha) * r0["pD"],
                            "pA": alpha * r1["pA"] + (1 - alpha) * r0["pA"]})
        records[f"prob-blend a={alpha}"] = blended
        summaries[f"prob-blend a={alpha}"] = report(f"prob-blend a={alpha} (xG share)", blended)

    with open(OUT, "w") as f:
        json.dump({"summaries": {k: v for k, v in summaries.items()},
                   "records": records}, f)
    print(f"\nrecords -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
