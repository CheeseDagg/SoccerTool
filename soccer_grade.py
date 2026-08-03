"""
soccer_grade.py — log every published match probability pre-match, settle it
against results from the SAME football-data CSVs the model trains on, and keep
the market-disagreement study running live. Grader-from-line-one, house rule.
Outcomes: H / D / A / pending. Idempotent by (div,date,home,away).
"""
import os, csv, datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PLOG = os.path.join(DATA, "soccer_predictions.csv")
GRADED = os.path.join(DATA, "soccer_graded.csv")
COLS = ["logged", "div", "date", "home", "away", "pH", "pD", "pA",
        "qH", "qD", "qA", "outcome"]

def ensure_files():
    """Both artifacts must exist from run one — offseason has nothing to log,
    and the workflow's git-add of missing paths is a hard failure otherwise."""
    os.makedirs(DATA, exist_ok=True)
    for p, cols in ((PLOG, COLS[:-1]), (GRADED, COLS)):
        if not os.path.exists(p):
            with open(p, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=cols).writeheader()

def load_csv(p):
    """Read a ledger, dropping any row that is a re-written header.

    log_predictions used to write a header even when ensure_files() had already
    written one, so the first real logging day appended a SECOND header line and
    every read after that parsed it as a match: div="div", date="date", pH="pH".
    The writer is fixed below; this filter is here so a file already carrying the
    junk row heals itself on the next run instead of needing hand editing."""
    if not os.path.exists(p): return []
    with open(p) as f:
        return [r for r in csv.DictReader(f) if r.get("div") != "div"]

def _key(r): return (r["div"], r["date"], r["home"], r["away"])

def log_predictions(fixture_rows):
    """fixture_rows: dicts with div,date(iso),home,away,pH,pD,pA and optional
    market qH,qD,qA. Appends rows not already logged."""
    ensure_files()
    existing = load_csv(PLOG)
    have = {_key(r) for r in existing}
    today = dt.date.today().isoformat()
    new = 0
    # NO writeheader() HERE. ensure_files() above has already guaranteed the header
    # exists. The old `write_header = not existing` was true whenever the file held
    # only a header -- i.e. on the very first day fixtures post -- and appended a
    # duplicate header line that every later read parsed as a phantom match.
    with open(PLOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS[:-1])
        for r in fixture_rows:
            if _key(r) in have: continue
            w.writerow({"logged": today, "div": r["div"], "date": r["date"],
                        "home": r["home"], "away": r["away"],
                        "pH": r["pH"], "pD": r["pD"], "pA": r["pA"],
                        "qH": r.get("qH", ""), "qD": r.get("qD", ""), "qA": r.get("qA", "")})
            new += 1
    return new

RESCHED_DAYS = 45   # a postponed match is replayed within weeks — well short of a season,
                    # so this can't collide with the same fixture in a different season.

def _rescheduled_result(pred, res_by_match):
    """Settle a prediction whose match was POSTPONED: the played date then differs from
    the scheduled date it was logged under, so the exact key misses. Each div/home/away
    pairing is played once per season, so match to the closest result within RESCHED_DAYS.
    Returns (hg, ag) or None."""
    cands = res_by_match.get((pred["div"], pred["home"], pred["away"]))
    if not cands: return None
    try: pdte = dt.date.fromisoformat(pred["date"])
    except (ValueError, TypeError): return None
    best = None
    for di, hgag in cands:
        try: d = dt.date.fromisoformat(di)
        except (ValueError, TypeError): continue
        delta = abs((d - pdte).days)
        if delta <= RESCHED_DAYS and (best is None or delta < best[0]):
            best = (delta, hgag)
    return best[1] if best else None

def grade_all(results):
    """results: iterable of match dicts (div,date(date),home,away,hg,ag) — the
    freshly fetched season files. Settles logged predictions whose result is in."""
    ensure_files()
    preds = load_csv(PLOG)
    if not preds: return 0, summarize([])
    done = {_key(r) for r in load_csv(GRADED)}
    res, res_by_match = {}, {}
    for m in results:
        di = m["date"].isoformat()
        res[(m["div"], di, m["home"], m["away"])] = (m["hg"], m["ag"])
        res_by_match.setdefault((m["div"], m["home"], m["away"]), []).append((di, (m["hg"], m["ag"])))
    new = []
    for r in preds:
        k = _key(r)
        if k in done: continue
        got = res.get(k)
        if got is None:
            got = _rescheduled_result(r, res_by_match)  # postponed match: tolerant date match
        if got is None: continue                        # not played / not in files yet
        hg, ag = got
        r2 = dict(r)
        r2["outcome"] = "H" if hg > ag else ("A" if ag > hg else "D")
        new.append(r2)
    if new:
        # Same rule as the prediction log: ensure_files() owns the header, the
        # append path never writes one. (This one happened to be safe only because
        # os.path.exists was always True by the time it ran.)
        with open(GRADED, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            for r in new: w.writerow(r)
    return len(new), summarize(load_csv(GRADED))

def _pick(h, d, a):
    return max(((float(h), "H"), (float(d), "D"), (float(a), "A")))[1]

def summarize(rows):
    """Panel per league + overall: n, acc, Brier(3-way), market acc + LIVE
    disagreement record."""
    panel = {"n": len(rows)}
    if not rows: return panel
    def block(rs):
        n = len(rs)
        acc = sum(1 for r in rs if _pick(r["pH"], r["pD"], r["pA"]) == r["outcome"]) / n
        bs = sum((float(r["pH"]) - (r["outcome"] == "H")) ** 2 +
                 (float(r["pD"]) - (r["outcome"] == "D")) ** 2 +
                 (float(r["pA"]) - (r["outcome"] == "A")) ** 2 for r in rs) / n
        out = {"n": n, "acc": round(100 * acc, 1), "brier3": round(bs, 4)}
        M = [r for r in rs if r.get("qH") not in ("", None)]
        if M:
            macc = sum(1 for r in M if _pick(r["qH"], r["qD"], r["qA"]) == r["outcome"]) / len(M)
            # Like for like: `acc` above is over EVERY graded row, `market.acc` can
            # only be over the rows that carried a closing price. Report the model's
            # rate on that same subset so the two numbers describe one match set.
            pacc = sum(1 for r in M if _pick(r["pH"], r["pD"], r["pA"]) == r["outcome"]) / len(M)
            dis = [r for r in M if _pick(r["pH"], r["pD"], r["pA"]) != _pick(r["qH"], r["qD"], r["qA"])]
            out["market"] = {"n": len(M), "acc": round(100 * macc, 1),
                             "model_acc": round(100 * pacc, 1),
                             "disagree_n": len(dis),
                             "disagree_model_right": (round(100 * sum(
                                 1 for r in dis if _pick(r["pH"], r["pD"], r["pA"]) == r["outcome"]) / len(dis), 1)
                                 if dis else None),
                             # The number this panel was missing. On a THREE-way
                             # market the market's rate on the disagreements is not
                             # 100 minus the model's -- a draw beats both -- so it
                             # cannot be derived and has to be counted. The walk-
                             # forward backtest says model 24.8% / market 44.7% over
                             # 467 disagreements; this is the live version of that
                             # same measurement, and it is the number that decides
                             # whether a disagreement with the price is worth acting
                             # on. Reporting only the model's half made a fatal
                             # result look merely mediocre.
                             "disagree_market_right": (round(100 * sum(
                                 1 for r in dis if _pick(r["qH"], r["qD"], r["qA"]) == r["outcome"]) / len(dis), 1)
                                 if dis else None)}
        return out
    panel.update(block(rows))
    panel["by_league"] = {}
    for div in sorted({r["div"] for r in rows}):
        panel["by_league"][div] = block([r for r in rows if r["div"] == div])
    return panel

def selftest():
    import tempfile, json
    global PLOG, GRADED
    tmp = tempfile.mkdtemp()
    PLOG, GRADED = os.path.join(tmp, "p.csv"), os.path.join(tmp, "g.csv")
    fx = [{"div": "E0", "date": "2026-08-16", "home": "Arsenal", "away": "Leeds",
           "pH": 0.62, "pD": 0.22, "pA": 0.16, "qH": 0.60, "qD": 0.24, "qA": 0.16},
          {"div": "SP1", "date": "2026-08-17", "home": "Girona", "away": "Betis",
           "pH": 0.40, "pD": 0.30, "pA": 0.30, "qH": 0.35, "qD": 0.28, "qA": 0.37}]
    assert log_predictions(fx) == 2
    assert log_predictions(fx) == 0                       # idempotent
    res = [{"div": "E0", "date": dt.date(2026, 8, 16), "home": "Arsenal", "away": "Leeds",
            "hg": 2, "ag": 0}]                            # SP1 match not played yet
    n, p = grade_all(res)
    assert n == 1 and p["n"] == 1 and p["acc"] == 100.0   # model picked H, H happened
    assert p["market"]["n"] == 1 and p["market"]["disagree_n"] == 0
    n2, _ = grade_all(res)
    assert n2 == 0                                        # idempotent grading
    res.append({"div": "SP1", "date": dt.date(2026, 8, 17), "home": "Girona", "away": "Betis",
                "hg": 0, "ag": 1})                        # A wins: model said H, market said A
    n3, p3 = grade_all(res)
    assert n3 == 1 and p3["n"] == 2 and p3["acc"] == 50.0
    m = p3["market"]
    assert m["disagree_n"] == 1 and m["disagree_model_right"] == 0.0
    # the market picked A on that disagreement and A happened
    assert m["disagree_market_right"] == 100.0, m
    assert m["model_acc"] == 50.0 and m["acc"] == 100.0, m
    assert set(p3["by_league"]) == {"E0", "SP1"}
    # NO DUPLICATE HEADER. log_predictions used to writeheader() whenever the file
    # held only a header -- exactly the state on the first day fixtures post -- and
    # every read afterwards parsed that second header line as a match with div="div"
    # and pH="pH". Both ledgers must carry exactly one header line, and load_csv must
    # drop such a row if some older file already has one.
    for path in (PLOG, GRADED):
        lines = [ln for ln in open(path).read().splitlines() if ln.strip()]
        heads = [ln for ln in lines if ln.startswith("logged,div,date")]
        assert len(heads) == 1, (path, len(heads), lines[:3])
    with open(PLOG, "a", newline="") as f:                 # simulate an already-corrupted file
        f.write(",".join(COLS[:-1]) + "\n")
    assert all(r["div"] != "div" for r in load_csv(PLOG)), "load_csv let a header row through"
    # and the phantom row must not be counted as an unlogged fixture either
    assert log_predictions(fx) == 0
    # THREE-WAY RATES ARE NOT COMPLEMENTS. Build a disagreement both sides lose.
    globals()["PLOG"], globals()["GRADED"] = (os.path.join(tempfile.mkdtemp(), "p.csv"),
                                              os.path.join(tempfile.mkdtemp(), "g.csv"))
    log_predictions([{"div": "E0", "date": "2026-08-16", "home": "A", "away": "B",
                      "pH": 0.6, "pD": 0.2, "pA": 0.2, "qH": 0.2, "qD": 0.2, "qA": 0.6}])
    _, pd_ = grade_all([{"div": "E0", "date": dt.date(2026, 8, 16), "home": "A",
                         "away": "B", "hg": 1, "ag": 1}])     # DRAW: both wrong
    md = pd_["market"]
    assert md["disagree_n"] == 1
    assert md["disagree_model_right"] == 0.0 and md["disagree_market_right"] == 0.0, md
    assert md["disagree_model_right"] + md["disagree_market_right"] != 100.0, \
        "three-way disagreement rates are NOT complements — that is why both are measured"
    bs_expected = round((((0.62-1)**2 + 0.22**2 + 0.16**2) +
                         ((0.40)**2 + 0.30**2 + (0.30-1)**2)) / 2, 4)
    assert p3["brier3"] == bs_expected, (p3["brier3"], bs_expected)
    json.dumps(p3)
    # RESCHEDULE: a postponed match settles even though its played date differs from the
    # scheduled date the prediction was logged under; a season-away date must NOT match.
    globals()["PLOG"], globals()["GRADED"] = os.path.join(tempfile.mkdtemp(),"p.csv"), os.path.join(tempfile.mkdtemp(),"g.csv")
    log_predictions([{"div":"E0","date":"2026-09-01","home":"Chelsea","away":"Fulham",
                      "pH":0.5,"pD":0.3,"pA":0.2,"qH":0.5,"qD":0.3,"qA":0.2}])
    nr, pr = grade_all([{"div":"E0","date":dt.date(2026,9,6),"home":"Chelsea","away":"Fulham","hg":1,"ag":1}])
    assert nr == 1 and pr["n"] == 1, (nr, pr)            # settled despite the 5-day postponement
    globals()["PLOG"], globals()["GRADED"] = os.path.join(tempfile.mkdtemp(),"p.csv"), os.path.join(tempfile.mkdtemp(),"g.csv")
    log_predictions([{"div":"E0","date":"2026-09-01","home":"Chelsea","away":"Fulham",
                      "pH":0.5,"pD":0.3,"pA":0.2,"qH":0.5,"qD":0.3,"qA":0.2}])
    nf, _ = grade_all([{"div":"E0","date":dt.date(2027,9,1),"home":"Chelsea","away":"Fulham","hg":1,"ag":1}])
    assert nf == 0                                       # >45 days away (next season) -> no false match
    # offseason path: fresh dir, zero fixtures -> both files still materialize
    tmp2 = tempfile.mkdtemp()
    globals()["PLOG"], globals()["GRADED"] = os.path.join(tmp2,"p.csv"), os.path.join(tmp2,"g.csv")
    n0, p0 = grade_all([])
    assert n0 == 0 and p0["n"] == 0
    assert os.path.exists(PLOG) and os.path.exists(GRADED)
    assert open(PLOG).readline().startswith("logged,div,date")
    print("SOCCER GRADER SELFTEST PASS — log/settle idempotent, 3-way Brier exact, "
          "market disagreement live, per-league split")
    return 0

if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv: sys.exit(selftest())
    print("run via soccer_publish.py")
