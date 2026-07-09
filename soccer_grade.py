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

def load_csv(p):
    if not os.path.exists(p): return []
    with open(p) as f: return list(csv.DictReader(f))

def _key(r): return (r["div"], r["date"], r["home"], r["away"])

def log_predictions(fixture_rows):
    """fixture_rows: dicts with div,date(iso),home,away,pH,pD,pA and optional
    market qH,qD,qA. Appends rows not already logged."""
    os.makedirs(DATA, exist_ok=True)
    existing = load_csv(PLOG)
    have = {_key(r) for r in existing}
    today = dt.date.today().isoformat()
    new = 0
    write_header = not existing
    with open(PLOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS[:-1])
        if write_header: w.writeheader()
        for r in fixture_rows:
            if _key(r) in have: continue
            w.writerow({"logged": today, "div": r["div"], "date": r["date"],
                        "home": r["home"], "away": r["away"],
                        "pH": r["pH"], "pD": r["pD"], "pA": r["pA"],
                        "qH": r.get("qH", ""), "qD": r.get("qD", ""), "qA": r.get("qA", "")})
            new += 1
    return new

def grade_all(results):
    """results: iterable of match dicts (div,date(date),home,away,hg,ag) — the
    freshly fetched season files. Settles logged predictions whose result is in."""
    preds = load_csv(PLOG)
    if not preds: return 0, summarize([])
    done = {_key(r) for r in load_csv(GRADED)}
    res = {}
    for m in results:
        res[(m["div"], m["date"].isoformat(), m["home"], m["away"])] = (m["hg"], m["ag"])
    new = []
    for r in preds:
        k = _key(r)
        if k in done: continue
        if k not in res: continue                       # not played / not in files yet
        hg, ag = res[k]
        r2 = dict(r)
        r2["outcome"] = "H" if hg > ag else ("A" if ag > hg else "D")
        new.append(r2)
    if new:
        exists = os.path.exists(GRADED)
        with open(GRADED, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            if not exists: w.writeheader()
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
            dis = [r for r in M if _pick(r["pH"], r["pD"], r["pA"]) != _pick(r["qH"], r["qD"], r["qA"])]
            out["market"] = {"n": len(M), "acc": round(100 * macc, 1),
                             "disagree_n": len(dis),
                             "disagree_model_right": (round(100 * sum(
                                 1 for r in dis if _pick(r["pH"], r["pD"], r["pA"]) == r["outcome"]) / len(dis), 1)
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
    assert set(p3["by_league"]) == {"E0", "SP1"}
    bs_expected = round((((0.62-1)**2 + 0.22**2 + 0.16**2) +
                         ((0.40)**2 + 0.30**2 + (0.30-1)**2)) / 2, 4)
    assert p3["brier3"] == bs_expected, (p3["brier3"], bs_expected)
    json.dumps(p3)
    print("SOCCER GRADER SELFTEST PASS — log/settle idempotent, 3-way Brier exact, "
          "market disagreement live, per-league split")
    return 0

if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv: sys.exit(selftest())
    print("run via soccer_publish.py")
