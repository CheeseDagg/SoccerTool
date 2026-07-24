"""
soccer_xg_experiment.py — DOES xG BEAT GOALS FOR 1X2?  (standalone experiment)
==============================================================================
QUESTION
--------
The production model (soccer_model.py) rates teams by *goals* via a weighted
Dixon-Coles Poisson MLE with time decay, home advantage and the low-score rho
correction, then turns attack/defence ratings into home/draw/away (1X2)
probabilities with match_probs(). Its cached holdout multiclass Brier is ~0.589
on E0 and it LOSES to the closing market.

This harness tests a single hypothesis, cleanly and leak-free:

    Do attack/defence ratings fitted on **expected goals (xG)** produce MORE
    ACCURATE 1X2 predictions than the SAME model fitted on actual goals?

Treatment = the *identical* Dixon-Coles fit and the *identical* match_probs
1X2 mapping, but with per-match xG substituted for goals as the Poisson
response. Nothing else changes. Baseline = the production goals fit. Both
predict the *real* match result. We compare multiclass Brier, log-loss and
accuracy on a temporal holdout, per season.

DATA
----
understat.com per-match xG. Understat embeds the whole season in the league
page as `var datesData = JSON.parse('<hex-escaped-json>')`; each entry carries
h/a team titles, goals{h,a}, xG{h,a}, datetime and isResult. We decode the
\\xNN byte-escapes back to UTF-8 and json.loads it.

EGRESS IS BLOCKED IN THIS SANDBOX. understat returns 403 here — that is
EXPECTED. The pull is written defensively: if understat is unreachable it
prints a clear one-line message and exits 0. The real pull happens on GitHub
Actions (and Ryan's machine), which can reach understat.

This file NEVER imports-and-runs production side effects and NEVER modifies
soccer_model.py. It only *reuses* fit_league() and match_probs() from it.

Run offline (NO NETWORK), must pass:
    python3 soccer_xg_experiment.py --selftest

Run the real experiment (needs network -> Actions):
    python3 soccer_xg_experiment.py
"""
import os, sys, re, json, math, argparse, urllib.request, urllib.error
import datetime as dt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, HERE)
import soccer_model as sm          # reuse fit_league + match_probs; NO modification

# understat league slug <- production div code, and understat season years.
# Understat seasons are named by starting year: 2023 == the 2023/24 campaign,
# matching production's rolling window "2324","2425","2526".
UNDERSTAT_LEAGUE = {"E0": "EPL", "SP1": "La_liga", "D1": "Bundesliga", "F1": "Ligue_1"}
UNDERSTAT_SEASONS = [2023, 2024, 2025]
UNDERSTAT_URL = "https://understat.com/league/{league}/{year}"
CACHE = os.path.join(DATA, "understat_xg.json")

MIN_TRAIN_DAYS = 180
REFIT_DAYS = 30
FIT_ITERS = 250

# FiveThirtyEight SPI fallback — understat Cloudflare-walls GitHub runners (run 1 died
# in 24s), but 538's static CSV carries per-match xG for these exact leagues, 2016-2023.
# Historical-only is fine for the VERDICT (does xG beat goals out-of-sample?).
SPI_URL = "https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv"
SPI_LEAGUE = {"Barclays Premier League": "E0", "Spanish Primera Division": "SP1",
              "German Bundesliga": "D1", "French Ligue 1": "F1"}


def parse_spi_csv(text):
    """538 spi_matches.csv -> harness match dicts (div/date/home/away/g_h/g_a/xgh/xga).
    Keeps only the four production leagues and rows with both scores and both xG."""
    import csv as _csv, io as _io
    out = []
    for r in _csv.DictReader(_io.StringIO(text)):
        div = SPI_LEAGUE.get((r.get("league") or "").strip())
        if not div:
            continue
        try:
            d = dt.date.fromisoformat((r.get("date") or "")[:10])
            g_h, g_a = int(float(r["score1"])), int(float(r["score2"]))
            xgh, xga = float(r["xg1"]), float(r["xg2"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append({"div": div, "date": d, "home": r["team1"].strip(),
                    "away": r["team2"].strip(), "g_h": g_h, "g_a": g_a,
                    "xgh": xgh, "xga": xga})
    return out


def fetch_538():
    """Pull the 538 SPI match file. Raises on unreachable (caller degrades)."""
    text = _http(SPI_URL)
    matches = parse_spi_csv(text)
    if not matches:
        raise RuntimeError("538 SPI reachable but produced 0 usable xG matches")
    per = {}
    for m in matches:
        per[m["div"]] = per.get(m["div"], 0) + 1
    note = " · ".join(f"{k}: {v}" for k, v in sorted(per.items()))
    return matches, f"538 SPI ({note})"

# ----------------------------------------------------------------- data layer
def _http(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (SoccerTool xG experiment)"})
    with urllib.request.urlopen(req, timeout=45) as r:
        return r.read().decode("utf-8", "replace")

def decode_jsonparse(blob):
    """Decode the argument understat passes to JSON.parse('...').

    Understat escapes essentially every byte as \\xNN (and occasionally \\uNNNN);
    a \\xNN is a *raw UTF-8 byte*, so multi-byte characters (accents in team
    names) must be reassembled at the byte level before UTF-8 decoding — decoding
    each \\xNN to a codepoint individually would corrupt them.
    """
    out = bytearray()
    i, n = 0, len(blob)
    simple = {"n": b"\n", "t": b"\t", "r": b"\r", '"': b'"', "'": b"'",
              "\\": b"\\", "/": b"/", "b": b"\b", "f": b"\f"}
    while i < n:
        c = blob[i]
        if c == "\\" and i + 1 < n:
            nxt = blob[i + 1]
            if nxt == "x" and i + 3 < n:
                out.append(int(blob[i + 2:i + 4], 16)); i += 4; continue
            if nxt == "u" and i + 5 < n:
                out += chr(int(blob[i + 2:i + 6], 16)).encode("utf-8"); i += 6; continue
            if nxt in simple:
                out += simple[nxt]; i += 2; continue
            out += nxt.encode("utf-8"); i += 2; continue
        out += c.encode("utf-8"); i += 1
    return out.decode("utf-8", "replace")

def parse_understat_dates(html, div, season):
    """Extract per-match xG from an understat league page's datesData block
    (LEGACY layout — understat now serves data via the getLeagueData JSON API)."""
    m = re.search(r"datesData\s*=\s*JSON\.parse\('(.*?)'\)", html, re.S)
    if not m:
        raise ValueError("datesData block not found (understat layout changed?)")
    rows = json.loads(decode_jsonparse(m.group(1)))
    return _matches_from_dates(rows, div, season)

def _matches_from_dates(rows, div, season):
    """understat match dicts (from datesData or getLeagueData['dates']) -> harness rows."""
    out = []
    for r in rows:
        if not r.get("isResult"):
            continue
        try:
            xgh = float(r["xG"]["h"]); xga = float(r["xG"]["a"])
            gh = int(float(r["goals"]["h"])); ga = int(float(r["goals"]["a"]))
        except (KeyError, TypeError, ValueError):
            continue
        date = _parse_dt(r.get("datetime"))
        home = (r.get("h") or {}).get("title", "").strip()
        away = (r.get("a") or {}).get("title", "").strip()
        if not (date and home and away):
            continue
        out.append({"div": div, "season": season, "date": date,
                    "home": home, "away": away,
                    "g_h": gh, "g_a": ga, "xgh": xgh, "xga": xga})
    return out

def _parse_dt(s):
    if not s:
        return None
    try:
        return dt.datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None

# understat's 2026 redesign: match data moved out of the page into a JSON API the
# league page's own JS calls — GET getLeagueData/{select-value}/{year} returning
# {"dates": [match dicts], "teams": ..., "players": ...}. The select values differ
# from the URL slugs ("La liga" with a space, not "La_liga").
UNDERSTAT_API = "https://understat.com/getLeagueData/{league}/{year}"
UNDERSTAT_API_LEAGUE = {"E0": "EPL", "SP1": "La liga", "D1": "Bundesliga", "F1": "Ligue 1"}


def _http_api(url, referer):
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest",     # jQuery marker the endpoint expects
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Encoding": "gzip",                # explicit; we decompress ourselves
        "Referer": referer,
    })
    with urllib.request.urlopen(req, timeout=45) as r:
        raw = r.read()
    if raw[:2] == b"\x1f\x8b":                    # gzip magic (also covers forced gzip)
        import gzip
        raw = gzip.decompress(raw)
    text = raw.decode("utf-8", "replace").lstrip("﻿")
    if not text.lstrip().startswith(("{", "[")):
        # not JSON — surface what we actually got so failures are diagnosable
        raise ValueError(f"non-JSON response (starts: {text.lstrip()[:80]!r})")
    return text


def fetch_understat(divs=("E0", "SP1", "D1", "F1")):
    """Pull per-match xG for every (div, season) — JSON API first (2026 layout),
    legacy embedded-datesData scrape as fallback. Returns (matches, note).

    Raises RuntimeError if understat is unreachable so the caller can degrade."""
    import urllib.parse
    matches, notes = [], []
    reached_any = False
    blocked = False
    for div in divs:
        got = 0
        for year in UNDERSTAT_SEASONS:
            api_url = UNDERSTAT_API.format(
                league=urllib.parse.quote(UNDERSTAT_API_LEAGUE[div]), year=year)
            referer = UNDERSTAT_URL.format(league=UNDERSTAT_LEAGUE[div], year=year)
            rows = None
            try:                                   # 1) the JSON API the site itself uses
                data = json.loads(_http_api(api_url, referer))
                reached_any = True
                rows = _matches_from_dates(data.get("dates") or [], div, year)
                if not rows:
                    notes.append(f"{div} {year}: api 0 rows")
            except Exception as e:
                notes.append(f"{div} {year}: api {type(e).__name__}")
                if isinstance(e, urllib.error.HTTPError) and e.code in (403, 407, 429):
                    blocked = True
                if isinstance(e, urllib.error.URLError):
                    blocked = True
            if not rows:
                try:                               # 2) legacy embedded-JSON page scrape
                    html = _http(referer)
                    reached_any = True
                    rows = parse_understat_dates(html, div, year)
                except Exception as e:
                    notes.append(f"{div} {year}: page {type(e).__name__}")
                    rows = []
            matches += rows; got += len(rows)
        notes.append(f"{div}: {got} xG matches")
    if not reached_any:
        raise RuntimeError(
            "understat UNREACHABLE from here (egress blocked / 403) — run on Actions. "
            + (" · ".join(notes) if blocked else ""))
    return matches, " · ".join(notes)

# --------------------------------------------------------- fit-target adapter
def _fit_view(m, mode):
    """A match dict shaped for soccer_model.fit_league, with the Poisson response
    set to goals (baseline) or xG (treatment). fit_league reads only
    date/home/away/hg/ag."""
    if mode == "xg":
        hg, ag = m["xgh"], m["xga"]          # xG floats fed as the Poisson counts
    else:
        hg, ag = m["g_h"], m["g_a"]
    return {"date": m["date"], "home": m["home"], "away": m["away"], "hg": hg, "ag": ag}

def _result(m):
    return "H" if m["g_h"] > m["g_a"] else ("A" if m["g_a"] > m["g_h"] else "D")

# ------------------------------------------------------ leak-free backtest
def backtest(matches, mode, refit_days=REFIT_DAYS, min_train=MIN_TRAIN_DAYS,
             iters=FIT_ITERS):
    """Expanding-window walk-forward, mirroring soccer_model.backtest_league's
    leak-free contract: for every predicted match the model is fitted ONLY on
    matches with a strictly earlier date. Refits every `refit_days`.

    `mode` selects the Poisson response ('goals' or 'xg'); the *predicted target*
    is always the real 1X2 result. Returns a list of prediction records."""
    ms = sorted(matches, key=lambda m: (m["date"], m["home"], m["away"]))
    if len(ms) < 80:
        return []
    start = ms[0]["date"] + dt.timedelta(days=min_train)
    preds = []
    fit = fit_date = None
    for m in ms:
        if m["date"] < start:
            continue
        if fit is None or (m["date"] - fit_date).days >= refit_days:
            train = [_fit_view(x, mode) for x in ms if x["date"] < m["date"]]  # STRICTLY prior
            if len(train) < 40:
                continue
            try:
                fit = sm.fit_league(train, asof=m["date"], iters=iters)
                fit_date = m["date"]
            except Exception:
                continue
        if fit is None:
            continue
        ratings, home_adv, rho, mu = fit
        if m["home"] not in ratings or m["away"] not in ratings:
            continue
        p = sm.match_probs(ratings, home_adv, rho, mu, m["home"], m["away"])
        preds.append({"date": m["date"], "season": m.get("season"),
                      "div": m.get("div"), "home": m["home"], "away": m["away"],
                      "pH": p["pH"], "pD": p["pD"], "pA": p["pA"], "res": _result(m)})
    return preds

# --------------------------------------------------------------- metrics
def brier3(rec):
    """Multiclass Brier for one prediction: sum over H/D/A of (p - y)^2."""
    return ((rec["pH"] - (rec["res"] == "H")) ** 2 +
            (rec["pD"] - (rec["res"] == "D")) ** 2 +
            (rec["pA"] - (rec["res"] == "A")) ** 2)

def logloss(rec, eps=1e-12):
    p = {"H": rec["pH"], "D": rec["pD"], "A": rec["pA"]}[rec["res"]]
    return -math.log(min(max(p, eps), 1.0))

def _pick(rec):
    return max((rec["pH"], "H"), (rec["pD"], "D"), (rec["pA"], "A"))[1]

def metrics(preds):
    n = len(preds)
    if n == 0:
        return None
    return {"n": n,
            "brier": sum(brier3(r) for r in preds) / n,
            "logloss": sum(logloss(r) for r in preds) / n,
            "acc": sum(1 for r in preds if _pick(r) == r["res"]) / n}

# --------------------------------------------------------------- experiment
def _key(r):
    return (r["date"], r["div"], r["home"], r["away"])

def run_experiment(matches):
    seasons = sorted({m["season"] for m in matches})
    if not seasons:
        print("No xG matches parsed — nothing to test."); return
    holdout_season = seasons[-1]                       # newest full season = temporal holdout

    base_all = backtest(matches, "goals")
    xg_all   = backtest(matches, "xg")

    # Compare on the IDENTICAL set of matches (intersection of both prediction sets).
    bkey = {_key(r): r for r in base_all}
    xkey = {_key(r): r for r in xg_all}
    common = [k for k in bkey if k in xkey]
    base = [bkey[k] for k in common]
    xg   = [xkey[k] for k in common]

    def split(preds, seas, keep):
        return [r for r in preds if (r["season"] == seas) == keep]
    base_hold = split(base, holdout_season, True)
    xg_hold   = split(xg,   holdout_season, True)

    print("=" * 74)
    print("xG-vs-GOALS 1X2 EXPERIMENT — leak-free expanding-window backtest")
    print("Model: soccer_model Dixon-Coles fit + match_probs (unchanged). Only the")
    print("Poisson response differs: actual goals (baseline) vs xG (treatment).")
    print("=" * 74)
    print(f"leagues={sorted({m['div'] for m in matches})}  seasons={seasons}  "
          f"matches_compared={len(common)}  holdout_season={holdout_season}")
    print()

    print("PER-SEASON  (multiclass Brier / log-loss / acc%, on identical matches)")
    print(f"{'season':>8} {'n':>5} | {'base Brier':>10} {'xg Brier':>9} {'Δ':>7} | "
          f"{'base LL':>8} {'xg LL':>7} | {'base acc':>8} {'xg acc':>7}")
    xg_wins = 0; seasons_counted = 0
    for s in seasons:
        b = metrics(split(base, s, True)); x = metrics(split(xg, s, True))
        if not (b and x):
            continue
        seasons_counted += 1
        if x["brier"] < b["brier"]:
            xg_wins += 1
        print(f"{s:>8} {b['n']:>5} | {b['brier']:>10.4f} {x['brier']:>9.4f} "
              f"{b['brier']-x['brier']:>+7.4f} | {b['logloss']:>8.4f} {x['logloss']:>7.4f} | "
              f"{100*b['acc']:>7.1f}% {100*x['acc']:>6.1f}%")

    bh = metrics(base_hold); xh = metrics(xg_hold)
    print()
    print("=" * 74)
    print("VERDICT — HOLDOUT (newest season, out-of-sample, identical matches)")
    print("=" * 74)
    if not (bh and xh):
        print("Insufficient holdout data for a verdict."); return
    print(f"  holdout matches ............. {bh['n']}")
    print(f"  baseline (goals) Brier ...... {bh['brier']:.4f}   acc {100*bh['acc']:.1f}%   LL {bh['logloss']:.4f}")
    print(f"  treatment (xG)   Brier ...... {xh['brier']:.4f}   acc {100*xh['acc']:.1f}%   LL {xh['logloss']:.4f}")
    d_brier = bh["brier"] - xh["brier"]
    d_ll = bh["logloss"] - xh["logloss"]
    print(f"  Δ Brier (base - xg) ......... {d_brier:+.4f}   ({'xG better' if d_brier>0 else 'goals better'})")
    print(f"  Δ log-loss (base - xg) ...... {d_ll:+.4f}   ({'xG better' if d_ll>0 else 'goals better'})")
    print(f"  seasons where xG wins Brier . {xg_wins}/{seasons_counted}")
    robust = (d_brier > 0 and d_ll > 0 and xg_wins > seasons_counted / 2)
    print()
    if robust:
        print("  >>> VERDICT: xG-based ratings ROBUSTLY WIN — lower holdout Brier AND")
        print("      log-loss, and win the Brier in a majority of seasons. Worth")
        print("      promoting to a candidate model (then re-test vs the closing market).")
    elif d_brier > 0:
        print("  >>> VERDICT: xG is BETTER ON HOLDOUT Brier but NOT robust across all")
        print("      three checks (log-loss / per-season majority). Promising, not proven.")
    else:
        print("  >>> VERDICT: xG does NOT beat goals for 1X2 here. Keep the goals model.")
    print("  NOTE: this only compares the two models to each other. Beating the")
    print("        closing market is a separate, harder bar and is NOT claimed here.")

# =====================================================================
#                         OFFLINE  SELFTEST
# =====================================================================
def _mock_understat_escape(s):
    """Re-encode a JSON string the way understat does (\\xNN for every char) so
    the parser can be exercised with NO network."""
    out = []
    for ch in s:
        for b in ch.encode("utf-8"):
            out.append("\\x%02x" % b)
    return "".join(out)

def _synth_xg(n_teams=14, rounds=3, seed=7, home=math.log(1.3), mu=math.log(1.35),
              season=2023, noise=0.35):
    """Double round-robin with KNOWN att/dfn. xG is the true Poisson mean times
    positive multiplicative noise (mean 1) — so E[xG] encodes att/dfn exactly —
    and actual goals are Poisson(mean) for the 1X2 label."""
    rng = np.random.default_rng(seed)
    att = rng.normal(0, 0.30, n_teams); att -= att.mean()
    dfn = rng.normal(0, 0.25, n_teams); dfn -= dfn.mean()
    teams = [f"Club {chr(65 + i)}" for i in range(n_teams)]
    ms = []
    day = dt.date(2023, 8, 12); k = 0
    for _ in range(rounds):
        pairs = [(i, j) for i in range(n_teams) for j in range(n_teams) if i != j]
        rng.shuffle(pairs)
        for i, j in pairs:
            lh = math.exp(mu + home + att[i] - dfn[j])
            la = math.exp(mu + att[j] - dfn[i])
            # gamma multiplicative noise, mean 1 (shape=1/noise^2, scale=noise^2)
            shape = 1.0 / (noise * noise)
            xgh = lh * rng.gamma(shape, 1.0 / shape)
            xga = la * rng.gamma(shape, 1.0 / shape)
            gh = int(rng.poisson(lh)); ga = int(rng.poisson(la))
            ms.append({"div": "E0", "season": season, "date": day,
                       "home": teams[i], "away": teams[j],
                       "g_h": gh, "g_a": ga, "xgh": float(xgh), "xga": float(xga)})
            k += 1
            if k % 7 == 0:
                day += dt.timedelta(days=3)
        day += dt.timedelta(days=14)
    truth = {teams[i]: (float(att[i]), float(dfn[i])) for i in range(n_teams)}
    return ms, truth

def selftest():
    # ---- 1. PARSER (offline): build a fake understat datesData, escape it \xNN,
    #         and prove decode+parse round-trips, incl. an accented team name.
    fake = [
        {"isResult": True, "datetime": "2023-08-12 15:00:00",
         "h": {"title": "Atlético Madrid"}, "a": {"title": "Cádiz"},
         "goals": {"h": "2", "a": "1"}, "xG": {"h": "1.85", "a": "0.90"}},
        {"isResult": False, "datetime": "2024-05-19 15:00:00",   # future -> dropped
         "h": {"title": "X"}, "a": {"title": "Y"},
         "goals": {"h": None, "a": None}, "xG": {"h": None, "a": None}},
    ]
    raw = json.dumps(fake, ensure_ascii=False)
    page = "something var datesData = JSON.parse('" + _mock_understat_escape(raw) + "'); more"
    rows = parse_understat_dates(page, "SP1", 2023)
    assert len(rows) == 1, rows
    r0 = rows[0]
    assert r0["home"] == "Atlético Madrid" and r0["away"] == "Cádiz", r0    # UTF-8 intact
    assert r0["g_h"] == 2 and r0["g_a"] == 1
    assert abs(r0["xgh"] - 1.85) < 1e-9 and abs(r0["xga"] - 0.90) < 1e-9
    assert r0["date"] == dt.date(2023, 8, 12) and r0["div"] == "SP1"

    # ---- 2. BRIER / LOG-LOSS math on a hand fixture
    rec = {"pH": 0.5, "pD": 0.3, "pA": 0.2, "res": "H"}
    assert abs(brier3(rec) - (0.25 + 0.09 + 0.04)) < 1e-12, brier3(rec)      # = 0.38
    assert abs(logloss(rec) - (-math.log(0.5))) < 1e-12
    rec2 = {"pH": 0.1, "pD": 0.7, "pA": 0.2, "res": "D"}
    assert abs(brier3(rec2) - (0.01 + 0.09 + 0.04)) < 1e-12                  # = 0.14
    m3 = metrics([rec, rec2])
    assert abs(m3["brier"] - (0.38 + 0.14) / 2) < 1e-12 and m3["n"] == 2
    assert _pick(rec) == "H" and _pick(rec2) == "D"

    # ---- 3. xG-AS-GOALS DIXON-COLES RECOVERY: fit on xG must recover the known
    #         attack/defence strengths (statistical, across seeds).
    corrs, maes = [], []
    for seed in (7, 11, 13, 21, 42):
        ms, truth = _synth_xg(rounds=3, seed=seed)
        view = [_fit_view(m, "xg") for m in ms]
        ratings, home_adv, rho, mu = sm.fit_league(view, iters=600)
        corrs.append(float(np.corrcoef([truth[t][0] for t in truth],
                                        [ratings[t][0] for t in truth])[0, 1]))
        errs = ([abs(ratings[t][0] - truth[t][0]) for t in truth] +
                [abs(ratings[t][1] - truth[t][1]) for t in truth])
        maes.append(sum(errs) / len(errs))
    corr = float(np.median(corrs)); mae = float(np.median(maes))
    assert corr > 0.90 and min(corrs) > 0.85, corrs
    assert mae < 0.12, maes

    # ---- 4. LEAK-FREENESS: fitting for a match uses ONLY strictly-earlier
    #         matches. Poison every match AFTER a cut date with absurd values;
    #         every prediction on/before the cut must be byte-identical.
    ms, _ = _synth_xg(rounds=3, seed=7)
    base = backtest(ms, "xg", refit_days=15, min_train=40, iters=120)
    assert len(base) > 40, len(base)
    cut = base[len(base) // 2]["date"]
    poisoned = [dict(m) for m in ms]
    for m in poisoned:
        if m["date"] > cut:
            m["xgh"] = 99.0; m["xga"] = 0.0; m["g_h"] = 9; m["g_a"] = 0
    pois = backtest(poisoned, "xg", refit_days=15, min_train=40, iters=120)
    pk = {_key(r): r for r in pois}
    checked = 0
    for r in base:
        if r["date"] <= cut:
            q = pk[_key(r)]
            assert abs(q["pH"] - r["pH"]) < 1e-12 and abs(q["pD"] - r["pD"]) < 1e-12 \
                and abs(q["pA"] - r["pA"]) < 1e-12, ("LEAK", _key(r), r, q)
            checked += 1
    assert checked > 20, checked

    # ---- 5. END-TO-END plumbing: both modes produce valid distributions and a
    #         metrics block over a synthetic two-season set.
    two = _synth_xg(rounds=2, seed=5, season=2023)[0] + _synth_xg(rounds=2, seed=6, season=2024)[0]
    for mode in ("goals", "xg"):
        preds = backtest(two, mode, refit_days=30, min_train=60, iters=150)
        assert preds, mode
        for r in preds:
            assert abs(r["pH"] + r["pD"] + r["pA"] - 1) < 1e-6
        mm = metrics(preds)
        assert mm and 0 < mm["brier"] < 2 and 0.15 < mm["acc"] < 0.9, (mode, mm)

    # getLeagueData API path: the 'dates' list feeds the same extractor
    api_rows = [
        {"isResult": True, "datetime": "2024-08-16 19:00:00",
         "h": {"title": "Manchester United"}, "a": {"title": "Fulham"},
         "goals": {"h": "1", "a": "0"}, "xG": {"h": "1.84", "a": "0.62"}},
        {"isResult": False, "datetime": "2027-01-01 15:00:00",
         "h": {"title": "X"}, "a": {"title": "Y"},
         "goals": {"h": None, "a": None}, "xG": {"h": None, "a": None}},
    ]
    api_m = _matches_from_dates(api_rows, "E0", 2024)
    assert len(api_m) == 1 and api_m[0]["home"] == "Manchester United"
    assert abs(api_m[0]["xgh"] - 1.84) < 1e-9 and api_m[0]["g_a"] == 0
    assert api_m[0]["date"] == dt.date(2024, 8, 16)

    # 538 SPI fallback parser: league filter, type coercion, junk-row tolerance
    spi_fix = ("season,date,league_id,league,team1,team2,spi1,spi2,prob1,prob2,probtie,"
               "proj_score1,proj_score2,importance1,importance2,score1,score2,xg1,xg2,"
               "nsxg1,nsxg2,adj_score1,adj_score2\n"
               "2022,2022-08-05,2411,Barclays Premier League,Crystal Palace,Arsenal,"
               "70.1,82.3,0.2,0.55,0.25,1.0,1.8,30,60,0,2,1.2,1.31,0.9,1.4,0.0,2.1\n"
               "2022,2022-08-06,9999,Some Other League,X,Y,1,1,0.3,0.4,0.3,1,1,1,1,1,1,0.5,0.5,0.4,0.4,1,1\n"
               "2022,2022-08-07,2411,Barclays Premier League,Leeds,Wolves,60,60,0.4,0.3,0.3,1.4,1.1,20,20,,,,,,,,\n")
    spi = parse_spi_csv(spi_fix)
    assert len(spi) == 1, f"538 parser: expected 1 usable row, got {len(spi)}"
    assert spi[0]["div"] == "E0" and spi[0]["g_a"] == 2 and abs(spi[0]["xgh"] - 1.2) < 1e-9
    assert spi[0]["date"] == dt.date(2022, 8, 5) and spi[0]["home"] == "Crystal Palace"

    print("XG-EXPERIMENT SELFTEST PASS — understat parser (UTF-8 \\xNN decode), "
          f"Brier/log-loss math, xG-as-goals recovery (corr {corr:.2f}, MAE {mae:.3f}), "
          f"leak-freeness ({checked} predictions unchanged under future poisoning), "
          "backtest plumbing (both modes), 538 SPI fallback parser.")
    return 0

# --------------------------------------------------------------- entrypoint
def main():
    ap = argparse.ArgumentParser(description="xG-vs-goals 1X2 accuracy experiment")
    ap.add_argument("--selftest", action="store_true", help="offline synthetic tests, no network")
    ap.add_argument("--save-cache", action="store_true",
                    help="on a successful pull, write data/understat_xg.json for Actions to reuse")
    ap.add_argument("--refresh-cache", action="store_true",
                    help="fetch understat and update the cache ONLY (no backtest) — used by "
                         "the daily workflow, fail-soft: keeps the old cache on any failure")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if args.refresh_cache:
        try:
            matches, note = fetch_understat()
            if not matches:
                print("refresh-cache: pull produced 0 matches — keeping existing cache")
                return 0
            os.makedirs(DATA, exist_ok=True)
            with open(CACHE, "w") as f:
                json.dump([{**m, "date": m["date"].isoformat()} for m in matches], f)
            print(f"refresh-cache: {note} -> {CACHE}")
        except Exception as e:
            print(f"refresh-cache: understat unavailable ({type(e).__name__}) — keeping existing cache")
        return 0

    # Real experiment path — network required. Degrade cleanly if blocked.
    try:
        matches, note = fetch_understat()
        print("understat pull:", note)
        if not matches:
            # understat answered but with challenge/changed HTML (0 parsed) — that is
            # just as dead as unreachable. Route to the 538 fallback.
            raise RuntimeError("understat reachable but parsed 0 matches")
        if args.save_cache and matches:
            os.makedirs(DATA, exist_ok=True)
            with open(CACHE, "w") as f:
                json.dump([{**m, "date": m["date"].isoformat()} for m in matches], f)
            print("cached ->", CACHE)
    except RuntimeError as e:
        # Fall back to a committed cache if one exists (Actions may have saved it).
        if os.path.exists(CACHE):
            print("understat live pull failed — using committed cache", CACHE)
            with open(CACHE) as f:
                matches = [{**m, "date": dt.date.fromisoformat(m["date"])} for m in json.load(f)]
        else:
            print(str(e))
            print("understat unavailable — trying FiveThirtyEight SPI xG (2016-2023)...")
            try:
                matches, note = fetch_538()
                print("538 pull:", note)
                if args.save_cache and matches:
                    os.makedirs(DATA, exist_ok=True)
                    with open(CACHE, "w") as f:
                        json.dump([{**m, "date": m["date"].isoformat()} for m in matches], f)
                    print("cached ->", CACHE)
            except Exception as e2:
                print(f"538 SPI also unreachable ({type(e2).__name__}).")
                print("Nothing to do from here. Exiting 0 (expected in a blocked sandbox).")
                return 0
    run_experiment(matches)
    return 0

if __name__ == "__main__":
    sys.exit(main())
