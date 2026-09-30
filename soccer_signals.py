#!/usr/bin/env python3
"""soccer_signals.py — the Signals tab's data file.

    python soccer_signals.py              # write data/signals.json
    python soccer_signals.py --selftest

WHERE THE NUMBERS COME FROM. The parlay repo's socextra builds league tables
for the countries football-data.co.uk does not publish at all -- Colombia,
Peru, Chile, Uruguay, Bolivia -- by reading Wikipedia's published season
results. This file does NOT re-implement any of that. It fetches the built
socextra.json and trims it to what the page renders.

That split is deliberate. socextra took six bug fixes to get right (a rowspan
that silently dropped half of every league's dates, a header row that was not
row 0, a reversed scoreline caught only because two readers disagreed) and
every one of those would have had to be found twice if the parser lived in
both repos. One builder, one source of truth, and this end stays a reader.

WHAT THE PAGE IS ALLOWED TO SHOW. Six signals: home/away splits, head to
head, opponent rank, recent form, injury impact, starter. The first three
come off the results, recent form comes off dated rows, and the last two are
not in any results table anywhere -- so the page prints the REASON a signal
is dark rather than an empty cell. A card with two invented rows is worse
than one with two honest blanks.

FAIL-SOFT ON PURPOSE. If the fetch fails, the committed signals.json is left
exactly as it is. A stale Signals tab is a small problem; a tab that empties
itself because GitHub had a bad minute is a page that looks broken.
"""
import json, os, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'data', 'signals.json')
SRC = 'https://raw.githubusercontent.com/CheeseDagg/parlay/main/socextra.json'
UA = 'Mozilla/5.0 (compatible; soccertool-signals/1.0)'


def fetch(url=SRC, timeout=40):
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())


def trim(doc):
    """socextra.json -> the page payload. Drops per-match rows (the page never
    shows them) and keeps only what a signal card reads."""
    out = {}
    for league, v in (doc or {}).items():
        rates = v.get('rates') or {}
        res = rates.get('result')
        if not res:
            continue
        sp = v.get('splits') or {}
        home, away = sp.get('home') or {}, sp.get('away') or {}
        form = {k: {'form': x['form'], 'gf': x['gf'], 'ga': x['ga'], 'newest': x['newest']}
                for k, x in (v.get('form') or {}).items()}
        teams = sorted(set(home) | set(away))
        if not teams:
            continue
        out[league] = {
            'base': res,
            'under': rates.get('under') or {},
            'teams': teams,
            'home': home,
            'away': away,
            'form': form,
            'h2h': sp.get('h2h') or {},
            'src': v.get('rates_source') or '',
            'pages': v.get('es_pages') or [],
        }
    return out


def selftest():
    ok = [0, 0]

    def chk(c, m):
        ok[1] += 1
        ok[0] += bool(c)
        print(('PASS  ' if c else 'FAIL  ') + m)

    doc = {
        'Testland Primera': {
            'rates': {'result': {'home': .45, 'draw': .3, 'away': .25,
                                 'mean_goals': 2.5, 'n': 100},
                      'under': {'2.5': .5, '3.5': .75}},
            'splits': {'home': {'Alpha': {'p': 5, 'w': 4, 'd': 1, 'l': 0,
                                          'gf': 10, 'ga': 2, 'ppg': 2.6}},
                       'away': {'Beta': {'p': 5, 'w': 1, 'd': 1, 'l': 3,
                                         'gf': 3, 'ga': 9, 'ppg': 0.8}},
                       'h2h': {'Alpha|Beta': [{'home': 'Alpha', 'away': 'Beta',
                                               'hg': 2, 'ag': 0}]}},
            'form': {'Alpha': {'form': 'WWDLW', 'n': 5, 'gf': 9, 'ga': 4,
                               'newest': '2026-09-20', 'matches': [{'x': 1}]}},
            'matches': [['Alpha', 'Beta', 2, 0]] * 40,
            'dated': [['2026-09-20', 'Alpha', 'Beta', 2, 0]] * 40,
            'rates_source': 'dated rounds',
            'es_pages': ['Torneo_X'],
        },
        'No Rates League': {'splits': {'home': {'X': {}}}},
        'No Teams League': {'rates': {'result': {'home': .4, 'draw': .3, 'away': .3,
                                                 'mean_goals': 2, 'n': 10}},
                            'splits': {}},
    }
    p = trim(doc)
    chk(list(p) == ['Testland Primera'],
        'a league with no rates, and one with no teams, are both dropped')
    t = p['Testland Primera']
    chk(t['teams'] == ['Alpha', 'Beta'], 'teams come from BOTH home and away splits')
    chk(t['base']['n'] == 100 and t['under']['2.5'] == .5, 'base and under ladder carry through')
    chk(t['h2h']['Alpha|Beta'][0]['hg'] == 2, 'head to head survives the trim')
    chk('matches' not in t and 'dated' not in t,
        'per-match rows are DROPPED -- the page never shows them')
    chk('matches' not in t['form']['Alpha'],
        "and form keeps its string, not its match list")
    chk(t['form']['Alpha']['newest'] == '2026-09-20',
        'the newest date rides along -- the page gates stale form on it')
    chk(t['src'] == 'dated rounds', 'provenance is carried to the page, not hidden')

    small = json.dumps(p)
    chk(len(small) < len(json.dumps(doc)), 'the trim is actually smaller than the source')
    chk(json.loads(small) == p, 'the payload round-trips through JSON for the browser')
    chk(trim({}) == {} and trim(None) == {}, 'an empty or missing source yields {} , not a crash')

    print(f'\n{ok[0]}/{ok[1]} checks pass')
    return 0 if ok[0] == ok[1] else 1


def main():
    try:
        doc = fetch()
    except Exception as e:
        print(f'signals: could not fetch socextra.json ({type(e).__name__}: {e})')
        print('signals: KEEPING the committed data/signals.json unchanged')
        return 0
    payload = trim(doc)
    if not payload:
        print('signals: source had no usable leagues -- keeping the committed file')
        return 0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(',', ':'))
    tot = sum(v['base']['n'] for v in payload.values())
    print(f'signals: {len(payload)} leagues, {tot} matches -> {OUT}')
    for lg, v in sorted(payload.items()):
        newest = max((f['newest'] for f in v['form'].values() if f.get('newest')), default='—')
        print(f"  {lg:26} {v['base']['n']:4} matches, {len(v['teams']):2} clubs, form to {newest}")
    return 0


if __name__ == '__main__':
    sys.exit(selftest() if '--selftest' in sys.argv else main())
