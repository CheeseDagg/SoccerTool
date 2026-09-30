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
PAT = 'https://raw.githubusercontent.com/CheeseDagg/parlay/main/socpatterns.json'
PLY = 'https://raw.githubusercontent.com/CheeseDagg/parlay/main/socplayers.json'
# Player head-to-head is a SEPARATE file because it comes from a separate
# source. socplayers is openfootball, which has one season of scorers and
# therefore cannot produce a head-to-head at all; socph2h is understat's
# per-match appearance record, which spans seasons and can.
H2H = 'https://raw.githubusercontent.com/CheeseDagg/parlay/main/socph2h.json'
UA = 'Mozilla/5.0 (compatible; soccertool-signals/1.0)'


def fetch(url=SRC, timeout=40):
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())


def trim_patterns(pdoc, keep=140):
    """socpatterns.json -> the ranked list the tab leads with.

    Rarest-first upstream, so this keeps the head of the list AND the count of
    chances examined. The count is what makes a rarity honest: a run that
    chance alone would produce a dozen times over is not a finding."""
    rows = (pdoc or {}).get('patterns') or []
    # 'text' IS the finding. The rarity fields the first build carried are gone
    # from upstream: that statistic ranks the list and never reaches a reader.
    keepers = ('league', 'club', 'category', 'text', 'hits', 'n', 'streak',
               'rate', 'opponent', 'newest')
    out = []
    for r in rows[:keep]:
        row = {k: r[k] for k in keepers if k in r}
        row['evidence'] = (r.get('evidence') or [])[:6]
        out.append(row)
    return {'chances': (pdoc or {}).get('chances'), 'rows': out}


def trim_players(pdoc, keep=90):
    """socplayers.json -> player rows for the list.

    Three families, in the order a reader can use them, and each one names its
    source because they cannot all make the same kind of claim:

      Player        per-match streaks -- the only one that can say "of the
                    last 6", and the only one that expires
      Player split  home/away and head-to-head -- a standing record, so it is
                    built from a completed season and carries the span it
                    covers rather than being stale-refused
      Player rate   season totals -- cannot say WHICH games, only how many

    The page must never blur those, which is why the category rides on every
    row instead of being inferred from the wording.
    """
    d = pdoc or {}
    rows = []

    # PER-FAMILY CAPS, NOT ONE CAP ON THE TOTAL. The families arrive ranked and
    # concatenated, so a single `keep` lets whichever family is longest eat the
    # list: 118 splits against a keep of 90 would have pushed every season rate
    # off the page and left the tab looking like it only does home/away. Each
    # family gets its own room and the total is still bounded.
    CAP = {'streaks': 30, 'splits': 40, 'h2h': 40, 'rates': 40}

    def ev(r, n=6):
        return [{'date': e.get('date'), 'opp': e.get('opp'),
                 'gf': e.get('goals', 0), 'ga': 0, 'side': e.get('venue', '')}
                for e in (r.get('evidence') or [])[:n]]

    for r in (d.get('streaks') or [])[:CAP['streaks']]:
        rows.append({'club': r.get('player'), 'team': r.get('team'),
                     'league': r.get('league'), 'category': 'Player',
                     'text': r.get('text'), 'source': r.get('source'),
                     'evidence': ev(r)})
    # A PLAYER'S HOME/AWAY RECORD IS A HOME/AWAY RECORD. These rode in a
    # category of their own called 'Player split', so clicking "Home/Away"
    # showed only club season averages and clicking "Head-to-head" only
    # club-vs-club -- the player rows were filed under a third name nobody would
    # think to open, and the two categories a reader actually wants were left
    # looking like vague team stats. The category is what the split IS, not who
    # the subject happens to be; the row already names a player rather than a
    # club, so nothing is lost by filing it where it belongs.
    SPLIT_CAT = {'venue': 'Home/Away', 'h2h': 'Head-to-head'}
    for r in (d.get('splits') or [])[:CAP['splits']]:
        rows.append({'club': r.get('player'), 'team': r.get('team'),
                     'league': r.get('league'),
                     'category': SPLIT_CAT.get(r.get('split'), 'Home/Away'),
                     'text': r.get('text'), 'source': r.get('source'),
                     'split': r.get('split'), 'span': r.get('span'),
                     'evidence': ev(r)})
    for r in (d.get('h2h') or [])[:CAP['h2h']]:
        rows.append({'club': r.get('player'), 'team': r.get('team'),
                     'league': r.get('league'), 'category': 'Head-to-head',
                     'text': r.get('text'), 'source': r.get('source'),
                     'split': 'h2h', 'span': r.get('span'), 'evidence': ev(r)})
    for r in (d.get('rates') or [])[:CAP['rates']]:
        rows.append({'club': r.get('player'), 'team': r.get('team'),
                     'league': r.get('league'), 'category': 'Player rate',
                     'text': r.get('text'), 'source': r.get('source'),
                     'evidence': []})
    return rows[:keep]


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

    pd = {'chances': 870, 'patterns': [
        {'league':'L','club':'UTC','category':'Form','key':'winless',
         'text':'failed to win in all of the last 10 - 21 in a row',
         'hits':10,'n':10,'streak':21,'rate':1.0,'newest':'2026-09-23',
         'evidence':[{'date':'2026-09-23','gf':0,'ga':1,'opp':'X','side':'A'}]*9},
        {'league':'L','club':'Y','category':'Goals','key':'under',
         'text':'under 2.5 in 8 of last 10','hits':8,'n':10,'streak':2,'rate':0.8,
         'newest':'2026-09-20','evidence':[]}]}
    tp = trim_patterns(pd)
    chk(tp['chances'] == 870, 'the chances-examined count reaches the page')
    chk(len(tp['rows']) == 2 and tp['rows'][0]['club'] == 'UTC', 'rows carry through in order')
    chk('one_in' not in tp['rows'][0] and 'expected_by_chance' not in tp['rows'][0],
        'no rarity field survives the trim -- the page never sees that statistic')
    chk(tp['rows'][0]['text'].startswith('failed to win'),
        'the sentence is what carries through')
    chk(len(tp['rows'][0]['evidence']) == 6, 'evidence is capped so the payload stays small')
    chk('base' not in tp['rows'][0] and 'key' not in tp['rows'][0],
        'fields the page does not render are dropped')
    chk(trim_patterns({})['rows'] == [] and trim_patterns(None)['rows'] == [],
        'a missing patterns file yields an empty list, not a crash')
    pl = trim_players({'streaks': [{'player':'A. Semenyo','team':'Bournemouth',
            'league':'England Premier League','text':"scored in 4 of Bournemouth's last 6",
            'source':'openfootball per-match','evidence':[{'date':'2026-09-20','opp':'X','goals':1}]}],
        'rates': [{'player':'Erling Haaland','team':'Man City','league':'England Premier League',
            'text':'leads the league with 5 goals in 5 games','source':'understat season totals'}]})
    chk(len(pl) == 2 and pl[0]['club'] == 'A. Semenyo',
        'streaks lead the player rows, rates follow')
    chk(pl[0]['category'] == 'Player' and pl[1]['category'] == 'Player rate',
        'the two kinds are separate categories so the filter can tell them apart')
    chk(pl[0]['source'] != pl[1]['source'],
        'each names its source -- one can say "of last 6", the other cannot')
    chk(pl[0]['evidence'][0]['date'] == '2026-09-20', 'streak evidence carries through')
    chk(trim_players({}) == [] and trim_players(None) == [],
        'a missing player file yields nothing, not a crash')

    # A PLAYER'S HOME/AWAY RECORD FILES UNDER Home/Away. These rode in a
    # category called 'Player split', so the two categories a reader actually
    # opens showed only club rows and read as vague team stats. Nothing asserted
    # the category, which is why moving it broke no test.
    ps = trim_players({'splits': [
        {'player': 'H. Wilson', 'team': 'Fulham', 'league': 'L', 'split': 'venue',
         'span': '2025-08 to 2026-05', 'source': 'openfootball per-match',
         'text': "scored in 8 of Fulham's 19 home matches, 2 of 19 away"},
        {'player': 'A. Nemesis', 'team': 'Cats', 'league': 'L', 'split': 'h2h',
         'span': '2025-08 to 2026-05', 'source': 'openfootball per-match',
         'text': "scored in 3 of Cats' 4 meetings with Rival"}]})
    chk([r['category'] for r in ps] == ['Home/Away', 'Head-to-head'],
        'a player venue split is Home/Away and a player h2h is Head-to-head')
    chk(not any(r['category'] == 'Player split' for r in ps),
        "no row is filed under a third name a reader would not think to open")
    chk(all(r.get('span') for r in ps),
        'and each still carries the span, because it is a finished record')
    # Player head-to-head arrives as its own family from its own source, and
    # lands in the Head-to-head category beside the club records.
    ph = trim_players({'h2h': [
        {'player': 'E. Haaland', 'team': 'Manchester City', 'league': 'L',
         'span': '2023-08 to 2026-02', 'source': 'understat per-match appearances',
         'text': 'scored in 5 of his 6 against Wolves — 8 goals',
         'evidence': [{'date': '2026-02-01', 'opp': 'Wolves', 'goals': 2,
                       'venue': 'home'}]}]})
    chk(len(ph) == 1 and ph[0]['category'] == 'Head-to-head',
        'a player head-to-head record files under Head-to-head')
    chk(ph[0]['span'] == '2023-08 to 2026-02' and 'appearances' in ph[0]['source'],
        'it carries a multi-season span and names the appearance source that '
        'makes "his" an honest denominator')
    chk(ph[0]['evidence'][0]['side'] == 'home',
        'and the venue of each meeting rides along')
    both = trim_players({'splits': [{'player': 'A', 'team': 'T', 'split': 'venue',
                                     'text': 'v', 'source': 's'}],
                         'h2h': [{'player': 'B', 'team': 'T', 'text': 'h',
                                  'source': 's'}]})
    chk([r['category'] for r in both] == ['Home/Away', 'Head-to-head'],
        'the two split families do not collide -- each lands in its own category')

    odd = trim_players({'splits': [{'player': 'P', 'team': 'T', 'split': 'something_new',
                                    'text': 't', 'source': 's'}]})
    chk(odd[0]['category'] == 'Home/Away',
        'an unrecognised split kind lands in a real category rather than inventing one')

    mixed = dict(trim({'L': {'rates': {'result': {'home':.4,'draw':.3,'away':.3,
                                                  'mean_goals':2,'n':9}},
                             'splits': {'home': {'A': {}}, 'away': {'B': {}}}}}))
    mixed['_patterns'] = tp
    lg_only = {k: v for k, v in mixed.items() if not k.startswith('_')}
    chk(list(lg_only) == ['L'],
        "'_patterns' rides in the same file but is never counted as a league")
    chk(sum(v['base']['n'] for v in lg_only.values()) == 9,
        'and the league summary still totals correctly beside it')

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
    try:
        payload['_patterns'] = trim_patterns(fetch(PAT))
    except Exception as e:
        print(f'signals: patterns unavailable ({type(e).__name__}) -- fixture card only')
    try:
        pdoc = fetch(PLY)
        # Merge the head-to-head file in as another family. Fail-soft on its own:
        # it is a different source with a different failure mode, and losing it
        # must not cost the splits and rates that did come back.
        try:
            pdoc = dict(pdoc, h2h=(fetch(H2H) or {}).get('h2h') or [])
            print(f"signals: {len(pdoc['h2h'])} player head-to-head records fetched")
        except Exception as e:
            print(f'signals: player head-to-head unavailable ({type(e).__name__}) '
                  f'-- the other player families are unaffected')
        pl = trim_players(pdoc)
        payload.setdefault('_patterns', {'chances': None, 'rows': []})
        payload['_patterns']['rows'] = pl + payload['_patterns']['rows']
        payload['_players'] = len(pl)
        print(f'signals: {len(pl)} player rows merged in')
    except Exception as e:
        print(f'signals: player signals unavailable ({type(e).__name__})')
    if not any(k for k in payload if not k.startswith('_')):
        print('signals: source had no usable leagues -- keeping the committed file')
        return 0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(',', ':'))
    # Iterate LEAGUES only: '_patterns' shares this dict and is not a league,
    # which broke this summary the moment it was added.
    leagues = {k: v for k, v in payload.items() if not k.startswith('_')}
    tot = sum(v['base']['n'] for v in leagues.values())
    pats = (payload.get('_patterns') or {}).get('rows') or []
    print(f'signals: {len(leagues)} leagues, {tot} matches, '
          f'{len(pats)} ranked patterns -> {OUT}')
    for lg, v in sorted(leagues.items()):
        newest = max((f['newest'] for f in v['form'].values() if f.get('newest')), default='—')
        print(f"  {lg:26} {v['base']['n']:4} matches, {len(v['teams']):2} clubs, form to {newest}")
    return 0


if __name__ == '__main__':
    sys.exit(selftest() if '--selftest' in sys.argv else main())
