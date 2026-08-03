"""
soccer_props.py — anytime-goalscorer props from the team goals model.
P(player scores) = 1 - exp(-lambda_player), where
  lambda_player = lambda_team (Dixon-Coles) x goal_share x availability
  goal_share    = blended (goals 60% / xG 40%) share of team output
  availability  = player minutes / team's most-played minutes

Player data: understat (per-player goals, xG, minutes) — the standard free
source for exactly these four leagues. UNREACHABLE from the dev sandbox, so the
parsers are fixture-tested against understat's known formats and fail LOUD in
production: on any parse miss the error carries the page context into the Action
log, making the first run its own probe.

SOURCE ORDER MATTERS AND WAS WRONG. Understat's 2026 redesign moved the data out
of the page and into the JSON endpoint the league page's own JS calls:

    GET /getLeagueData/{select-value}/{year}
      -> {"dates": [...], "teams": {...}, "players": [...]}

soccer_xg_experiment.py was ported to that endpoint; THIS module was not. It kept
scraping `playersData` out of the HTML and, when that missed, POSTing to a
/main/getPlayersStats/ route that no longer exists. The result was visible in
production and had been for weeks: data/slate.json carried

    props_src: "E0:off(ValueError) · SP1:off(ValueError) · D1:off(ValueError) · F1:off(ValueError)"

i.e. the Props tab was empty in every league, on every run, and the only symptom
on the site was the neutral-sounding "no fixtures carry player shares yet". The
API is now tried FIRST and the legacy scrape is the fallback, which is the order
the live evidence supports.
"""
import re, json, gzip, urllib.request, urllib.parse

UNDERSTAT = {"E0": "EPL", "SP1": "La_liga", "D1": "Bundesliga", "F1": "Ligue_1"}
# The API's league selector is NOT the URL slug: it uses spaces, not underscores
# ("La liga", not "La_liga"), and must be percent-encoded. Getting this wrong
# yields a 404 that looks exactly like "understat is down".
UNDERSTAT_API = "https://understat.com/getLeagueData/{league}/{year}"
UNDERSTAT_API_LEAGUE = {"E0": "EPL", "SP1": "La liga", "D1": "Bundesliga", "F1": "Ligue 1"}
GOAL_W, XG_W = 0.60, 0.40

# football-data name  <-  understat team_title (only the known deltas)
ALIASES = {
    "Manchester United": "Man United", "Manchester City": "Man City",
    "Newcastle United": "Newcastle", "Wolverhampton Wanderers": "Wolves",
    "Nottingham Forest": "Nott'm Forest", "Tottenham": "Tottenham",
    "West Bromwich Albion": "West Brom", "Sheffield United": "Sheffield United",
    "Atletico Madrid": "Ath Madrid", "Athletic Club": "Ath Bilbao",
    "Real Sociedad": "Sociedad", "Celta Vigo": "Celta", "Cadiz": "Cadiz",
    "Alaves": "Alaves", "Real Betis": "Betis",
    # understat "Espanyol" vs football-data "Espanol" -- ONE letter apart, and
    # neither normalises to a substring of the other, so the fuzzy fallback never
    # caught it. Every Espanyol match was dropped from the xG overlay and every
    # Espanyol player from the props board, in silence, for the whole history.
    "Espanyol": "Espanol",
    "Borussia M.Gladbach": "M'gladbach", "Borussia Dortmund": "Dortmund",
    "Bayern Munich": "Bayern Munich", "RasenBallsport Leipzig": "RB Leipzig",
    "Eintracht Frankfurt": "Ein Frankfurt", "Bayer Leverkusen": "Leverkusen",
    "FC Cologne": "FC Koln", "VfB Stuttgart": "Stuttgart",
    "Paris Saint Germain": "Paris SG", "Saint-Etienne": "St Etienne",
}

def _norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())

def map_team(understat_title, fd_teams):
    """understat team_title -> the football-data team string used by our model."""
    if understat_title in ALIASES and ALIASES[understat_title] in fd_teams:
        return ALIASES[understat_title]
    if understat_title in fd_teams:
        return understat_title
    nu = _norm(understat_title)
    if not nu:
        return None
    # fd_teams is a SET at every call site, so iterating it and returning the first
    # substring hit made the mapping depend on hash order: when two football-data
    # names both contain (or are contained by) the understat name -- "Union Berlin"
    # vs "FC Union Berlin", "Leeds" vs "Leeds United" -- the winner could differ
    # between runs of the same code on the same data. Every player on that club
    # would then be attributed to a different team, silently, on some days only.
    # Score all candidates and break ties deterministically.
    exact = [t for t in fd_teams if _norm(t) == nu]
    if exact:
        return sorted(exact)[0]
    # Prefer the closest length match, then alphabetical. A one- or two-character
    # normalised name is not evidence of anything; substring-matching on it would
    # attach a whole squad to whichever club happens to contain those letters.
    if len(nu) < 4:
        return None
    cand = [t for t in fd_teams
            if _norm(t) and (nu in _norm(t) or _norm(t) in nu) and len(_norm(t)) >= 4]
    if not cand:
        return None
    return sorted(cand, key=lambda t: (abs(len(_norm(t)) - len(nu)), t))[0]

def parse_players_page(html):
    """understat league page -> [{name, team, goals, xg, minutes}].
    Tries known embeddings in order; on total miss the error carries a window
    AROUND the token so the Action log itself hands over the fix."""
    data = None
    m = re.search(r"playersData\s*=\s*JSON\.parse\('(.*?)'\)", html, re.S)
    if m:                                   # legacy hex-escaped embed
        data = json.loads(m.group(1).encode("utf-8").decode("unicode_escape"))
    if data is None:
        m = re.search(r"playersData\s*=\s*(\[.*?\])\s*[;,<]", html, re.S)
        if m:                               # direct JSON array assignment
            try: data = json.loads(m.group(1))
            except ValueError: data = None
    if data is None:                        # any-script scan for the array
        for sm in re.finditer(r"<script[^>]*>(.*?)</script>", html, re.S):
            body = sm.group(1)
            if "playersData" not in body: continue
            m = re.search(r"JSON\.parse\('(.*?)'\)", body, re.S)
            if m:
                try:
                    data = json.loads(m.group(1).encode("utf-8").decode("unicode_escape")); break
                except ValueError: pass
    if data is None:                        # rename-proof: any parse blob w/ player rows
        for pm in re.finditer(r"JSON\.parse\('(.*?)'\)", html, re.S):
            try:
                cand = json.loads(pm.group(1).encode("utf-8").decode("unicode_escape"))
            except ValueError:
                continue
            if isinstance(cand, list) and cand and isinstance(cand[0], dict) \
               and "player_name" in cand[0]:
                data = cand; break
    if data is None:
        i = html.find("playersData")
        ctx = (html[max(0, i-40): i+160].replace("\n", " ")
               if i >= 0 else "TOKEN ABSENT; head: " + html[:120].replace("\n", " "))
        raise ValueError("playersData unparsed; context: " + ctx)
    out = []
    for p in data:
        try:
            out.append({"name": p["player_name"], "team": p["team_title"].split(",")[0],
                        "goals": int(p.get("goals", 0) or 0),
                        "xg": float(p.get("xG", 0) or 0),
                        "minutes": int(p.get("time", 0) or 0)})
        except (KeyError, TypeError, ValueError):
            continue
    if not out:
        raise ValueError("playersData parsed but yielded no rows")
    return out

def rows_to_players(data):
    """understat player dicts -> our row shape. Shared by every source so a fix to
    one field name cannot apply to the API path and miss the scrape path."""
    out = []
    for p in data:
        try:
            out.append({"name": p["player_name"], "team": p["team_title"].split(",")[0],
                        "goals": int(p.get("goals", 0) or 0),
                        "xg": float(p.get("xG", p.get("xg", 0)) or 0),
                        "minutes": int(p.get("time", p.get("minutes", 0)) or 0)})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _http_api(url, referer):
    """GET understat's JSON endpoint the way its own page does.

    Copied deliberately from soccer_xg_experiment._http_api, which is the version
    that demonstrably works against the 2026 layout. The headers are not cargo
    cult: without X-Requested-With the endpoint 404s, and it will serve gzip
    whether or not you ask, so the magic-byte check is mandatory rather than an
    optimisation."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Encoding": "gzip",
        "Referer": referer,
    })
    with urllib.request.urlopen(req, timeout=45) as r:
        raw = r.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    text = raw.decode("utf-8", "replace").lstrip("﻿")
    if not text.lstrip().startswith(("{", "[")):
        raise ValueError(f"non-JSON response (starts: {text.lstrip()[:80]!r})")
    return text


def players_from_league_data(payload):
    """Pull the player list out of a getLeagueData response.

    The endpoint returns {"dates": [...], "teams": {...}, "players": [...]}. Accept
    a bare list too, and hunt any list-of-dicts carrying player_name, so a key
    rename degrades to a slower path instead of to zero props."""
    data = payload
    if isinstance(payload, dict):
        data = payload.get("players") or payload.get("playersData")
        if not isinstance(data, list):
            data = None
            for v in payload.values():
                if isinstance(v, list) and v and isinstance(v[0], dict) \
                   and "player_name" in v[0]:
                    data = v
                    break
    if not isinstance(data, list) or not data:
        raise ValueError("getLeagueData carried no player list; keys: "
                         + str(sorted(payload)[:8] if isinstance(payload, dict) else type(payload)))
    out = rows_to_players(data)
    if not out:
        raise ValueError(f"player list present ({len(data)} rows) but none mapped; "
                         f"first row keys: {sorted(data[0])[:10] if isinstance(data[0], dict) else '?'}")
    return out


def _players_from_post(slug, season_year, route="/main/getPlayersStats/"):
    """LEGACY: understat's pre-2026 POST AJAX source. Kept as a last resort only."""
    body = urllib.parse.urlencode({"league": slug, "season": str(season_year)}).encode()
    req = urllib.request.Request("https://understat.com" + route,
        data=body, headers={"User-Agent": "Mozilla/5.0 (SoccerTool props)",
                            "X-Requested-With": "XMLHttpRequest",
                            "Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=45) as r:
        j = json.loads(r.read().decode("utf-8", "replace"))
    data = j.get("response", {}).get("players") if isinstance(j, dict) else j
    if not isinstance(data, list) or not data:
        raise ValueError(f"api shape unexpected: {str(j)[:200]}")
    out = rows_to_players(data)
    if not out:
        raise ValueError("api rows empty after mapping")
    return out

def _page_endpoints(html):
    """what does the redesigned page actually load? name it in the log."""
    srcs = re.findall(r'<script[^>]+src="([^"]+)"', html)[:8]
    toks = []
    for t in ("getPlayers", "players", "PlayersStats", "api/"):
        i = html.find(t)
        if i >= 0: toks.append(html[max(0,i-40):i+120].replace("\n"," "))
    return " | scripts: " + ",".join(srcs) + (" | tokens: " + " ~ ".join(toks[:3]) if toks else "")

def fetch_league_players(div, season_year):
    """Per-player goals/xG/minutes for one league-season.

    Order is: JSON API (2026 layout) -> embedded-JSON page scrape (legacy) ->
    POST route (pre-2026). It used to be scrape-first with no API path at all,
    which is why every league logged off(ValueError) in production. If ALL THREE
    miss, the raised error names what each one did, because a props outage that
    only says "ValueError" costs another whole day to diagnose."""
    slug = UNDERSTAT[div]
    referer = f"https://understat.com/league/{slug}/{season_year}"
    errs = []

    # 1) the JSON endpoint the league page's own JS calls
    api_url = UNDERSTAT_API.format(
        league=urllib.parse.quote(UNDERSTAT_API_LEAGUE[div]), year=season_year)
    try:
        return players_from_league_data(json.loads(_http_api(api_url, referer)))
    except Exception as e:
        errs.append(f"api:{type(e).__name__}:{str(e)[:120]}")

    # 2) legacy page scrape. A network failure here is NOT the same as a parse
    #    failure and must not be reported as one -- if understat is unreachable,
    #    say so instead of blaming the parser.
    try:
        req = urllib.request.Request(referer, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"})
        with urllib.request.urlopen(req, timeout=45) as r:
            html = r.read().decode("utf-8", "replace")
    except Exception as e:
        raise ValueError(f"understat unreachable for {div}/{season_year}: "
                         f"page {type(e).__name__}:{str(e)[:100]} | " + " ; ".join(errs))
    try:
        return parse_players_page(html)
    except ValueError as e_page:
        errs.append(f"page:{str(e_page)[:120]}")

    # 3) pre-2026 POST routes, harvested from whatever the page still references
    routes = sorted(set(re.findall(r"/main/[A-Za-z]+/", html))) or ["/main/getPlayersStats/"]
    for route in routes[:3]:
        try:
            return _players_from_post(slug, season_year, route)
        except Exception as e:
            errs.append(f"post {route}:{type(e).__name__}:{str(e)[:60]}")

    varnames = re.findall(r"var\s+(\w+)\s*=\s*JSON\.parse", html)[:6]
    raise ValueError("all props sources failed for " + f"{div}/{season_year} :: "
                     + " ;; ".join(errs)
                     + " || routes on page: " + (",".join(routes) or "NONE")
                     + " || parse vars: " + (",".join(varnames) or "none")
                     + " || " + _page_endpoints(html)[:300])

def team_shares(players, fd_teams):
    """-> {fd_team: [{name, share, avail}]} using blended goals/xG shares."""
    by_team = {}
    for p in players:
        fd = map_team(p["team"], fd_teams)
        if fd: by_team.setdefault(fd, []).append(p)
    out = {}
    for fd, ps in by_team.items():
        blend = {p["name"]: GOAL_W * p["goals"] + XG_W * p["xg"] for p in ps}
        tot = sum(blend.values())
        if tot <= 0: continue
        max_min = max(p["minutes"] for p in ps) or 1
        rows = []
        for p in ps:
            share = blend[p["name"]] / tot
            avail = min(p["minutes"] / max_min, 1.0)
            if share > 0.01:
                rows.append({"name": p["name"], "share": round(share, 4),
                             "avail": round(avail, 3)})
        rows.sort(key=lambda r: -r["share"])
        out[fd] = rows
    return out

def anytime_probs(lam_team, shares_rows, top=6):
    """-> [{name, p}] anytime-goalscorer for one side of one fixture."""
    import math
    out = []
    for r in shares_rows[:top + 4]:
        lam_p = lam_team * r["share"] * r["avail"]
        out.append({"name": r["name"], "p": round(1 - math.exp(-lam_p), 3)})
    out.sort(key=lambda x: -x["p"])
    return out[:top]

def selftest():
    import math
    # 1. parser on a faithful understat-format fixture (hex-escaped embedded JSON)
    payload = [{"id": "1", "player_name": "Star Striker", "team_title": "Manchester United",
                "goals": "20", "xG": "17.5", "time": "3000"},
               {"id": "2", "player_name": "Rotation Kid", "team_title": "Manchester United",
                "goals": "5", "xG": "6.2", "time": "1200"},
               {"id": "3", "player_name": "Two Club Guy", "team_title": "Celta Vigo,Real Betis",
                "goals": "7", "xG": "5.0", "time": "2100"}]
    esc = json.dumps(payload).encode("unicode_escape").decode()
    html = f"<html><script>var playersData = JSON.parse('{esc}');</script></html>"
    rows = parse_players_page(html)
    assert len(rows) == 3 and rows[0]["goals"] == 20 and rows[2]["team"] == "Celta Vigo"
    rows2 = parse_players_page("<html><script>var playersData = "
        + json.dumps(payload) + ";</script></html>")
    assert len(rows2) == 3 and rows2[0]["goals"] == 20    # direct-array variant
    try:
        parse_players_page("<html>redesigned page</html>"); assert False
    except ValueError as e:
        assert "TOKEN ABSENT" in str(e)
    try:
        parse_players_page("<html><script>window.playersData = load('/api/x')</script></html>")
        assert False
    except ValueError as e:
        assert "context:" in str(e) and "load('/api/x')" in str(e)   # window shows the truth
    # 1b. getLeagueData (the 2026 layout) is the shape that actually matters now.
    #     This is the path whose absence left props_src reading off(ValueError) in
    #     every league, in production, for weeks.
    api_payload=[{"player_name":"Api Guy","team_title":"Girona","goals":"9","xG":"7.1","time":"2400"}]
    assert players_from_league_data({"dates": [], "teams": {}, "players": api_payload})[0]["name"] == "Api Guy"
    assert players_from_league_data(api_payload)[0]["name"] == "Api Guy"          # bare list
    assert players_from_league_data({"dates": [], "somethingNew": api_payload})[0]["name"] == "Api Guy"
    for bad, why in (({"dates": [], "teams": {}}, "no player list"),
                     ({"players": []},            "empty player list"),
                     ({"players": [{"nope": 1}]}, "rows that do not map")):
        try:
            players_from_league_data(bad); assert False, why
        except ValueError:
            pass
    # the API league selector is NOT the URL slug -- this is the whole bug class
    assert UNDERSTAT_API_LEAGUE["SP1"] == "La liga" and UNDERSTAT["SP1"] == "La_liga"
    assert urllib.parse.quote(UNDERSTAT_API_LEAGUE["F1"]) == "Ligue%201"
    assert set(UNDERSTAT_API_LEAGUE) == set(UNDERSTAT), "every league needs an API selector"
    inv=_page_endpoints('<script src="/static/app.9f2.js"></script> fetch("/main/getPlayersStats/")')
    assert "app.9f2.js" in inv and "getPlayersStats" in inv
    # 1c. renamed inline var still parses; POST routes are still harvested
    esc2 = json.dumps(payload).encode("unicode_escape").decode()
    rowsR = parse_players_page(f"<script>var statsBlob = JSON.parse('{esc2}');</script>")
    assert rowsR[0]["name"] == "Star Striker"                     # rename-proof
    fakepage = 'x $.post("/main/getTeamStats/") y $.post("/main/getLeaguePlayers/") z'
    rts = sorted(set(re.findall(r"/main/[A-Za-z]+/", fakepage)))
    assert rts == ["/main/getLeaguePlayers/", "/main/getTeamStats/"]
    # 2. team mapper on the notorious deltas + fuzzy fallback
    fd = {"Man United", "Wolves", "Nott'm Forest", "Ath Madrid", "Dortmund", "Paris SG", "Girona"}
    assert map_team("Manchester United", fd) == "Man United"
    assert map_team("Wolverhampton Wanderers", fd) == "Wolves"
    assert map_team("Atletico Madrid", fd) == "Ath Madrid"
    assert map_team("Borussia Dortmund", fd) == "Dortmund"
    assert map_team("Girona", fd) == "Girona"            # exact passthrough
    assert map_team("Unknown FC", fd) is None
    # 2b. DETERMINISM. fd_teams is a set at every call site; the old mapper
    #     returned the first substring hit in iteration order, so a club with two
    #     plausible football-data spellings could be resolved differently between
    #     runs of identical code on identical data -- and every one of its players
    #     would move to another team on those days. Same input, same answer, always.
    amb = {"Union Berlin", "FC Union Berlin", "Leeds", "Leeds United"}
    first = map_team("Union Berlin", amb)
    assert first == "Union Berlin"                       # exact beats substring
    assert all(map_team("Union Berlin", set(amb)) == first for _ in range(50))
    lu = map_team("Leeds United FC", amb)
    assert lu == "Leeds United" and all(
        map_team("Leeds United FC", set(amb)) == lu for _ in range(50))
    # 2c. a very short name is not evidence; it must not drag a squad onto a club
    #     that merely contains those letters.
    assert map_team("PSG", {"Paris SG", "Le Havre"}) is None
    assert map_team("", fd) is None
    # 2e. the one-letter spellings the fuzzy rule can never reach
    assert map_team("Espanyol", {"Espanol", "Girona"}) == "Espanol"
    # 2d. the selftest must not sabotage the process it runs in. This block used
    #     to do `urllib.request.urlopen = None` to "ensure tests never hit the
    #     network" -- on the SHARED urllib module object, so every later caller in
    #     the same process, in any module, got NoneType-is-not-callable.
    assert callable(urllib.request.urlopen), \
        "selftest left urllib.request.urlopen patched out"
    # 3. shares + availability + anytime math, hand-checked
    shares = team_shares(rows, {"Man United"})["Man United"]
    star = shares[0]
    exp_share = (0.6*20 + 0.4*17.5) / ((0.6*20 + 0.4*17.5) + (0.6*5 + 0.4*6.2))
    assert abs(star["share"] - exp_share) < 1e-3 and star["avail"] == 1.0
    pr = anytime_probs(1.60, shares)
    exp_p = 1 - math.exp(-1.60 * exp_share)
    assert abs(pr[0]["p"] - exp_p) < 1e-3, (pr[0], exp_p)
    assert pr[0]["p"] > pr[-1]["p"]                      # likelihood-first ordering
    # 4. sanity bounds: share*avail can never exceed team lambda -> p < 1-exp(-lam)
    assert all(x["p"] <= 1 - math.exp(-1.60) + 1e-9 for x in pr)
    print(f"PROPS SELFTEST PASS — parser/diagnostic, name-map deltas, share blend, "
          f"anytime math exact (star {pr[0]['p']:.1%} at lam 1.6)")
    return 0

if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv: sys.exit(selftest())
    print("library module — used by soccer_publish.py")
