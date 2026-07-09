"""
soccer_props.py — anytime-goalscorer props from the team goals model.
P(player scores) = 1 - exp(-lambda_player), where
  lambda_player = lambda_team (Dixon-Coles) x goal_share x availability
  goal_share    = blended (goals 60% / xG 40%) share of team output
  availability  = player minutes / team's most-played minutes

Player data: understat league pages (per-player goals, xG, minutes) — the
standard free source for exactly these four leagues. UNREACHABLE from the dev
sandbox, so the parser is fixture-tested against understat's known embedded-
JSON format and fails LOUD in production: on any parse miss it prints the raw
page head into the Action log, making the first run its own probe.
"""
import re, json, urllib.request

UNDERSTAT = {"E0": "EPL", "SP1": "La_liga", "D1": "Bundesliga", "F1": "Ligue_1"}
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
    for t in fd_teams:
        nt = _norm(t)
        if nu == nt or nu in nt or nt in nu:
            return t
    return None

def parse_players_page(html):
    """understat league page -> [{name, team, goals, xg, minutes}].
    Raises ValueError with a diagnostic head if the format shifted."""
    m = re.search(r"playersData\s*=\s*JSON\.parse\('(.*?)'\)", html, re.S)
    if not m:
        raise ValueError("playersData block not found; page head: " + html[:400].replace("\n", " "))
    raw = m.group(1).encode("utf-8").decode("unicode_escape")
    data = json.loads(raw)
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

def fetch_league_players(div, season_year):
    slug = UNDERSTAT[div]
    url = f"https://understat.com/league/{slug}/{season_year}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (SoccerTool props)"})
    with urllib.request.urlopen(req, timeout=45) as r:
        return parse_players_page(r.read().decode("utf-8", "replace"))

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
    try:
        parse_players_page("<html>redesigned page</html>"); assert False
    except ValueError as e:
        assert "page head" in str(e)                     # loud diagnostic path
    # 2. team mapper on the notorious deltas + fuzzy fallback
    fd = {"Man United", "Wolves", "Nott'm Forest", "Ath Madrid", "Dortmund", "Paris SG", "Girona"}
    assert map_team("Manchester United", fd) == "Man United"
    assert map_team("Wolverhampton Wanderers", fd) == "Wolves"
    assert map_team("Atletico Madrid", fd) == "Ath Madrid"
    assert map_team("Borussia Dortmund", fd) == "Dortmund"
    assert map_team("Girona", fd) == "Girona"            # exact passthrough
    assert map_team("Unknown FC", fd) is None
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
