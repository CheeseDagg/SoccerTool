# SoccerTool — EPL · La Liga · Bundesliga · Ligue 1

Time-decayed Dixon-Coles goals model, one engine for four leagues. Full
scoreline grid prices 1X2 / over-2.5 / BTTS on every upcoming fixture, with
devigged closing-market anchors alongside. Every published probability is
logged pre-match and graded against the final result; each league carries a
walk-forward backtest of the model **versus the closing market on identical
matches** — the disagreement record is published, win or lose.

Player props: anytime-goalscorer per fixture — team goal rate x blended
goal/xG share x availability (understat), labeled as a model read and logged
for grading.

Pipeline: `soccer_model.py` (+ `soccer_props.py`) (fetch + fit + backtest) → `soccer_grade.py`
(log/settle) → `soccer_publish.py` (slate.json) → `index.html`.
Runs daily via `.github/workflows/soccer-daily.yml`. Data: football-data.co.uk.

Engine validated by synthetic recovery: the fitter provably recovers known
attack/defense/home/rho parameters from generated seasons (median attack
correlation 0.96, MAE 0.064 across seeds) before ever touching real data.
