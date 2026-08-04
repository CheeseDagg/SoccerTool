# xG / decay / shrinkage sweep — RESULT: NOTHING BEATS THE SHIPPED MODEL

**Date:** 2026-08-04 · **Harness:** `experiments/xg_sweep.py` · **Verdict: no change shipped.**

## What was tested

Leak-free expanding-window walk-forward over the cached understat matches
(`data/understat_xg.json`, 4,115 matches, 2023-24 → 2025-26, all four leagues),
mirroring `soccer_model.backtest_league` exactly: refit every 30 days,
min_train 180 days, `fit_league(iters=250)`, fit strictly on earlier matches,
graded on the real result. 3,222 out-of-sample predictions per config — the
same splits behind the printed record.

- **a. xG weight in the ratings response** — `hg = w·xG + (1−w)·goals`
  (production = w=1.0: pure xG shape + goals level calibration + rho on goals)
- **b. time-decay half-life** — production 240d, swept 90–600d, plus honest
  per-league tuning (chosen on burn-in, scored on eval)
- **c. xG for strength / goals for level** — *this is already the shipped
  model* (the LEVEL CALIBRATION block in `fit_league`, shipped `a23e7a6`).
  Additionally tested: probability-level blends of the goals-fit and
  xG-fit models.
- **d. promoted-club shrinkage** — low-data teams pulled toward (i) the mean
  of the bottom-quintile established teams, several strengths K, and (ii) the
  league average, K = 3/8.
- extra: iters=400 (convergence check), w=0.9, and a probability
  recalibration layer (temperature + draw multiplier) tuned on burn-in only.

**Tuning rule:** any parameter choice was made on the BURN-IN window
(predictions from the 2023-24 season, n=545) and had to also win on the EVAL
window (2024-25 + 2025-26, n=2,677) to be shippable. Full-window numbers are
what is tabled, for comparability with the printed record.

**What could NOT be re-measured locally:** the disagreements-vs-market
record. No closing odds exist in the local cache (football-data.co.uk is
unreachable from this sandbox and no odds CSVs are committed); that number
regenerates on Actions. Since no model change shipped, the printed
market-comparison record (model 24.8% vs market 44.7% on 467 disagreements)
is unchanged and still describes the live model.

## Full table (walk-forward, n=3,222 pooled; Brier3 = primary)

| config | Brier3 | acc% | logloss | burn Brier | eval Brier | ΔBrier vs base (±SE) | Δacc |
|---|---|---|---|---|---|---|---|
| **baseline (w=1.0, hl=240) — SHIPPED MODEL** | **0.5854** | **52.9** | 0.9838 | 0.5756 | 0.5874 | — | — |
| w=0.0 (pure goals) | 0.5950 | 51.6 | 0.9976 | 0.5913 | 0.5957 | +0.0096 (±0.0021) | −1.27pp |
| w=0.25 | 0.5904 | 52.3 | 0.9907 | 0.5840 | 0.5917 | +0.0050 (±0.0015) | −0.56pp |
| w=0.5 | 0.5874 | 52.8 | 0.9864 | 0.5790 | 0.5891 | +0.0020 (±0.0010) | −0.09pp |
| w=0.75 | 0.5858 | 53.1 | 0.9841 | 0.5763 | 0.5877 | +0.0004 (±0.0005) | +0.22pp |
| w=0.9 | 0.5854 | 53.1 | 0.9837 | 0.5757 | 0.5874 | −0.0000 (±0.0002) | +0.19pp |
| hl=90 | 0.5881 | 53.3 | 0.9881 | 0.5764 | 0.5904 | +0.0027 (±0.0010) | +0.37pp |
| hl=120 | 0.5866 | 53.4 | 0.9858 | 0.5759 | 0.5888 | +0.0012 (±0.0007) | +0.56pp |
| hl=180 | 0.5856 | 53.1 | 0.9842 | 0.5757 | 0.5877 | +0.0002 (±0.0003) | +0.25pp |
| hl=330 | 0.5854 | 52.9 | 0.9837 | 0.5757 | 0.5874 | −0.0000 (±0.0002) | +0.00pp |
| hl=420 | 0.5855 | 52.9 | 0.9838 | 0.5757 | 0.5875 | +0.0001 (±0.0003) | +0.03pp |
| hl=600 | 0.5856 | 52.9 | 0.9840 | 0.5758 | 0.5877 | +0.0002 (±0.0005) | +0.00pp |
| shrink→bottom-prior K=5 | 0.5887 | 52.7 | 0.9887 | 0.5826 | 0.5899 | +0.0033 (±0.0008) | −0.22pp |
| shrink→bottom-prior K=15 | 0.5968 | 52.5 | 1.0006 | 0.5966 | 0.5969 | +0.0114 (±0.0017) | −0.37pp |
| shrink→bottom-prior K=30 | 0.6069 | 51.3 | 1.0148 | 0.6113 | 0.6060 | +0.0215 (±0.0024) | −1.58pp |
| shrink→bottom-prior K=15 pf=0.35 | 0.5971 | 52.4 | 1.0009 | 0.5967 | 0.5971 | +0.0117 (±0.0017) | −0.47pp |
| shrink→league-avg K=3 | 0.5873 | 52.9 | 0.9866 | 0.5793 | 0.5889 | +0.0019 (±0.0006) | −0.03pp |
| shrink→league-avg K=8 | 0.5917 | 52.8 | 0.9932 | 0.5863 | 0.5928 | +0.0063 (±0.0012) | −0.06pp |
| prob-blend α=0.25 xG | 0.5906 | 52.3 | 0.9910 | 0.5845 | 0.5919 | +0.0053 (±0.0016) | −0.62pp |
| prob-blend α=0.5 xG | 0.5876 | 52.8 | 0.9867 | 0.5796 | 0.5893 | +0.0022 (±0.0010) | −0.06pp |
| prob-blend α=0.75 xG | 0.5859 | 53.1 | 0.9843 | 0.5766 | 0.5877 | +0.0005 (±0.0005) | +0.19pp |
| iters=400 | 0.5854 | 52.9 | 0.9838 | 0.5756 | 0.5874 | +0.0000 (±0.0000) | +0.03pp |

Per-league baseline (Brier / acc%): D1 0.5828/52.9 · E0 0.5816/53.3 ·
F1 0.6002/51.5 · SP1 0.5793/53.7. (The committed `backtest_cache.json`
numbers differ in the third decimal because that run walked the
football-data join rather than the understat-native list; same model, same
splits, slightly different match keys/roster.)

## Per-league tuning: tried honestly, failed honestly

Per-league argmin of BURN-IN Brier picked hl=90 for D1/F1 and hl=600 for
E0/SP1. Composite scored on the EVAL window: **0.5890 vs baseline 0.5874**
— the per-league tuning is burn-in noise, not signal. Not shipped.

Recalibration layer (temperature 1.10, draw ×0.90, argmin on burn-in):
burn 0.5743 → looked good; eval **0.5886 vs 0.5874** (+0.0012 ±0.0009).
Not shipped. The base probabilities are already calibrated: over all 3,222
predictions, mean pH/pD/pA = .433/.245/.322 vs actual frequencies
.436/.245/.319.

## Conclusions

1. **The current config is at a flat optimum on every axis swept.** Best
   challengers (w=0.9, hl=330) tie to the 4th decimal — differences of
   −0.000004 against a paired SE of 0.0002. Nothing satisfies the ship rule
   (improve Brier AND not worsen accuracy, out-of-sample).
2. **The xG response weight belongs at 1.0.** Brier degrades monotonically as
   goals are mixed in (w=0 is +0.0096, 4.6 SE worse). The goals signal the
   model needs is already extracted by the level calibration and the
   rho-on-goals step; adding goals to the *shape* only adds finishing noise.
3. **Shorter half-lives buy accuracy, not Brier.** hl=120 gains +0.56pp
   accuracy but costs +0.0012 Brier — sharper but worse-calibrated
   probabilities. Under the house metric order (Brier primary) that is a
   worse model wearing a better hit rate.
4. **Promoted-club shrinkage hurts, every version, both priors.** The
   "new teams are bad" prior (bottom-quintile) is far worse than the implicit
   "new teams are average" the fitter already uses, and even gentle shrinkage
   toward league average (K=3, ~3 effective matches) loses +0.0019 (3 SE).
   With a 240d half-life the fitter simply out-learns any prior within a few
   matchdays, and the prior poisons early-window predictions for every
   returning club, not just promoted ones.
5. **Variant (c) of the brief — xG for strength, goals for level — is the
   model that is already shipped** (`fit_league`'s LEVEL CALIBRATION +
   rho-on-goals, commit `a23e7a6`). This sweep confirms it beats every
   neighbour tested.

No model change, no REBUILD trigger, red honesty verdict untouched — it
still describes the live model exactly. Reproduce with
`python3 experiments/xg_sweep.py` (writes per-prediction records to
`experiments/xg_sweep_records.json`, gitignored at 13MB).
