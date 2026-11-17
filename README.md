# GTA VI launch-day Twitch CCU forecaster

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for make.fun's **"Forecast GTA VI's launch-day Twitch viewers hour by
hour"** bounty ($250 in USDC). The program reads one case file — everything
Twitch recorded for the Grand Theft Auto VI category up to the end of the
first hour of release day, 2026-11-19 (UTC) — and forecasts, for each of
hours 2–24, five quantiles (5%, 25%, 50%, 75%, 95%) of that hour's
concurrent viewers. Ranked by mean pinball loss on the log scale; lower is
better.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network, deterministic: the same
case always yields the same bytes. One file, `main.py`, at the repository
root; it reads every `*.in` in the input directory and writes one `<id>.out`
per case.

## The model: launch-curve regression on rolling-midnight launches

A launch day's Twitch curve is not a random walk around hour 1 — it is a
diurnal rollout. GTA VI unlocks at midnight local time, region by region
(New Zealand first, then Europe, then the US East at 05:00 UTC and the US
West at 08:00 UTC), so viewership dips in the small hours, jumps when the
US wakes up, and peaks in the US evening (hours 19–21 UTC). How steep that
climb is depends on how much of the launch demand hour 1 already captured:
for the two biggest rolling-midnight launches in the data (ELDEN RING,
Cyberpunk 2077, hour-1 prints ~850k) hour 1 *was* the peak, while a small
hour-1 print (tens of thousands) typically grows 2–6× by the evening.

For each hour h = 2..24 the model fits, over past big launches, a weighted
least-squares line in log space:

```
ln(viewers_h / viewers_1) = a_h + b_h * ln(viewers_1) + noise
```

and forecasts the median as `viewers_1 * exp(a_h + b_h * ln(viewers_1))`
with a Gaussian spread from that hour's residual standard deviation:

```
q_tau(h) = viewers_1 * exp(mu_h + z_tau * sd_h)
```

**Training set:** only "rolling midnight local" console launches — the
unlock pattern GTA VI itself uses — identified by name (the known Sony /
Rockstar / Nintendo / Call of Duty midnight-local launches, including the
four practice launches) or by "midnight local" in the comparable's unlock
note. Launches with a single simultaneous global unlock (a sharp step at
one UTC hour) or with early-access-contaminated release days are a
different shape and are left out. If fewer than 8 rolling launches are
usable, the model falls back to all usable comparables with quality
weights; if none are usable, it falls back to the house persistence
baseline.

**Guards for the final case:** `ln(viewers_1)` is capped to the range seen
in training, so a GTA VI hour-1 print above ELDEN RING's 875k does not
extrapolate the line into a collapse (mega-launch curves are flat, not
falling); the residual spread is floored at 0.08 and capped at 1.0 log
units; quantiles are rounded half-up to 2 decimals, clamped to (0, 1e9]
and forced non-decreasing.

## Validation (local backtest)

Practice answers were taken from the same public source the house used
(SullyGnome hourly category averages; the fetched hour-1 values match the
cases' `known` prints exactly).

| model | mean pinball loss, 4 practice cases |
|---|---|
| House persistence baseline | **0.3871** |
| Empirical median-ratio curve | 0.3240 |
| Log(hour-1) regression, all comparables | 0.2272 |
| **This entry (regression, rolling-midnight only)** | **0.1739** |

Per case: Wolverine 0.2360, Red Dead Redemption 2 0.1319, God of War
Ragnarök 0.1259, Spider-Man 2 0.2018. Typical miss ~29% (baseline: ~83%),
90% coverage 0.74–0.96.

Leave-one-out over the 19 rolling-midnight launches (the 15 in the kit's
comparables plus the four practice launches): **0.2040** (median 0.1511).
The worst held-out launch is Resident Evil Requiem (0.9277), whose hour-1
print was small yet whose evening peak was 12× it — the one shape this
family of curves genuinely struggles with.

Things tried and rejected (see `main.py` for the numbers): a pre-release
baseline predictor, hour-1 channel count, Huber robust regression,
wider spreads, smoothing coefficients across hours, training on all
comparables, and an n−2 residual variance.

## Files

- `main.py` — the whole entry (single file, stdlib only)
- `LICENSE` — MIT
- `.makefun/owner.json` — payout wallet for the entry

## Note on the practice score

The practice is graded on past launches anyone can look up and pays
nothing; per the rules, a practice score is no guide to the final. The
final is one case (h00), cut at 2026-11-19 01:00 UTC, on the house
recorder's minute reads — a different measure from SullyGnome's hourly
averages, so expect the final loss to differ from 0.174 in either
direction.

---

<sub>Entered on [make.fun](https://make.fun/?ref=lively-owl-81&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
