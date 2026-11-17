"""GTA VI launch-day Twitch CCU forecaster (make.fun gta6-twitch-ccu-forecast-01).

Reads one case file (/input/<id>.in, schema "makefun/twitch-ccu-case/v1") and
writes /output/<id>.out: for each of hours 2..24 of the release day, five
quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of that hour's concurrent Twitch
viewers in the game's category.

Model: launch-curve regression on the case's own comparables
------------------------------------------------------------
A launch day's Twitch curve is not a random walk around hour 1 -- it is a
diurnal rollout. The game unlocks at midnight local time, region by region
(New Zealand first, then Europe, then the US East at 05:00 UTC and the US
West at 08:00 UTC), so viewership dips in the small hours, jumps when the US
wakes up, and peaks in the US evening (hours 19-21 UTC). How steep that
climb is depends on how much of the launch demand hour 1 already captured:
for the two biggest rolling-midnight launches in the data (ELDEN RING,
Cyberpunk 2077, hour-1 prints ~850k) hour 1 *was* the peak, while a small
hour-1 print (a few tens of thousands) typically grows 2-6x by the evening.

So, for each hour h = 2..24, the model fits, over past big launches, a
weighted least-squares line in log space

    ln(viewers_h / viewers_1) = a_h + b_h * ln(viewers_1) + noise,

and forecasts the median as viewers_1 * exp(a_h + b_h * ln(viewers_1)) with a
Gaussian spread from that hour's residual standard deviation:

    q_tau(h) = viewers_1 * exp(mu_h + z_tau * sd_h).

Training set: only "rolling midnight local" console launches -- the unlock
pattern GTA VI itself uses -- identified by name (the known Sony / Rockstar /
Nintendo / Call of Duty midnight-local launches, including the four practice
launches) or by "midnight local" in the comparable's unlock note. Launches
with a single simultaneous global unlock (a sharp step at one UTC hour) or
with early-access-contaminated release days are a different shape and are
left out. If fewer than 8 rolling launches are usable, the model falls back
to all usable comparables with quality weights; if none are usable at all,
it falls back to the house persistence baseline.

Guards that keep the forecast sane on the final case:
  * ln(viewers_1) is capped to the range seen in training, so a GTA VI
    hour-1 print above ELDEN RING's 875k does not extrapolate the line into
    a collapse -- the biggest launches' curves are flat, not falling.
  * the residual spread is floored at 0.08 and capped at 1.0 log units.
  * quantiles are rounded half-up to 2 decimals, clamped to (0, 1e9] and
    forced non-decreasing.

Validation (local backtest, see README.md): on the four practice cases the
model scores a mean pinball loss of 0.1739 (per case: Wolverine 0.2360,
Red Dead Redemption 2 0.1319, God of War Ragnarok 0.1259, Spider-Man 2
0.2018; typical miss ~29%), against 0.3871 for the house persistence
baseline. Leave-one-out over the 19 rolling-midnight launches (the 15 in
the kit's comparables plus the four practice launches) scores 0.2040
(median 0.1511); the worst held-out launch is Resident Evil Requiem
(0.9277), whose hour-1 print was small yet whose evening peak was 12x it --
the one shape this family of curves genuinely struggles with.

Things tried and rejected, so a later reader does not re-try them blind:
  * a second predictor, ln(hour-1 / pre-release baseline): practice 0.29.
  * hour-1 channel count as a second predictor: practice 0.1716, a gain
    inside the noise of four cases, for an extra failure mode -- dropped.
  * Huber (robust) regression instead of OLS: practice 0.1837, LOO 0.2033.
  * widening the spread x1.15 / x1.3: practice 0.178 / 0.184.
  * smoothing the per-hour coefficients across neighbouring hours: 0.1759.
  * training on all comparables with quality weights instead of rolling
    launches only: practice 0.2298.
  * residual variance with n-2 instead of n-1 degrees of freedom: 0.1748.

Python 3.13, standard library only, no network, deterministic: the same
case always yields the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports. (This entry is a single file, but the
# line keeps the layout identical to the kit's starter.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Rolling-midnight-local console launches: the unlock pattern GTA VI uses.
# Names are matched case-insensitively against the comparable's "name".
ROLLING_NAMES = {
    "elden ring",
    "resident evil requiem",
    "ghost of yotei",
    "monster hunter wilds",
    "call of duty: modern warfare 4",
    "the legend of zelda: tears of the kingdom",
    "cyberpunk 2077",
    "call of duty: black ops 6",
    "call of duty: modern warfare iii",
    "call of duty: black ops 7",
    "final fantasy vii rebirth",
    "the last of us part ii",
    "ghost of tsushima",
    "stellar blade",
    "saros",
    # The four practice launches (in the final case's comparables).
    "marvel's wolverine",
    "red dead redemption 2",
    "god of war ragnarok",
    "marvel's spider-man 2",
}

# Quality weights used only by the all-comparables fallback.
FALLBACK_WEIGHTS = {
    "black myth: wukong": 0.45,
    "hollow knight: silksong": 0.45,
    "battlefield 6": 0.45,
    "007 first light": 0.45,
    "baldur's gate 3": 0.45,
    "hogwarts legacy": 0.6,
    "starfield": 0.6,
    "diablo iv": 0.6,
    "assassin's creed shadows": 0.6,
    "doom: the dark ages": 0.6,
    "death stranding 2: on the beach": 0.6,
    "forza horizon 5": 0.6,
    "forza horizon 6": 0.6,
    "marvel rivals": 0.3,
    "helldivers 2": 0.3,
    "marathon": 0.3,
    "arc raiders": 0.3,
    "elden ring nightreign": 0.3,
    "clair obscur: expedition 33": 0.3,
    "kingdom come: deliverance ii": 0.3,
    "borderlands 4": 0.3,
    "halo infinite": 0.3,
}

MIN_ROLLING = 8          # below this, fall back to all comparables
MIN_SD = 0.08            # floor on the per-hour residual spread (log units)
MAX_SD = 1.0             # cap on the per-hour residual spread (log units)


def parse_time(text):
    """ISO 8601 UTC timestamp -> aware datetime (handles the trailing Z)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def is_rolling(comp):
    """True when a comparable launched with a rolling midnight-local unlock."""
    name = (comp.get("name") or "").strip().lower()
    if name in ROLLING_NAMES:
        return True
    unlock = (comp.get("unlock") or "").lower()
    return "midnight local" in unlock and "early access" not in unlock


def release_day(comp):
    """The comparable's 24 hourly viewer counts on its release day, or None.

    Requires a complete day: all 24 hours present and positive. This also
    drops the GTA VI category's own pre-release history, which has no
    release day in it.
    """
    try:
        day_start = parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    hours = {}
    for row in comp.get("hourly") or []:
        try:
            hours[parse_time(row["from"])] = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
    day = []
    for h in range(24):
        value = hours.get(day_start + timedelta(hours=h))
        if value is None or value <= 0:
            return None
        day.append(value)
    return day


def wols_1d(xs, ys, ws):
    """Weighted least squares for y = a + b*x. Returns (a, b)."""
    sw = sum(ws)
    if sw <= 0:
        return 0.0, 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx if sxx > 1e-12 else 0.0
    return my - b * mx, b


def round2(x):
    """x rounded to 2 decimals, half up (like the kit's starter)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def baseline_quantiles(p1, h):
    """The house persistence baseline: every hour stays at hour 1's print."""
    return [round2(p1 * math.exp(z * 0.15 * math.sqrt(h - 1))) for z in Z]


def build_training(case):
    """Usable (day, weight, rolling?) rows from the case's comparables."""
    rows = []
    for comp in case.get("comparables") or []:
        day = release_day(comp)
        if day is None:
            continue
        name = (comp.get("name") or "").strip().lower()
        rows.append({
            "day": day,
            "log_p1": math.log(day[0]),
            "rolling": is_rolling(comp),
            "weight": FALLBACK_WEIGHTS.get(name, 0.5),
        })
    return rows


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    log_p1 = math.log(p1)

    rows = build_training(case)
    rolling = [r for r in rows if r["rolling"]]
    if len(rolling) >= MIN_ROLLING:
        train, weights = rolling, [1.0] * len(rolling)
    elif rows:
        train, weights = rows, [r["weight"] for r in rows]
    else:
        train, weights = [], []

    hours_out = []
    if not train:
        for row in case["forecast"]:
            hours_out.append({"hour": row["hour"], "q": baseline_quantiles(p1, row["hour"])})
        return {"hours": hours_out}

    lo = min(r["log_p1"] for r in train)
    hi = max(r["log_p1"] for r in train)
    # Cap the feature at the training range: no linear extrapolation past
    # the biggest launch seen (ELDEN RING, 875k) -- mega-launch curves are
    # flat, and the line would otherwise predict a collapse.
    x_target = min(max(log_p1, lo), hi)

    xs = [r["log_p1"] for r in train]
    for row in case["forecast"]:
        h = row["hour"]
        ys = [math.log(r["day"][h - 1] / r["day"][0]) for r in train]
        a, b = wols_1d(xs, ys, weights)
        mu = a + b * x_target
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        n = len(resid)
        mean_resid = sum(resid) / n
        var = sum((r - mean_resid) ** 2 for r in resid) / max(1, n - 1)
        sd = min(MAX_SD, max(MIN_SD, math.sqrt(var)))
        qs = []
        prev = 0.0
        for z in Z:
            q = p1 * math.exp(mu + z * sd)
            q = min(1_000_000_000, max(1.0, q))
            q = max(q, prev)  # keep the five quantiles non-decreasing
            qs.append(round2(q))
            prev = qs[-1]
        hours_out.append({"hour": h, "q": qs})
    return {"hours": hours_out}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
