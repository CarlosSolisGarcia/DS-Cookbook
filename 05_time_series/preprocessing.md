# Time-Series Preprocessing: Index, Missing Timestamps, Outliers, Transformations, Differencing

> **Folder:** 05_time_series · **Prev:** [time_series_understanding.md](time_series_understanding.md) · **Next:** [exploratory_analysis.md](exploratory_analysis.md) · **Related:** [missing_values.md](../01_data_understanding/missing_values.md), [outliers.md](../01_data_understanding/outliers.md), [transformations.md](../02_preprocessing/transformations.md)
>
> *Order note:* in practice you alternate between this recipe and EDA. You need a regular index before most plots and statistics make sense, and EDA tells you which transformations are needed. This recipe comes first because a correct index is a prerequisite.

---

## 1. Purpose
Produce a **regular, correctly indexed, clean** series without introducing look-ahead information. Then apply the transformations (log/Box-Cox, differencing) that models need, with a reliable way to invert them.

## 2. When to use it
Always before time-series modelling. It is especially important with sensor/meter data (gaps, duplicates, timezone/DST issues, glitches) and with irregular event data.

## 3. Conceptual foundation
- **Missing timestamps ≠ missing values.** A missing *row* breaks lag alignment (lag 1 becomes "the previous available row"). First reindex to a complete regular grid, which turns absent rows into NaN. Then decide how to fill them.
- **Irregular sampling.** Event-driven data must be **resampled/aggregated** to a regular grid (mean, sum, last, count, depending on the meaning), or modelled with methods for irregular data (not covered here).
- **Aggregation semantics.** *Flows* (energy, production) are summed. *States* (temperature, pressure) are averaged, or taken as the last value. Counts are summed. Choose by physics, not convenience.
- **Gap filling and leakage.** Linear interpolation uses the value *after* the gap. That is acceptable for cleaning the **training history** used to fit a model, because at any forecast origin in the past those gaps were in the past. It is **not** acceptable for creating features at the forecast origin: at t₀ you don't know t₀+1. For real-time features, use forward-fill (last observation carried forward) or model-based imputation that uses past data only.
- **Variance-stabilising transforms.** log/Box-Cox make multiplicative seasonality additive and stabilise variance ([transformations.md](../02_preprocessing/transformations.md)). Forecasts must be back-transformed; intervals transform correctly, but means need a bias adjustment.
- **Differencing.** Δy_t = y_t − y_{t−1} removes a stochastic trend. Seasonal differencing y_t − y_{t−m} removes stable seasonality. It is used for ARIMA-type models and sometimes as the target of ML models, so they can handle trends.

## 4. Assumptions and limitations
- Interpolating long gaps fabricates data. Mark the filled values (a `was_missing` flag) and consider excluding long gaps from evaluation.
- Outlier replacement in time series should use local (rolling) robust statistics, because global thresholds fail when there is trend and seasonality.
- Timezones: store timestamps in UTC. Local time creates duplicated or missing hours at DST changes (hourly data).

## 5. Step-by-step procedure
1. **Parse, sort and deduplicate** timestamps. Decide how to combine duplicates (mean? last?).
2. **Timezone:** convert to UTC for storage; derive local-time calendar features later if behaviour follows local time.
3. **Regularise:** for regular data, reindex with `asfreq(freq)`. For irregular data, `resample(freq).agg(...)` with the correct semantics.
4. **Quantify the missingness:** number of missing steps, longest gap, gaps by period.
5. **Detect outliers** with rolling robust statistics. Confirm them with the context (event logs) and replace confirmed glitches with NaN.
6. **Fill gaps** (for the training history): time interpolation for short gaps; seasonal interpolation for gaps longer than a season; flag the filled values.
7. **Transformations,** if EDA indicates multiplicative behaviour: log or Box-Cox (fitted on training data).
8. **Differencing,** only for models that need it (ARIMA), or as a deliberate ML target choice. Keep the information needed to invert it.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cookbook_data import make_energy_demand

raw = make_energy_demand()

# 1. Sort, deduplicate (keep the mean of duplicated timestamps if any)
s = raw.sort_values("timestamp").groupby("timestamp")["demand_mwh"].mean()

# 3. Regularise to a complete daily grid → absent rows become NaN
s_reg = s.asfreq("D")
print(f"rows before: {len(s)}, after reindexing: {len(s_reg)}, missing steps: {s_reg.isna().sum()}")

# 4. Gap structure: lengths of consecutive NaN runs
is_na = s_reg.isna()
run_id = (is_na != is_na.shift()).cumsum()
gap_lengths = is_na.groupby(run_id).sum()[lambda g: g > 0]
print("gap length distribution:", gap_lengths.value_counts().sort_index().to_dict())
```

```python
# Irregular data example: event-driven readings → regular hourly grid with correct semantics
rng = np.random.default_rng(0)
events = pd.DataFrame({"time": pd.Timestamp("2024-01-01") + pd.to_timedelta(np.sort(rng.uniform(0, 72, 300)), unit="h"),
                       "energy_kwh": rng.gamma(2, 1.5, 300),          # a FLOW (sum it)
                       "temperature_c": 20 + rng.normal(0, 1, 300)})   # a STATE (average or last)
hourly = events.set_index("time").resample("h").agg({"energy_kwh": "sum", "temperature_c": "mean"})
hourly["n_events"] = events.set_index("time").resample("h").size()       # keep the count: 0 events ≠ 0 energy measured
print(hourly.head())
```

```python
# 5. Outliers with ROLLING, SEASON-AWARE robust statistics (global thresholds fail under trend/seasonality)
from cookbook_data import HOLIDAYS_MMDD

def seasonal_robust_outliers(s, period_key, n_cycles=5, k=6.0, known_events=None):
    """Compare each value with the rolling median of the SAME position in the cycle
    (e.g. same weekday over the surrounding 5 weeks); flag |residual| > k robust SDs.
    Known events (holidays, planned shutdowns) are excluded: they are real, predictable effects."""
    ref = s.groupby(period_key).transform(lambda g: g.rolling(n_cycles, center=True, min_periods=2).median())
    resid = s - ref
    mad = resid.abs().rolling(61, center=True, min_periods=10).median()
    flags = resid.abs() > k * 1.4826 * mad
    if known_events is not None:
        flags &= ~known_events
    return flags, resid

holiday = pd.Series(s_reg.index.strftime("%m-%d").isin(HOLIDAYS_MMDD), index=s_reg.index)
flags_naive, _ = seasonal_robust_outliers(s_reg, s_reg.index.dayofweek)
flags, resid = seasonal_robust_outliers(s_reg, s_reg.index.dayofweek, known_events=holiday)
print(f"flagged without domain knowledge: {flags_naive.sum()}  (of which holidays: {(flags_naive & holiday).sum()})")
print("flagged after excluding known holidays:", s_reg[flags].round(1).to_dict())

# Before removing anything, check the context: are the remaining flags explained by a driver?
temp = raw.set_index("timestamp")["temperature_c"].asfreq("D")
print(pd.DataFrame({"demand": s_reg[flags], "temperature": temp[flags]}).round(1))
# NOTE: centred windows are fine for OFFLINE cleaning of history; for online detection use trailing windows.
glitch = flags & (resid > 0) & (s_reg > 3 * s_reg.median())       # confirmed meter glitch (domain check)
s_clean = s_reg.mask(glitch)
```

```python
# 6. Gap filling for the training history, with a flag
def fill_history(s, max_interp=3, season=7):
    """Short gaps: time interpolation. Longer gaps: value from one season earlier (+ interpolation fallback)."""
    was_missing = s.isna()
    out = s.copy()
    run_id = (was_missing != was_missing.shift()).cumsum()
    run_len = was_missing.groupby(run_id).transform("sum")
    short = was_missing & (run_len <= max_interp)
    out[short] = s.interpolate(method="time")[short]
    long_ = out.isna()
    out[long_] = out.shift(season)[long_]                           # seasonal fill (same weekday last week)
    out = out.interpolate(method="time", limit_direction="both")    # fallback
    return out, was_missing.astype(int)

y, y_was_missing = fill_history(s_clean)
fig, ax = plt.subplots(figsize=(12, 3))
y["2021-02-25":"2021-03-25"].plot(ax=ax, marker=".", label="filled")
s_reg["2021-02-25":"2021-03-25"].plot(ax=ax, marker="o", ls="", label="observed"); ax.legend(); plt.show()

# Real-time feature construction must NOT use future values. At a forecast origin use ffill:
y_realtime = s_clean.ffill()
```

```python
# 7–8. Transformations and differencing, with inversion
from scipy.stats import boxcox
from scipy.special import inv_boxcox

y_log = np.log(y)
y_bc, lam = boxcox(y.values)                                 # fit λ on TRAINING data only in real use
print(f"Box-Cox λ = {lam:.2f} (λ≈1: no transform needed; λ≈0: log)")

d1 = y.diff(1)                     # removes trend / level changes
d7 = y.diff(7)                     # removes weekly seasonality
d7_1 = y.diff(7).diff(1)           # both
# Inverting a seasonal difference forecast: y_hat[t] = d7_hat[t] + y[t-7] (recursively for multi-step)
def invert_seasonal_diff(d_hat, history, m=7):
    hist = list(history[-m:])
    out = []
    for v in d_hat:
        nxt = v + hist[-m]
        out.append(nxt); hist.append(nxt)
    return np.array(out)

check = invert_seasonal_diff(d7.iloc[-14:].values, y.iloc[:-14].values)
print("inversion error:", np.abs(check - y.iloc[-14:].values).max().round(10))
```

## 7. Interpretation
- **Missing steps after reindexing** (here 21) are invisible before reindexing. Lags computed on the raw rows would have been wrong around every gap.
- **Gap lengths:** mostly 1-day gaps plus one 6-day outage. Short gaps → interpolation. The 6-day outage → seasonal fill. The weekly pattern is then preserved, which plain linear interpolation would flatten.
- **Outlier flags without domain knowledge** pick up about 30 days, and almost all are **holidays**. Statistically they are outliers (≈ 35% lower demand). Physically they are a known, predictable effect, so they must be modelled with a holiday feature and not removed. A statistical rule cannot know this: domain knowledge supplies the list of known events.
- **After excluding holidays,** three days remain. The 422 MWh spike is ~3.5× normal: a meter glitch, so remove it. Two consecutive high days in July 2021 coincide with **high temperature** (cooling demand). They are real behaviour explained by a driver. Keep them; the temperature feature will explain them. Rule: *flag statistically, decide with context.*
- **Box-Cox λ near 1:** the variance is roughly stable, so no transform is required. A λ near 0 would suggest a log transform (multiplicative behaviour).
- **Differenced series:** d1 removes the level shift and trend. d7 removes the weekly pattern. Check with EDA (ACF) which differencing makes the series look stationary ([exploratory_analysis.md](exploratory_analysis.md)).

## 8. Decision framework
| Situation | Action |
|---|---|
| Absent rows in regular data | `asfreq(freq)` → NaN, then fill/flag |
| Irregular timestamps | `resample` with flow = sum, state = mean/last, plus a count column |
| Gap ≤ a few steps | Time interpolation (history), ffill (real-time features) |
| Gap > one seasonal period | Seasonal fill or model-based fill; flag it; consider excluding it from evaluation |
| Very long gap / outage | Treat as a break: start training after it, or add a regime flag |
| Isolated spike confirmed as a glitch | NaN + fill; log the decision |
| Real but unusual event (holiday, shutdown) | Keep it; model it with event features |
| Seasonal amplitude grows with the level | log / Box-Cox |
| ARIMA-type model, trend/seasonality present | Differencing (d, D) chosen via EDA/tests |
| ML model with a trending target | Difference or de-trend the target, or add trend features; trees cannot extrapolate levels |

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Computing lags on rows with gaps | Misaligned features | Reindex first |
| Interpolating with future values in real-time features | Look-ahead leakage | ffill or past-only methods at the origin |
| Removing holidays as outliers | Model blind to predictable effects | Event features instead |
| Global z-score outlier detection on a trending series | Flags normal high/low seasons | Rolling robust statistics |
| Summing a state variable / averaging a flow | Physically wrong aggregates | Aggregation semantics by variable |
| Forgetting to invert transformations or differences | Forecasts on the wrong scale | Keep the inversion functions and test them |
| Local time with DST in hourly data | Duplicated/missing hours | Store UTC |

## 10. Checklist
- [ ] Timestamps sorted, deduplicated, timezone handled.
- [ ] Regular grid (reindex/resample) with the correct aggregation semantics.
- [ ] Gap structure quantified; fills chosen by gap length; filled values flagged.
- [ ] Outliers detected with rolling robust statistics and confirmed; real events kept.
- [ ] Transformations justified by EDA; λ fitted on training data.
- [ ] Differencing only when needed; inversion tested.
- [ ] Real-time features use only past data.

## 11. References
- pandas User Guide, *Time series / date functionality* (resample, asfreq, time zones). https://pandas.pydata.org/docs/user_guide/timeseries.html
- Hyndman, R. J. & Athanasopoulos, G. (2021). *FPP3*, §3.1 "Transformations and adjustments", §9.1 "Stationarity and differencing". https://otexts.com/fpp3/
- Moritz, S. & Bartz-Beielstein, T. (2017). "imputeTS: Time Series Missing Value Imputation in R." *The R Journal* 9(1). A good overview of TS imputation methods (language-independent). https://doi.org/10.32614/RJ-2017-009
- statsmodels, `seasonal_decompose` / `STL` (also useful for seasonal imputation). https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html
