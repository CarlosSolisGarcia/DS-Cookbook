# Understanding Temporal Data and Defining the Forecasting Task

> **Folder:** 05_time_series · **Prev:** [time_series_workflow.md](time_series_workflow.md) · **Next:** [exploratory_analysis.md](exploratory_analysis.md) · **Related:** [problem_framing.md](../00_methodology/problem_framing.md), [16_industrial_sensor_data](../16_industrial_sensor_data/)

---

## 1. Purpose
Learn the vocabulary and structure of time series: frequency, components, horizon and origin. Classify the problem correctly (forecasting, temporal regression, time-series classification, anomaly detection), and write a precise forecasting task definition before touching models.

## 2. When to use it
At the start of any project whose data have timestamps. It matters most when you are unsure whether time order actually matters.

## 3. Conceptual foundation

### 3.1 Vocabulary
| Term | Meaning | Example |
|---|---|---|
| **Frequency / sampling interval** | Time between consecutive observations | Daily; hourly; 1 s sensor data |
| **Regular vs. irregular** | Fixed interval vs. event-driven timestamps | Meter readings every day vs. lab samples when taken |
| **Forecast origin** (t₀) | The last time point whose data are available when forecasting | Sunday 23:59 |
| **Horizon** (h = 1…H) | How many steps ahead | h = 1…14 days |
| **Lead time / latency** | Delay between an event and its data being available | Meter data arrive with a 1-day delay → effectively forecast from t₀ − 1 |
| **Aggregation level** | The resolution at which forecasts are needed | Hourly data, but daily planning |
| **Exogenous variable** | An external driver | Temperature, holidays, production plan |
| **Known in advance** vs. **must be forecast** | Whether the future value of the exogenous variable is available at t₀ | Holidays (known); temperature (only a weather forecast) |

### 3.2 Components
- **Trend:** a long-term increase or decrease (e.g. the site growing). It can be linear, damped or change abruptly (a **level shift**, like the new production line in June 2022).
- **Seasonality:** a pattern with a **fixed, known period** (weekly 7 days, yearly 365.25 days, daily 24 h). Several seasonalities can coexist.
- **Cycles:** rises and falls *without* a fixed period (business cycles). They are often confused with seasonality.
- **Noise / remainder:** what is left. Ideally it is unpredictable from the series' own past. In practice it is often **autocorrelated** (shocks persist).
- **Additive vs. multiplicative:** whether the seasonal swings stay constant (additive) or grow with the level (multiplicative). A log transform turns multiplicative into additive.

### 3.3 Stationarity, in one paragraph
A series is (weakly) stationary if its mean, variance and autocorrelation structure do not change over time. Many classical models (ARMA) assume stationarity, after removing trend and seasonality by differencing. ML models with lag features do **not** formally require it. They do need the relationships they learn to persist, and they **cannot extrapolate a trend** beyond the training range unless you give them de-trended targets or trend features. Details: [exploratory_analysis.md](exploratory_analysis.md).

### 3.4 Which temporal problem do you have?

| Problem type | Question | Target | Typical approach | Recipe |
|---|---|---|---|---|
| **Forecasting** | What will y be at t₀+1…t₀+H? | Future values of the series itself | Baselines, ETS/ARIMA, ML with lags | This folder |
| **Temporal regression / classification** | What is the property of this row (batch, machine-month), whose features happen to be time-stamped? | A per-row target, not the series' future | Tabular workflow + **chronological validation** + past-window features | [03](../03_regression/regression_workflow.md) / [04](../04_classification/classification_workflow.md) |
| **Time-series classification** | Which class does this whole sequence/window belong to? | A label per sequence (e.g. fault type from a vibration signal) | Window features (statistics, spectra) + classifier; deep learning for raw signals | [16_industrial_sensor_data](../16_industrial_sensor_data/) |
| **Anomaly / change detection** | Is the current behaviour unusual? Did the regime change? | Usually no labels | Residuals from a forecast model, statistical control, isolation methods | [10_anomaly_detection](../10_anomaly_detection/) |
| **Nowcasting / soft sensors** | What is the current value of a slow or expensive measurement? | Current value from fast signals | Regression with lagged inputs | [16_industrial_sensor_data](../16_industrial_sensor_data/) |

**Decision cues:**
- If future values of the *same* quantity are needed → forecasting.
- If each row has its own outcome (yield of a batch, failure of a machine in 30 days) → temporal regression/classification, even if you use lagged inputs.
- If labels are attached to windows → time-series classification.
- If there are no labels and the interest is in "unusual" → anomaly detection.

## 4. Assumptions and limitations
- The forecast can only be as good as the information available at the origin. Unforeseeable events (outages, policy changes) are not forecastable from history.
- Forecasting at the wrong aggregation level wastes effort. Forecasting hourly when decisions are daily adds noise, but daily forecasts cannot be disaggregated reliably either. Match the decision.

## 5. Step-by-step procedure
1. **Identify the problem type** with the table in §3.4.
2. **Define the forecasting task** (template in §6): target, unit, frequency, aggregation, horizon H, origin schedule, latency, decision, loss/metric.
3. **List the exogenous variables** and classify each as *known in advance*, *forecastable* (with its forecast error), or *only observed after the fact* (unusable for forecasting beyond what it contributes through lags).
4. **Check the history length** relative to the seasonality. You need at least two full cycles of the longest seasonality to estimate it (two years for yearly seasonality).
5. **Agree on the evaluation:** origins, horizons and the metric weighting across horizons.

## 6. Python implementation: task definition and a first structural look

```text
FORECASTING TASK — Site electricity demand (v1)
Target            : demand_mwh, daily total (site meter)
Frequency         : daily; data available by 06:00 next day (latency 1 day)
Origin schedule   : every Monday 08:00, using data up to Sunday
Horizon           : h = 1..14 days (2-week energy purchasing)
Decision          : volume of energy bought on the day-ahead / week-ahead market
Loss              : under-forecast → buy at the spot price (expensive); over-forecast → resell at a discount
Metric            : MAE per horizon (primary), MASE vs seasonal naive, bias
Exogenous (known) : calendar, holidays, planned shutdowns
Exogenous (fcst)  : temperature (weather forecast; error grows with h)
History           : 2020-01-01 .. 2023-12-31 (4 yearly cycles)
Known events      : new production line 2022-06-01 (level shift); meter outage 2021-03-10..15
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cookbook_data import make_energy_demand

raw = make_energy_demand()                         # raw data: missing rows, NaN, a spike
s = raw.set_index("timestamp")["demand_mwh"]

# 1. Frequency: infer from the timestamp differences (do not assume)
deltas = s.index.to_series().diff().value_counts().head(5)
print("most common sampling intervals:\n", deltas)
print("pandas inferred freq:", pd.infer_freq(s.index))   # None → irregular (gaps present)

# 2. History length vs the longest seasonality
span_days = (s.index.max() - s.index.min()).days
print(f"history: {span_days} days ≈ {span_days/365.25:.1f} yearly cycles, {span_days/7:.0f} weekly cycles")

# 3. A first look at several time scales (the single most informative step)
fig, ax = plt.subplots(3, 1, figsize=(13, 8))
s.plot(ax=ax[0], lw=0.6, title="full history (trend, level shift, yearly pattern)")
s["2022-05-01":"2022-07-31"].plot(ax=ax[1], marker=".", lw=0.8, title="3 months (weekly pattern, level shift on 2022-06-01)")
s["2021-02-20":"2021-03-31"].plot(ax=ax[2], marker="o", lw=0.8, title="6 weeks around the meter outage (gap)")
plt.tight_layout(); plt.show()
```

## 7. Interpretation
- **Most common interval = 1 day, with some 2-day and a 7-day gap:** a regular daily series with missing timestamps. Fix this in [preprocessing.md](preprocessing.md) before computing lags (otherwise "lag 1" sometimes means 2 days).
- **`infer_freq` returns None:** irregular because of the gaps. Not a problem once it has been reindexed.
- **Four yearly cycles:** enough to estimate yearly seasonality, but each year is only one sample of "January". Expect yearly effects to be estimated less precisely than weekly ones.
- **Full-history plot:** an upward trend, a step up in mid-2022 and a yearly U-shape (heating in winter, cooling in summer). Each needs to be represented (trend term, level-shift indicator or a shorter training window, temperature/yearly features).
- **Zoomed plot:** a clear weekly pattern (lower weekends). Seasonal naive with m = 7 will be a strong baseline.

## 8. Decision framework
- **IF** the target is per-row and only the features are temporal → do *not* use forecasting models. Use the tabular workflow with chronological validation.
- **IF** an exogenous variable is known only after the fact → do not use its future values. Its lags can still be used.
- **IF** an exogenous variable must itself be forecast (weather) → evaluate with *forecasted* values (or simulated forecast errors), not actuals. Using actuals gives an optimistic "oracle" estimate. Report both if useful.
- **IF** history covers fewer than two cycles of a seasonality → do not try to estimate that seasonality from data. Use domain knowledge or external drivers (temperature for yearly patterns).
- **IF** the decision is at a coarser level than the data → aggregate first and forecast at the decision level. Compare with summing finer forecasts if both are needed.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Treating temporal regression as forecasting (or vice versa) | Wrong models and validation | Problem-type table §3.4 |
| Assuming the data are regular | Lags misaligned | Check the timestamp deltas |
| Ignoring latency | Using data not yet available at the origin | Encode the latency as a gap in validation |
| Using actual future weather | Optimistic, undeployable accuracy | Forecasted or perturbed exogenous inputs |
| Confusing cycles with seasonality | Spurious seasonal models | Seasonality = fixed known period |

## 10. Checklist
- [ ] Problem type identified (forecasting vs temporal regression vs classification vs anomaly).
- [ ] Task definition written (target, frequency, H, origins, latency, decision, metric).
- [ ] Exogenous variables classified (known / forecast / after-the-fact).
- [ ] Frequency and regularity checked from the data.
- [ ] History length ≥ 2 cycles of each modelled seasonality.
- [ ] Known events (breaks, outages, interventions) listed.

## 11. References
- Hyndman, R. J. & Athanasopoulos, G. (2021). *FPP3*, Ch. 1 (what can be forecast), Ch. 2 "Time series graphics", Ch. 3 "Time series decomposition". https://otexts.com/fpp3/
- Shumway, R. H. & Stoffer, D. S. (2017). *Time Series Analysis and Its Applications*, 4th ed. Springer. Ch. 1–2 (stationarity, ACF). Free PDF from the authors: https://www.stat.pitt.edu/stoffer/tsa4/
- pandas User Guide, *Time series / date functionality*. https://pandas.pydata.org/docs/user_guide/timeseries.html
- Ruiz, A. P., Flynn, M., Large, J., Middlehurst, M. & Bagnall, A. (2021). "The great multivariate time series classification bake off." *Data Mining and Knowledge Discovery* 35, 401–449. On time-series classification. https://doi.org/10.1007/s10618-020-00727-3
