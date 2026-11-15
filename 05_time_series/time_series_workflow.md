# Time-Series Forecasting Workflow: End-to-End Playbook

> **Folder:** 05_time_series · **Start here for forecasting.** · **Next:** [time_series_understanding.md](time_series_understanding.md) · **Related:** [project_lifecycle.md](../00_methodology/project_lifecycle.md), [regression_workflow.md](../03_regression/regression_workflow.md) (compare: what changes when time matters)
>
> *Added to the original architecture: it gives the time-series folder the same entry point as regression and classification.*

---

## 1. Purpose
This is the master procedure for forecasting future values of one or more time series. It shows the order in which to use the time-series recipes, the questions to answer at each stage, and the points where forecasting differs from ordinary regression.

## 2. When to use it
- You need values of a quantity at **future time points** (demand next 14 days, production next month, temperature next hour).
- For other temporal problems (temporal regression, time-series classification, anomaly detection), first read [time_series_understanding.md](time_series_understanding.md) §3.4 to confirm the problem type.

## 3. Conceptual foundation
**What changes compared with tabular regression:**

| Aspect | Tabular regression | Forecasting |
|---|---|---|
| Rows | Exchangeable (roughly i.i.d.) | Ordered and dependent (autocorrelation) |
| Validation | (Stratified/grouped) K-fold | **Chronological**: rolling origin ([validation.md](validation.md)) |
| Features | Measured attributes | Lags, rolling statistics, calendar, exogenous **known at the forecast origin** |
| Baseline | Mean/median | Naive, **seasonal naive** ([baseline_models.md](baseline_models.md)) |
| Target | One value per row | Values at horizons h = 1…H, with errors that usually grow with h |
| Metrics | MAE/RMSE/R² | MAE/RMSE per horizon, **MASE**, sMAPE ([evaluation.md](evaluation.md)) |
| Main leakage risk | Target proxies, preprocessing | **Look-ahead**: future values in features, interpolation or scaling |

## 4. Assumptions and limitations
- Forecasting assumes the future resembles the past in some learnable way (patterns persist). Structural breaks, regime changes and interventions limit any model. Detect and handle them; do not pretend they are noise.
- The cookbook covers univariate forecasting and forecasting with exogenous variables for one or a few series. Large collections of related series ("global models", hierarchical reconciliation) and deep-learning forecasters are pointed to as extensions.

## 5. Step-by-step procedure

| Step | Action | Recipe | Key question |
|---|---|---|---|
| 1 | Define the forecasting task: target, frequency, **horizon H**, **origin schedule**, latency, decision supported, which exogenous variables are known in advance | [time_series_understanding.md](time_series_understanding.md) | What exactly is forecast, when, how far ahead? |
| 2 | Build a regular time index; handle missing timestamps, duplicates, timezones, outliers | [preprocessing.md](preprocessing.md) | Is the index complete and correct? |
| 3 | Reserve the final test period (≥ 1–2 × H, ideally covering a full seasonal cycle) | [validation.md](validation.md) | Which period is untouchable? |
| 4 | EDA: plots, decomposition, seasonality, ACF/PACF, stationarity, breaks | [exploratory_analysis.md](exploratory_analysis.md) | Trend? Seasonality (which periods)? Breaks? Drivers? |
| 5 | Baselines: naive, seasonal naive, (drift), with rolling-origin evaluation | [baseline_models.md](baseline_models.md) | What is the score to beat? |
| 6 | Statistical models: ETS, (S)ARIMA(X) | [forecasting_models.md](forecasting_models.md) | Does modelling level/trend/seasonality help? |
| 7 | ML models: features (lags, rolling, calendar, exogenous) + linear / GBM | [feature_engineering.md](feature_engineering.md), [forecasting_models.md](forecasting_models.md) | Do exogenous drivers and non-linearities help? |
| 8 | Multi-step strategy: recursive vs direct | [multi_step_forecasting.md](multi_step_forecasting.md) | How do we produce H steps? |
| 9 | Evaluate per horizon, against the baselines; residual diagnostics; intervals | [evaluation.md](evaluation.md) | Is it better, at which horizons, and how uncertain is it? |
| 10 | Iterate; final test once; refit on all data for deployment | [iterative_improvement.md](../06_model_improvement/iterative_improvement.md) | What is the next best improvement? |

## 6. Python implementation: compact end-to-end skeleton

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from cookbook_data import load_regular_energy_demand
from cookbook_utils import evaluate_forecaster, summarize_forecasts

df = load_regular_energy_demand()            # regular daily index (see preprocessing.md)
y = df["demand_mwh"]
H = 14                                       # forecast the next 14 days, every 7 days
test_start = len(y) - 182                    # last ~6 months reserved as the final test period
y_dev = y.iloc[:test_start]                  # development data for model selection

def seasonal_naive(y_tr, h, X_tr=None, X_fu=None, m=7):
    last = y_tr.values[-m:]
    return np.array([last[i % m] for i in range(h)])

def holt_winters(y_tr, h, X_tr=None, X_fu=None):
    fit = ExponentialSmoothing(y_tr.values, trend="add", damped_trend=True,
                               seasonal="add", seasonal_periods=7).fit()
    return fit.forecast(h)

results = pd.concat([
    evaluate_forecaster(y_dev, seasonal_naive, initial=730, horizon=H, step=7, name="seasonal_naive"),
    evaluate_forecaster(y_dev, holt_winters, initial=730, horizon=H, step=7, name="holt_winters"),
])
print(summarize_forecasts(results).round(3))
print(summarize_forecasts(results, by=["model", "h"])["MAE"].unstack(0).round(2).iloc[[0, 6, 13]])
```

## 7. Interpretation
- **Seasonal naive** is the yardstick. Its MASE is near 1 by construction for one-step forecasts, and higher for longer horizons.
- **Holt-Winters beating seasonal naive** at all horizons means modelling the level and trend helps. If it wins only at short horizons, the trend extrapolation may hurt further out (damping helps).
- **The MAE-by-horizon table** shows how quickly accuracy decays. It is often more decision-relevant than the overall average.

## 8. Decision framework
- **IF** the series is short (< 2 seasonal cycles) → simple methods (naive, ETS with few components); avoid ML with many lags.
- **IF** strong, stable seasonality and no useful exogenous data → ETS / SARIMA are hard to beat.
- **IF** there are drivers known in advance (calendar, holidays, planned production) → regression or GBM with these features; compare with SARIMAX.
- **IF** there are many related series → consider one global ML model across series (future extension; see [16_industrial_sensor_data](../16_industrial_sensor_data/) for multi-sensor practice).
- **IF** forecasts drive inventory/capacity buffers → add prediction intervals ([evaluation.md](evaluation.md)).
- **IF** nothing beats seasonal naive → check for breaks or data problems, or accept the naive forecast (it is a legitimate result).

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Random train/test split | Leakage via autocorrelation; optimistic | Chronological rolling origin |
| No seasonal-naive baseline | Cannot tell if the model adds value | Always include it |
| Using future values of exogenous variables that won't be known | Optimistic, undeployable | Use only inputs known at the origin, or forecasted inputs |
| One overall metric only | Hides horizon decay | Metrics by horizon |
| Evaluating at a single origin | Results depend on one period | Many rolling origins |

## 10. Checklist
- [ ] Target, frequency, horizon H, origin schedule and latency defined.
- [ ] Regular index; gaps and outliers handled with past-only logic.
- [ ] Final test period reserved.
- [ ] EDA: decomposition, ACF/PACF, breaks.
- [ ] Naive and seasonal-naive baselines under rolling origin.
- [ ] Statistical and ML candidates compared on the same origins.
- [ ] Metrics by horizon; MASE vs baseline; residual checks.
- [ ] Intervals if needed; final test once; refit plan.

## 11. References
- Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd ed. OTexts. https://otexts.com/fpp3/ (Ch. 1 "Getting started", Ch. 5 "The forecaster's toolbox")
- Petropoulos, F. et al. (2022). "Forecasting: theory and practice." *IJF* 38(3), 705–871. https://doi.org/10.1016/j.ijforecast.2021.11.001
- Makridakis, S., Spiliotis, E. & Assimakopoulos, V. (2020). "The M4 Competition: 100,000 time series and 61 forecasting methods." *IJF* 36(1), 54–74. https://doi.org/10.1016/j.ijforecast.2019.04.014
- statsmodels, *Time Series Analysis (tsa)*. https://www.statsmodels.org/stable/tsa.html
