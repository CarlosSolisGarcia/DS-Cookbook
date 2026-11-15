# Forecasting Baselines: Naive, Seasonal Naive, Mean, Drift, Moving Average

> **Folder:** 05_time_series · **Prev:** [validation.md](validation.md) · **Next:** [forecasting_models.md](forecasting_models.md) · **Related:** [modeling_fundamentals.md](../00_methodology/modeling_fundamentals.md) §2.4

---

## 1. Purpose
Establish forecasts that any serious model must beat. In forecasting, simple baselines are surprisingly hard to beat, and several sophisticated models in published comparisons fail to beat seasonal naive.

## 2. When to use it
Before any forecasting model, with the same rolling-origin evaluation. The seasonal-naive baseline also defines the scale of MASE ([evaluation.md](evaluation.md)).

## 3. Conceptual foundation
| Baseline | Forecast for t₀+h | Good when | Fails when |
|---|---|---|---|
| **Mean** | mean of the training history | Stationary noise around a constant | Trend, seasonality, shifts |
| **Naive (random walk)** | y_{t₀} | Random-walk-like series (prices, slowly drifting levels) | Strong seasonality |
| **Seasonal naive** | y_{t₀+h−m·k}, the last observed value at the same season position | Strong, stable seasonality | Trend growth over the season, holidays on different dates |
| **Drift** | y_{t₀} + h·(y_{t₀} − y_1)/(T−1) | Linear trend | Breaks, seasonality |
| **Moving average** | mean of the last w values | Noisy, level-only series | Seasonality (unless applied per season position) |
| **Seasonal moving average** | mean of the same weekday over the last k weeks | Seasonal series with noise | Fast changes |

## 4. Assumptions and limitations
- Baselines have no parameters to fit (or only w, k), so they are immune to overfitting. That is part of their strength.
- Seasonal naive repeats last season's anomalies (e.g. last week's holiday). Seasonal moving averages smooth this out.

## 5. Step-by-step procedure
1. Implement the baselines as forecaster functions `f(y_train, h, X_train, X_future) -> array`.
2. Evaluate all of them with identical rolling origins.
3. Keep the best baseline (usually seasonal naive or the seasonal moving average) as **the** reference in every later comparison.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from cookbook_data import load_regular_energy_demand
from cookbook_utils import evaluate_forecaster, summarize_forecasts

df = load_regular_energy_demand()
y_dev = df["demand_mwh"].iloc[:-182]
H, M = 14, 7

def mean_fc(y, h, X=None, Xf=None):
    return np.full(h, y.mean())

def naive_fc(y, h, X=None, Xf=None):
    return np.full(h, y.iloc[-1])

def seasonal_naive_fc(y, h, X=None, Xf=None, m=M):
    last = y.values[-m:]
    return np.array([last[i % m] for i in range(h)])

def drift_fc(y, h, X=None, Xf=None):
    slope = (y.iloc[-1] - y.iloc[0]) / (len(y) - 1)
    return y.iloc[-1] + slope * np.arange(1, h + 1)

def moving_average_fc(y, h, X=None, Xf=None, w=28):
    return np.full(h, y.iloc[-w:].mean())

def seasonal_ma_fc(y, h, X=None, Xf=None, m=M, k=4):
    """Mean of the same season position over the last k seasons (robust version of seasonal naive)."""
    vals = y.values[-m * k:].reshape(k, m)          # rows = past weeks, columns = weekday position
    profile = np.median(vals, axis=0)               # median → robust to a holiday in one of the weeks
    return np.array([profile[i % m] for i in range(h)])

baselines = {"mean": mean_fc, "naive": naive_fc, "seasonal_naive": seasonal_naive_fc,
             "drift": drift_fc, "moving_avg_28": moving_average_fc, "seasonal_median_4w": seasonal_ma_fc}
res = pd.concat([evaluate_forecaster(y_dev, f, initial=730, horizon=H, step=7, name=n) for n, f in baselines.items()])
print(summarize_forecasts(res).sort_values("MAE").round(3))
print(summarize_forecasts(res, by=["model", "h"])["MAE"].unstack(0)[["seasonal_naive", "seasonal_median_4w"]].round(2).T)
```

## 7. Interpretation
- **Mean, naive, drift and moving average** do poorly, because they ignore the weekly pattern (weekends are about 30% lower).
- **Seasonal naive** is much better. The **seasonal median over 4 weeks** is usually better still, because it is not fooled by a holiday or noise in the single previous week.
- **MAE by horizon for seasonal baselines** is roughly flat within the 14 days. Their errors do not compound, unlike recursive models.
- **The reference to beat** is the best seasonal baseline. A model with MASE < 1 beats the in-sample one-step seasonal naive. What matters is beating the *out-of-sample* seasonal baseline at the same horizons.

## 8. Decision framework
- **IF** the series is seasonal → the seasonal naive / seasonal median is the reference.
- **IF** non-seasonal and trending → drift / naive.
- **IF** non-seasonal and noisy around a level → moving average / mean.
- **IF** a sophisticated model does not beat the best baseline beyond the noise across origins → keep the baseline. It is cheap, robust and explainable.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Using only the mean as a baseline | Every model looks great | Seasonal naive at minimum |
| Different origins for baselines and models | Invalid comparison | Same rolling-origin function |
| Dismissing baselines as "too simple" | Over-engineering | Respect the baseline; quantify the improvement |

## 10. Checklist
- [ ] Naive, seasonal naive, drift, mean, (seasonal) moving average evaluated.
- [ ] Same rolling origins as the models.
- [ ] Best baseline chosen as the reference; MAE by horizon recorded.

## 11. References
- Hyndman, R. J. & Athanasopoulos, G. (2021). *FPP3*, §5.2 "Some simple forecasting methods". https://otexts.com/fpp3/simple-methods.html
- Makridakis, S., Spiliotis, E. & Assimakopoulos, V. (2018). "Statistical and Machine Learning forecasting methods: Concerns and ways forward." *PLoS ONE* 13(3): e0194889. https://doi.org/10.1371/journal.pone.0194889
