# Temporal Feature Engineering: Lags, Rolling Windows, Calendar and Exogenous Variables

> **Folder:** 05_time_series · **Prev:** [exploratory_analysis.md](exploratory_analysis.md) · **Next:** [validation.md](validation.md) · **Related:** [feature_engineering.md (tabular)](../02_preprocessing/feature_engineering.md), [multi_step_forecasting.md](multi_step_forecasting.md), [data_leakage.md](../00_methodology/data_leakage.md)

---

## 1. Purpose
Turn a time series and its drivers into a feature matrix, so that regression models (linear, GBM) can forecast. This must be done **without** using information that is unavailable at the forecast origin.

## 2. When to use it
- Forecasting with ML/regression models instead of (or as well as) ETS/ARIMA.
- Temporal regression/classification, where rows need context from the past (e.g. vibration trend over the last 7 days for a machine-month).

## 3. Conceptual foundation
**The core rule.** A feature for predicting y at target time t = o + h may only use (a) data observed up to the origin o, and (b) information about t that is **known in advance** (calendar, planned events) or **forecast** (weather forecasts).

| Feature family | Examples | Leakage trap |
|---|---|---|
| **Lags** | y_{o}, y_{o−1}, …; y at the same weekday last week | For horizon h, "lag 1 of the target" (y_{t−1}) is **unknown** when h > 1 |
| **Rolling statistics** | Mean/std/min/max over the last 7/28 days ending at o | `rolling()` without `shift` includes the current value; centred windows include the future |
| **Calendar** | Day of week, month, day of year, week of year, hour; Fourier terms sin/cos(2πk·t/P) | None (known in advance) |
| **Holidays / known events** | Holiday flag, bridge days, planned shutdowns, the new-line indicator | Only events known in advance; not "unplanned outage" |
| **Exogenous drivers** | Temperature, production plan, prices | Use the *forecast* available at o, not the realised value |
| **Trend / level** | Time index, post-break indicator | Trees cannot extrapolate a time index; use it with linear models, or de-trend the target |
| **Interactions** | Holiday × weekday; temperature × weekend | — |

**Fourier terms** represent a long seasonality (yearly with daily data) with a few smooth columns instead of 365 dummies. K pairs of sin/cos capture increasingly detailed shapes. Choose K by validation (typically 1–5 for yearly).

**Two ways to arrange the data for multi-step ML forecasting** (details: [multi_step_forecasting.md](multi_step_forecasting.md)):
- **Recursive:** a one-step model with lags of y_{t−1}, y_{t−2}…; for h > 1 it feeds its own predictions back in.
- **Direct ("pooled"):** one row per (origin, horizon). The features are known at the origin, `h` is a feature, and the target is y_{o+h}. This cookbook uses the pooled direct layout as the default, because it is leakage-safe by construction. It is implemented in `cookbook_utils.make_direct_features`.

## 4. Assumptions and limitations
- Lags of up to L steps cost you the first L rows of history.
- Many lags and rolling windows create correlated features. That is fine for regularised linear models and GBMs, but it complicates interpretation.
- Tree models cannot extrapolate. With a trending target, model the target relative to a recent level (y − rolling mean) or use linear models (see [forecasting_models.md](forecasting_models.md)).
- Simulating exogenous forecast errors (as done here for temperature) is an approximation. When available, use archived real forecasts ("as-of" weather forecasts).

## 5. Step-by-step procedure
1. **Start from the EDA decisions:** which seasonal periods, which lags (PACF), which drivers (shape).
2. **Define the information set at the origin** (latency!) and the target-time known-in-advance variables.
3. **Build the lag and rolling features from data up to the origin.** Always `shift` before `rolling` for row-aligned features.
4. **Build the target-time features:** calendar, Fourier, holidays, events, exogenous forecasts.
5. **Assemble the table** (pooled direct or recursive layout); drop the incomplete rows at the start.
6. **Audit for leakage:** for a few rows, check by hand that every feature value was available at the origin.
7. **Evaluate feature groups** with rolling origin ([validation.md](validation.md)): lags only → + calendar → + exogenous.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from cookbook_data import load_regular_energy_demand

df = load_regular_energy_demand()
y = df["demand_mwh"]

# --- Row-aligned features (for one-step / recursive models and for temporal regression) ---
feat = pd.DataFrame(index=df.index)
for k in (1, 2, 7, 14):
    feat[f"lag_{k}"] = y.shift(k)                         # value k days before t
feat["roll7_mean_past"] = y.shift(1).rolling(7).mean()    # mean of t-7..t-1 (shift FIRST)
feat["roll28_std_past"] = y.shift(1).rolling(28).std()
feat["ewm14_past"] = y.shift(1).ewm(span=14).mean()       # exponentially weighted past mean
# calendar (known in advance)
feat["dow"] = df.index.dayofweek
feat["is_weekend"] = (feat["dow"] >= 5).astype(int)
doy = df.index.dayofyear
for k in (1, 2):
    feat[f"sin_y{k}"], feat[f"cos_y{k}"] = np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25)
feat["is_holiday"] = df["is_holiday"]
feat["post_new_line"] = (df.index >= "2022-06-01").astype(int)   # known, planned event
# exogenous with the physically motivated shape (heating/cooling degree days)
feat["hdd"] = np.clip(15 - df["temperature_c"], 0, None)
feat["cdd"] = np.clip(df["temperature_c"] - 22, 0, None)
print(feat.dropna().head(3).T)
```

```python
# Leakage audit by hand: for one row, recompute the features from raw values
t = pd.Timestamp("2023-03-15")
manual_roll7 = y.loc[t - pd.Timedelta("7D"): t - pd.Timedelta("1D")].mean()
assert np.isclose(manual_roll7, feat.loc[t, "roll7_mean_past"]), "rolling feature includes the current/future value!"
assert feat.loc[t, "lag_1"] == y.loc[t - pd.Timedelta("1D")]
print("audit passed for", t.date())
# NOTE: hdd/cdd above use ACTUAL temperature on day t. That is fine for one-step nowcasting with
# observed weather, but NOT for forecasting h days ahead (use a temperature FORECAST, below).
```

```python
# --- Pooled direct layout for multi-step forecasting (H = 14) ---
from cookbook_utils import make_direct_features

H = 14
direct = make_direct_features(df, H=H, n_lags=14, events={"post_new_line": "2022-06-01"})
print(direct.shape)
print(direct.loc[direct["h"].isin([1, 7, 14]),
                 ["origin", "h", "target_time", "y_lag0", "y_same_weekday", "dow", "is_holiday_t",
                  "temp_fc", "y_target"]].head(6))
# Each row: features known at `origin`, target = demand on `target_time` = origin + h.
# temp_fc = actual temperature on target_time + noise growing with h (a simulated weather forecast).
```

```python
# Evaluate feature groups with rolling origin (the ML model itself is discussed in forecasting_models.md)
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from cookbook_utils import evaluate_direct_model, summarize_forecasts

TEST = 182
dev_end = df.index[-TEST - 1]
dev = direct[direct["target_time"] <= dev_end]
origins = df.index[730:-TEST - H:28]                     # every 4 weeks (keeps runtime low)

lags = [f"y_lag{k}" for k in range(14)] + ["roll7_mean", "roll28_mean", "roll7_std", "y_same_weekday", "h"]
calendar = ["dow", "sin_year1", "cos_year1", "sin_year2", "cos_year2", "is_holiday_t", "post_new_line"]
weather = ["heat_fc", "cool_fc"]

def ridge_factory(cols):
    num = [c for c in cols if c != "dow"]
    def make():
        pre = ColumnTransformer([("num", StandardScaler(), num)] +
                                ([("dow", OneHotEncoder(handle_unknown="ignore"), ["dow"])] if "dow" in cols else []))
        return make_pipeline(pre, RidgeCV(alphas=np.logspace(-2, 3, 12)))
    return make

res = []
for name, cols in [("lags", lags), ("lags+calendar", lags + calendar), ("lags+calendar+weather", lags + calendar + weather)]:
    res.append(evaluate_direct_model(dev, ridge_factory(cols), origins, cols, H, y, name=name))
print(summarize_forecasts(pd.concat(res)).round(3))
```

## 7. Interpretation
- **The leakage audit** is the most important output. If an assertion fails, a window includes the current or future value.
- **Feature-group comparison:** adding calendar features should help a lot (weekday and holiday effects are large and known in advance). Adding the weather forecast helps further, because the yearly pattern and the hot/cold days are temperature-driven. Because the temperature forecast is noisy, its benefit should shrink with the horizon (check MAE by `h`).
- **If adding a feature group makes the results worse,** the features may be redundant with the lags (seasonality already carried by `y_same_weekday`), or too noisy (temperature forecasts at long horizons). Keep only what helps, beyond the noise across origins.

## 8. Decision framework
| Situation | Feature choice |
|---|---|
| Strong weekly seasonality | Day-of-week (one-hot for linear models) and/or same-weekday lag |
| Long seasonality (yearly with daily data; weekly with hourly data) | Fourier terms (K by validation) or a driver that explains it (temperature) |
| Multiple seasonalities | Calendar features for each (hour, weekday, Fourier yearly) |
| Holidays / events | Flags for the event day and its neighbours (bridge days) |
| Weather-driven series | Degree-day (hinge) features from **forecasts** |
| Trending series + trees | Target relative to the recent level, or a linear model with a trend term |
| Level shift with a known date | Indicator after the date; or train only after the break |
| Hourly or finer data with long memory | Lags at 1..k plus seasonal lags (24, 168); rolling windows at several scales |
| Temporal classification (per-row target) | Past-window aggregates per entity, computed as-of the row's time |

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| `y.rolling(7).mean()` without `shift(1)` | Includes y_t (the target) | Shift first; hand audit |
| Lag-1 features for h-step forecasts | Unavailable at forecast time | Direct layout, or recursive prediction |
| Actual future exogenous values | Optimistic accuracy | Forecasts (as-of) or simulated errors |
| Time index as a feature for trees | Flat extrapolation | De-trend, or linear model |
| Including events not known in advance (unplanned outages) | Leakage | Only planned/known events |
| Hundreds of lags "just in case" | Overfitting, slow | PACF + validation-driven choice |

## 10. Checklist
- [ ] Information set at the origin defined (incl. latency).
- [ ] Lag and rolling features past-only (shift before rolling); audited by hand.
- [ ] Calendar, Fourier, holiday and event features for the target time.
- [ ] Exogenous variables as forecasts available at the origin.
- [ ] Layout chosen (direct/recursive) consistent with H.
- [ ] Feature groups evaluated by rolling origin; MAE by horizon checked.

## 11. References
- Hyndman, R. J. & Athanasopoulos, G. (2021). *FPP3*, Ch. 7 "Time series regression models" (§7.4 useful predictors, Fourier terms), §12.1 complex seasonality. https://otexts.com/fpp3/
- scikit-learn, *Lagged features for time series forecasting*. https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html
- scikit-learn, *Time-related feature engineering* (periodic splines, cyclical encodings). https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html
- Januschowski, T. et al. (2022). "Forecasting with trees." *IJF* 38(4), 1473–1481. https://doi.org/10.1016/j.ijforecast.2021.10.004
