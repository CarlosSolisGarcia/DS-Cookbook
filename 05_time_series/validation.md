# Time-Series Validation: Chronological Splits, Rolling Origin and Temporal Leakage

> **Folder:** 05_time_series · **Prev:** [feature_engineering.md](feature_engineering.md) · **Next:** [baseline_models.md](baseline_models.md) · **Related:** [cross_validation.md](../06_model_improvement/cross_validation.md), [data_leakage.md](../00_methodology/data_leakage.md), [validation_reference.md](../08_quick_references/validation_reference.md)
>
> *Reading order:* this recipe is numbered after feature engineering, but read it **before** comparing any models. The helper functions defined here (`rolling_origin_splits`, `evaluate_forecaster`) are reused by every later time-series recipe via `cookbook_utils.py`.

---

## 1. Purpose
Estimate how a forecasting method will perform **in deployment**: repeatedly forecasting H steps ahead from successive origins, using only data available at each origin. Do this without the subtle look-ahead leaks that make time-series results optimistic.

## 2. When to use it
For every forecasting comparison. Also use chronological validation for temporal regression/classification (rows ordered in time, model used on the future).

## 3. Conceptual foundation
- **Why not random K-fold?** Neighbouring observations are correlated. A random split puts t−1 and t+1 in training when t is being validated, so the model "interpolates" instead of forecasting. The estimate is optimistic, and trends/breaks can never be tested properly.
- **Rolling-origin evaluation** (time-series cross-validation, walk-forward): for origins o₁ < o₂ < …, train on data up to o_k and forecast o_k+1 … o_k+H. Then average the errors per horizon over the origins.
  - **Expanding window:** training always starts at the beginning. More data, but it assumes old data are still relevant.
  - **Sliding window:** a fixed training length. Adapts to change, uses less data. Compare both when there are breaks or drift.
- **Gap (latency):** if data arrive with a delay, leave a gap between the end of training and the first forecast step.
- **Re-fit frequency:** in production you may refit weekly while forecasting daily. Validation should mimic the same schedule (refit at each origin, or refit every k origins).
- **Final hold-out:** reserve the last period (covering at least H, ideally a seasonal cycle) as the test set. Do all model selection with rolling origins on the development period before it.
- **Statistical caveat:** for purely autoregressive ML models with no unmodelled dependence, standard K-fold has been shown to be valid (Bergmeir, Hyndman & Koo, 2018). In practice drift, trends and exogenous effects are almost always present, so rolling origin is the safe default.

`sklearn.model_selection.TimeSeriesSplit` implements expanding (or capped with `max_train_size`) folds, with `test_size` and `gap` parameters. It works directly with scikit-learn estimators on a feature matrix. For forecasters that consume a series and output H steps, use the explicit loop from `cookbook_utils.evaluate_forecaster`.

### Temporal leakage checklist
| Leak | Example | Fix |
|---|---|---|
| Future in features | Centred rolling mean; `bfill`; interpolation across the origin | Past-only windows (`shift(1).rolling`), ffill at the origin |
| Future in preprocessing statistics | Scaler/Box-Cox λ fitted on the full series | Fit inside each training window (Pipeline) |
| Future exogenous actuals | Using observed temperature on forecast days | Use forecasts, or simulated forecast error, or lags only |
| Feature selection/tuning on the test period | Choosing lags by test error | Choose within rolling origins on the development period |
| Overlapping targets | Direct H-step targets built across the split boundary | Drop training rows whose targets extend past the origin |
| Global target statistics | Target encoding of weekday with all data | Compute within the training window |

## 4. Assumptions and limitations
- Rolling-origin errors at different origins are correlated (overlapping training data, autocorrelated errors). Standard errors computed as if they were independent are too small.
- Few origins give noisy estimates. Many origins with long refits are expensive. Balance them with `step`, and refit every k origins for heavy models.
- Evaluation periods should cover the seasons that matter (e.g. both winter and summer).

## 5. Step-by-step procedure
1. **Fix H, the origin schedule, the latency (gap) and the refit schedule** from the task definition.
2. **Reserve the final test period** (≥ H; ideally ≥ one seasonal cycle).
3. **Choose the initial training size** (≥ 2 seasonal cycles if the yearly pattern is modelled) and the step between origins (e.g. 7 days).
4. **Run each candidate** through the same rolling origins. Store a long table (origin, h, y, ŷ).
5. **Summarise** per model and per horizon. Compare with the seasonal-naive baseline (MASE).
6. **Check stability:** errors per origin over time; expanding vs sliding window.
7. **Final:** refit the chosen configuration on all development data, then forecast the test period with the same origin schedule (rolling over the test period). Report once.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

from cookbook_data import load_regular_energy_demand
from cookbook_utils import evaluate_forecaster, rolling_origin_splits, summarize_forecasts

df = load_regular_energy_demand()
y = df["demand_mwh"]
TEST = 182
y_dev, y_test = y.iloc[:-TEST], y.iloc[-TEST:]
H = 14

# Visualise the rolling-origin scheme (expanding window, step 7, first 6 origins + last)
splits = list(rolling_origin_splits(len(y_dev), initial=730, horizon=H, step=7))
print(f"{len(splits)} origins; first origin {y_dev.index[splits[0][0][-1]].date()}, "
      f"last {y_dev.index[splits[-1][0][-1]].date()}")
fig, ax = plt.subplots(figsize=(11, 3))
for i, (tr, te) in enumerate(splits[:6] + splits[-1:]):
    ax.plot(y_dev.index[tr], [i] * len(tr), color="tab:blue", lw=4)
    ax.plot(y_dev.index[te], [i] * len(te), color="tab:orange", lw=4)
ax.set_yticks([]); ax.set_title("rolling origin: training (blue) / forecast window (orange)"); plt.show()
```

```python
# The same idea with scikit-learn's TimeSeriesSplit (for feature-matrix models; see feature_engineering.md)
tscv = TimeSeriesSplit(n_splits=10, test_size=H, gap=0)     # gap=1 would mimic 1-day latency
for k, (tr, te) in enumerate(tscv.split(y_dev)):
    if k in (0, 9):
        print(f"fold {k}: train ..{y_dev.index[tr[-1]].date()} ({len(tr)} rows), "
              f"test {y_dev.index[te[0]].date()}..{y_dev.index[te[-1]].date()}")
```

```python
# Demonstration: random K-fold vs rolling origin for a lag-feature model (why random CV is optimistic)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score

feat = pd.DataFrame({f"lag_{k}": y_dev.shift(k) for k in (1, 2, 7, 14)})
feat["lead_1"] = y_dev.shift(-1)                      # a LEAKY feature: tomorrow's value (deliberate bug)
data = feat.assign(y=y_dev).dropna()
X_ok, X_leaky, t = data.drop(columns=["y", "lead_1"]), data.drop(columns=["y"]), data["y"]
gbm = HistGradientBoostingRegressor(random_state=0)
for name, X in [("lags only", X_ok), ("with leaky lead_1", X_leaky)]:
    r_rand = -cross_val_score(gbm, X, t, cv=KFold(5, shuffle=True, random_state=0), scoring="neg_mean_absolute_error").mean()
    r_time = -cross_val_score(gbm, X, t, cv=TimeSeriesSplit(5, test_size=91), scoring="neg_mean_absolute_error").mean()
    print(f"{name:18s} MAE random-KFold={r_rand:5.2f}  time-ordered={r_time:5.2f}")
# Note: one-step-ahead with true lags is easier than the real 14-day task (see multi_step_forecasting.md)
```

```python
# Expanding vs sliding window, with the level shift in the data
def seasonal_naive(y_tr, h, X_tr=None, X_fu=None, m=7):
    return np.array([y_tr.values[-m:][i % m] for i in range(h)])

from statsmodels.tsa.holtwinters import ExponentialSmoothing
def holt_winters(y_tr, h, X_tr=None, X_fu=None):
    return ExponentialSmoothing(y_tr.values, trend="add", damped_trend=True, seasonal="add",
                                seasonal_periods=7).fit().forecast(h)

res = pd.concat([
    evaluate_forecaster(y_dev, seasonal_naive, initial=730, horizon=H, step=7, name="seasonal_naive"),
    evaluate_forecaster(y_dev, holt_winters, initial=730, horizon=H, step=7, name="HW expanding"),
    evaluate_forecaster(y_dev, holt_winters, initial=730, horizon=H, step=7, window=365, name="HW sliding 365d"),
])
print(summarize_forecasts(res).round(3))

# Stability over time: MAE per origin
per_origin = res.groupby(["model", "origin"])["abs_err"].mean().unstack(0)
per_origin.rolling(4).mean().plot(figsize=(11, 3), title="MAE per origin (4-origin rolling mean)"); plt.show()
```

## 7. Interpretation
- **Random K-fold vs time-ordered (lags only):** random K-fold reports MAE ≈ 6.0, time-ordered ≈ 8.5. Random CV is about 30% optimistic here, without any explicit leak, because it lets the model interpolate between neighbouring days and never tests the drift.
- **The leaky `lead_1` feature** (a `shift(-1)` bug) improves the score under **both** schemes. The CV scheme protects against *split* leakage, not *feature* leakage. Only a feature audit (§3 checklist, [data_leakage.md](../00_methodology/data_leakage.md)) catches the latter.
- **Expanding vs sliding:** if sliding wins, older data are less relevant (drift, breaks). If expanding wins, more data help (stable process). Here they are practically tied (MAE 6.80 vs 6.88). Holt-Winters adapts its level anyway, so the 2022 shift does not penalise the expanding window much. (MASE differs slightly between them because each window computes its own scale.)
- **MAE per origin over time:** spikes around holidays or the level shift show where the models struggle. A model that is better on average but much worse right after breaks may be riskier.

## 8. Decision framework
- **IF** the series has drift or breaks → compare expanding vs sliding windows; consider training only after the break.
- **IF** data arrive with latency → set `gap` equal to the latency.
- **IF** the model is expensive to fit → refit every k origins (mimicking production), or reduce the number of origins (keep enough to cover all seasons).
- **IF** there are several related series → do rolling origin per series (or across series with the same origins); avoid splitting series randomly.
- **IF** the target is per-row (temporal regression) → `TimeSeriesSplit` on the sorted rows, plus grouping if entities repeat.
- **IF** n origins is small → report the per-origin spread and avoid over-interpreting small differences.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Random K-fold for forecasting | Optimistic, can't test extrapolation | Rolling origin |
| A single train/test split | Result depends on one period | Many origins |
| Preprocessing fitted on the whole series | Look-ahead | Fit inside the training window |
| Using actual future exogenous values | Optimistic | Forecasted/perturbed exogenous inputs |
| Tuning on the final test period | Optimistic final report | Dev-period rolling origin for all choices |
| Ignoring latency | Using data not yet available | `gap` |

## 10. Checklist
- [ ] H, origins, gap and refit schedule match deployment.
- [ ] Final test period reserved and untouched.
- [ ] Initial window ≥ 2 cycles of the modelled seasonality.
- [ ] All candidates on identical origins; results stored in a long table.
- [ ] Per-horizon summaries; MASE vs seasonal naive.
- [ ] Stability across origins checked; expanding vs sliding compared if drift exists.
- [ ] Temporal leakage checklist (§3) reviewed.

## 11. References
- Hyndman, R. J. & Athanasopoulos, G. (2021). *FPP3*, §5.10 "Time series cross-validation". https://otexts.com/fpp3/tscv.html
- Tashman, L. J. (2000). "Out-of-sample tests of forecasting accuracy: an analysis and review." *IJF* 16(4), 437–450. https://doi.org/10.1016/S0169-2070(00)00065-0
- Bergmeir, C., Hyndman, R. J. & Koo, B. (2018). "A note on the validity of cross-validation for evaluating autoregressive time series prediction." *CSDA* 120, 70–83. https://doi.org/10.1016/j.csda.2017.11.003
- Cerqueira, V., Torgo, L. & Mozetič, I. (2020). "Evaluating time series forecasting models: an empirical study on performance estimation methods." *Machine Learning* 109, 1997–2028. https://doi.org/10.1007/s10994-020-05910-7
- scikit-learn, `TimeSeriesSplit`. https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split
