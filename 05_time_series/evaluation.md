# Forecast Evaluation: Metrics, Residual Diagnostics, Prediction Intervals and Model Comparison

> **Folder:** 05_time_series · **Prev:** [multi_step_forecasting.md](multi_step_forecasting.md) · **Next:** [iterative_improvement.md](../06_model_improvement/iterative_improvement.md) · **Related:** [metrics_reference.md](../08_quick_references/metrics_reference.md), [regression uncertainty.md](../03_regression/uncertainty.md), [validation.md](validation.md)

---

## 1. Purpose
Judge forecasts on the criteria that matter: accuracy relative to baselines and per horizon, bias, whether the residuals still contain structure, and the reliability of the prediction intervals. Then perform the final test-period evaluation and comparison.

## 2. When to use it
At every model comparison (on rolling-origin results) and once on the final test period.

## 3. Conceptual foundation

### 3.1 Metrics
| Metric | Definition | Scale-free? | Strengths | Weaknesses |
|---|---|---|---|---|
| **MAE** | mean \|e\| | No | Interpretable in units; optimal for median forecasts | Not comparable across series |
| **RMSE** | √mean e² | No | Penalises large misses; optimal for mean forecasts | Dominated by a few large errors |
| **MAPE** | mean \|e/y\| × 100 | Yes | Intuitive % | Undefined/explosive near 0; asymmetric; favours under-forecasting |
| **sMAPE** | mean 2\|e\|/(\|y\|+\|ŷ\|) × 100 | Yes | Bounded (0–200%) | Still unstable near 0; not truly symmetric; several definitions exist (state which one) |
| **MASE** | MAE / (in-sample MAE of the seasonal-naive one-step forecast) | Yes | Comparable across series; well-defined near 0; < 1 = better than in-sample seasonal naive | Less intuitive; depends on the chosen m |
| **WAPE** | Σ\|e\| / Σ\|y\| | Yes | Volume-weighted % | Hides per-period errors |
| **Bias** | mean(ŷ − y) | No | Systematic over/under-forecasting | Can be 0 with large errors |
| **Pinball loss / coverage / interval width** | — | — | Quantile and interval quality | — |

Use MAE (or RMSE, if large errors are costly) as the primary metric for a single series, and MASE to compare across series and with the literature. Always show the **metric by horizon**.

### 3.2 Residual diagnostics
For a well-specified model, the **one-step-ahead** residuals should be uncorrelated (no remaining predictable structure), have mean ≈ 0 and roughly constant variance.
- **ACF of the residuals** plus the **Ljung–Box test** (H₀: no autocorrelation up to lag L). A small p-value means structure remains: missing lags or seasonality, or an unmodelled driver.
- Residuals vs time: drift, breaks, seasonal patterns left over.
- Residuals vs drivers (temperature, holiday): a mis-specified driver effect.
- Multi-step residuals (h > 1) are *naturally* autocorrelated (overlapping horizons). Apply Ljung–Box to one-step residuals.

### 3.3 Prediction intervals
- **Model-based** (ETS/ARIMA state space): analytic intervals that assume the model is correct and the errors are Gaussian. Often too narrow in practice, because they ignore parameter and model uncertainty.
- **Empirical (from rolling-origin errors):** for each horizon h, take quantiles of the past forecast errors at that horizon and add them to the point forecast. This approach is simple, works for any model, and its width grows with h automatically. It is a form of split-conformal prediction, and it inherits the exchangeability caveat: under drift, recent errors are more representative than old ones.
- Evaluate with **empirical coverage per horizon** and the mean width. Narrowest at the nominal coverage wins.

## 4. Assumptions and limitations
- Rolling-origin metrics are correlated across origins, so differences need care. Look at the per-origin win rates and the stability over time, not just the averages.
- A good forecast metric does not guarantee a good *decision*. Asymmetric costs call for quantile forecasts and a pinball loss at the relevant quantile.

## 5. Step-by-step procedure
1. **Collect rolling-origin results** for all candidates (long table).
2. **Summary metrics** (MAE, RMSE, MASE, bias), overall and **by horizon**. Compare with the best baseline.
3. **Per-origin comparison:** the fraction of origins where the model beats the baseline, and errors over time.
4. **Residual diagnostics** of the chosen model: one-step residual ACF, Ljung–Box, and residuals vs drivers/time.
5. **Prediction intervals:** empirical per-horizon error quantiles (and model-based ones if available). Check coverage per horizon.
6. **Final test:** forecast the test period with the same origin schedule (rolling over the test period, refitting as in production). Report the metrics, coverage and comparison with the baseline once.

## 6. Python implementation

```python
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

from cookbook_data import load_regular_energy_demand
from cookbook_utils import (evaluate_forecaster, summarize_forecasts, make_direct_features,
                            evaluate_direct_model, mase)

warnings.filterwarnings("ignore")
df = load_regular_energy_demand()
y = df["demand_mwh"]
TEST, H, INITIAL = 182, 14, 730
dev_end = y.index[-TEST - 1]
y_dev = y[:dev_end]

def seasonal_median_fc(y_, h, X=None, Xf=None, m=7, k=4):
    prof = np.median(y_.values[-m * k:].reshape(k, m), axis=0)
    return np.array([prof[i % m] for i in range(h)])

direct = make_direct_features(df, H=H, events={"post_new_line": "2022-06-01"})
cols = [c for c in direct.columns if c not in ("origin", "target_time", "y_target")]
def ridge():
    num = [c for c in cols if c != "dow"]
    return make_pipeline(ColumnTransformer([("num", StandardScaler(), num),
                                            ("dow", OneHotEncoder(handle_unknown="ignore"), ["dow"])]),
                         RidgeCV(alphas=np.logspace(-2, 3, 12)))

# Rolling-origin results on the development period (weekly origins)
origins_dev = y_dev.index[INITIAL - 1: len(y_dev) - H: 7]
res = pd.concat([
    evaluate_forecaster(y_dev, seasonal_median_fc, INITIAL, H, 7, name="baseline"),
    evaluate_direct_model(direct[direct["target_time"] <= dev_end], ridge, origins_dev, cols, H, y_dev, name="ridge"),
])
print(summarize_forecasts(res).round(3))
```

```python
# By horizon, and per-origin win rate vs the baseline
by_h = summarize_forecasts(res, by=["model", "h"])[["MAE", "MASE"]].unstack(0)
print(by_h.iloc[[0, 3, 6, 9, 13]].round(3))
per_origin = res.groupby(["origin", "model"])["abs_err"].mean().unstack()
print(f"ridge beats baseline at {np.mean(per_origin['ridge'] < per_origin['baseline']):.0%} of origins")
per_origin.plot(figsize=(11, 3), title="MAE per origin"); plt.show()
```

```python
# Residual diagnostics on ONE-STEP residuals. They must be consecutive DAILY residuals,
# so we fit once at an origin one year before the end of dev and predict the h=1 rows of the following year
# (the weekly-origin table above has only one h=1 residual per week, which is useless for ACF at lags 1-14).
O1 = dev_end - pd.Timedelta("365D")
d_dev = direct[direct["target_time"] <= dev_end]
m1 = ridge().fit(d_dev.loc[d_dev["target_time"] <= O1, cols], d_dev.loc[d_dev["target_time"] <= O1, "y_target"])
one = d_dev[(d_dev["h"] == 1) & (d_dev["origin"] >= O1)].set_index("target_time")
resid = one["y_target"] - m1.predict(one[cols])
print(acorr_ljungbox(resid, lags=[7, 14], return_df=True).round(4))
fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
resid.plot(ax=ax[0], title="one-step residuals over time"); ax[0].axhline(0, c="k")
plot_acf(resid, lags=21, ax=ax[1])
temp = df.loc[resid.index, "temperature_c"]
ax[2].scatter(temp, resid, s=6); ax[2].set(xlabel="temperature", title="residual vs temperature")
plt.tight_layout(); plt.show()
```

```python
# Empirical prediction intervals: per-horizon quantiles of past rolling-origin errors
def interval_table(res_model, alpha=0.1, recent=None):
    r = res_model.copy()
    if recent is not None:                                    # use only the most recent origins (drift-aware)
        r = r[r["origin"] >= sorted(r["origin"].unique())[-recent]]
    e = r["y"] - r["yhat"]
    return (r.assign(e=e).groupby("h")["e"].quantile([alpha / 2, 1 - alpha / 2]).unstack()
            .rename(columns={alpha / 2: "lo", 1 - alpha / 2: "hi"}))

ridge_dev = res[res["model"] == "ridge"]
q = interval_table(ridge_dev, alpha=0.1)
print(q.iloc[[0, 6, 13]].round(2))                        # widths grow with h
```

```python
# FINAL TEST: rolling origins over the test period, refitting each week (as in production),
# with intervals from development-period errors. Done ONCE.
origins_test = y.index[len(y_dev) - 1: len(y) - H: 7]
res_test = pd.concat([
    evaluate_forecaster(y, seasonal_median_fc, len(y_dev), H, 7, name="baseline"),
    evaluate_direct_model(direct, ridge, origins_test, cols, H, y, name="ridge"),
])
print(summarize_forecasts(res_test).round(3))

rt = res_test[res_test["model"] == "ridge"].merge(q, left_on="h", right_index=True)
rt["inside"] = (rt["y"] >= rt["yhat"] + rt["lo"]) & (rt["y"] <= rt["yhat"] + rt["hi"])
print("90% interval coverage on test:", round(rt["inside"].mean(), 3))
print(rt.groupby(pd.cut(rt["h"], [0, 4, 9, 14]), observed=True)["inside"].mean().round(3))

# Plot one test origin with its interval
o = rt["origin"].unique()[5]
one = rt[rt["origin"] == o]
hist = y[o - pd.Timedelta("28D"): o]
plt.figure(figsize=(10, 3.5)); plt.plot(hist.index, hist, label="history")
plt.plot(one["timestamp"], one["y"], "k.-", label="actual")
plt.plot(one["timestamp"], one["yhat"], label="forecast")
plt.fill_between(one["timestamp"], one["yhat"] + one["lo"], one["yhat"] + one["hi"], alpha=0.3, label="90% PI")
plt.legend(); plt.title(f"forecast from {pd.Timestamp(o).date()}"); plt.show()
```

## 7. Interpretation
- **Overall:** the ridge model has a MASE well below the baseline's and a small bias. Check whether the bias is consistently negative. A remaining under-forecast suggests the trend or level is not fully captured.
- **By horizon:** the errors grow with h for the ridge model, because the weather forecast degrades and the lags become stale. The baseline is roughly flat. The *relative* advantage is usually largest at short horizons.
- **Per-origin win rate:** a high win rate (e.g. > 80%) together with a lower average MAE is strong evidence. A model that wins on average but loses at half of the origins depends on a few periods. Investigate those periods.
- **Ljung–Box on one-step residuals:** here p ≈ 0 at lags 7 and 14, so clear structure remains. Two causes are plausible and both are instructive. (1) A *pooled* linear direct model uses the **same lag coefficients for every horizon**. At h = 1 the most recent value (`y_lag0`) deserves much more weight than at h = 14, so the AR(1)-type persistence of the noise is under-used. Fixes: interactions `h × y_lag0`, separate models for short horizons, or a GBM, which can learn horizon-dependent effects. (2) The diagnostic model was fitted once and not refitted for a year, so slow level drift shows up as autocorrelation. Use refits that match production. At large n, also check the size of the ACF spikes: tiny correlations can be "significant" without being worth modelling.
- **Residual vs temperature:** a remaining V or curve shape means the degree-day thresholds (15/22 °C) are mis-specified. Try splines.
- **Interval coverage on test:** close to 90% means the empirical intervals are reliable. In this run the coverage is ≈ 85%, and only ≈ 76% at h ≤ 4. The test-period errors are somewhat larger than the development errors, and the bias is more negative (≈ −1.3), which is a sign of drift. Remedies: calibrate on recent origins only (`recent=` argument), re-check the trend handling, or widen the intervals. Always check coverage by horizon.
- **Final test vs development metrics:** similar values mean the evaluation procedure was honest. Clearly worse values point to drift or overfitting of the model selection to the development period.

## 8. Decision framework
| Situation | Metric / action |
|---|---|
| One series, costs ∝ error | MAE primary; MAE by horizon |
| Large misses especially costly | RMSE + worst-case analysis |
| Comparing across series / with the literature | MASE (state m) |
| Series near zero / intermittent | MASE or MAE; avoid MAPE/sMAPE |
| Asymmetric costs | Quantile forecasts at the cost-implied quantile; pinball loss |
| Residuals autocorrelated (Ljung–Box) | Add lags/AR errors; check for missing drivers |
| Intervals under-cover on recent data | Drift-aware (recent-origin) calibration; review model drift |
| A model wins on average but not consistently | Investigate the losing periods; consider combination |
| Final test much worse than dev | Re-check leakage and drift; do not re-tune on the test set |

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Only aggregate metrics | Horizon decay hidden | Metrics by h |
| MAPE on near-zero series | Exploding metric | MASE/MAE |
| Ljung–Box on multi-step residuals | Spurious autocorrelation | One-step residuals |
| Trusting model-based intervals blindly | Under-coverage | Empirical/conformal intervals; check coverage |
| A single test origin | Luck dominates | Rolling origins over the test period |
| Re-tuning after seeing the test results | Optimistic final report | Test once; document |

## 10. Checklist
- [ ] MAE/RMSE/MASE/bias overall and by horizon vs the best baseline.
- [ ] Per-origin win rate and errors over time.
- [ ] One-step residual ACF + Ljung–Box; residuals vs drivers and time.
- [ ] Prediction intervals with per-horizon coverage and width.
- [ ] Final test with rolling origins and refits as in production, done once.
- [ ] Report: metrics, intervals, comparison with the baseline, known limitations.

## 11. References
- Hyndman, R. J. & Koehler, A. B. (2006). "Another look at measures of forecast accuracy." *IJF* 22(4), 679–688. https://doi.org/10.1016/j.ijforecast.2006.03.001
- Hyndman, R. J. & Athanasopoulos, G. (2021). *FPP3*, §5.4 "Residual diagnostics", §5.5 "Distributional forecasts and prediction intervals", §5.8 "Evaluating point forecast accuracy". https://otexts.com/fpp3/
- Ljung, G. M. & Box, G. E. P. (1978). "On a measure of lack of fit in time series models." *Biometrika* 65(2), 297–303.
- Stankevičiūtė, K., Alaa, A. & van der Schaar, M. (2021). "Conformal Time-series Forecasting." *NeurIPS 34*.
- Gneiting, T. & Raftery, A. E. (2007). "Strictly Proper Scoring Rules, Prediction, and Estimation." *JASA* 102(477), 359–378. https://doi.org/10.1198/016214506000001437
