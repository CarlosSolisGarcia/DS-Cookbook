# Forecasting Models: Exponential Smoothing, ARIMA/SARIMA(X), Regression and Gradient Boosting

> **Folder:** 05_time_series · **Prev:** [baseline_models.md](baseline_models.md) · **Next:** [multi_step_forecasting.md](multi_step_forecasting.md) · **Related:** [feature_engineering.md](feature_engineering.md), [evaluation.md](evaluation.md), [regression model_selection.md](../03_regression/model_selection.md)

---

## 1. Purpose
Choose, fit and compare the main families of forecasting models. Understand what structure each one captures, how to configure it from the EDA findings, and how to compare the families fairly against the baselines.

## 2. When to use it
After the baselines are established and the rolling-origin evaluation is set up ([validation.md](validation.md)).

## 3. Conceptual foundation

### 3.1 Exponential smoothing (ETS)
Forecasts are weighted averages of past observations, with weights decaying exponentially. The model has a **level**, optionally a **trend** (additive or damped) and a **seasonal** component (additive or multiplicative), each updated by a smoothing parameter (α, β, γ) estimated by maximum likelihood.
- **SES** (level only) → flat forecasts. **Holt** (+ trend). **Holt-Winters** (+ seasonality).
- **Damped trend** (φ < 1) flattens the trend over longer horizons. It is usually more accurate than a linear trend, which extrapolates forever.
- Strengths: few parameters, robust, it adapts to level changes, and the state-space form (`ETSModel`) gives prediction intervals.
- Limits: one seasonal period, no exogenous variables, and long seasonal periods (365 for daily data) are impractical.

### 3.2 ARIMA / SARIMA / SARIMAX
- **AR(p):** y_t depends on its own p past values. **MA(q):** on the past q forecast errors. **I(d):** differencing d times to remove a trend/unit root.
- **Seasonal SARIMA(p,d,q)(P,D,Q)_m** adds the same terms at seasonal lags m (e.g. D = 1 for y_t − y_{t−7}).
- **SARIMAX** adds exogenous regressors (temperature degree-days, holiday flags). Formally, this is a regression with ARIMA errors.
- **Identification:** d and D from the stationarity analysis ([exploratory_analysis.md](exploratory_analysis.md)). p and q from the PACF/ACF of the differenced series, or by comparing AIC over a small grid. **Always confirm with rolling-origin error.** AIC is an in-sample criterion.
- Limits: linear dynamics, one seasonal period (long periods are slow), and fitting cost grows with the seasonal order.

### 3.3 Regression-based forecasting
A linear model on calendar, Fourier, holiday, exogenous and lag features ([feature_engineering.md](feature_engineering.md)). It is transparent and fast, it extrapolates trends linearly, and it handles several seasonalities through features. With residual autocorrelation, adding lags (or ARIMA errors via SARIMAX) improves it.

### 3.4 Tree-based / gradient boosting with lag features
GBMs on the same features capture non-linear effects and interactions (temperature × weekday, holiday × season) automatically. The key limit is that they **cannot extrapolate levels**. Under a trend, predictions saturate at the training range, giving a systematic under-forecast for a growing series. Remedy: model the target **relative to a recent level** (y_{t} − rolling mean at the origin) or as a ratio, and add the level back afterwards.

### 3.5 Which family usually wins?
Results from the M-competitions and practice:
- Simple statistical methods (ETS, Theta, ARIMA) are strong for single series with stable seasonality.
- ML (especially GBMs) wins when there are informative exogenous variables, non-linear effects, or many related series that can be pooled (M5).
- **Combinations** (averaging forecasts from different families) are consistently robust.

## 4. Assumptions and limitations
- ETS/ARIMA assume the dynamics are stable over the estimation window. Breaks need intervention terms or a shorter window.
- SARIMAX with exogenous inputs needs **future values** of those inputs (forecasts) for the horizon.
- GBMs need enough history (hundreds of rows at least). With a few dozen observations, use ETS or simple regression.

## 5. Step-by-step procedure
1. **ETS:** choose the components from EDA (trend? damped? seasonal additive/multiplicative?). Evaluate by rolling origin.
2. **SARIMA(X):** set d and D from the EDA. Compare a small grid of (p,q)(P,Q) by AIC on one training window. Evaluate the top 1–2 candidates by rolling origin. Add exogenous regressors for SARIMAX.
3. **Linear regression** on the direct features (calendar + lags + exogenous forecasts).
4. **GBM** on the same features: first the raw target, then a level-relative target.
5. **Compare** all of them with the best baseline: MAE, MASE, bias, MAE by horizon.
6. **Try a simple combination** (mean of the 2–3 best).
7. **Residual diagnostics** for the chosen model ([evaluation.md](evaluation.md)).

## 6. Python implementation

```python
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from cookbook_data import load_regular_energy_demand
from cookbook_utils import evaluate_forecaster, summarize_forecasts, make_direct_features, evaluate_direct_model

warnings.filterwarnings("ignore")                     # statsmodels convergence chatter; check fits manually in real work
df = load_regular_energy_demand()
TEST, H = 182, 14
dev = df.iloc[:-TEST]
y_dev = dev["demand_mwh"]
INITIAL, STEP = 730, 28                               # 4-weekly origins keep the runtime modest

def seasonal_median_fc(y, h, X=None, Xf=None, m=7, k=4):
    prof = np.median(y.values[-m * k:].reshape(k, m), axis=0)
    return np.array([prof[i % m] for i in range(h)])
```

```python
# 1. Exponential smoothing variants
def ets_factory(trend, damped, seasonal):
    def fc(y, h, X=None, Xf=None):
        m = ExponentialSmoothing(y.values, trend=trend, damped_trend=damped,
                                 seasonal=seasonal, seasonal_periods=7 if seasonal else None)
        return m.fit().forecast(h)
    return fc

ets_variants = {"SES": (None, False, None), "Holt-Winters additive": ("add", False, "add"),
                "Holt-Winters damped": ("add", True, "add")}
res = [evaluate_forecaster(y_dev, seasonal_median_fc, INITIAL, H, STEP, name="seasonal_median (baseline)")]
for name, cfg in ets_variants.items():
    res.append(evaluate_forecaster(y_dev, ets_factory(*cfg), INITIAL, H, STEP, name=name))
print(summarize_forecasts(pd.concat(res)).round(3))
```

```python
# 2. SARIMA order selection by AIC on ONE training window (first origin), then rolling-origin check
y0 = y_dev.iloc[:INITIAL]
grid = [((1, 0, 1), (0, 1, 1, 7)), ((1, 0, 0), (0, 1, 1, 7)), ((2, 0, 1), (0, 1, 1, 7)),
        ((1, 1, 1), (0, 1, 1, 7)), ((1, 0, 1), (1, 1, 1, 7))]
aic = {}
for order, sorder in grid:
    fit = SARIMAX(y0.values, order=order, seasonal_order=sorder, trend="c" if order[1] == 0 else "n").fit(disp=False)
    aic[f"{order}{sorder}"] = fit.aic
print(pd.Series(aic).sort_values().round(1))
```

```python
# SARIMA and SARIMAX (exogenous: holiday flag + heating/cooling degree days from a SIMULATED temperature forecast)
rng = np.random.default_rng(0)
def exog_matrix(d, forecast=False):
    temp = d["temperature_c"].to_numpy()
    if forecast:                                           # forecast error grows with the horizon
        temp = temp + rng.normal(0, 0.8 + 0.15 * np.arange(1, len(d) + 1))
    return pd.DataFrame({"holiday": d["is_holiday"].to_numpy(),
                         "hdd": np.clip(15 - temp, 0, None), "cdd": np.clip(temp - 22, 0, None)})

def sarima_fc(y, h, X=None, Xf=None):
    fit = SARIMAX(y.values, order=(1, 0, 1), seasonal_order=(0, 1, 1, 7), trend="c").fit(disp=False)
    return fit.forecast(h)

def sarimax_fc(y, h, X, Xf):
    fit = SARIMAX(y.values, exog=exog_matrix(X), order=(1, 0, 1), seasonal_order=(0, 1, 1, 7),
                  trend="c").fit(disp=False)
    return fit.forecast(h, exog=exog_matrix(Xf, forecast=True))

X_dev = dev[["temperature_c", "is_holiday"]]
res.append(evaluate_forecaster(y_dev, sarima_fc, INITIAL, H, STEP, name="SARIMA(1,0,1)(0,1,1)7"))
res.append(evaluate_forecaster(y_dev, sarimax_fc, INITIAL, H, STEP, X=X_dev, name="SARIMAX + holiday + degree days"))
print(summarize_forecasts(pd.concat(res)).round(3))
```

```python
# 3–4. Regression and gradient boosting on the pooled direct features (feature_engineering.md)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

direct = make_direct_features(df, H=H, events={"post_new_line": "2022-06-01"})
direct_dev = direct[direct["target_time"] <= y_dev.index[-1]]
cols = [c for c in direct.columns if c not in ("origin", "target_time", "y_target")]
origins = y_dev.index[INITIAL - 1: len(y_dev) - H: STEP]      # same origins as evaluate_forecaster above

def ridge():
    num = [c for c in cols if c != "dow"]
    return make_pipeline(ColumnTransformer([("num", StandardScaler(), num),
                                            ("dow", OneHotEncoder(handle_unknown="ignore"), ["dow"])]),
                         RidgeCV(alphas=np.logspace(-2, 3, 12)))
def gbm():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=30, random_state=0)

res.append(evaluate_direct_model(direct_dev, ridge, origins, cols, H, y_dev, name="ridge (direct features)"))
res.append(evaluate_direct_model(direct_dev, gbm, origins, cols, H, y_dev, name="GBM raw target"))
res.append(evaluate_direct_model(direct_dev, gbm, origins, cols, H, y_dev, name="GBM target − roll28_mean",
                                 target_transform="roll28_mean"))
allres = pd.concat(res)
print(summarize_forecasts(allres).sort_values("MAE").round(3))
```

```python
# 5–6. Per-horizon comparison and a simple combination
summary_h = summarize_forecasts(allres, by=["model", "h"])["MAE"].unstack(0)
print(summary_h.iloc[[0, 6, 13]].T.sort_values(14).round(2))

combo_members = ["ridge (direct features)", "SARIMAX + holiday + degree days", "GBM target − roll28_mean"]
combo = (allres[allres["model"].isin(combo_members)]
         .groupby(["origin", "h", "timestamp"], as_index=False)
         .agg(y=("y", "first"), yhat=("yhat", "mean"), scale=("scale", "first")))
combo["model"], combo["abs_err"] = "combination (mean of 3)", (combo["y"] - combo["yhat"]).abs()
print(summarize_forecasts(pd.concat([allres, combo])).sort_values("MAE").round(3).head(5))
```

## 7. Interpretation
Check the outputs against these expectations:
- **ETS:** Holt-Winters improves hugely on SES (it models the weekly pattern), and it is only marginally better than the seasonal-median baseline. Damping makes no difference at a 14-day horizon here. It matters more for longer horizons. ETS adapts its level after the June 2022 shift, but it cannot anticipate holidays or temperature extremes, which are the main remaining error sources.
- **AIC grid:** the grid ranks (1,1,1)(0,1,1)₇ best. Note that **AIC values are only comparable between models with the same differencing (d, D)**, because differencing changes the data the likelihood is computed on. Within the d = 0 models, (2,0,1) and (1,0,1) are close. The code continues with the simpler (1,0,1)(0,1,1)₇ plus a constant. *Exercise:* evaluate the d = 1 variant by rolling origin. AIC is an in-sample criterion; the rolling-origin error decides.
- **SARIMAX vs SARIMA:** adding the holiday flag and degree days cuts the MAE by about a third (≈ 7.3 → 4.8). Holidays (−35%) and hot/cold days are the main drivers that pure seasonal dynamics cannot anticipate. Plain SARIMA is no better than the baseline here.
- **Ridge on direct features:** often the best single model here. It combines calendar, holidays, the weather forecast and the recent level, all linear, with no extrapolation problem.
- **GBM on the raw target:** clearly negatively biased (under-forecasts). It cannot extrapolate the upward trend and the post-shift level from the training range. **GBM on a level-relative target** removes much of this bias. This is the single most important practical trick for tree models in forecasting.
- **Per-horizon table:** models using the weather forecast lose some of their edge at long horizons (larger forecast errors). Seasonal baselines stay roughly flat.
- **Combination:** often matches or beats the best member, with lower risk. Choose it when no single model dominates across horizons and origins.

## 8. Decision framework
| Situation | Recommended family |
|---|---|
| Single series, stable seasonality, no useful drivers | ETS (damped) or SARIMA; compare with the seasonal-median baseline |
| Strong known drivers (calendar, holidays, weather, plans) | SARIMAX or linear regression on direct features |
| Non-linear driver effects / interactions, enough history | GBM on direct features with a **level-relative target**; compare with linear |
| Multiple seasonalities | Regression/GBM with calendar + Fourier; or MSTL + ETS on the seasonally adjusted series |
| Very short history (< 2 seasonal cycles) | Seasonal naive, ETS without seasonality, or regression with domain-known patterns |
| Many related series | A global GBM pooled across series (future extension) |
| No single model dominates | Simple average of 2–3 diverse models |

- **IF** a GBM shows systematic negative bias on a growing series → level-relative target or differencing; or switch to linear.
- **IF** SARIMAX coefficients for the drivers have the wrong sign → check collinearity and the specification; do not trust the forecasts.
- **IF** the best model beats the baseline only at short horizons → consider using it for h ≤ k and the baseline/ETS beyond (horizon-specific selection), validated by rolling origin.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Choosing ARIMA orders by AIC only | Good in-sample, worse out-of-sample | Rolling-origin confirmation |
| Undamped trend over long horizons | Over-forecasting | `damped_trend=True`; compare |
| GBM on a trending target | Systematic bias | Level-relative target |
| Using actual future exogenous values | Optimistic SARIMAX/ML | Forecasted exogenous values |
| Comparing models on different origins | Invalid comparisons | Same origin list for all |
| Seasonal period 365 in SARIMA with daily data | Very slow and unstable | Fourier terms / drivers for the yearly pattern |

## 10. Checklist
- [ ] ETS variants chosen from the EDA; evaluated by rolling origin.
- [ ] SARIMA orders from EDA + AIC grid; confirmed by rolling origin.
- [ ] SARIMAX with known/forecasted exogenous inputs only.
- [ ] Linear and GBM models on leakage-audited direct features.
- [ ] GBM bias checked; level-relative target if the series trends.
- [ ] All compared with the best baseline on the same origins: MAE, MASE, bias, by horizon.
- [ ] Combination tried.

## 11. References
- Hyndman, R. J. & Athanasopoulos, G. (2021). *FPP3*, Ch. 8 "Exponential smoothing", Ch. 9 "ARIMA models", Ch. 10 "Dynamic regression models", §13.4 "Forecast combinations". https://otexts.com/fpp3/
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C. & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*, 5th ed. Wiley.
- Gardner, E. S. (2006). "Exponential smoothing: The state of the art — Part II." *IJF* 22(4), 637–666. https://doi.org/10.1016/j.ijforecast.2006.03.005
- Makridakis, S., Spiliotis, E. & Assimakopoulos, V. (2022). "M5 accuracy competition: Results, findings, and conclusions." *IJF* 38(4), 1346–1364. https://doi.org/10.1016/j.ijforecast.2021.11.013
- statsmodels: `ExponentialSmoothing`, `ETSModel`, `SARIMAX`. https://www.statsmodels.org/stable/tsa.html
