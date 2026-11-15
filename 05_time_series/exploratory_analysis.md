# Time-Series Exploratory Analysis: Trend, Seasonality, Autocorrelation, Stationarity

> **Folder:** 05_time_series · **Prev:** [preprocessing.md](preprocessing.md) · **Next:** [feature_engineering.md](feature_engineering.md) · **Related:** [exploratory_data_analysis.md](../01_data_understanding/exploratory_data_analysis.md), [visualization_reference.md](../08_quick_references/visualization_reference.md)

---

## 1. Purpose
Discover the structure that a forecasting model must capture: trend, level shifts, the seasonal periods and their shape, the dependence on the series' own past (autocorrelation), the relationships with drivers and changes over time. Each finding should translate into a modelling decision.

## 2. When to use it
After the index is regular ([preprocessing.md](preprocessing.md)), on the development period only (never the final test period). Revisit when residuals show unexplained structure.

## 3. Conceptual foundation
- **Time plot at several zoom levels:** the most important plot. Trend and breaks show at the full scale, seasonality at intermediate scales, noise and anomalies when zoomed in.
- **Seasonal plots / subseries plots:** the values grouped by season position (weekday, month) reveal the seasonal *shape* and whether it changes over the years.
- **Decomposition (STL):** y = trend + seasonal + remainder. STL (Seasonal-Trend decomposition using LOESS) is robust, handles any period and lets the seasonal component evolve. MSTL handles several seasonalities.
- **Rolling statistics:** a rolling mean and std over a window. A changing mean means trend or shift; a changing std means heteroscedasticity (consider log).
- **Autocorrelation function (ACF):** the correlation between y_t and y_{t−k} for each lag k. Spikes at multiples of m mean seasonality. A slowly decaying ACF means a trend or non-stationarity.
- **Partial ACF (PACF):** the correlation at lag k *after* removing the effects of lags 1…k−1. It helps identify the AR order. A sharp PACF cut-off after lag p suggests an AR(p) structure.
- **Stationarity tests:**
  - **ADF** (Augmented Dickey–Fuller): H₀ = unit root (non-stationary). A small p-value suggests stationarity.
  - **KPSS:** H₀ = (trend-)stationary. A small p-value suggests non-stationarity.
  - Use them **together and with plots**. ADF failing to reject *and* KPSS rejecting → difference. Conflicting results → the series may be trend-stationary or fractionally integrated; let the plots and validation decide.
- **Cross-correlation with drivers:** the correlation of y_t with x_{t−k} identifies leading drivers. Pre-whitening (removing each series' own autocorrelation first) is needed for clean inference. As a quick EDA it is fine with caution.

## 4. Assumptions and limitations
- ACF/PACF and stationarity tests assume a constant structure over the analysed window. Level shifts and breaks distort them (e.g. a break looks like a unit root).
- Statistical tests at large n flag small deviations, and at small n they lack power. They support decisions; they do not make them.
- Decomposition is descriptive. STL uses centred smoothing (future values), so its components must **not** be used directly as features for forecasting.

## 5. Step-by-step procedure
1. **Time plot at three scales**; mark known events (breaks, outages, interventions).
2. **Rolling mean/std** (window = the longest seasonality) to see trend and variance changes.
3. **Seasonal subseries plots** for each candidate period (weekday, month; hour for hourly data).
4. **STL/MSTL decomposition.** Assess the relative strength of trend and seasonality.
5. **ACF/PACF** of the series and of its differenced versions (d = 1, D = 1 at the seasonal lag).
6. **Stationarity tests** (ADF + KPSS) on the levels and the differences.
7. **Driver analysis:** scatter of y vs. temperature (non-linear?), the holiday effect, cross-correlation.
8. **Write EDA notes → decisions** (§6 template).

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import MSTL, STL
from statsmodels.tsa.stattools import adfuller, kpss

from cookbook_data import load_regular_energy_demand

df = load_regular_energy_demand()
dev = df.iloc[:-182]                       # EDA on the development period only
y = dev["demand_mwh"]

# 1–2. Time plot + rolling statistics
fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
y.plot(ax=ax[0], lw=0.5, label="daily"); y.rolling(28).mean().plot(ax=ax[0], label="28-day mean")
ax[0].axvline(pd.Timestamp("2022-06-01"), color="r", ls="--", label="new production line"); ax[0].legend()
y.rolling(28).std().plot(ax=ax[1], title="28-day rolling std")
plt.tight_layout(); plt.show()
```

```python
# 3. Seasonal subseries: weekday and month profiles (and whether they change by year)
tmp = dev.assign(weekday=dev.index.day_name().str[:3], month=dev.index.month, year=dev.index.year)
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
sns.boxplot(data=tmp, x="weekday", y="demand_mwh", order=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], ax=ax[0])
sns.lineplot(data=tmp, x="month", y="demand_mwh", hue="year", estimator="mean", errorbar=None, ax=ax[1], palette="viridis")
plt.tight_layout(); plt.show()
print(tmp.groupby("weekday")["demand_mwh"].mean().round(1))
```

```python
# 4. Decomposition: MSTL with weekly and yearly periods
mstl = MSTL(y, periods=(7, 365)).fit()
mstl.plot(); plt.show()
comp = pd.DataFrame({"trend": mstl.trend, "resid": mstl.resid})
seasonal = mstl.seasonal.sum(axis=1)
var = lambda s: np.var(s)
# Strength measures (Wang, Smith & Hyndman 2006; FPP3 §4.3): 1 - Var(R)/Var(component + R)
F_trend = max(0, 1 - var(comp["resid"]) / var(comp["trend"] + comp["resid"]))
F_season = {f"seasonal_{p}": max(0, 1 - var(comp["resid"]) / var(mstl.seasonal[f"seasonal_{p}"] + comp["resid"]))
            for p in (7, 365)}
print(f"trend strength={F_trend:.2f}", {k: round(v, 2) for k, v in F_season.items()})
```

```python
# 5. ACF / PACF of levels and of differenced series
fig, ax = plt.subplots(3, 2, figsize=(13, 9))
for row, (name, s) in enumerate([("level", y), ("diff(7)", y.diff(7)), ("diff(7).diff(1)", y.diff(7).diff())]):
    plot_acf(s.dropna(), lags=42, ax=ax[row, 0], title=f"ACF {name}")
    plot_pacf(s.dropna(), lags=42, ax=ax[row, 1], title=f"PACF {name}", method="ywm")
plt.tight_layout(); plt.show()
```

```python
# 6. Stationarity tests (use together; interpret with plots)
def stationarity_report(s, name):
    s = s.dropna()
    adf_p = adfuller(s, autolag="AIC")[1]
    kpss_p = kpss(s, regression="c", nlags="auto")[1]     # p-value is truncated to [0.01, 0.1] by statsmodels
    verdict = ("stationary" if adf_p < 0.05 and kpss_p > 0.05 else
               "non-stationary" if adf_p >= 0.05 and kpss_p <= 0.05 else "inconclusive")
    return {"series": name, "ADF p": round(adf_p, 4), "KPSS p": round(kpss_p, 3), "verdict": verdict}

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")                       # KPSS warns when p is outside its lookup table
    print(pd.DataFrame([stationarity_report(y, "level"), stationarity_report(y.diff(7), "diff(7)"),
                        stationarity_report(y.diff(7).diff(), "diff(7).diff(1)")]))
```

```python
# 7. Drivers: temperature (non-linear: heating & cooling), holidays, lagged correlation
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
weekday_mask = dev.index.dayofweek < 5
ax[0].scatter(dev.loc[weekday_mask, "temperature_c"], dev.loc[weekday_mask, "demand_mwh"], s=4, alpha=0.4)
ax[0].set(xlabel="temperature (°C)", ylabel="demand (weekdays)", title="V-shape: heating & cooling")
hol = dev.groupby("is_holiday")["demand_mwh"].mean()
ax[1].bar(["normal", "holiday"], hol.values); ax[1].set_title("mean demand")
plt.tight_layout(); plt.show()

# Correlation of demand with lagged temperature deviation from comfort (|T - 18|)
dev_t = (dev["temperature_c"] - 18).abs()
print({k: round(y.corr(dev_t.shift(k)), 3) for k in [0, 1, 2, 7]})
```

**EDA notes template (deliverable):**
```text
TS EDA NOTES — site demand (dev period 2020-01 .. 2023-06)
Trend      : upward (~+4.4 MWh/yr) + level shift +15% on 2022-06-01 → shift indicator or train on post-shift data
Seasonality: weekly strong (Sat −18%, Sun −30%); yearly via temperature (V-shape, min ≈ 15–22 °C)
Holidays   : −35% → holiday flag (known in advance)
Dependence : PACF of diff(7) shows short-memory AR structure (lags 1–2) → lags 1–7 useful; AR(1–2) noise
Stationarity: level non-stationary (trend/shift); diff(7) close to stationary → SARIMA with D=1 candidate
Variance   : roughly stable (Box-Cox λ≈1.3) → no log needed
Drivers    : temperature non-linear → features max(0, 15−T), max(0, T−22) or splines; same-day effect strongest
Next       : baselines (seasonal naive m=7), ETS (A,Ad,A), SARIMAX with temperature & holidays, GBM with lags + calendar + temperature
```

## 7. Interpretation
- **Rolling mean:** a gentle upward trend, a jump in June 2022 and a yearly oscillation. Models must handle the jump. Options: a level-shift indicator, training on the post-shift period only, or models that adapt their level (ETS, differencing).
- **Weekday box plots:** Saturday and Sunday are clearly lower, and the weekday profile is stable across years. Weekly seasonality is fixed, so day-of-week features or a seasonal period of 7 are enough.
- **Monthly lines by year:** the curves are shifted upwards over the years (trend) with the same shape. The yearly seasonality is stable and temperature-driven.
- **STL/MSTL strengths:** a weekly seasonal strength close to 1 means very strong weekly seasonality. A yearly strength well above 0 means a clear yearly pattern. Trend strength is high because of the growth and the shift.
- **ACF of the level:** high and slowly decaying, with spikes at 7, 14, 21… This is trend plus weekly seasonality. **ACF of diff(7):** decays quickly. Remaining short-lag correlation suggests AR terms or lags 1–2.
- **Stationarity:** the level is non-stationary, and diff(7) is likely stationary or borderline. A "inconclusive" verdict is common around breaks.
- **Temperature scatter:** a V-shape (heating below ~15 °C, cooling above ~22 °C). A linear temperature term would be wrong; use hinge features or splines. Same-day correlation is the strongest, so same-day temperature matters, and that requires a temperature *forecast* at prediction time.

## 8. Decision framework
| Finding | Modelling decision |
|---|---|
| Strong, stable seasonality at period m | Seasonal naive baseline; ETS/SARIMA with period m; ML with day-of-week or lag m features |
| Several seasonalities (daily + weekly + yearly) | MSTL-based models, Fourier terms, or ML with calendar features (ETS/SARIMA handle only one period well) |
| Seasonal amplitude grows with the level | Log / Box-Cox, or multiplicative ETS |
| Slowly decaying ACF | Trend or unit root → differencing (ARIMA) or trend features / differenced target (ML) |
| Level shift / break | Shift indicator, or train after the break; intervention variable in SARIMAX; validate around the break |
| Significant PACF at lags 1..p after seasonal differencing | AR(p) terms; lags 1..p as ML features |
| Clear driver relationship (non-linear) | Exogenous features with the right shape; check that the driver is known/forecastable at the origin |
| Variance changing over time | Transformation; interval methods that adapt |
| Remainder with outliers | Check against events; robust models/losses |

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| EDA including the test period | Choices tuned to the test period | Development period only |
| Using STL components as features | Look-ahead leakage (centred smoothing) | STL for understanding; past-only features for models |
| Trusting a single stationarity test | Wrong differencing decisions | ADF + KPSS + plots + validation |
| Ignoring breaks when reading ACF/tests | Spurious unit roots | Analyse sub-periods; mark breaks |
| Linear temperature effect | Misses the V-shape | Scatter plot first; hinge/spline features |
| Over-differencing | Introduces negative autocorrelation and inflates variance | Difference only as much as needed; check the ACF at lag 1 (≈ −0.5 suggests over-differencing) |

## 10. Checklist
- [ ] Time plots at 3 scales with events marked.
- [ ] Rolling mean/std.
- [ ] Seasonal subseries for each candidate period.
- [ ] STL/MSTL decomposition + strength measures.
- [ ] ACF/PACF on levels and differences.
- [ ] ADF + KPSS interpreted together.
- [ ] Drivers analysed (shape, lag, availability at the origin).
- [ ] EDA notes with decisions.

## 11. References
- Hyndman, R. J. & Athanasopoulos, G. (2021). *FPP3*, Ch. 2 "Time series graphics", Ch. 3 "Time series decomposition" (§3.6 STL, §4.3 strength features), §9.1 (stationarity, unit-root tests). https://otexts.com/fpp3/
- Cleveland, R. B., Cleveland, W. S., McRae, J. E. & Terpenning, I. (1990). "STL: A Seasonal-Trend Decomposition Procedure Based on Loess." *J. Official Statistics* 6(1), 3–73.
- Bandara, K., Hyndman, R. J. & Bergmeir, C. (2021). "MSTL: A Seasonal-Trend Decomposition Algorithm for Time Series with Multiple Seasonal Patterns." https://arxiv.org/abs/2107.13462
- statsmodels: `STL`, `MSTL`, `adfuller`, `kpss`, `plot_acf`, `plot_pacf`. https://www.statsmodels.org/stable/tsa.html
- Kwiatkowski, D., Phillips, P. C. B., Schmidt, P. & Shin, Y. (1992). "Testing the null hypothesis of stationarity against the alternative of a unit root." *J. Econometrics* 54, 159–178.
