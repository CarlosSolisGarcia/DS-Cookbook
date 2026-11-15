# Regression Error Analysis: Residuals, Heteroscedasticity and What to Fix Next

> **Folder:** 03_regression · **Prev:** [uncertainty.md](uncertainty.md) · **Next:** [iterative_improvement.md](../06_model_improvement/iterative_improvement.md) · **Related:** [evaluation.md](evaluation.md), [model_interpretation.md](../06_model_improvement/model_interpretation.md)

---

## 1. Purpose
Find **systematic** patterns in the errors and translate them into concrete next actions: a feature, a transformation, a different loss, a data fix or a different validation. Error analysis is the engine of iterative improvement.

## 2. When to use it
After every meaningful model evaluation, on out-of-fold (validation) predictions, **never** on training predictions. Use the test set only at the very end.

## 3. Conceptual foundation
Residual e = y − ŷ. For a well-specified model the residuals should look like **unstructured noise**:
- **Mean zero everywhere.** No trend against ŷ, against any feature, against time, or within any segment. A trend shows a missing effect (wrong functional form, a missing variable, an interaction).
- **Constant spread** (homoscedasticity). A funnel shape (spread growing with ŷ) indicates multiplicative noise. Consequences: RMSE is dominated by the high-level points, constant-width intervals are miscalibrated, and squared loss gives too much weight to the high-variance region.
- **No autocorrelation** for ordered data. Correlated residuals mean dynamics or drift have not been modelled.
- **Distribution:** heavy tails or skew in the residuals point to outliers or the need for a robust loss. For *prediction*, normality of the residuals is not required. For parametric intervals it matters.

| Plot | Reveals |
|---|---|
| Predicted vs. actual (with the y = x line) | Overall fit, shrinkage towards the mean, saturation at the extremes |
| Residuals vs. predicted | Non-linearity, heteroscedasticity (funnel) |
| Residuals vs. each feature (binned means) | Under-used features, wrong functional form |
| Residuals vs. time / order | Drift, autocorrelation, regime changes |
| Residuals by category (box plot) | Segment bias |
| Q–Q plot / histogram of residuals | Heavy tails, skew, outliers |
| Largest errors table | Data errors, rare conditions |

## 4. Assumptions and limitations
- Residual patterns found with many plots can be noise (multiple comparisons). Confirm them by fixing the pattern and checking the validation improvement.
- A model with regularisation or a log target shows some shrinkage ("over-predicts lows, under-predicts highs"). Mild shrinkage is expected and not a bug.
- Formal tests (Breusch–Pagan for heteroscedasticity, Ljung–Box for autocorrelation) are useful confirmations, but at large n they flag trivial deviations. Effect size matters more.

## 5. Step-by-step procedure
1. **Collect out-of-fold predictions** with all metadata (features, time, IDs).
2. **Global plots:** predicted vs. actual; residuals vs. predicted.
3. **Feature-wise residual plots** (binned means with CIs) for all important features, including features *not* in the model.
4. **Temporal plots:** residuals vs. time, rolling mean of the residuals, autocorrelation.
5. **Segment table:** bias and MAE per category.
6. **Worst cases:** inspect the 20 largest absolute errors and look for common factors.
7. **Formulate hypotheses** (one per pattern) and the corresponding action (§8). Prioritise by the error mass the pattern accounts for.
8. **Test the top hypothesis** in a tracked experiment and repeat.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from cookbook_data import load_clean_process_yield

df = load_clean_process_yield()
train = df.iloc[: int(0.8 * len(df))].copy()
T = "yield_kg"
# A deliberately simplistic model, to show what residual analysis reveals:
# temperature, catalyst and supplier omitted.
features = ["pressure_bar", "feed_rate_kg_h", "residence_time_min", "catalyst_age_days", "reactor"]
X = train[features].astype({"reactor": "category"})

def oof_predictions(model, X, y, cv):
    p = pd.Series(np.nan, index=X.index)
    for tr, va in cv.split(X):
        p.iloc[va] = model.fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[va])
    return p

model = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=0)
train["pred"] = oof_predictions(model, X, train[T], TimeSeriesSplit(5))
res = train.dropna(subset=["pred"]).assign(resid=lambda d: d[T] - d["pred"])
```

```python
# Global diagnostics
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].scatter(res["pred"], res[T], s=5, alpha=0.4)
lims = [res[[T, "pred"]].min().min(), res[[T, "pred"]].max().max()]
ax[0].plot(lims, lims, "k--"); ax[0].set(xlabel="predicted", ylabel="actual", title="pred vs actual")
ax[1].scatter(res["pred"], res["resid"], s=5, alpha=0.4); ax[1].axhline(0, c="k")
ax[1].set(xlabel="predicted", ylabel="residual", title="residuals vs predicted (funnel = heteroscedastic)")
from scipy import stats
stats.probplot(res["resid"], dist="norm", plot=ax[2]); ax[2].set_title("Q-Q plot of residuals")
plt.tight_layout(); plt.show()

# Quantify heteroscedasticity: spread of residuals by predicted-level bins
res["pred_bin"] = pd.qcut(res["pred"], 5)
print(res.groupby("pred_bin", observed=True)["resid"].agg(["mean", "std", "count"]).round(1))
```

```python
# Residuals vs features: INCLUDE features not in the model. A trend means signal is being left behind.
def residual_trend(res, col, bins=10):
    if res[col].dtype.kind in "OUSb" or str(res[col].dtype) in ("category", "str", "string"):
        g = res.groupby(col, observed=True)["resid"]
    else:
        g = res.groupby(pd.qcut(res[col], bins, duplicates="drop"), observed=True)["resid"]
    out = g.agg(["mean", "count"])
    out["se"] = g.std() / np.sqrt(out["count"])
    return out

for col in ["temperature_c", "catalyst", "impurity_ppm", "supplier", "feed_rate_kg_h"]:
    t = residual_trend(res, col)
    spread = t["mean"].max() - t["mean"].min()
    print(f"{col:16s} range of mean residual across bins/levels = {spread:6.1f} kg "
          f"(typical se ≈ {t['se'].median():.1f})")
```

```python
# Temporal structure of the residuals
from statsmodels.graphics.tsaplots import plot_acf
fig, ax = plt.subplots(1, 2, figsize=(13, 3.5))
res.set_index("start_time")["resid"].rolling(50).mean().plot(ax=ax[0], title="rolling mean residual (50 batches)")
ax[0].axhline(0, c="k")
plot_acf(res["resid"], lags=30, ax=ax[1]); plt.tight_layout(); plt.show()

# Formal heteroscedasticity check (Breusch–Pagan) on residuals vs predicted value
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm
lm, lm_p, f, f_p = het_breuschpagan(res["resid"], sm.add_constant(res[["pred"]]))
print(f"Breusch–Pagan p-value: {lm_p:.2g}  (small → variance depends on predicted level)")
```

```python
# Worst cases: look for common factors
worst = res.reindex(res["resid"].abs().sort_values(ascending=False).index).head(15)
print(worst[["start_time", "reactor", "catalyst", "supplier", "temperature_c", T, "pred", "resid"]].round(1))
print(worst["catalyst"].value_counts(normalize=True).round(2), "\nvs overall:\n",
      res["catalyst"].value_counts(normalize=True).round(2))
```

## 7. Interpretation
For the deliberately incomplete model above you should find:
- **A residual trend vs. `temperature_c` and `catalyst`** (features not in the model). Mean residuals differ by tens of kg across bins, far beyond the standard errors. **Action:** add temperature and catalyst, ideally the catalyst-specific optimum feature ([feature_engineering.md](../02_preprocessing/feature_engineering.md)).
- **A residual trend vs. `supplier` / `impurity_ppm`:** real effects. The supplier range is large, but each level has few rows (larger standard errors), so part of that range is noise. **Action:** add them, with a missing indicator for impurity and rare-level grouping for supplier.
- **A funnel in residuals vs. predicted**, and a Breusch–Pagan p-value ≪ 0.05: multiplicative noise. **Actions:** a log target or gamma loss ([transformations.md](../02_preprocessing/transformations.md)); adaptive prediction intervals ([uncertainty.md](uncertainty.md)); MAE or relative metrics alongside RMSE.
- **The rolling mean residual oscillates in time:** some temporal effect is not captured. Here it is partly catalyst-replacement cycles, which `catalyst_age_days` should capture. If not, check the feature or add interactions.
- **Heavy-tailed Q–Q plot:** a few large errors. Inspect the worst cases. Are they data errors (fix them) or rare real conditions (robust loss, or a feature that identifies them)?

General reading rules:
- **Pattern in residuals vs. an included feature** → wrong functional form or a missing interaction with that feature.
- **Pattern vs. an excluded feature** → include it (if it is available at t₀).
- **Pattern vs. time** → drift or missing dynamics. Consider time-aware features and shorter training windows.
- **No pattern anywhere, error ≈ known measurement noise** → you are near the noise floor. Stop modelling; improve the data.

## 8. Decision framework
| Residual pattern | Likely cause | Action |
|---|---|---|
| Curve in residuals vs. ŷ | Wrong functional form | Transformations, splines, a flexible model |
| Funnel (spread ∝ level) | Multiplicative / heteroscedastic noise | Log target / gamma loss; adaptive intervals; relative metrics |
| Trend vs. a feature | Under-used feature | Transform it, add interactions, or a flexible model |
| Trend vs. an omitted variable | Missing feature | Add it (check availability at t₀) |
| Bias in one category | Segment effect not captured | Category feature / interaction; check segment data quality |
| Drift over time | Process change, ageing, drift | Time features; retraining strategy; chronological validation |
| Autocorrelated residuals | Unmodelled dynamics | Lag features of the inputs (past only); time-series methods |
| Large isolated errors | Data errors or rare regimes | Inspect; fix the data; robust loss; regime flag |
| Pred vs. actual flattened at the extremes | Over-regularisation, tree saturation, or a missing driver of extremes | Less regularisation; features for extreme conditions; check extrapolation |
| No structure, error ≈ noise level | Near the irreducible error | Stop tuning; consider better measurements |

Prioritise patterns by **error mass**: the fraction of the total absolute error coming from the affected rows × the size of the bias.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Analysing training-set residuals | Patterns hidden by overfitting | Out-of-fold predictions |
| Only looking at features in the model | Missing omitted variables | Plot residuals vs. all candidate variables |
| Fixing every pattern with model complexity | Overfitting, opacity | Prefer features / data fixes |
| Ignoring time | Drift unnoticed | Always plot residuals vs. time |
| Using the test set for error analysis | Test contamination | Validation predictions only |

## 10. Checklist
- [ ] Out-of-fold predictions with metadata.
- [ ] Pred vs. actual; residuals vs. predicted; Q–Q.
- [ ] Residuals vs. all candidate features (in and out of the model).
- [ ] Residuals vs. time; autocorrelation.
- [ ] Segment bias table.
- [ ] Worst cases inspected.
- [ ] Hypotheses → actions, prioritised by error mass.
- [ ] Next experiment logged.

## 11. References
- Kutner, M., Nachtsheim, C., Neter, J. & Li, W. (2005). *Applied Linear Statistical Models*, 5th ed., Ch. 3 "Diagnostics and Remedial Measures".
- Fox, J. (2016). *Applied Regression Analysis and Generalized Linear Models*, 3rd ed., Ch. 11–12 (diagnostics).
- statsmodels, *Regression diagnostics* (`het_breuschpagan`, `acorr_ljungbox`). https://www.statsmodels.org/stable/diagnostic.html
- scikit-learn, `PredictionErrorDisplay`. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PredictionErrorDisplay.html
- Ng, A. *Machine Learning Yearning*, chapters on error analysis (free draft). https://info.deeplearning.ai/machine-learning-yearning-book
