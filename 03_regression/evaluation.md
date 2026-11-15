# Regression Evaluation: Metrics, Uncertainty of Scores and Model Comparison

> **Folder:** 03_regression · **Prev:** [model_selection.md](model_selection.md) · **Next:** [uncertainty.md](uncertainty.md) · **Related:** [metrics_reference.md](../08_quick_references/metrics_reference.md), [error_analysis.md](error_analysis.md), [cross_validation.md](../06_model_improvement/cross_validation.md)

---

## 1. Purpose
Measure how good a regression model is **in the units and terms that matter**, quantify how certain that measurement is, and compare models without being fooled by noise or by metric artefacts.

## 2. When to use it
At every model comparison, and for the final test-set report. Residual *diagnostics* (patterns, heteroscedasticity) are in [error_analysis.md](error_analysis.md). *Prediction* intervals are in [uncertainty.md](uncertainty.md).

## 3. Conceptual foundation

| Metric | Formula (intuition) | Units | Sensitive to | Hides | Use when |
|---|---|---|---|---|---|
| **MAE** | mean \|y − ŷ\| | target | Typical error; robust | Rare large errors | Costs roughly proportional to the error size; outliers present |
| **MSE** | mean (y − ŷ)² | target² | Large errors (quadratically) | Interpretability (squared units) | Optimisation; large errors costly |
| **RMSE** | √MSE | target | Large errors | Typical error when the tail is heavy | Large errors costly; comparable to the std of y |
| **R²** | 1 − SSE/SST | — | Variance explained relative to the mean baseline | Absolute error size; depends on the variance of the test set | Communicating relative skill; same dataset only |
| **MAPE** | mean \|y − ŷ\|/\|y\| | % | Relative error | — | All y well away from zero, relative errors matter |
| **WAPE** | Σ\|y − ŷ\| / Σ\|y\| | % | Relative error weighted by volume | Per-item relative errors | Aggregated volumes (production, demand) |
| **Median AE / P90 AE** | quantiles of \|e\| | target | Distribution of errors | — | "90% of batches within ±X kg" |
| **Pinball loss (α)** | asymmetric absolute | target | Quantile accuracy | — | Quantile predictions/intervals |

**Key facts:**
- MAE is minimised by the **median**, MSE by the **mean**. Comparing a median-predicting model with a mean-predicting model on one metric favours one of them by construction.
- RMSE ≥ MAE always. The ratio RMSE/MAE indicates how heavy-tailed the errors are: about 1.25 for normal errors, larger with occasional big misses.
- R² is **not comparable across datasets.** The same model has a higher R² on a test set with more spread in y. A temporal test set covering a narrow range can show a low R² with a small MAE.
- **MAPE limitations:** it is undefined at y = 0, explodes near zero, and is asymmetric: an over-prediction can produce an error far above 100%, while an under-prediction is capped at 100%. Minimising MAPE therefore favours under-prediction. Use WAPE, or MAE on the original scale.

**Uncertainty of a metric.** A test MAE computed from 400 batches is itself an estimate. Report a confidence interval, via a bootstrap over test rows (or blocks, for temporal data), or the spread across CV folds.

**Comparing two models.** Compare them **on the same rows**, using paired per-row differences in loss. A bootstrap of the mean difference is much more sensitive than comparing two separate intervals.

## 4. Assumptions and limitations
- The bootstrap assumes rows are exchangeable. For temporal or grouped data, resample blocks or groups.
- CV fold scores are correlated (overlapping training sets), so the std across folds underestimates the true uncertainty somewhat. Use it as a practical noise floor, not as an exact standard error.
- No single metric captures everything. Choose one **primary** metric for decisions and report secondary ones for context.

## 5. Step-by-step procedure
1. **Primary metric** from the problem statement (cost structure). Secondary metrics for the tail (P90 AE, RMSE) and for relative size (WAPE).
2. **Validation scores:** mean ± std across folds, for every candidate, on identical folds.
3. **Paired comparison** of the top candidates (per-row loss differences on out-of-fold predictions).
4. **Segment evaluation:** metrics per reactor, supplier, time period and target range. A good overall score can hide a failing segment.
5. **Final test:** compute the metrics once, with bootstrap CIs. Compare with the CV estimate.
6. **Communicate in business terms:** "median error ±12 kg per batch; 90% of batches within ±35 kg; 40% lower error than the current rule".

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, median_absolute_error, r2_score, root_mean_squared_error)
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from cookbook_data import load_clean_process_yield

df = load_clean_process_yield()
cut = int(0.8 * len(df))
train, test = df.iloc[:cut], df.iloc[cut:]
T = "yield_kg"

def domain_features(d):
    d = d.copy()
    d["log_feed"] = np.log(d["feed_rate_kg_h"] * 8.0)
    d["log_res"] = np.log(d["residence_time_min"])
    d["temp_dev_sq"] = (d["temperature_c"] - d["catalyst"].map({"Pt": 190.0, "Pd": 182.0, "Ni": 195.0})) ** 2
    return d

lin_num = ["log_feed", "log_res", "temp_dev_sq", "pressure_bar", "catalyst_age_days", "impurity_ppm"]
raw_num = ["temperature_c", "pressure_bar", "feed_rate_kg_h", "residence_time_min", "catalyst_age_days", "impurity_ppm"]
cat = ["reactor", "catalyst", "shift", "supplier"]

ridge = TransformedTargetRegressor(make_pipeline(
    FunctionTransformer(domain_features),
    ColumnTransformer([("n", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), lin_num),
                       ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat)]),
    RidgeCV(alphas=np.logspace(-3, 3, 13))), func=np.log, inverse_func=np.exp)
hgb = make_pipeline(FunctionTransformer(lambda d: d[raw_num + cat].astype({c: "category" for c in cat})),
                    HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=0))

def regression_report(y, p) -> dict:
    e = np.asarray(y) - np.asarray(p)
    return {"MAE": mean_absolute_error(y, p), "RMSE": root_mean_squared_error(y, p),
            "MedAE": median_absolute_error(y, p), "P90_AE": np.quantile(np.abs(e), 0.9),
            "R2": r2_score(y, p), "WAPE_%": 100 * np.abs(e).sum() / np.abs(y).sum(),
            "MAPE_%": 100 * mean_absolute_percentage_error(y, p), "bias(mean e)": e.mean()}

# Out-of-fold predictions on the training period: TimeSeriesSplit does not cover the first block,
# so collect predictions fold by fold.
cv = TimeSeriesSplit(5)
oof = {}
for name, m in [("ridge", ridge), ("hgb", hgb)]:
    p = pd.Series(np.nan, index=train.index)
    for tr, va in cv.split(train):
        m.fit(train.iloc[tr], train[T].iloc[tr])
        p.iloc[va] = m.predict(train.iloc[va])
    oof[name] = p
mask = oof["ridge"].notna()
print(pd.DataFrame({k: regression_report(train.loc[mask, T], v[mask]) for k, v in oof.items()}).round(2))
```

```python
# Paired bootstrap of the difference in MAE (ridge - hgb) on the same rows.
# For temporal data, a block bootstrap is more honest (block = 20 consecutive batches).
def paired_block_bootstrap(y, p1, p2, block=20, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.abs(y - p1) - np.abs(y - p2)                      # per-row loss difference
    n_blocks = int(np.ceil(len(d) / block))
    starts = np.arange(0, len(d) - block + 1)
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, n_blocks)])[: len(d)]
        stats.append(d[idx].mean())
    return d.mean(), np.percentile(stats, [2.5, 97.5])

yv = train.loc[mask, T].to_numpy()
diff, ci = paired_block_bootstrap(yv, oof["ridge"][mask].to_numpy(), oof["hgb"][mask].to_numpy())
print(f"MAE(ridge) - MAE(hgb) = {diff:.2f} kg, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]")
```

```python
# Segment evaluation: where is the model good or bad?
res = train.loc[mask].assign(pred=oof["ridge"][mask], abs_err=lambda d: (d[T] - d["pred"]).abs())
res["yield_band"] = pd.qcut(res[T], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
for seg in ["reactor", "catalyst", "yield_band"]:
    g = res.groupby(seg, observed=True).agg(n=("abs_err", "size"), MAE=("abs_err", "mean"),
                                            bias=("pred", lambda p: (p - res.loc[p.index, T]).mean()))
    print(g.round(2), "\n")
```

```python
# Final test evaluation (once), with a block-bootstrap CI for MAE
final = ridge.fit(train, train[T])
pt = final.predict(test)
rep = regression_report(test[T], pt)
e = np.abs(test[T].to_numpy() - pt)
rng = np.random.default_rng(1)
boots = [e[np.concatenate([np.arange(s, s + 20) for s in rng.choice(len(e) - 20, len(e) // 20 + 1)])[: len(e)]].mean()
         for _ in range(2000)]
print({k: round(v, 2) for k, v in rep.items()})
print("test MAE 95% CI:", np.percentile(boots, [2.5, 97.5]).round(2))
```

## 7. Interpretation
- **RMSE/MAE ratio:** about 1.25–1.4 means fairly regular errors. A much higher ratio means a few large misses dominate RMSE. Find them in [error_analysis.md](error_analysis.md).
- **MAPE vs WAPE:** if MAPE ≫ WAPE, the relative errors are large on *small* targets. Check whether small batches matter for the business.
- **Bias (mean error) ≠ 0.** A systematic over- or under-prediction. For log-target models a positive bias (y > ŷ on average) is expected; see the smearing correction in [transformations.md](../02_preprocessing/transformations.md).
- **Paired bootstrap CI excludes 0:** the difference is real for this data distribution. A CI including 0 means "not distinguishable"; choose by simplicity.
- **Segments:** in-sample, larger absolute errors in high-yield bands are expected with multiplicative noise, so check the *relative* error too. A segment with high bias (not just high MAE) suggests a missing feature for that segment.
- **Test vs CV:** a test MAE within the CV range is consistent. A test MAE much worse means the validation did not mimic deployment, or there is drift in the test period.

## 8. Decision framework
| Situation | Metric decision |
|---|---|
| Costs ∝ error size, outliers present | Primary MAE; report P90 AE |
| Large errors especially costly (safety, penalties) | Primary RMSE (or a custom cost); analyse the worst cases |
| Targets span orders of magnitude; relative error matters | WAPE, or MAE on the log scale (≈ relative error); avoid plain MAPE near zero |
| Asymmetric costs (under-prediction worse) | Pinball loss at a suitable quantile; quantile model ([uncertainty.md](uncertainty.md)) |
| Need to explain skill to non-experts | R² relative to the mean *and* % improvement over the informed baseline |
| Two models within the paired CI | Choose the simpler / more robust / more interpretable |
| Segment failing badly | Error analysis on that segment before further tuning |

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Comparing R² across datasets/periods | Misleading conclusions | Use MAE/RMSE in units; R² only within one dataset |
| MAPE with near-zero targets | Exploding, unstable metric | WAPE / MAE |
| Reporting test metrics without uncertainty | Over-confident claims | Bootstrap CIs (block/group-aware) |
| Comparing models with unpaired intervals | Missing real differences | Paired per-row differences |
| Only aggregate metrics | Failing segments hidden | Segment table every evaluation |
| Metric on the log scale when the business needs units | Wrong model choice | Metrics on the original scale |

## 10. Checklist
- [ ] Primary metric aligned with costs; secondary metrics for tail and relative size.
- [ ] CV mean ± std for all candidates on identical folds.
- [ ] Paired comparison of the top candidates.
- [ ] Segment metrics (categories, time, target range).
- [ ] Bias checked.
- [ ] Final test metrics once, with CIs, compared with CV.
- [ ] Results communicated in business units and relative to the baseline.

## 11. References
- scikit-learn User Guide, *Regression metrics*. https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
- Hyndman, R. J. & Koehler, A. B. (2006). "Another look at measures of forecast accuracy." *IJF* 22(4), 679–688. Problems with MAPE. https://doi.org/10.1016/j.ijforecast.2006.03.001
- Gneiting, T. (2011). "Making and Evaluating Point Forecasts." *JASA* 106(494). https://doi.org/10.1198/jasa.2011.r10138
- Efron, B. & Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Künsch, H. R. (1989). "The Jackknife and the Bootstrap for General Stationary Observations." *Annals of Statistics* 17(3). The block bootstrap.
