# Feature Selection and Multicollinearity (Supervised, Tabular)

> **Folder:** 06_model_improvement · **Prev:** [hyperparameter_optimization.md](hyperparameter_optimization.md) · **Next:** [model_interpretation.md](model_interpretation.md) · **Related:** [11_dimensionality_reduction](../11_dimensionality_reduction/) (PCA, extraction, very high-dimensional data), [exploratory_data_analysis.md](../01_data_understanding/exploratory_data_analysis.md) (VIF), [data_leakage.md](../00_methodology/data_leakage.md)

---

## 1. Purpose
Decide **whether** to remove features and, if so, which ones and how. The aim is to improve generalisation, robustness, interpretability or cost, without leaking information and without discarding useful signal because of correlation alone.

## 2. When to use it
- Many features relative to the number of rows (p comparable to n, or p ≫ n).
- Features are expensive to obtain or maintain in production.
- Interpretability requires a compact model.
- Correlated features make linear coefficients unstable.
- **Often unnecessary** for regularised linear models and GBMs with moderate p. They handle redundancy reasonably well, and selection mainly adds risk and work.

## 3. Conceptual foundation
| Family | How | Examples (scikit-learn) | Pros | Cons |
|---|---|---|---|---|
| **Filter** | Score each feature independently of the model | Variance threshold, correlation, `mutual_info_*`, `f_regression`, `SelectKBest` | Fast | Ignores interactions and redundancy; univariate |
| **Wrapper** | Search feature subsets using the model's CV score | `RFECV`, `SequentialFeatureSelector` | Model-aware | Expensive; high variance; overfits the selection without nesting |
| **Embedded** | Selection happens during fitting | Lasso/ElasticNet (L1), tree importances + `SelectFromModel` | Efficient | Model-specific; L1 is unstable with correlated features |
| **Permutation-based** | Drop features whose permutation does not hurt validation performance | `permutation_importance` | Model-agnostic; measured on validation data | Correlated features share importance (each looks unimportant) |

**Multicollinearity:** strong linear dependence among features.
- **Effect on prediction:** usually small, provided the correlation structure persists in production.
- **Effect on interpretation:** large. Linear coefficients become unstable (they can flip sign) and importance is split between correlated features.
- **Diagnostics:** the correlation matrix, VIF (variance inflation factor, 1/(1 − R²_j)), and the condition number.
- **Options:** keep everything with regularisation (ridge); keep one representative per correlated cluster (domain choice); combine the cluster into one feature (average, physical index, first principal component); or explain at the cluster level (grouped permutation importance).

**Selection must be inside the CV loop.** Selecting on all the data, then cross-validating, is a classic leakage (see the noise demonstration in [data_leakage.md](../00_methodology/data_leakage.md) §6.2).

## 4. Assumptions and limitations
- Selection results are **unstable**: different samples select different subsets, especially with correlated features. Check stability across folds before drawing conclusions ("feature X is irrelevant").
- "Not selected" ≠ "no relationship with the target". It means "not needed given the other features" for this model.
- Removing features changes the meaning of the remaining coefficients.

## 5. Step-by-step procedure
1. **Remove junk:** constant/near-constant columns, identifiers, duplicates, leakage columns.
2. **Diagnose redundancy:** Spearman correlation clusters; VIF if a linear model will be interpreted.
3. **Decide the objective:** accuracy, cost or interpretability. If it is accuracy, first check whether regularisation alone suffices.
4. **Choose a method** (§8), implemented inside a pipeline.
5. **Evaluate selection as part of the model** with CV (nested if the number of features is tuned).
6. **Check stability:** how often is each feature selected across folds/bootstraps?
7. **Decide.** Keep a reduced set only if it is as good (within the noise) and simpler, or if it is required for cost or interpretability.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFECV, SelectFromModel, SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cookbook_data import load_clean_process_yield

df = load_clean_process_yield()
rng = np.random.default_rng(0)
# Add 20 pure-noise features to make the selection problem visible
noise = pd.DataFrame(rng.normal(size=(len(df), 20)), columns=[f"noise_{i}" for i in range(20)])
df = pd.concat([df, noise], axis=1)
df["log_feed"] = np.log(df["feed_rate_kg_h"])
df["temp_dev_sq"] = (df["temperature_c"] - df["catalyst"].map({"Pt": 190, "Pd": 182, "Ni": 195})) ** 2
real = ["temperature_c", "jacket_temperature_c", "temp_dev_sq", "pressure_bar", "log_feed",
        "residence_time_min", "catalyst_age_days", "impurity_ppm", "feed_purity_pct", "ambient_humidity_pct"]
features = real + list(noise.columns)
X, y = df[features], np.log(df["yield_kg"])
cv = KFold(5, shuffle=True, random_state=42)
prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

def cv_mae(model):
    s = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
    return f"{-s.mean():.4f} ± {s.std():.4f}"

print("ridge, all 30 features      :", cv_mae(make_pipeline(prep, RidgeCV(alphas=np.logspace(-3, 3, 13)))))
print("ridge, 10 real features only :", cv_mae(make_pipeline(
    ColumnTransformer([("r", prep, real)]), RidgeCV(alphas=np.logspace(-3, 3, 13)))))
```

```python
# Filter (mutual information), embedded (Lasso) and wrapper (RFECV): all INSIDE the pipeline
pipes = {
    "MI top-10 → ridge": make_pipeline(prep, SelectKBest(lambda X_, y_: mutual_info_regression(X_, y_, random_state=0), k=10),
                                       RidgeCV(alphas=np.logspace(-3, 3, 13))),
    "Lasso-selected → ridge": make_pipeline(prep, SelectFromModel(LassoCV(cv=5, random_state=0)),
                                            RidgeCV(alphas=np.logspace(-3, 3, 13))),
    "RFECV(ridge) → ridge": make_pipeline(prep, RFECV(RidgeCV(alphas=np.logspace(-3, 3, 13)), step=2, cv=3,
                                                      scoring="neg_mean_absolute_error"),
                                          RidgeCV(alphas=np.logspace(-3, 3, 13))),
}
for name, p in pipes.items():
    print(f"{name:24s}", cv_mae(p))
```

```python
# Stability of selection across folds: how often is each feature kept by the Lasso?
counts = pd.Series(0, index=features)
for tr, _ in cv.split(X):
    sel = make_pipeline(prep, SelectFromModel(LassoCV(cv=5, random_state=0))).fit(X.iloc[tr], y.iloc[tr])
    counts += sel[-1].get_support().astype(int)
print(counts.sort_values(ascending=False).head(14))
```

```python
# Multicollinearity: correlation clusters + one representative per cluster
corr = np.array(X[real].corr(method="spearman").abs().fillna(0), copy=True)   # writable copy (pandas 3 returns read-only views)
np.fill_diagonal(corr, 1.0)
dist = squareform(1 - corr, checks=False)
link = hierarchy.linkage(dist, method="average")
clusters = pd.Series(hierarchy.fcluster(link, t=0.3, criterion="distance"), index=real)   # |ρ| > 0.7 → same cluster
print(clusters.sort_values().groupby(clusters.sort_values()).apply(list).to_dict())

# Grouped permutation importance: permute a whole correlated cluster together
model = make_pipeline(prep, RidgeCV(alphas=np.logspace(-3, 3, 13))).fit(X.iloc[:1500], y.iloc[:1500])
Xv, yv = X.iloc[1500:].copy(), y.iloc[1500:]
base_err = np.mean(np.abs(yv - model.predict(Xv)))
for cid, cols in clusters.groupby(clusters).groups.items():
    Xp = Xv.copy()
    perm = rng.permutation(len(Xp))
    Xp[list(cols)] = Xp[list(cols)].to_numpy()[perm]          # same permutation for all columns of the cluster
    print(f"cluster {list(cols)}: ΔMAE = {np.mean(np.abs(yv - model.predict(Xp))) - base_err:+.4f}")
```

## 7. Interpretation
- **Ridge with 30 vs 10 features:** MAE(log) 0.146 vs 0.145, a negligible difference. Ridge shrinks the 20 noise coefficients, so selection is not needed for accuracy here.
- **Lasso and RFECV pipelines** match the "real features only" result. **MI top-10 is clearly worse** (0.178). Mutual information ranks features one at a time and missed features that matter only in combination, or it kept redundant pairs. Univariate filters are the riskiest selectors.
- **Selection stability (Lasso, 5 folds):** the true drivers (pressure, `temp_dev_sq`, `log_feed`, residence time, impurity, catalyst age) are selected in 5/5 folds. The correlated pairs behave as expected: `feed_purity_pct` is kept in 5/5 folds as a near-duplicate of impurity, while `temperature_c` and `jacket_temperature_c` appear only once, because `temp_dev_sq` already carries the temperature information. **`noise_17` is selected in 5/5 folds, a false discovery** that happens to correlate with the target in this sample. With 20 noise features this is expected. That is why "selected" must never be read as "real", and why stability selection and domain review matter.
- **Correlation clusters and grouped permutation:** {impurity, feed purity} and {temperature, jacket temperature} form clusters. Permuting the whole impurity cluster costs ΔMAE ≈ 0.03. Permuting the temperature cluster costs ≈ 0, because the model uses temperature through `temp_dev_sq` (its own cluster, ΔMAE ≈ 0.57). Importance therefore belongs to the *information*, here encoded in the engineered feature, not to the raw column.

## 8. Decision framework
| Goal / situation | Approach |
|---|---|
| Accuracy, p ≪ n, regularised or tree model | Usually no selection; remove junk/leakage only |
| p ≈ n or p ≫ n (spectra, omics, wide sensor sets) | L1/ElasticNet, or extraction (PCA/PLS; [11_dimensionality_reduction](../11_dimensionality_reduction/)), inside nested CV |
| Production cost per feature | Wrapper/embedded with a cost-aware stopping rule; validate the reduced model |
| Interpretable linear model | Domain-driven: one representative per correlated cluster, or a combined index; check VIF |
| Correlated features and importance questions | Grouped (cluster-level) permutation importance |
| Need a ranked list for stakeholders | Permutation importance on validation data + stability across folds |
| Many noise features suspected | Compare the CV with vs without selection (nested); Boruta-like shadow features (optional) |

- **IF** selection does not improve CV beyond the noise **AND** there is no cost/interpretability need → do not select.
- **IF** you must present "the important features" → present stability frequencies and cluster-level importance, not a single subset.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Selecting on all data, then CV | Optimistic scores, even on noise | Selection inside the pipeline (nested) |
| Dropping correlated features automatically (VIF > 10) | Loses information; arbitrary choice | Decide by goal and domain |
| Interpreting "not selected" as "irrelevant" | Wrong scientific conclusions | Stability analysis; cluster-level importance |
| Using impurity-based tree importance to select | Biased towards high-cardinality / continuous features | Permutation importance on validation data |
| Univariate filters only | Miss interaction-only features | Model-based checks |

## 10. Checklist
- [ ] Junk and leakage columns removed.
- [ ] Redundancy clusters identified (Spearman, VIF if needed).
- [ ] Objective of the selection stated (accuracy / cost / interpretability).
- [ ] Selection inside the pipeline; evaluated with CV (nested if tuned).
- [ ] Stability across folds reported.
- [ ] Reduced model kept only if within the noise of the full model and simpler, or if required.

## 11. References
- Guyon, I. & Elisseeff, A. (2003). "An Introduction to Variable and Feature Selection." *JMLR* 3, 1157–1182. https://jmlr.org/papers/v3/guyon03a.html
- scikit-learn User Guide, *Feature selection*. https://scikit-learn.org/stable/modules/feature_selection.html
- scikit-learn, *Permutation Importance with Multicollinear or Correlated Features*. https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html
- Kuhn, M. & Johnson, K. (2019). *Feature Engineering and Selection*, Ch. 10–12. http://www.feat.engineering/
- Meinshausen, N. & Bühlmann, P. (2010). "Stability selection." *JRSS B* 72(4), 417–473.
- Ambroise, C. & McLachlan, G. J. (2002). "Selection bias in gene extraction on the basis of microarray gene-expression data." *PNAS* 99(10), 6562–6566. https://doi.org/10.1073/pnas.102102699
