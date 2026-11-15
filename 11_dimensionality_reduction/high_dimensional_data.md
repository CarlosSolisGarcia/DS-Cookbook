# High-Dimensional Data (p ≫ n): Regularisation, Selection, Mutual Information and Multicollinearity

> **Folder:** 11_dimensionality_reduction · **Prev:** [pca.md](pca.md) · **Next:** [visualization_embeddings.md](visualization_embeddings.md) · **Related:** [feature_selection.md](../06_model_improvement/feature_selection.md) (method families, stability, grouped importance), [small_data_and_uncertainty.md](../18_scientific_machine_learning/small_data_and_uncertainty.md), [cross_validation.md](../06_model_improvement/cross_validation.md) (nested CV)

---

## 1. Purpose
Build reliable models when the number of features is comparable to or larger than the number of samples (spectra, omics, wide sensor sets, text, molecular descriptors), and avoid the false discoveries that high dimensionality makes easy.

## 2. When to use it
p ≳ n/10 in general, and certainly p > n. Also when many candidate features are generated automatically (descriptor libraries, lag/rolling feature grids).

## 3. Conceptual foundation
- **The curse of dimensionality:** volume grows exponentially with p, so the data become sparse. Distances concentrate (nearest and farthest neighbours become almost equidistant), and *some* feature will correlate with the target by chance. With 1000 noise features and 50 samples, the best one typically reaches |r| ≈ 0.5.
- **Ill-posedness:** with p > n, infinitely many models fit the training data exactly. Generalisation therefore depends entirely on the **inductive bias** (regularisation, low-rank structure, sparsity).
- **Three families of remedies:**
  1. **Shrinkage:** ridge (dense, stable with correlated features), lasso (sparse, unstable), elastic net (a compromise).
  2. **Latent structure:** PCR/PLS ([pca.md](pca.md)).
  3. **Selection:** filters (MI, correlation), wrappers (RFE) and embedded methods ([feature_selection.md](../06_model_improvement/feature_selection.md)). They **must** be nested inside CV.
- **Mutual information (MI):** a non-parametric measure of dependence (captures non-linear relationships). It is useful as a filter. Its estimates are noisy for small n and it ignores interactions.
- **Multicollinearity in p ≫ n:** VIF is undefined (the design matrix is singular). Think in terms of groups/blocks of correlated features (spectral regions, sensor groups) and of **grouped** importance and selection.

## 4. Assumptions and limitations
- Selection results with p ≫ n are notoriously unstable. Report frequencies across resamples, and avoid strong claims about individual features.
- Nested CV is mandatory for honest estimates when selecting or tuning. Non-nested estimates can be wildly optimistic.
- Small n also means noisy performance estimates. Use repeated CV and report the spread ([small_data_and_uncertainty.md](../18_scientific_machine_learning/small_data_and_uncertainty.md)).

## 5. Step-by-step procedure
1. **Establish a baseline** (mean predictor) and a **shrinkage model** (ridge with CV) inside a pipeline.
2. **Try latent-variable models** (PLS) and sparse models (lasso/elastic net).
3. **If selection is needed** (cost, interpretation): MI or univariate filters → model, *inside* the pipeline; or RFE with a linear model. Evaluate with **nested** CV.
4. **Assess stability:** selection frequency over bootstrap/CV resamples; prefer blocks/regions over single features.
5. **Run a noise sanity check:** add permuted (shadow) copies of features, or permute y. Selected "signal" should beat shadows clearly, and the permuted-y performance should be ≈ the baseline.
6. **Interpret at the level of groups**, with domain knowledge (spectral bands, sensor subsystems).

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import RFECV, SelectKBest, mutual_info_regression, f_regression
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from cookbook_data import make_nir_spectra

X, t = make_nir_spectra(n=80)                   # fewer samples: p = 400 ≫ n = 80
y = t["analyte_pct"].to_numpy()
wl = np.array([int(c.split("_")[1]) for c in X.columns])
outer = RepeatedKFold(n_splits=5, n_repeats=3, random_state=0)
def score(model):
    s = -cross_val_score(model, X, y, cv=outer, scoring="neg_root_mean_squared_error", n_jobs=-1)
    return f"{s.mean():.3f} ± {s.std():.3f}"

print("baseline (std of y):", round(y.std(), 3))
inner = KFold(5, shuffle=True, random_state=1)
models = {
    "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 40))),
    "lasso": make_pipeline(StandardScaler(), LassoCV(cv=inner, max_iter=50000, random_state=0)),
    "elastic net": make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=inner, max_iter=50000, random_state=0)),
    "PLS (k tuned, nested)": GridSearchCV(make_pipeline(StandardScaler(), PLSRegression(scale=False)),
                                          {"plsregression__n_components": range(1, 11)}, cv=inner,
                                          scoring="neg_root_mean_squared_error"),
    "MI top-30 → ridge (nested)": make_pipeline(StandardScaler(),
        SelectKBest(lambda A, b: mutual_info_regression(A, b, random_state=0), k=30), RidgeCV(alphas=np.logspace(-4, 4, 40))),
}
for name, m in models.items():
    print(f"{name:28s} RMSE = {score(m)}")
```

```python
# Selection stability: which wavelengths does the lasso pick across resamples?
freq = np.zeros(X.shape[1])
for i, (tr, _) in enumerate(KFold(10, shuffle=True, random_state=2).split(X)):
    m = make_pipeline(StandardScaler(), LassoCV(cv=5, max_iter=50000, random_state=0)).fit(X.iloc[tr], y[tr])
    freq += (m[-1].coef_ != 0)
freq /= 10
sel = pd.Series(freq, index=wl)
print("wavelengths selected in ≥ 80% of resamples:", sel[sel >= 0.8].index.tolist())
print("selected in 10–70% (unstable):", int(((sel > 0) & (sel < 0.8)).sum()))
```

```python
# Noise sanity checks: (1) chance correlations with pure noise, (2) permuted target
rng = np.random.default_rng(0)
noise = rng.normal(size=(80, 1000))
max_r = np.max(np.abs([np.corrcoef(noise[:, j], y)[0, 1] for j in range(1000)]))
print(f"max |corr| between y and 1000 PURE NOISE features (n=80): {max_r:.2f}")

y_perm = rng.permutation(y)
s_perm = -cross_val_score(models["ridge"], X, y_perm, cv=outer, scoring="neg_root_mean_squared_error").mean()
print(f"ridge on permuted y: RMSE = {s_perm:.3f} (should be ≈ std of y = {y.std():.3f})")

# WRONG way (selection outside CV) vs RIGHT way (inside), on noise features only
wrong_X = SelectKBest(f_regression, k=20).fit_transform(noise, y)
wrong = -cross_val_score(RidgeCV(), wrong_X, y, cv=outer, scoring="neg_root_mean_squared_error").mean()
right = -cross_val_score(make_pipeline(SelectKBest(f_regression, k=20), RidgeCV()), noise, y, cv=outer,
                         scoring="neg_root_mean_squared_error").mean()
print(f"noise features: selection outside CV RMSE={wrong:.3f} (looks predictive!) vs inside CV RMSE={right:.3f}")
```

## 7. Interpretation
- **Model comparison with n = 80:** ridge, elastic net and PLS should all be far below the baseline. Lasso can be somewhat worse or more variable: with strongly correlated wavelengths it picks a few somewhat arbitrary representatives.
- **MI top-30 → ridge** loses a lot of accuracy here (RMSE ≈ 0.96 vs 0.25 for ridge, with a high variance). MI evaluates each wavelength separately and favours the strong water/solvent bands, which correlate with the analyte only indirectly.
- **Selection stability:** the stable picks cluster in *regions*: around the analyte bands (≈ 1205–1216, 1711–1756, 2293–2318 nm), plus solvent-band regions (≈ 1616–1668, 2132–2202 nm). The solvent is informative too, because the composition sums to 100% (closure), so the model uses it indirectly. About 100 further wavelengths are selected only occasionally, which is the instability of single-feature selection. Report *regions*, and interpret them with chemistry.
- **Chance correlation:** with n = 80 and 1000 noise features, the maximum |r| is ≈ 0.35 here (typically 0.35–0.5). This is why univariate screening produces false discoveries in wide data.
- **Permuted target:** the RMSE should return to ≈ std(y). If it does not, the pipeline leaks (e.g. selection or scaling outside CV).
- **Wrong vs right selection on pure noise:** selection outside CV produces an apparently predictive model on noise. Inside CV, it shows no skill. This is the most important demonstration on this page.

## 8. Decision framework
| Situation | Approach |
|---|---|
| p ≫ n, correlated features (spectra, sensors) | Ridge / PLS / elastic net; nested CV |
| Need a sparse model for cost/interpretability | Elastic net (more stable than lasso) or lasso + stability selection; report frequencies |
| Non-linear relations suspected, p large | Regularised GBM with feature subsampling, *after* a linear baseline; or a kernel method; nested CV |
| Many automatically generated features | Group them; filter by domain sense; strong regularisation; beware chance findings |
| Selection claims ("biomarkers", "key wavelengths") | Stability selection + permutation checks + external validation |
| Very small n (< ~30) | Few, domain-chosen features; regularised linear model; LOO/repeated CV; report uncertainty |

- **IF** the performance on the permuted target is better than the baseline → there is leakage. Fix it before anything else.
- **IF** the selected features change completely across resamples → stop interpreting individual features; use group-level statements.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Univariate screening on all data, then CV | Spurious "predictive" models | Selection inside the pipeline; nested CV |
| Reporting the top feature list as findings | Non-replicable claims | Stability + permutation + external validation |
| Using VIF thresholds with p > n | Undefined / meaningless | Group-level thinking; regularisation |
| Single CV split with small n | Noisy, optimistic estimates | Repeated (nested) CV |
| Distance-based models on raw wide data | Concentrated distances; poor performance | PCA/PLS first, or linear models |

## 10. Checklist
- [ ] Baseline and ridge/PLS references in pipelines.
- [ ] All selection/tuning nested in CV; repeated CV for small n.
- [ ] Permuted-target and noise sanity checks.
- [ ] Selection stability reported as frequencies/regions.
- [ ] Interpretation at the group level with domain knowledge.

## 11. References
- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *ESL*, Ch. 18 "High-Dimensional Problems: p ≫ N". https://hastie.su.domains/ElemStatLearn/
- Hastie, T., Tibshirani, R. & Wainwright, M. (2015). *Statistical Learning with Sparsity: The Lasso and Generalizations*. CRC. Free PDF: https://hastie.su.domains/StatLearnSparsity/
- Ambroise, C. & McLachlan, G. J. (2002). "Selection bias in gene extraction on the basis of microarray gene-expression data." *PNAS* 99(10), 6562–6566.
- Meinshausen, N. & Bühlmann, P. (2010). "Stability selection." *JRSS B* 72(4), 417–473.
- Kraskov, A., Stögbauer, H. & Grassberger, P. (2004). "Estimating mutual information." *Physical Review E* 69, 066138. The estimator behind `mutual_info_regression`.
- scikit-learn, *Feature selection* and `mutual_info_regression`. https://scikit-learn.org/stable/modules/feature_selection.html
