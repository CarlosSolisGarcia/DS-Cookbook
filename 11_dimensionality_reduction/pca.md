# Principal Component Analysis (PCA) and PLS in Practice

> **Folder:** 11_dimensionality_reduction · **Prev:** [dimensionality_reduction_workflow.md](dimensionality_reduction_workflow.md) · **Next:** [high_dimensional_data.md](high_dimensional_data.md) · **Related:** [statistical_methods.md](../10_anomaly_detection/statistical_methods.md) (PCA monitoring), [scaling.md](../02_preprocessing/scaling.md)

---

## 1. Purpose
Use PCA to understand and compress correlated data: explained variance, the number of components, loadings and scores, reconstruction. Use PCA and PLS as preprocessing for prediction and monitoring.

## 2. When to use it
- Correlated numeric variables (sensors, spectra, lab measurements).
- To see the main directions of variation, find outliers, compress, denoise, or prepare data for distance-based methods.
- PLS when a target exists and prediction is the goal.

## 3. Conceptual foundation
- **PCA** finds orthogonal directions (principal components, PCs) of maximal variance. PC1 explains the most, PC2 the most of what remains, and so on. Mathematically it is the SVD of the centred (and usually scaled) data matrix X = U S Vᵀ.
  - **Scores** (T = XV): the coordinates of the samples on the PCs.
  - **Loadings** (V): the weight of each original variable in each PC.
  - **Explained variance ratio:** the share of the total variance captured per PC.
  - **Reconstruction:** X̂ = T_k V_kᵀ. The residual X − X̂ gives SPE (see the anomaly recipes).
- **Scaling decides what "variance" means.** Unscaled PCA on mixed units is dominated by the large-unit variables. Standardisation gives every variable equal prior weight. For spectra (common units), centring without scaling is often preferred, because scaling inflates noisy, flat regions.
- **Choosing k:** the scree plot elbow; cumulative explained variance (e.g. 90–99%); parallel analysis (compare with eigenvalues from permuted data); or, for prediction, **cross-validation of the downstream model**.
- **PLS** finds components that maximise the *covariance* between X scores and y. It is supervised, so it reaches better predictions with fewer components. It must be fitted inside CV.

## 4. Assumptions and limitations
- Linear structure; sensitive to outliers (use robust scaling, or inspect the score plots for influential points).
- PCs are mathematical constructs. Their interpretation through loadings is an *interpretation*, not a physical identity, and the signs are arbitrary.
- PCA directions are unstable when eigenvalues are close (PCs can swap or rotate between samples).

## 5. Step-by-step procedure
1. **Preprocess:** handle missing values; scale (standardise for mixed units; centre only, or SNV/derivatives, for spectra).
2. **Fit PCA** on training data; inspect the explained-variance curve.
3. **Score plot** (PC1 vs PC2, coloured by metadata): clusters, trends, outliers, batch effects.
4. **Loadings:** which variables or wavelengths drive each PC?
5. **Choose k** for the goal (CV for prediction; the variance or reconstruction criterion for compression/monitoring).
6. **For prediction,** compare PCR and PLS inside pipelines, with k tuned by CV.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from cookbook_data import make_nir_spectra

X, t = make_nir_spectra()
wl = np.array([int(c.split("_")[1]) for c in X.columns])
y = t["analyte_pct"]

# Preprocessing typical for spectra: SNV (removes multiplicative scatter/offset per spectrum),
# Savitzky–Golay first derivative (removes baseline offset and slope). Both are row-wise → no leakage.
def snv(A):
    A = np.asarray(A, float)
    return (A - A.mean(1, keepdims=True)) / A.std(1, keepdims=True)
def sg_derivative(A, window=15, poly=2):
    return savgol_filter(np.asarray(A, float), window_length=window, polyorder=poly, deriv=1, axis=1)

fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
for a, (name, A) in zip(ax, [("raw", X.values), ("SNV", snv(X)), ("SNV + 1st derivative", sg_derivative(snv(X)))]):
    a.plot(wl, A[:20].T, lw=0.6); a.set_title(name); a.set_xlabel("nm")
plt.tight_layout(); plt.show()
```

```python
# PCA: explained variance (raw vs preprocessed) and parallel analysis for k
def pca_curve(A):
    return PCA().fit(A - A.mean(0)).explained_variance_ratio_

raw_ev, pre_ev = pca_curve(X.values), pca_curve(sg_derivative(snv(X)))
print("cumulative explained variance, first 8 PCs")
print(pd.DataFrame({"raw": np.cumsum(raw_ev)[:8], "SNV+deriv": np.cumsum(pre_ev)[:8]}, index=range(1, 9)).round(4))

rng = np.random.default_rng(0)
A = sg_derivative(snv(X)); A = A - A.mean(0)
perm_ev = np.mean([PCA(10).fit(np.column_stack([rng.permutation(col) for col in A.T])).explained_variance_ratio_
                   for _ in range(5)], axis=0)
k_parallel = int(np.sum(pre_ev[:10] > perm_ev))
print("parallel analysis suggests k =", k_parallel)
```

```python
# Score plot coloured by metadata (batch) and by target; loadings of PC1–PC2
pca = PCA(5).fit(A)
T = pca.transform(A)
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
sc0 = ax[0].scatter(T[:, 0], T[:, 1], c=t["batch"], cmap="tab10", s=15); ax[0].set_title("scores: colour = batch")
sc1 = ax[1].scatter(T[:, 0], T[:, 1], c=y, cmap="viridis", s=15); ax[1].set_title("scores: colour = analyte %")
plt.colorbar(sc1, ax=ax[1])
for i in range(2):
    ax[2].plot(wl, pca.components_[i], label=f"PC{i+1} loading")
ax[2].legend(); ax[2].set_xlabel("nm"); ax[2].set_title("loadings")
plt.tight_layout(); plt.show()
print("corr(PC scores, analyte):", np.round([np.corrcoef(T[:, i], y)[0, 1] for i in range(5)], 2))
```

```python
# Prediction: PCR vs PLS, k tuned by CV, preprocessing in the pipeline
prep = FunctionTransformer(lambda Z: sg_derivative(snv(Z)))
cv = KFold(5, shuffle=True, random_state=0)
pcr = GridSearchCV(Pipeline([("prep", prep), ("pca", PCA()), ("ols", LinearRegression())]),
                   {"pca__n_components": list(range(1, 16))}, cv=cv, scoring="neg_root_mean_squared_error")
pls = GridSearchCV(Pipeline([("prep", prep), ("pls", PLSRegression(scale=False))]),
                   {"pls__n_components": list(range(1, 16))}, cv=cv, scoring="neg_root_mean_squared_error")
pls_raw = GridSearchCV(Pipeline([("sc", StandardScaler()), ("pls", PLSRegression(scale=False))]),
                       {"pls__n_components": list(range(1, 16))}, cv=cv, scoring="neg_root_mean_squared_error")
for name, gs in [("PCR (SNV+deriv)", pcr), ("PLS (SNV+deriv)", pls), ("PLS (raw, standardised)", pls_raw)]:
    gs.fit(X, y)
    k = list(gs.best_params_.values())[0]
    print(f"{name:25s} best k={k:2d}  CV RMSE={-gs.best_score_:.3f}")
# NOTE: reporting best_score_ after tuning k is mildly optimistic → nested CV for the final number.
```

## 7. Interpretation
- **Raw vs preprocessed spectra:** the raw spectra differ by vertical offsets and scaling (scatter, baseline). After SNV + derivative they align, and the band structure stands out.
- **Explained variance (raw):** PC1 explains 75%, and PC1–PC2 explain 98%. That is mostly baseline/batch offsets and the strong water/solvent bands. **The analyte, the quantity of interest, is in PC3** (corr ≈ 0.99 with the analyte score), which carries only ~1% of the variance. A "keep 95% of the variance" rule would discard it. Explained variance measures *variance*, not *relevance*.
- **Parallel analysis** suggests k = 2. That is right for *describing* the dominant variation, and wrong for *predicting* the analyte. For prediction, CV decides.
- **Score plots:** colouring by batch shows whether instrument days separate along a PC (batch effects to handle). Colouring by the target shows which PC tracks the analyte.
- **Loadings:** a PC whose loadings peak at the analyte band positions (≈ 1210, 1730, 2310 nm) and whose scores correlate with the analyte confirms a chemically sensible model.
- **PCR vs PLS, and the preprocessing surprise:** PLS needs far fewer components than PCR (4 vs 13 on the preprocessed spectra), as expected. But **SNV + derivative made the prediction worse** here (RMSE ≈ 0.32 vs ≈ 0.23 for PLS on the standardised raw spectra). In this simulation, the multiplicative scatter is mild and the baseline offsets are linear, so PLS models them with a component or two. SNV removes part of the concentration information (it rescales every spectrum), and the derivative amplifies noise. In real NIR data the opposite is often true. **Spectral preprocessing is a hypothesis to validate, not a ritual.** Compare the options under the same CV.

## 8. Decision framework
- **IF** the variables have mixed units → standardise before PCA. **IF** they share units (spectra) → centre, then consider SNV/derivatives instead of standardisation.
- **IF** the goal is prediction → tune k by (nested) CV; prefer PLS over PCR for the same k.
- **IF** the goal is description or monitoring → k by parallel analysis / explained variance / SPE behaviour on validation data.
- **IF** the score plot shows batch or instrument clusters → handle the batch effects (preprocessing, batch-wise validation, or including the batch info) before modelling.
- **IF** outliers dominate the PCs → investigate them; use robust scaling, or fit on a cleaned reference.
- **IF** the data are non-negative and additive (spectra, counts) and parts-based components are wanted → try NMF.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Standardising spectra blindly | Noise in flat regions inflated | Centre + SNV/derivatives |
| Equating explained variance with importance | Keeping baseline components, dropping weak chemical signals | CV for prediction; inspect loadings |
| Interpreting PC signs | Arbitrary conclusions | Signs are arbitrary; interpret patterns |
| Fitting PCA/PLS on all data before CV | Leakage (strong for PLS) | Pipeline |
| Reporting `best_score_` as the final performance | Optimistic | Nested CV / held-out test |

## 10. Checklist
- [ ] Preprocessing chosen for the data type (scaling vs SNV/derivatives).
- [ ] Explained-variance curve + parallel analysis.
- [ ] Score plots coloured by metadata and target; outliers/batches investigated.
- [ ] Loadings interpreted against domain knowledge.
- [ ] k tuned for the goal; PCR vs PLS compared inside pipelines.

## 11. References
- Jolliffe, I. T. (2002). *Principal Component Analysis*, 2nd ed. Springer.
- Rinnan, Å., van den Berg, F. & Engelsen, S. B. (2009). "Review of the most common pre-processing techniques for near-infrared spectra." *TrAC Trends in Analytical Chemistry* 28(10), 1201–1222. https://doi.org/10.1016/j.trac.2009.07.007
- Horn, J. L. (1965). "A rationale and test for the number of factors in factor analysis." *Psychometrika* 30, 179–185 (parallel analysis).
- scikit-learn, *Principal Component Regression vs Partial Least Squares Regression*. https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html
- SciPy, `savgol_filter`. https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
