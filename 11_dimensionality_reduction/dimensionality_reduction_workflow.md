# Dimensionality Reduction and Feature Selection: When, Which and Why

> **Folder:** 11_dimensionality_reduction · **Start here.** · **Next:** [pca.md](pca.md) · **Related:** [feature_selection.md](../06_model_improvement/feature_selection.md) (filter/wrapper/embedded methods, RFE, mutual information, multicollinearity: the basics are there and are not repeated here), [high_dimensional_data.md](high_dimensional_data.md), [visualization_embeddings.md](visualization_embeddings.md)

---

## 1. Purpose
Decide whether reducing the number of dimensions will help a given goal: prediction, visualisation, monitoring, compression or interpretation. If it will, choose between **selecting** original features and **extracting** new ones. Also know when dimensionality reduction (DR) mainly makes interpretation harder.

## 2. When to use it
- Many features relative to the number of samples (spectra, omics, wide sensor sets, embeddings).
- Strong redundancy among features (correlated sensors, neighbouring wavelengths).
- A need to visualise high-dimensional structure, or to monitor many variables with a few statistics (MSPC).
- Downstream methods that suffer in high dimensions (kNN, clustering, density-based anomaly detection).

## 3. Conceptual foundation
| | **Feature selection** | **Feature extraction** |
|---|---|---|
| Output | A subset of the original variables | New variables = combinations of the originals |
| Examples | Filter (MI, correlation), wrapper (RFE), embedded (L1, tree importance); see [feature_selection.md](../06_model_improvement/feature_selection.md) | PCA, PLS, ICA, NMF, autoencoders; t-SNE/UMAP (visualisation only) |
| Interpretability | Kept: the variables keep their meaning | Reduced: components are mixtures (loadings help) |
| Handles redundancy | Picks representatives (unstable with correlated features) | Merges correlated variables into one component (stable) |
| Supervised? | Often | PCA/ICA/NMF: no. PLS: yes |
| Typical use | Cost reduction, interpretable models | Spectra, sensors, compression, monitoring, noise reduction |

**Linear vs non-linear:**
- **PCA/PLS** (linear) preserve global structure and are invertible (reconstruction). They are suitable for modelling and monitoring.
- **t-SNE/UMAP** (non-linear, neighbourhood-preserving) are for **visualisation**. Distances between clusters, cluster sizes and densities in the map are not reliable, and there is no faithful inverse ([visualization_embeddings.md](visualization_embeddings.md)).

**When DR improves a model:** when p is large relative to n, features are noisy and redundant, and the model is variance-limited (e.g. unregularised linear models, kNN). **When it does not:** with regularised linear models or GBMs on moderate p, DR often brings no gain, and it obscures which original variables matter. Supervised DR (PLS) usually beats unsupervised DR (PCA) for prediction, because PCA keeps directions of high *variance*, which need not be predictive.

## 4. Assumptions and limitations
- PCA assumes that variance is the interesting signal (scaling choices change everything) and that linear combinations suffice.
- Any DR fitted on the full data before CV leaks (unsupervised leakage is mild; supervised DR such as PLS or MI selection leaks strongly). Put it inside the pipeline.
- Components fitted on one period/instrument may not transfer to another (instrument drift, new batches).

## 5. Step-by-step procedure
1. **State the goal:** prediction, visualisation, monitoring, compression or interpretation.
2. **Quantify the redundancy:** the correlation structure; the PCA explained-variance curve (how many components carry most of the variance?).
3. **Pick the approach** from the decision framework (§8).
4. **Implement inside a pipeline;** tune the number of components/features by CV (prediction) or by the explained variance / monitoring performance (other goals).
5. **Compare against no reduction** (and regularisation only) under the same CV.
6. **Interpret:** loadings for PCA/PLS; selection stability for feature selection.

## 6. Python implementation: does DR help this model?

```python
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cookbook_data import make_nir_spectra

X, targets = make_nir_spectra()                 # 150 samples × 400 wavelengths (p >> n)
y = targets["analyte_pct"]
cv = KFold(5, shuffle=True, random_state=0)
def rmse(model):
    s = -cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    return f"{s.mean():.3f} ± {s.std():.3f}"

print("target std (baseline RMSE ≈ this):", round(y.std(), 3))
models = {
    "OLS on 400 raw features (p > n)": make_pipeline(StandardScaler(), LinearRegression()),
    "kNN on raw features": make_pipeline(StandardScaler(), KNeighborsRegressor(10)),
    "kNN on 6 PCA components": make_pipeline(StandardScaler(), PCA(6), KNeighborsRegressor(10)),
    "PCR (PCA 6 → OLS)": make_pipeline(StandardScaler(), PCA(6), LinearRegression()),
    "PLS (6 components)": make_pipeline(StandardScaler(), PLSRegression(6)),
    "Ridge on all features (no DR)": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 3, 30))),
}
for name, m in models.items():
    print(f"{name:34s} RMSE = {rmse(m)}")
```

## 7. Interpretation
- **OLS with p > n works here (RMSE ≈ 0.24), surprisingly.** With more features than samples, scikit-learn returns the *minimum-norm* least-squares solution, which acts as implicit regularisation. These synthetic spectra are also low-noise and low-rank. Do not rely on this: with more noise or nuisance variation, min-norm OLS degrades sharply. Use explicit regularisation or DR, whose behaviour you control.
- **kNN fails with and without PCA** (RMSE ≈ the target std, i.e. no skill). This is the key lesson. The first PCs capture the directions of largest *variance*: baseline offsets, scatter and the water and solvent bands. The analyte is a weak signal (it appears only in PC3, see [pca.md](pca.md)). Euclidean distances in the PCA space are dominated by irrelevant variation, so "neighbours" are not chemically similar in analyte content. **Unsupervised DR keeps variance, not relevance.**
- **PCR, PLS and ridge** all reach RMSE ≈ 0.23 (vs a target std of 3.0). The linear models can weight the weak analyte direction. PLS gets there with fewer components because it is supervised. Choose by the secondary goals: PLS gives a low-dimensional score space for monitoring and outlier detection; ridge gives one coefficient per wavelength.

## 8. Decision framework
**Choose by goal:**
| Goal | Recommended |
|---|---|
| Prediction, p ≫ n, correlated features (spectra) | PLS or ridge/ElasticNet; tune by CV; spectral preprocessing ([high_dimensional_data.md](high_dimensional_data.md)) |
| Prediction, moderate p, tree models | Usually no DR; remove junk only |
| Prediction with distance-based models (kNN, SVM-RBF, clustering) | PCA (whitening optional) before the model |
| Interpretability / measurement cost | Feature selection (L1, RFE, domain choice); report stability |
| Process monitoring with many correlated sensors | PCA → T² / SPE ([statistical_methods.md](../10_anomaly_detection/statistical_methods.md)) |
| Visualising clusters/structure | PCA first (global), then UMAP/t-SNE (local), with caveats |
| Compression / denoising | PCA (choose k by reconstruction error), autoencoders for non-linear data |
| Non-negative additive parts (spectra, topics, images) | NMF |

- **IF** DR does not beat "no DR + regularisation" beyond the noise → skip it (simpler, more interpretable).
- **IF** stakeholders need statements about original variables → prefer selection, or translate the components back through their loadings, with care.
- **IF** you use t-SNE/UMAP coordinates as model features → do so only with the reducer fitted inside CV, and expect instability. Usually prefer PCA.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| PCA/PLS/selection fitted before CV | Leakage (severe for supervised DR) | Pipeline |
| Choosing k by "95% explained variance" for prediction | Components chosen for variance, not prediction | Tune k by CV |
| Unscaled PCA on mixed units | Components dominated by large-unit variables | Standardise (or scale by domain-meaningful units) |
| Interpreting t-SNE distances/cluster sizes | False conclusions | Use for qualitative neighbourhoods only |
| DR "because the data is wide" without comparison | Lost interpretability, no gain | Compare with regularisation only |

## 10. Checklist
- [ ] Goal stated; redundancy quantified.
- [ ] Selection vs extraction decided, with a reason.
- [ ] DR inside the pipeline; k tuned for the goal.
- [ ] Compared with no DR / regularisation only.
- [ ] Loadings or selection stability interpreted.

## 11. References
- Jolliffe, I. T. & Cadima, J. (2016). "Principal component analysis: a review and recent developments." *Phil. Trans. R. Soc. A* 374: 20150202. https://doi.org/10.1098/rsta.2015.0202
- Wold, S., Sjöström, M. & Eriksson, L. (2001). "PLS-regression: a basic tool of chemometrics." *Chemometrics and Intelligent Laboratory Systems* 58(2), 109–130. https://doi.org/10.1016/S0169-7439(01)00155-1
- James, G. et al. (2023). *ISLP*, §6.3 "Dimension Reduction Methods" (PCR, PLS), §12.2 PCA. https://www.statlearning.com/
- scikit-learn User Guide, *Decomposing signals in components*; *Cross decomposition* (PLS). https://scikit-learn.org/stable/modules/decomposition.html · https://scikit-learn.org/stable/modules/cross_decomposition.html
- Guyon, I. & Elisseeff, A. (2003). "An Introduction to Variable and Feature Selection." *JMLR* 3, 1157–1182.
