# Cluster Validation and Choosing the Number of Clusters

> **Folder:** 09_unsupervised_learning · **Prev:** [clustering_algorithms.md](clustering_algorithms.md) · **Next:** [cluster_interpretation.md](cluster_interpretation.md) · **Related:** [metrics_reference.md](../08_quick_references/metrics_reference.md) §5

---

## 1. Purpose
Judge whether a clustering is real, stable and useful, and choose the number of clusters k with several complementary kinds of evidence instead of a single "best" index.

## 2. When to use it
After fitting candidate clusterings, before interpreting or using them.

## 3. Conceptual foundation
**Internal indices** use only the data and the labels:

| Index | Measures | Range / best | Biases |
|---|---|---|---|
| **Silhouette** s(i) = (b − a)/max(a, b) | How much closer a point is to its own cluster (a) than to the nearest other cluster (b) | −1…1, higher is better; per-point values available | Favours convex, well-separated clusters; often prefers small k |
| **Davies–Bouldin** | Average similarity of each cluster with its most similar one | ≥ 0, lower is better | Convex-cluster bias |
| **Calinski–Harabasz** | Between- vs within-cluster dispersion ratio | Higher is better | Grows with k for some data; convex bias |
| **Inertia (elbow)** | Within-cluster sum of squares (K-Means) | Always decreases with k | Elbows are often ambiguous |
| **BIC / AIC** (GMM) | Likelihood penalised by complexity | Lower is better | Assumes Gaussian components; large n favours many components |

**Stability:** cluster subsamples or bootstraps, then compare the partitions (adjusted Rand index, ARI) on the shared points. Real structure is stable; arbitrary partitions of a continuum are not.

**External validation** (when metadata or a few labels exist): ARI / normalised mutual information (NMI) with known categories, and cross-tabulations. The metadata need not be "the truth", but strong disagreement must be explained.

**Null reference:** compare an index with its value on structureless data of the same shape (uniform in the bounding box or in the PCA box). This is the idea behind the gap statistic. A silhouette of 0.3 means little if random data also give 0.3.

**Usefulness** is the final criterion: does the clustering improve the downstream task (regime-specific monitoring, a stratified model)?

## 4. Assumptions and limitations
- No internal index is neutral. Each one encodes a notion of a "good cluster" (mostly compact and convex). Density-based clusters can score badly while being correct.
- Indices are computed in the feature space you clustered in, so they cannot detect bad feature choices.
- Silhouette is O(n²). Use `sample_size` for large n.

## 5. Step-by-step procedure
1. **For each candidate k (or parameter setting),** compute silhouette, Davies–Bouldin, Calinski–Harabasz, and inertia/BIC as applicable.
2. **Compare with a null reference** (uniform data, same shape).
3. **Check stability** across bootstraps for the most promising settings.
4. **Check external consistency** with any metadata.
5. **Look at per-cluster silhouettes** (silhouette plot): are some clusters poorly separated, or tiny?
6. **Decide k** using the convergence of evidence *and* the intended use. Where the evidence is ambiguous, prefer the k whose clusters are interpretable and actionable.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_samples, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from cookbook_data import make_process_sensors

df = make_process_sensors()
normal = df[df["fault"] == "none"]
features = ["flow_m3h", "pressure_drop_bar", "pump_current_a", "outlet_temp_c"]
X = StandardScaler().fit_transform(normal[features])
rng = np.random.default_rng(0)
X_null = rng.uniform(X.min(0), X.max(0), X.shape)          # structureless reference

rows = []
for k in range(2, 9):
    km = KMeans(k, n_init=10, random_state=0).fit(X)
    km0 = KMeans(k, n_init=10, random_state=0).fit(X_null)
    gm = GaussianMixture(k, covariance_type="full", random_state=0).fit(X)
    rows.append({"k": k, "inertia": km.inertia_,
                 "silhouette": silhouette_score(X, km.labels_, sample_size=3000, random_state=0),
                 "silhouette_null": silhouette_score(X_null, km0.labels_, sample_size=3000, random_state=0),
                 "davies_bouldin": davies_bouldin_score(X, km.labels_),
                 "calinski_harabasz": calinski_harabasz_score(X, km.labels_),
                 "gmm_bic": gm.bic(X),
                 "ARI_vs_regime_log": adjusted_rand_score(normal["regime"], km.labels_)})
tab = pd.DataFrame(rows).set_index("k")
print(tab.round(3))
```

```python
# Stability: bootstrap pairs, ARI on the points shared by both resamples
def bootstrap_stability(X, k, n_pairs=10, seed=0):
    rng = np.random.default_rng(seed); scores = []
    for b in range(n_pairs):
        i1, i2 = rng.choice(len(X), len(X), replace=True), rng.choice(len(X), len(X), replace=True)
        m1 = KMeans(k, n_init=5, random_state=b).fit(X[i1]); m2 = KMeans(k, n_init=5, random_state=b + 100).fit(X[i2])
        scores.append(adjusted_rand_score(m1.predict(X), m2.predict(X)))   # compare the induced partitions of all points
    return np.mean(scores), np.std(scores)

for k in [2, 3, 4, 5, 6]:
    m, s = bootstrap_stability(X, k)
    print(f"k={k}: stability ARI = {m:.3f} ± {s:.3f}")
```

```python
# Silhouette plot for the chosen k: separation per cluster
k = 4
lab = KMeans(k, n_init=10, random_state=0).fit_predict(X)
idx = rng.choice(len(X), 3000, replace=False)
sil = silhouette_samples(X[idx], lab[idx])
fig, ax = plt.subplots(figsize=(6, 4)); y0 = 0
for c in range(k):
    v = np.sort(sil[lab[idx] == c]); ax.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v); y0 += len(v) + 20
    print(f"cluster {c}: n={np.sum(lab == c)}, mean silhouette={v.mean():.3f}")
ax.axvline(sil.mean(), color="k", ls="--"); ax.set_title("silhouette per cluster"); plt.show()
```

## 7. Interpretation
- **Silhouette and Davies–Bouldin both prefer k = 2** (shutdown vs everything else, the most extreme separation). Calinski–Harabasz keeps rising with k. Yet the operator log has 4 regimes, and k = 4 matches it almost perfectly (ARI ≈ 1). It is a local optimum of Davies–Bouldin (0.26, vs 0.34 at k = 3 and 0.27 at k = 5), it has a silhouette nearly as high as k = 2, it sits at the clearest inertia elbow, and it is perfectly stable. **No single index decides.** The convergence of several weaker signals, external consistency and the intended use (regime-specific limits) point to k = 4. Without the regime log you would still shortlist k = 2 and k = 4, and the domain discussion would decide.
- **Null silhouette** (uniform data) is around 0.2–0.35. The real data's values (0.6–0.8) are far above it, so the structure is real.
- **GMM BIC** keeps decreasing slowly beyond 4 components. With n ≈ 10 000, BIC can justify extra components that describe within-regime shape (daily temperature cycles) rather than new regimes. Read the BIC curve for its first big drop, not its minimum.
- **Stability:** k = 4 should give ARI ≈ 1 across bootstraps. Values that drop for k = 5–6 show the extra clusters split a regime arbitrarily.
- **Per-cluster silhouettes:** a cluster with a low mean silhouette, or many negative values, overlaps with its neighbour. Consider merging it, or check its features.

## 8. Decision framework
- **IF** the indices disagree → favour stability + external consistency + usefulness over any single index.
- **IF** all indices are close to their null-reference values → there is no meaningful cluster structure. Report that, and consider DR or a continuous description instead.
- **IF** density-based clusters (HDBSCAN) look right but score poorly on silhouette → trust a density-aware check (e.g. DBCV, or visual + stability) rather than silhouette.
- **IF** k must be fixed by the business (e.g. 3 service levels) → validate that this k is stable and interpretable, and report the separation honestly.
- **IF** using GMM → use BIC to shortlist k, then check stability and interpretability.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Choosing k by the silhouette maximum alone | Often too few clusters | Several indices + stability + use |
| No null reference | Over-interpreting weak structure | Compare with uniform/PCA-box data |
| Ignoring stability | Arbitrary partitions presented as findings | Bootstrap ARI |
| Elbow over-interpretation | Arbitrary k | Treat the elbow as one hint |
| Silhouette on full large data | Very slow | `sample_size` |

## 10. Checklist
- [ ] Internal indices for a range of k.
- [ ] Null-reference comparison.
- [ ] Bootstrap stability.
- [ ] External consistency with metadata (if available).
- [ ] Per-cluster silhouette inspection.
- [ ] k chosen with converging evidence and the intended use; rationale documented.

## 11. References
- Rousseeuw, P. J. (1987). "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis." *J. Comput. Appl. Math.* 20, 53–65. https://doi.org/10.1016/0377-0427(87)90125-7
- Tibshirani, R., Walther, G. & Hastie, T. (2001). "Estimating the number of clusters in a data set via the gap statistic." *JRSS B* 63(2), 411–423.
- Ben-Hur, A., Elisseeff, A. & Guyon, I. (2002). "A stability based method for discovering structure in clustered data." *Pacific Symposium on Biocomputing*.
- Hubert, L. & Arabie, P. (1985). "Comparing partitions." *Journal of Classification* 2, 193–218 (ARI).
- scikit-learn User Guide, *Clustering performance evaluation*. https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
- Moulavi, D. et al. (2014). "Density-Based Clustering Validation." *SIAM SDM* (DBCV).
