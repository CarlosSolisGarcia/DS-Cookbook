# Clustering Algorithms: K-Means, Hierarchical, DBSCAN/HDBSCAN and Gaussian Mixtures

> **Folder:** 09_unsupervised_learning · **Prev:** [unsupervised_workflow.md](unsupervised_workflow.md) · **Next:** [cluster_validation.md](cluster_validation.md) · **Related:** [scaling.md](../02_preprocessing/scaling.md), [pca.md](../11_dimensionality_reduction/pca.md)

---

## 1. Purpose
Choose the clustering algorithm whose **assumptions about cluster shape, size and density** match the data. Configure it correctly and recognise the typical failure modes.

## 2. When to use it
After the unsupervised task is defined and the features are prepared ([unsupervised_workflow.md](unsupervised_workflow.md)).

## 3. Conceptual foundation
| Algorithm | Idea | Assumes | Needs | Handles noise/outliers? | Scales to |
|---|---|---|---|---|---|
| **K-Means** | Minimise within-cluster squared distance to k centroids | Convex, roughly spherical, similar-size clusters (Euclidean) | k; scaling | No: every point is assigned; outliers pull the centroids | Millions (MiniBatchKMeans) |
| **Agglomerative (hierarchical)** | Merge the closest clusters step by step; the dendrogram shows all levels | Depends on the linkage: *Ward* ≈ K-Means-like; *single* → chains; *average/complete* in between | Linkage, cut level (k or a distance) | Single linkage isolates outliers; Ward does not | ~10⁴ (O(n²) memory) |
| **DBSCAN** | Dense regions (≥ `min_samples` within `eps`) form clusters; the rest is noise | Clusters of **similar density**; arbitrary shapes | `eps`, `min_samples` | **Yes**: label −1 = noise | ~10⁵–10⁶ with indexing |
| **HDBSCAN** (`sklearn.cluster.HDBSCAN`, ≥ 1.3) | Hierarchical DBSCAN over all eps; keeps stable clusters | Varying densities; arbitrary shapes | `min_cluster_size` (and `min_samples`) | Yes | ~10⁵–10⁶ |
| **Gaussian Mixture (GMM)** | Data = mixture of k Gaussians; soft memberships via EM | Elliptical clusters with their own covariance | k, covariance type | Partly (low likelihood = outlier score) | ~10⁵ |
| **Spectral, mean-shift, OPTICS, BIRCH** | (not detailed) | — | — | — | — |

**Practical intuitions:**
- K-Means answers "where should I place k prototypes?" It is excellent for regimes that are compact blobs, and wrong for elongated or nested shapes.
- GMM is K-Means with elliptical, soft clusters and a likelihood. You can use BIC to choose k, and you get membership probabilities ("70% regime A") for transition periods.
- DBSCAN/HDBSCAN find "islands of density" and declare everything else noise. They suit spatial data, trajectories and data with outliers. You do not need k. You do need a meaningful distance scale.
- Hierarchical clustering gives the whole tree. It is very useful when the number of groups is genuinely unclear, or when the hierarchy itself matters (product families → variants).

## 4. Assumptions and limitations
- All of them depend on the distance metric and the scaling. Changing the features changes the clusters.
- K-Means and GMM depend on the initialisation. Use several initialisations (`n_init`), and check the stability across seeds.
- DBSCAN's `eps` is hard to set in high dimensions, and it fails when cluster densities differ a lot (use HDBSCAN).
- With high-dimensional data (≫ 10–20 features) distances concentrate. Reduce the dimensionality (PCA) first, or select meaningful features.

## 5. Step-by-step procedure
1. **Visualise** the (scaled) data in 2-D (PCA; [visualization_embeddings.md](../11_dimensionality_reduction/visualization_embeddings.md) for t-SNE/UMAP) to form a hypothesis about the shapes.
2. **Pick 2–3 candidate algorithms** using the table and the decision framework.
3. **Set their parameters:**
   - K-Means/GMM: a range of k (validation in [cluster_validation.md](cluster_validation.md)); `n_init ≥ 10`.
   - DBSCAN: `min_samples` ≈ 2 × dimensions (rule of thumb); `eps` from the k-distance plot (the "elbow").
   - HDBSCAN: `min_cluster_size` = the smallest group size you would care about.
   - Hierarchical: Ward for compact clusters; inspect the dendrogram for the cut level.
4. **Fit and compare** the partitions (agreement measured with the adjusted Rand index between methods).
5. **Validate and interpret** (next recipes).

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# --- Assumptions made visible on three toy geometries ---
rng = np.random.default_rng(0)
blobs, yb = make_blobs(600, centers=3, cluster_std=[0.5, 1.0, 1.8], random_state=0)
moons, ym = make_moons(600, noise=0.06, random_state=0)
aniso = blobs @ np.array([[0.6, -0.6], [-0.4, 0.9]])                 # elongated clusters
datasets = {"blobs (unequal spread)": (blobs, yb), "moons (non-convex)": (moons, ym), "anisotropic": (aniso, yb)}
algos = {
    "KMeans": lambda X, k: KMeans(k, n_init=10, random_state=0).fit_predict(X),
    "GMM full": lambda X, k: GaussianMixture(k, covariance_type="full", random_state=0).fit_predict(X),
    "Ward": lambda X, k: AgglomerativeClustering(k, linkage="ward").fit_predict(X),
    "HDBSCAN": lambda X, k: HDBSCAN(min_cluster_size=20).fit_predict(X),
}
fig, axes = plt.subplots(3, 4, figsize=(16, 11))
for r, (dname, (X, ytrue)) in enumerate(datasets.items()):
    Xs = StandardScaler().fit_transform(X); k = len(np.unique(ytrue))
    for c, (aname, f) in enumerate(algos.items()):
        lab = f(Xs, k)
        axes[r, c].scatter(Xs[:, 0], Xs[:, 1], c=lab, s=5, cmap="tab10")
        axes[r, c].set_title(f"{aname} | {dname}\nARI={adjusted_rand_score(ytrue, lab):.2f}", fontsize=8)
plt.tight_layout(); plt.show()
```

```python
# --- Applied: operating regimes of the process loop (see unsupervised_workflow.md) ---
from cookbook_data import make_process_sensors
df = make_process_sensors()
normal = df[df["fault"] == "none"].copy()
features = ["flow_m3h", "pressure_drop_bar", "pump_current_a", "outlet_temp_c"]
X = StandardScaler().fit_transform(normal[features])

labels = {
    "kmeans_k4": KMeans(4, n_init=10, random_state=0).fit_predict(X),
    "gmm_k4": GaussianMixture(4, covariance_type="full", random_state=0).fit_predict(X),
    "hdbscan": HDBSCAN(min_cluster_size=100).fit_predict(X),
}
for name, lab in labels.items():
    n_noise = int((lab == -1).sum())
    print(f"{name:10s} clusters={len(set(lab) - {-1})} noise={n_noise:4d} "
          f"ARI vs operator regime log={adjusted_rand_score(normal['regime'], lab):.3f}")
```

```python
# DBSCAN: choosing eps with the k-distance plot (k = min_samples)
min_samples = 2 * X.shape[1]
d = np.sort(NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)[0][:, -1])
plt.figure(figsize=(6, 3)); plt.plot(d); plt.ylabel(f"distance to {min_samples}-th neighbour"); plt.xlabel("points (sorted)")
plt.title("k-distance plot: eps ≈ the 'knee'"); plt.show()
eps = float(np.quantile(d, 0.98))                 # a pragmatic knee estimate; inspect the plot
lab_db = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
print(f"DBSCAN eps={eps:.2f}: clusters={len(set(lab_db) - {-1})}, noise={(lab_db == -1).sum()}, "
      f"ARI={adjusted_rand_score(normal['regime'], lab_db):.3f}")
```

```python
# Hierarchical: dendrogram on a subsample to choose the cut level
sub = X[np.random.default_rng(0).choice(len(X), 400, replace=False)]
Z = linkage(sub, method="ward")
plt.figure(figsize=(10, 3)); dendrogram(Z, truncate_mode="lastp", p=30, no_labels=True)
plt.title("Ward dendrogram (large vertical gaps suggest natural cut levels)"); plt.show()

# GMM soft memberships: uncertain points lie in TRANSITIONS between regimes (ramps)
gmm = GaussianMixture(4, covariance_type="full", random_state=0).fit(X)
proba = gmm.predict_proba(X)
normal["max_membership"] = proba.max(1)
print("share of snapshots with max membership < 0.9:", round((normal["max_membership"] < 0.9).mean(), 4))
print(normal.loc[normal["max_membership"] < 0.9, "load_pct"].describe().round(1))
```

## 7. Interpretation
- **Toy grid:** K-Means and Ward recover the blobs but split the moons wrongly. GMM (full covariance) handles the anisotropic clusters best. HDBSCAN recovers the moons and labels sparse points as noise, but it may merge or drop the wide, low-density blob. Each algorithm succeeds where its assumptions hold.
- **Process regimes:** K-Means and GMM with k = 4 should agree almost perfectly with the operator's regime log (ARI ≈ 1). The regimes are compact blobs in this feature space. HDBSCAN may also find them, but it marks the sparse ramp periods as noise. Depending on the use, that can be useful (transitions) or a nuisance.
- **k-distance plot:** a clear knee helps you set eps. A smooth curve means no single density scale exists, so prefer HDBSCAN.
- **Dendrogram:** a few long vertical branches before merging suggest a natural number of groups. Many similar heights suggest a continuum.
- **GMM memberships below 0.9:** only a handful of snapshots, all at load values *between* the plateaus (e.g. 73% load, halfway between 60% and 80%). These are the ramps between regimes. Here the ramps last only 30 minutes, so they are rare. In processes with slow transitions (start-ups, grade changes) this share is much larger. Soft memberships then reveal meaningful "transition" states that a hard clustering hides.

## 8. Decision framework
| Data characteristic | First choice | Alternative |
|---|---|---|
| Compact, similar-size blobs; large n | K-Means (MiniBatchKMeans for very large n) | GMM |
| Elliptical/correlated clusters; need soft memberships or a likelihood | GMM (`covariance_type="full"`, BIC for k) | K-Means on PCA-whitened data |
| Arbitrary shapes, noise points, unknown k | HDBSCAN | DBSCAN (similar densities) |
| Spatial coordinates (distances in metres) | DBSCAN/HDBSCAN with a physical eps | — ([17_geospatial_data](../17_geospatial_data/geospatial_workflow.md)) |
| Unknown number of groups; hierarchy matters; n ≲ 10⁴ | Agglomerative + dendrogram | HDBSCAN |
| Categorical or mixed data | Gower distance + hierarchical/PAM (k-medoids), or k-modes | One-hot + PCA + K-Means (crude) |
| Time series as objects | Feature-based clustering (statistics per series) or DTW + hierarchical | — |
| High-dimensional | PCA (or domain feature selection) → any of the above | — |

- **IF** the methods disagree strongly → the structure is weak or the features are ambiguous. Revisit the features before tuning the algorithms.
- **IF** K-Means gives very unequal cluster sizes with outliers pulling centroids → remove/flag outliers first, or use HDBSCAN.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| K-Means on non-convex or very unequal clusters | Wrong partitions | Visualise; GMM/HDBSCAN |
| `n_init=1` | Local optimum depending on the seed | `n_init ≥ 10`; stability check |
| DBSCAN eps guessed blindly | All noise or one cluster | k-distance plot; HDBSCAN |
| Clustering unscaled mixed-unit features | Units define the clusters | Standardise or use domain weights |
| Treating DBSCAN noise as errors | Losing transition or rare-regime information | Inspect the noise points |
| Single linkage for compact groups | Chaining | Ward/average linkage |

## 10. Checklist
- [ ] Shape hypothesis from visualisation.
- [ ] 2–3 algorithms matching the hypothesis.
- [ ] Parameters set with diagnostics (k range, k-distance plot, dendrogram).
- [ ] Multiple initialisations; agreement between methods measured.
- [ ] Noise/uncertain points inspected.
- [ ] Validation and interpretation next.

## 11. References
- scikit-learn User Guide, *Clustering* (overview table of methods and use cases). https://scikit-learn.org/stable/modules/clustering.html
- scikit-learn User Guide, *Gaussian mixture models*. https://scikit-learn.org/stable/modules/mixture.html
- Ester, M., Kriegel, H.-P., Sander, J. & Xu, X. (1996). "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise." *KDD*.
- Campello, R. J. G. B., Moulavi, D. & Sander, J. (2013). "Density-Based Clustering Based on Hierarchical Density Estimates." *PAKDD*. HDBSCAN. https://doi.org/10.1007/978-3-642-37456-2_14
- Arthur, D. & Vassilvitskii, S. (2007). "k-means++: The Advantages of Careful Seeding." *SODA*.
- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *ESL*, §14.3 "Cluster Analysis". https://hastie.su.domains/ElemStatLearn/
