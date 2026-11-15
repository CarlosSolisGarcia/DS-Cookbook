# Unsupervised Learning Workflow: Problem Formulation Without Labels

> **Folder:** 09_unsupervised_learning · **Start here when there is no target.** · **Next:** [clustering_algorithms.md](clustering_algorithms.md) · **Related:** [10_anomaly_detection](../10_anomaly_detection/anomaly_detection_workflow.md), [11_dimensionality_reduction](../11_dimensionality_reduction/dimensionality_reduction_workflow.md), [exploratory_data_analysis.md](../01_data_understanding/exploratory_data_analysis.md)

---

## 1. Purpose
Turn a vague "find structure in this data" request into a concrete unsupervised task with a stated **use** and **success criterion**. Choose between clustering, anomaly detection and dimensionality reduction, and explore the data without the guidance of a target.

## 2. When to use it
- No labels exist, or labels exist only for a few examples.
- The goal is to discover groups (operating regimes, product families, customer or machine profiles), unusual cases, or a compact representation.
- **Not** when a target exists and prediction is the goal. Supervised learning is more reliable then, and clusters used as features rarely beat good supervised features.

## 3. Conceptual foundation
**Three unsupervised objectives (often confused):**

| Objective | Question | Output | Typical use | Recipe |
|---|---|---|---|---|
| **Clustering** | Which observations are similar to each other, forming groups? | A group label per row (or membership probabilities) | Operating regimes; segmentation; data stratification; label suggestions | This folder |
| **Anomaly detection** | Which observations are unlike the bulk? | A score per row + threshold | Fault detection; data-quality screening; novelty | [10_anomaly_detection](../10_anomaly_detection/anomaly_detection_workflow.md) |
| **Dimensionality reduction** | Can the variation be described by fewer variables? | Low-dimensional coordinates | Visualisation; compression; noise reduction; preprocessing | [11_dimensionality_reduction](../11_dimensionality_reduction/dimensionality_reduction_workflow.md) |

**No ground truth means validation is different.** Without labels, "is this clustering correct?" has no single answer. A clustering is *useful* if it is:
1. **Stable:** similar results on resampled data and with different seeds.
2. **Separated:** good internal metrics (silhouette etc.), interpreted with care.
3. **Interpretable:** clusters differ in ways a domain expert can name.
4. **Actionable:** the groups lead to a different decision, model or treatment.
5. **Externally consistent:** it agrees with known metadata (regime logs, product types) when these exist.

**Distance defines similarity.** Every clustering method relies on a notion of distance or density, so feature choice, transformation and scaling *are* the model. Two analysts with different scaling will find different clusters in the same data.

## 4. Assumptions and limitations
- Clustering algorithms always return clusters, even for structureless data. Check whether the structure exists at all (§6, and the null-reference comparison in [cluster_validation.md](cluster_validation.md)).
- Clusters in high-dimensional data suffer from distance concentration. Reduce the dimensionality or select features first.
- Clusters are descriptive. They do not prove distinct mechanisms. Validate with domain knowledge and, where possible, with follow-up data.

## 5. Step-by-step procedure
1. **State the use:** "Identify operating regimes so that the anomaly-detection thresholds can be regime-specific." Write down what a good result would allow you to do.
2. **Choose the objective** (clustering / anomaly detection / DR) with the table above.
3. **Define the unit and the features** that express the notion of similarity you care about. For regimes: process conditions (load, flow), not the fault symptoms (vibration) you want to detect later.
4. **Prepare the data:** data quality ([data_quality.md](../01_data_understanding/data_quality.md)); transformations of skewed features; scaling; possibly DR.
5. **Explore without labels:** univariate distributions (multi-modality hints at groups), pairwise scatter plots, PCA projection, nearest-neighbour distances.
6. **Fit 2–3 algorithms** suited to the expected cluster shapes ([clustering_algorithms.md](clustering_algorithms.md)).
7. **Validate:** internal metrics, stability, external consistency ([cluster_validation.md](cluster_validation.md)).
8. **Profile and name** the clusters with domain experts ([cluster_interpretation.md](cluster_interpretation.md)).
9. **Use the result** (regime-specific models, stratified sampling, reporting) and check that it improves the downstream task.

## 6. Python implementation: exploring structure without labels

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from cookbook_data import make_process_sensors

df = make_process_sensors()
normal = df[df["fault"] == "none"]               # regime discovery on normal operation
features = ["flow_m3h", "pressure_drop_bar", "pump_current_a", "outlet_temp_c"]   # process conditions
X = StandardScaler().fit_transform(normal[features])

# 1. Multi-modality in single variables hints at groups
fig, ax = plt.subplots(1, len(features), figsize=(16, 3))
for a, c in zip(ax, features):
    a.hist(normal[c], bins=60); a.set_title(c)
plt.tight_layout(); plt.show()

# 2. PCA projection: is there visible structure in 2-D?
pc = PCA(n_components=2, random_state=0).fit(X)
Z = pc.transform(X)
plt.figure(figsize=(5, 4)); plt.scatter(Z[:, 0], Z[:, 1], s=2, alpha=0.3)
plt.title(f"PCA (explained {pc.explained_variance_ratio_.sum():.0%})"); plt.show()
```

```python
# 3. Is there clustering tendency at all? Hopkins-like statistic: compare nearest-neighbour distances of
#    real points with those of uniformly random points in the same bounding box.
def hopkins(X, m=500, seed=0):
    rng = np.random.default_rng(seed)
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    idx = rng.choice(len(X), m, replace=False)
    w = nn.kneighbors(X[idx], 2)[0][:, 1]                         # real → nearest other real point
    U = rng.uniform(X.min(0), X.max(0), (m, X.shape[1]))
    u = nn.kneighbors(U, 1)[0][:, 0]                              # random → nearest real point
    return u.sum() / (u.sum() + w.sum())                          # ≈0.5: random-like, →1: clustered

rng = np.random.default_rng(1)
print(f"Hopkins (process data)  = {hopkins(X):.2f}")
print(f"Hopkins (uniform noise) = {hopkins(rng.uniform(size=(5000, 4))):.2f}")
```

**Problem statement template (deliverable):**
```text
UNSUPERVISED TASK — Operating regimes of heat-exchanger loop H-3
Use          : regime-specific normal ranges for monitoring (10_anomaly_detection); regime reporting
Unit         : one 5-minute snapshot of process conditions
Similarity   : flow, pressure drop, pump current, outlet temperature (standardised); NOT vibration
Data         : 45 days, fault periods excluded for discovery (labels from maintenance logs)
Success      : (1) clusters stable across bootstraps (ARI > 0.9); (2) operators can name each cluster;
               (3) regime-specific limits reduce false alarms vs global limits (measured in 10_anomaly_detection)
```

## 7. Interpretation
- **Histograms:** flow and current are clearly multi-modal (one peak per load level, plus a spike at zero for shutdowns). There are discrete operating regimes.
- **PCA scatter:** separated blobs along PC1 are the regimes. Elongated shapes within a blob come from continuous variation (daily temperature cycles).
- **Hopkins statistic:** ≈ 0.5 for uniform noise and close to 1 for clustered data. A value near 0.5 on your data means there may be no cluster structure worth looking for, so stop or reframe (perhaps as dimensionality reduction).

## 8. Decision framework
**IF the dataset has no labels:**
- **IF** the question is "which groups exist?" → clustering. Plan to validate by stability and expert interpretation.
- **IF** the question is "which cases are unusual / faulty?" → anomaly detection, *not* clustering. A clustering optimises the description of the bulk, not the detection of the rare.
- **IF** the question is "can I see/compress the data?" or the features are many and redundant → dimensionality reduction, often as a first step before either of the above.
- **IF** a few labels exist → use them for **external validation** of the unsupervised result, or consider semi-supervised/weak-supervision approaches ([19_advanced_tabular_ml](../19_advanced_tabular_ml/noisy_labels_weak_supervision.md)).

**IF clustering is proposed as a preprocessing step for supervised learning:**
→ Test it. Compare the supervised CV score with vs without cluster features (fitted inside the pipeline). It often does not help, because GBMs find these partitions themselves.

**IF stakeholders want "segments" for decisions:**
→ Insist on actionability. Each segment must imply a different action. Otherwise simple rules (quantile bins of a key variable) may be clearer.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Clustering without a stated use | Unverifiable results; endless tweaking | Write the use and success criteria first |
| Including the variables you want to explain/detect | Circular conclusions | Separate the "similarity" features from the "outcome" features |
| No scaling (or unjustified scaling) | Clusters driven by units | Standardise, or use domain-meaningful weights |
| Assuming clusters exist | Over-interpreting noise | Clustering-tendency check; null-reference comparison |
| Using clustering for anomaly detection | Rare cases absorbed into clusters | Dedicated anomaly methods |

## 10. Checklist
- [ ] Use and success criteria written.
- [ ] Objective chosen (clustering / anomaly detection / DR).
- [ ] Unit and similarity features justified; outcome variables kept separate.
- [ ] Data quality, transformations and scaling done.
- [ ] Clustering tendency checked.
- [ ] Algorithms, validation and interpretation done via the next recipes.
- [ ] Downstream benefit measured.

## 11. References
- Hennig, C. (2015). "What are the true clusters?" *Pattern Recognition Letters* 64, 53–62. https://doi.org/10.1016/j.patrec.2015.04.009
- von Luxburg, U., Williamson, R. C. & Guyon, I. (2012). "Clustering: Science or Art?" *JMLR W&CP* 27, 65–79. https://proceedings.mlr.press/v27/luxburg12a.html
- James, G. et al. (2023). *ISLP*, Ch. 12 "Unsupervised Learning". https://www.statlearning.com/
- scikit-learn User Guide, *Clustering*. https://scikit-learn.org/stable/modules/clustering.html
- Lawson, R. G. & Jurs, P. C. (1990). "New index for clustering tendency and its application to chemical problems." *J. Chem. Inf. Comput. Sci.* 30(1), 36–41. Hopkins statistic in chemistry. https://doi.org/10.1021/ci00065a010
