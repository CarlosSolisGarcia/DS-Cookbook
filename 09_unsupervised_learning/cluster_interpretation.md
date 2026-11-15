# Interpreting and Profiling Clusters

> **Folder:** 09_unsupervised_learning · **Prev:** [cluster_validation.md](cluster_validation.md) · **Next:** [anomaly_detection_workflow.md](../10_anomaly_detection/anomaly_detection_workflow.md) · **Related:** [model_interpretation.md](../06_model_improvement/model_interpretation.md)

---

## 1. Purpose
Describe what distinguishes each cluster, give the clusters meaningful names with domain experts, check that they are actionable, and put them to use: as labels, strata, regime indicators or features.

## 2. When to use it
After validation has shown a stable, meaningful clustering. Profiling also often exposes problems the indices missed (a "cluster" that is just a data-quality artefact).

## 3. Conceptual foundation
- **Profile = how the cluster differs from the rest.** Use robust statistics per cluster (median, IQR) for the clustering features *and* for variables that were **not** used in the clustering (outcomes, metadata). The latter are the real test of meaning.
- **Effect sizes, not p-values.** With thousands of points, every difference is "significant". Report standardised differences, e.g. (median_cluster − median_all)/IQR_all.
- **Surrogate model explanation:** train a shallow decision tree to predict the cluster label from the features. Its rules give a compact, human-readable description. Fidelity (the tree's accuracy) tells you how faithful the description is.
- **Prototypes:** show real example rows close to each centroid (medoids). Experts understand examples better than tables.

## 4. Assumptions and limitations
- Names and stories are hypotheses. A cluster called "fouled operation" has to be confirmed (inspection records, lab data).
- Post-hoc comparison of variables *used* to form the clusters is circular: of course they differ. Use the external variables for validation.
- Clusters drift over time. Re-assess them periodically, or use a supervised model trained on the cluster labels to assign new data consistently.

## 5. Step-by-step procedure
1. **Size and stability** of each cluster (from validation).
2. **Feature profiles:** median/IQR per cluster; a standardised-difference heatmap.
3. **External variables:** outcome rates, time, metadata per cluster.
4. **Surrogate tree rules** for a compact description; report the fidelity.
5. **Prototypes/medoids:** a few representative rows per cluster.
6. **Workshop with experts:** names, plausibility, actions.
7. **Operationalise:** assign new data (`predict` of K-Means/GMM, or a classifier trained on the labels) and monitor the cluster proportions over time.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from cookbook_data import make_pump_fleet

# Use case: profile pump UNITS by their operating behaviour in the first 60 days (fleet segmentation)
fleet = make_pump_fleet()
early = fleet[fleet["day"] <= 60]
profile = early.groupby("unit_id").agg(
    vib_mean=("vibration_rms", "mean"), temp_mean=("bearing_temp_c", "mean"),
    flow_mean=("flow_m3h", "mean"), current_mean=("motor_current_a", "mean"),
    share_B=("setting", lambda s: (s == "B").mean()), share_C=("setting", lambda s: (s == "C").mean()))
external = fleet.groupby("unit_id").agg(life_days=("day", "max"), site=("site", "first"))   # NOT used for clustering
features = list(profile.columns)
Xs = StandardScaler().fit_transform(profile)

km = KMeans(3, n_init=20, random_state=0).fit(Xs)
profile["cluster"] = km.labels_
print(profile["cluster"].value_counts().sort_index())
```

```python
# Standardised-difference heatmap: (cluster median − overall median) / overall IQR
overall_med = profile[features].median()
iqr = profile[features].quantile(0.75) - profile[features].quantile(0.25)
diff = (profile.groupby("cluster")[features].median() - overall_med) / iqr
sns.heatmap(diff, annot=True, fmt=".1f", cmap="vlag", center=0); plt.title("cluster profiles (robust z)"); plt.show()

# External validation: do the clusters differ in variables NOT used to build them?
ext = profile[["cluster"]].join(external)
print(ext.groupby("cluster")["life_days"].describe()[["count", "50%", "25%", "75%"]].round(0))
print(pd.crosstab(ext["cluster"], ext["site"], normalize="index").round(2))

# Is the difference in median life larger than chance? Permutation test (few units per cluster!)
def median_range(lbl, life):
    return life.groupby(lbl).median().agg(lambda m: m.max() - m.min())
obs = median_range(ext["cluster"], ext["life_days"])
rng = np.random.default_rng(0)
null = [median_range(pd.Series(rng.permutation(ext["cluster"].to_numpy()), index=ext.index), ext["life_days"])
        for _ in range(2000)]
print(f"observed range of cluster medians = {obs:.0f} days, permutation p ≈ {np.mean(np.array(null) >= obs):.2f}")
```

```python
# Surrogate tree: compact rules + fidelity
tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(profile[features], profile["cluster"])
print(f"surrogate fidelity (training accuracy) = {tree.score(profile[features], profile['cluster']):.2f}")
print(export_text(tree, feature_names=features))

# Prototypes: the real unit closest to each centroid
proto_idx, _ = pairwise_distances_argmin_min(km.cluster_centers_, Xs)
print(profile.iloc[proto_idx].round(2))
```

## 7. Interpretation
- **Heatmap:** clusters differ mainly in the **setting shares** and in the level variables that depend on the setting (flow, current). They describe *how the pumps are operated*, not their health. In the first 60 days the degradation is still small.
- **External validation:** cluster 0 (higher share of setting B, the high-duty setting) shows a shorter median life (≈ 194 vs ≈ 240 days). That is tempting to report, but there are only 18–32 units per cluster and the IQRs are wide. The **permutation test** shows how often random relabelling produces a difference this large. In the generator, lifetime is in fact independent of duty, so a non-small p-value is the correct, honest result. With small groups, always quantify the chance before telling a story.
- **Surrogate tree:** with high fidelity (> 0.9), the two or three rules are a faithful summary, e.g. "share_B > 0.4 → high-duty cluster". Low fidelity means the clusters have complex boundaries, so describe them with prototypes instead.
- **Prototypes** give experts concrete units to look at ("pump U017 is a typical member of cluster 2").
- **Red flag:** a cluster defined by one site, or by a calibration offset (e.g. bearing temperature +4 °C), is probably measuring a sensor or site artefact rather than behaviour. Check before naming it.

## 8. Decision framework
- **IF** the clusters differ only in the features that were used, and in no external variable → they may be valid descriptions but of limited use. Ask whether a simpler rule (binning one variable) is clearer.
- **IF** a cluster corresponds to a data artefact (one sensor, one site, one period) → fix the data or adjust the features, then re-cluster.
- **IF** the clusters are to be used operationally → train a classifier (or use `predict`) for assignment; monitor the proportions and the distance to the centroids for drift.
- **IF** the clusters will become labels for supervised learning → treat them as noisy, derived labels and document their origin.
- **IF** stakeholders want to act on the clusters → define the action per cluster, and evaluate whether outcomes improve (ideally with an experiment).

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Profiling only with the clustering features | Circular "validation" | External variables |
| p-values for cluster differences | Everything "significant" | Effect sizes |
| Names without expert confirmation | Misleading narratives | Workshops; prototypes |
| Static clusters in a drifting process | Wrong assignments over time | Monitor; re-cluster periodically |
| Ignoring artefact clusters | Decisions based on sensor quirks | Check site/sensor/period composition |

## 10. Checklist
- [ ] Cluster sizes and stability known.
- [ ] Robust profiles + standardised differences.
- [ ] External variables compared.
- [ ] Surrogate tree with fidelity; prototypes.
- [ ] Expert naming and plausibility check.
- [ ] Assignment method and monitoring defined.

## 11. References
- Hennig, C., Meila, M., Murtagh, F. & Rocci, R. (eds.) (2015). *Handbook of Cluster Analysis*. CRC Press. Chapters on cluster validation and interpretation.
- Molnar, C. (2022). *Interpretable Machine Learning*, "Global Surrogate". https://christophm.github.io/interpretable-ml-book/global.html
- Kaufman, L. & Rousseeuw, P. J. (1990). *Finding Groups in Data: An Introduction to Cluster Analysis*. Wiley (medoids/PAM).
