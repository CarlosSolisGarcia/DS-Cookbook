# ML Anomaly Detection: Isolation Forest, LOF, One-Class SVM, Density, Clustering and Autoencoders

> **Folder:** 10_anomaly_detection · **Prev:** [statistical_methods.md](statistical_methods.md) · **Next:** [time_series_anomalies.md](time_series_anomalies.md) · **Related:** [clustering_algorithms.md](../09_unsupervised_learning/clustering_algorithms.md), [evaluation_and_thresholds.md](evaluation_and_thresholds.md)

---

## 1. Purpose
Apply flexible, non-parametric detectors when normal behaviour is non-Gaussian, non-linear or high-dimensional. Understand what each one considers "anomalous", so you can pick a method that matches the anomaly types you must detect.

## 2. When to use it
After a statistical baseline ([statistical_methods.md](statistical_methods.md)), when normal data are multimodal or non-linear, or when there are many features. Not as a first step: their flexibility makes them harder to explain and calibrate.

## 3. Conceptual foundation
| Method | Anomaly = | Strengths | Weaknesses | Key parameters |
|---|---|---|---|---|
| **Isolation Forest** (IF) | Points isolated by few random axis-aligned splits | Fast, scales, few assumptions; good for **point/global** outliers in tabular data | Weak on **relationship breaks** (axis-aligned splits); weak in low-density valleys between clusters | `n_estimators`, `max_samples`, `max_features` |
| **Local Outlier Factor** (LOF) | Local density much lower than that of its neighbours | Handles clusters of different density; **local/contextual** anomalies | O(n²)-ish; sensitive to scaling and `n_neighbors`; needs `novelty=True` to score new data | `n_neighbors` |
| **One-Class SVM** | Outside a learned boundary enclosing (1 − ν) of the data | Flexible boundaries (RBF) | Sensitive to `gamma`, `nu`, scaling; slow for large n | `nu`, `gamma` |
| **Density (GMM / KDE)** | Low likelihood under a fitted density | Probabilistic, interpretable mixture components (≈ regimes) | KDE suffers in high dimensions; GMM needs k | `n_components`, covariance type / bandwidth |
| **Clustering-based** | Far from the nearest centroid/cluster, or DBSCAN noise | Simple; uses regime structure | Rare faults can form their own small cluster | k, eps |
| **Autoencoder** | Large reconstruction error after compressing and decompressing | Non-linear relationship breaks; high-dimensional data (signals, images) | Needs more data and tuning; can reconstruct anomalies too well | Architecture, bottleneck size, epochs |

**Isolation Forest intuition.** Anomalies are "few and different". Random splits isolate them quickly, so their average path length is short. A fault that only breaks a *correlation* (both values individually normal) is hard to isolate with axis-aligned splits.

**Autoencoder intuition (non-linear PCA).** Train a network to reproduce normal inputs through a narrow bottleneck. It learns the normal manifold. Inputs off the manifold reconstruct poorly. A linear autoencoder with squared loss spans the same subspace as PCA, so PCA-SPE is the natural baseline to beat.

## 4. Assumptions and limitations
- All of them assume that the training data are mostly normal. Contamination shifts the learned normal region.
- Scaling matters for LOF, OCSVM, GMM and autoencoders. It matters much less for IF.
- `contamination` (IF/LOF) sets only the default `predict` threshold. Use `score_samples` and calibrate your own threshold.
- Autoencoders on small tabular data rarely beat PCA-SPE or regression residuals. Use them when the data are large, non-linear or high-dimensional.

## 5. Step-by-step procedure
1. Use the same reference/monitoring split and scaling as the statistical baseline.
2. Fit IF, LOF (`novelty=True`), OCSVM (on a subsample if n is large) and GMM on the reference data.
3. Score the monitoring data with `score_samples` (negate it so that higher = more anomalous).
4. Set thresholds on the reference-score quantiles (same target false-alarm rate for all).
5. Compare detection per fault type with the statistical baseline (PCA-SPE).
6. Optionally train an autoencoder; compare it with PCA-SPE.
7. Choose or combine detectors by coverage of the required fault types and by explainability.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from cookbook_data import make_process_sensors

df = make_process_sensors()
S = ["flow_m3h", "inlet_temp_c", "outlet_temp_c", "pressure_drop_bar", "pump_current_a", "vibration_mm_s"]
op = df[df["regime"] != "shutdown"]
d = (op.index - df.index[0]) / pd.Timedelta("1D")
ref, mon = op[d < 9.5], op[d >= 9.5]
sc = StandardScaler().fit(ref[S]); Xr, Xm = sc.transform(ref[S]), sc.transform(mon[S])
FAULTS = ["sensor_drift", "process_upset", "bearing", "fouling", "spike"]

def detection_row(name, s_ref, s_mon, q=0.995):
    thr = np.quantile(s_ref, q); f = mon["fault"].to_numpy()
    return {"method": name, "false_alarm_rate": np.mean(s_mon[f == "none"] > thr),
            **{k: np.mean(s_mon[f == k] > thr) for k in FAULTS}}

detectors = {
    "Isolation Forest": IsolationForest(n_estimators=300, random_state=0).fit(Xr),
    "LOF (novelty, k=35)": LocalOutlierFactor(n_neighbors=35, novelty=True).fit(Xr),
    "One-Class SVM (nu=0.01)": OneClassSVM(nu=0.01, gamma="scale").fit(Xr[::2]),     # subsample for speed
    "GMM (4 comp.)": GaussianMixture(4, covariance_type="full", random_state=0).fit(Xr),
}
rows = [detection_row(n, -m.score_samples(Xr), -m.score_samples(Xm)) for n, m in detectors.items()]

km = KMeans(4, n_init=10, random_state=0).fit(Xr)                                # clustering-based score
rows.append(detection_row("KMeans distance to centroid", km.transform(Xr).min(1), km.transform(Xm).min(1)))

pca = PCA(2).fit(Xr)                                                              # statistical reference
spe = lambda X: ((X - pca.inverse_transform(pca.transform(X))) ** 2).sum(1)
rows.append(detection_row("PCA SPE (baseline)", spe(Xr), spe(Xm)))
```

```python
# Autoencoder (PyTorch, CPU): non-linear compression of normal data; anomaly score = reconstruction error
import torch
from torch import nn

torch.manual_seed(0)
class AE(nn.Module):
    def __init__(self, p, h=16, z=2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(p, h), nn.ReLU(), nn.Linear(h, z))
        self.dec = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Linear(h, p))
    def forward(self, x):
        return self.dec(self.enc(x))

def train_autoencoder(X, epochs=60, lr=1e-2, batch=256, val_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X)); n_val = int(val_frac * len(X))
    Xt, Xv = torch.tensor(X[idx[n_val:]], dtype=torch.float32), torch.tensor(X[idx[:n_val]], dtype=torch.float32)
    model, best, best_state = AE(X.shape[1]), np.inf, None
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        for b in torch.split(Xt[torch.randperm(len(Xt))], batch):
            opt.zero_grad(); loss = ((model(b) - b) ** 2).mean(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = ((model(Xv) - Xv) ** 2).mean().item()
        if v < best:                                                   # keep the best validation epoch
            best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best

ae, val_loss = train_autoencoder(Xr)
def ae_score(X):
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32)
        return ((ae(Xt) - Xt) ** 2).sum(1).numpy()
print(f"autoencoder validation MSE (normal data): {val_loss:.4f}")
rows.append(detection_row("Autoencoder (6-16-2-16-6)", ae_score(Xr), ae_score(Xm)))
print(pd.DataFrame(rows).set_index("method").round(3).to_string())
```

## 7. Interpretation
Typical findings (compare them with your table):
- **Isolation Forest is weak on these faults.** The bearing fault, the fouling and the sensor drift are *relationship* anomalies: each value lies within the range seen at some load. Axis-aligned isolation does not see that the combination is inconsistent. In this run IF even misses the spikes. The reference period itself contains a few glitches (realistic contamination), and IF's compressed score range puts its 99.5% reference quantile at the level of those glitches. IF can work well for gross outliers in wide tabular data, but it is **not** a universal default for process monitoring. Always compare it with a statistical baseline.
- **LOF, GMM and OCSVM** do much better. They model the *local* density of the multi-regime normal data, so "vibration high for this flow" lands in a low-density region. OCSVM's false-alarm rate is noticeably above nominal. Its boundary depends strongly on `gamma`/`nu` and is fitted on a subsample. Re-calibrate it, or tune it on reference data.
- **K-Means distance** captures regime-relative deviations cheaply, and is often a surprisingly good baseline when regimes dominate.
- **The autoencoder** behaves like a non-linear PCA. On six correlated sensors it rarely beats PCA-SPE by much. Its advantage shows with many sensors, non-linear manifolds or raw signal windows.
- **No ML method catches the slow sensor drift well** point-by-point. Slow faults need temporal accumulation (EWMA/CUSUM, [time_series_anomalies.md](time_series_anomalies.md)).

## 8. Decision framework
| Situation | Recommended detector |
|---|---|
| Tabular data, gross point outliers, large n, need speed | Isolation Forest |
| Multi-regime / clustered normal data, local anomalies | LOF (novelty) or GMM; K-Means distance as a cheap baseline |
| Correlated process variables, sensor faults | PCA-SPE or regression residuals first; autoencoder if non-linear |
| High-dimensional signals or images | Autoencoder (or embeddings + kNN distance) |
| Need probabilities or regime insight | GMM |
| Small n (< ~1000) | Robust statistics, PCA, LOF; avoid deep models |
| Slow drift | Not these alone; add temporal methods |

- **IF** several detectors are complementary → combine their **rank-normalised** scores (max or mean of the percentiles) and calibrate the threshold on the combined score.
- **IF** a method's false-alarm rate on validation-normal data is far above nominal → the reference is not representative (regime mix, drift), or the method overfits the reference. Re-calibrate on a separate normal validation period.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Isolation Forest as the default for everything | Missed relationship faults | Match the method to the anomaly type |
| LOF with `novelty=False` to score new data | Wrong API use (`predict` unavailable / meaningless) | `novelty=True` for new data |
| Trusting `contamination`-based `predict` | Arbitrary alarm rates | Calibrate thresholds on `score_samples` |
| Unscaled inputs for distance/density methods | Units dominate | Scale with reference statistics |
| Autoencoder trained to zero error | Reconstructs anomalies too | Bottleneck, early stopping on validation-normal data |
| Comparing methods at different false-alarm rates | Unfair comparisons | Same reference-quantile thresholds |

## 10. Checklist
- [ ] Same reference/monitoring split and scaling as the baseline.
- [ ] Scores via `score_samples`; thresholds from reference quantiles.
- [ ] Detection per fault type compared with PCA-SPE / robust z.
- [ ] False-alarm rate on separate normal data checked.
- [ ] Autoencoder only if justified; compared with PCA.
- [ ] Combination considered if detectors are complementary.

## 11. References
- Liu, F. T., Ting, K. M. & Zhou, Z.-H. (2008). "Isolation Forest." *ICDM*, 413–422. https://doi.org/10.1109/ICDM.2008.17
- Breunig, M. M., Kriegel, H.-P., Ng, R. T. & Sander, J. (2000). "LOF: Identifying Density-Based Local Outliers." *SIGMOD*. https://doi.org/10.1145/342009.335388
- Schölkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J. & Williamson, R. C. (2001). "Estimating the Support of a High-Dimensional Distribution." *Neural Computation* 13(7), 1443–1471.
- scikit-learn User Guide, *Novelty and Outlier Detection* (incl. comparison of methods). https://scikit-learn.org/stable/modules/outlier_detection.html
- Sakurada, M. & Yairi, T. (2014). "Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction." *MLSDA Workshop*. https://doi.org/10.1145/2689746.2689747
- Han, S., Hu, X., Huang, H., Jiang, M. & Zhao, Y. (2022). "ADBench: Anomaly Detection Benchmark." *NeurIPS Datasets and Benchmarks*. https://arxiv.org/abs/2206.09426
