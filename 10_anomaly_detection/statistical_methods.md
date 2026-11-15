# Statistical Anomaly Detection: Robust Univariate Scores, Mahalanobis, PCA T²/SPE and Control Charts

> **Folder:** 10_anomaly_detection · **Prev:** [anomaly_detection_workflow.md](anomaly_detection_workflow.md) · **Next:** [ml_methods.md](ml_methods.md) · **Related:** [outliers.md](../01_data_understanding/outliers.md), [pca.md](../11_dimensionality_reduction/pca.md), [process_monitoring.md](../16_industrial_sensor_data/process_monitoring.md)

---

## 1. Purpose
Detect anomalies with transparent statistical models of normal operation: univariate robust scores, multivariate distances, PCA-based monitoring statistics (Hotelling's T² and the squared prediction error, SPE/Q) and control charts for shifts and drifts. These methods are the backbone of industrial **multivariate statistical process control (MSPC)**.

## 2. When to use it
- As the first, interpretable detector for sensor and process data.
- When correlated variables carry most of the information (MSPC).
- When small persistent shifts matter more than single outliers (EWMA/CUSUM).

## 3. Conceptual foundation
| Method | Score | Catches | Misses | Assumes |
|---|---|---|---|---|
| **Robust z** per variable: (x − median)/(1.4826·MAD) or IQR-based | max \|z\| over the variables | Point anomalies in one variable | Relationship breaks; contextual (unless regime-aware) | Unimodal per context |
| **Mahalanobis distance** d² = (x − μ)ᵀ Σ⁻¹ (x − μ), robust via MCD | d²; ≈ χ²(p) under normality | Unusual *combinations* | Multimodal data (several regimes) unless modelled per regime | One elliptical cloud |
| **PCA T²** | Mahalanobis distance *within* the k retained components | Unusual magnitude along the normal correlation structure (e.g. extreme but consistent operation) | Breaks orthogonal to the model | Linear correlation structure |
| **PCA SPE (Q)** | ‖x − x̂‖² of the reconstruction from k components | **Relationship breaks**: the new data don't fit the normal correlations (sensor faults, drifts) | Large but consistent moves | Linear structure; the k choice |
| **Shewhart chart** | Sample vs mean ± 3σ | Large sudden shifts | Small persistent shifts | Independent samples |
| **EWMA chart** | z_t = λx_t + (1 − λ)z_{t−1} | Small persistent shifts and drifts | Isolated spikes (by design) | Approximately independent residuals |
| **CUSUM** | S_t = max(0, S_{t−1} + (x_t − μ − k)) | Small persistent shifts, with a quick detection | — | Known in-control mean |

**Autocorrelation warning:** control-chart limits assume independent observations. Sensor data at high frequency are autocorrelated, which produces far more false alarms than nominal. Apply charts to **residuals** of a model that removes the dynamics, or to aggregated or subsampled data, and calibrate the limits empirically on reference data.

## 4. Assumptions and limitations
- The reference period must represent all normal regimes. Regimes missing from the reference period will alarm.
- Gaussian-theory limits (χ², F) are approximations. Prefer empirical quantiles from reference data (e.g. the 99.5th percentile), especially with non-normal, multimodal or autocorrelated data.
- Linear methods (PCA) miss non-linear relationship breaks. Kernel PCA, autoencoders or regression residuals with non-linear models can help ([ml_methods.md](ml_methods.md)).

## 5. Step-by-step procedure
1. **Reference data:** a clean normal period, with planned events excluded; identify the regimes.
2. **Scale** using reference statistics (per regime if the regimes differ in level).
3. **Univariate robust z** per sensor (regime-aware) → a baseline detector.
4. **Multivariate:** robust Mahalanobis, *per regime* if the data are multimodal.
5. **PCA monitoring:** choose k (explained variance ~90–95%, or cross-validation), compute T² and SPE, and set empirical limits on the reference data.
6. **Control charts** (EWMA/CUSUM) on residuals or on monitoring statistics for slow drifts.
7. **Evaluate** per fault type and calibrate the false-alarm rate ([evaluation_and_thresholds.md](evaluation_and_thresholds.md)).

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cookbook_data import make_process_sensors

df = make_process_sensors()
S = ["flow_m3h", "inlet_temp_c", "outlet_temp_c", "pressure_drop_bar", "pump_current_a", "vibration_mm_s"]
op = df[df["regime"] != "shutdown"]
d = (op.index - df.index[0]) / pd.Timedelta("1D")
ref, mon = op[d < 9.5], op[d >= 9.5]
FAULTS = ["sensor_drift", "process_upset", "bearing", "fouling", "spike"]

def detection_table(name, s_ref, s_mon, q=0.995):
    """Threshold = q-quantile of the reference scores; report the alarm share per condition."""
    thr = np.quantile(s_ref, q)
    out = {"method": name, "false_alarm_rate": np.mean(s_mon[mon["fault"].to_numpy() == "none"] > thr)}
    out.update({f: np.mean(s_mon[mon["fault"].to_numpy() == f] > thr) for f in FAULTS})
    return out

rows = []
sc = StandardScaler().fit(ref[S])
Xr, Xm = sc.transform(ref[S]), sc.transform(mon[S])
rows.append(detection_table("global robust z", np.abs(Xr).max(1), np.abs(Xm).max(1)))
```

```python
# Mahalanobis: GLOBAL (one ellipse for multimodal data) vs PER REGIME
mcd = MinCovDet(random_state=0).fit(Xr)
rows.append(detection_table("Mahalanobis (global MCD)", mcd.mahalanobis(Xr), mcd.mahalanobis(Xm)))

def per_regime_mahalanobis(ref, mon, cols, normalise=True, q=0.995):
    """One robust ellipse per regime. With normalise=True each regime's distances are divided by that
    regime's reference q-quantile, i.e. the threshold is calibrated PER REGIME (regimes differ in spread:
    the 'low' regime also contains the ramps around shutdowns, so its distances are much larger)."""
    s_ref, s_mon = np.zeros(len(ref)), np.zeros(len(mon))
    for r in ref["regime"].unique():
        a, b = (ref["regime"] == r).to_numpy(), (mon["regime"] == r).to_numpy()
        scr = StandardScaler().fit(ref.loc[a, cols])
        m = MinCovDet(random_state=0).fit(scr.transform(ref.loc[a, cols]))
        d_ref = m.mahalanobis(scr.transform(ref.loc[a, cols]))
        scale = np.quantile(d_ref, q) if normalise else 1.0
        s_ref[a] = d_ref / scale
        if b.any():
            s_mon[b] = m.mahalanobis(scr.transform(mon.loc[b, cols])) / scale
    return s_ref, s_mon

rows.append(detection_table("Mahalanobis per regime, ONE global threshold", *per_regime_mahalanobis(ref, mon, S, normalise=False)))
rows.append(detection_table("Mahalanobis per regime, per-regime threshold", *per_regime_mahalanobis(ref, mon, S)))
```

```python
# PCA monitoring (MSPC): T² within the model, SPE (Q) outside it
pca = PCA().fit(Xr)
k = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.90) + 1)
pca = PCA(n_components=k).fit(Xr)
print(f"components kept: {k} (explained {pca.explained_variance_ratio_.sum():.1%})")

def t2_spe(X):
    T = pca.transform(X)
    t2 = np.sum(T ** 2 / pca.explained_variance_, axis=1)
    spe = np.sum((X - pca.inverse_transform(T)) ** 2, axis=1)
    return t2, spe

t2_r, spe_r = t2_spe(Xr); t2_m, spe_m = t2_spe(Xm)
rows.append(detection_table("PCA T²", t2_r, t2_m))
rows.append(detection_table("PCA SPE (Q)", spe_r, spe_m))

fig, ax = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
for a, (name, s, sr) in zip(ax, [("T²", t2_m, t2_r), ("SPE", spe_m, spe_r)]):
    a.semilogy(mon.index, s, lw=0.4); a.axhline(np.quantile(sr, 0.995), color="r", ls="--"); a.set_ylabel(name)
plt.suptitle("PCA monitoring statistics with 99.5% reference limits"); plt.tight_layout(); plt.show()
```

```python
# EWMA chart on the SPE (log scale) to catch SLOW drifts; limits calibrated on reference data
def ewma(x, lam=0.05):
    out = np.empty_like(x, dtype=float); out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = lam * x[i] + (1 - lam) * out[i - 1]
    return out

log_spe_r, log_spe_m = np.log(spe_r), np.log(spe_m)
e_r = ewma(log_spe_r)
e_m = ewma(np.concatenate([log_spe_r[-200:], log_spe_m]))[200:]       # warm start from the end of the reference
rows.append(detection_table("EWMA(log SPE), λ=0.05", e_r, e_m))

# CUSUM on the outlet-temperature robust z (one-sided, upward shifts); k = allowance, h = decision interval
def cusum(z, k=0.5):
    s = np.zeros_like(z); 
    for i in range(1, len(z)):
        s[i] = max(0.0, s[i - 1] + z[i] - k)
    return s
z_out_r = Xr[:, S.index("outlet_temp_c")]; z_out_m = Xm[:, S.index("outlet_temp_c")]
rows.append(detection_table("CUSUM(outlet temp z)", cusum(z_out_r), cusum(z_out_m)))

print(pd.DataFrame(rows).set_index("method").round(3).to_string())
```

## 7. Interpretation
Read the table by row (method) and by column (fault type). The patterns are typical:
- **Global robust z** catches the large single-sensor deviations (the upset, many spikes) but not the relationship break (sensor drift).
- **Global Mahalanobis (one MCD ellipse) fails badly:** the reference data contain *several regimes*. The robust estimator models one of them, and its threshold then becomes huge, so almost nothing is flagged. **Per-regime ellipses with one global threshold still fail** (e.g. the process upset is missed). The regimes differ in spread: the "low" regime contains the ramps around shutdowns, so its reference distances are an order of magnitude larger, and they dictate the global threshold. **Per-regime models *and* per-regime thresholds** restore part of the detection (bearing 0.25 → 0.75). The process upset, which happened in the low-load regime, is *still* missed. The "low" reference contains the **transient ramps** before and after shutdowns, whose distances are huge, so that regime's threshold is inflated. The final lesson: **model and calibrate normality per operating regime, and treat transients (start-up, shutdown, ramps) as their own regime or exclude them**, e.g. a "steady state" filter on |Δload|. This is the most important lesson for multivariate statistical detection in industry.
- **PCA SPE** is strong for faults that break the correlation structure (bearing: vibration rises while flow does not; fouling: outlet temperature and pressure drop no longer fit the flow), and it catches spikes. **T²** responds to unusual but *consistent* operation, and is weaker here.
- **The sensor drift** (+3 °C over 2 days on one sensor) is hard for all point-wise scores. The **EWMA of the SPE** and the **CUSUM** accumulate small persistent deviations and catch part of it. They also flag the fouling strongly, at the price of a slower response and some residual false alarms (autocorrelation).
- **The false-alarm rate** on normal monitoring data should be close to the nominal 0.5% for well-calibrated methods. Much higher values mean that normal behaviour changed (drift, a new regime mix) or that the method is sensitive to autocorrelation.

## 8. Decision framework
- **IF** the data have several operating regimes → per-regime statistics, or condition on load (regression residuals). Never use one global Gaussian.
- **IF** the variables are highly correlated (typical process data) → PCA monitoring with T² + SPE. SPE first for sensor and relationship faults.
- **IF** small persistent shifts or drifts matter → EWMA/CUSUM on residuals or on log-SPE, with limits calibrated on reference data.
- **IF** the data are strongly autocorrelated → model the dynamics first (residuals), or aggregate. Empirical limits, not textbook 3σ.
- **IF** p > n or the variables are many → PCA (or PLS if a quality variable exists) instead of the full covariance.
- **IF** non-linear relationships dominate → regression residuals with non-linear models, or autoencoders ([ml_methods.md](ml_methods.md)).

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Global Gaussian model on multi-regime data | Blindness or floods of alarms | Per-regime models |
| Textbook χ² / 3σ limits on autocorrelated, non-normal data | Many false alarms | Empirical quantile limits on reference data |
| Ignoring SPE (only T²) | Missed sensor faults | Monitor both |
| Reference data contaminated by faults | Inflated limits | Clean reference; robust estimators (MCD) |
| Scaling with statistics from the monitoring period | Faults normalised away | Scale with reference statistics only |

## 10. Checklist
- [ ] Clean, regime-covering reference data; planned events excluded.
- [ ] Scaling with reference statistics (per regime if needed).
- [ ] Univariate robust baseline.
- [ ] Per-regime Mahalanobis or PCA T²/SPE with empirical limits.
- [ ] EWMA/CUSUM for slow drifts.
- [ ] Per-fault-type detection and false-alarm rate evaluated.

## 11. References
- MacGregor, J. F. & Kourti, T. (1995). "Statistical process control of multivariate processes." *Control Engineering Practice* 3(3), 403–414. https://doi.org/10.1016/0967-0661(95)00014-L
- Qin, S. J. (2003). "Statistical process monitoring: basics and beyond." *J. Chemometrics* 17(8–9), 480–502. https://doi.org/10.1002/cem.800
- Montgomery, D. C. (2019). *Introduction to Statistical Quality Control*, 8th ed. Wiley (Shewhart, EWMA, CUSUM; Ch. on MSPC).
- NIST/SEMATECH *e-Handbook of Statistical Methods*, §6.3 "Univariate and Multivariate Control Charts". https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc3.htm
- Rousseeuw, P. J. & Van Driessen, K. (1999). "A Fast Algorithm for the Minimum Covariance Determinant Estimator." *Technometrics* 41(3).
- scikit-learn, `MinCovDet`. https://scikit-learn.org/stable/modules/covariance.html#robust-covariance
