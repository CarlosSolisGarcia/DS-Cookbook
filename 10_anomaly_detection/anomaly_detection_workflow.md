# Anomaly Detection Workflow: Framing, Data, Methods and Operation

> **Folder:** 10_anomaly_detection · **Start here.** · **Next:** [statistical_methods.md](statistical_methods.md) · **Related:** [outliers.md](../01_data_understanding/outliers.md) (outliers as a data-quality issue), [unsupervised_workflow.md](../09_unsupervised_learning/unsupervised_workflow.md), [16_industrial_sensor_data](../16_industrial_sensor_data/sensor_data_workflow.md), [class_imbalance.md](../04_classification/class_imbalance.md)

---

## 1. Purpose
Build systems that flag **unusual observations or behaviour** (equipment faults, sensor failures, process upsets, unusual operating conditions) when labelled anomalies are scarce or absent. They should catch what matters while producing a false-alarm rate operators will tolerate.

## 2. When to use it
- Monitoring equipment or processes for faults; screening data for errors; detecting novel conditions (inputs unlike the training data of a deployed model).
- Labels for the abnormal class are absent, very few, or not representative of future faults (new fault types will appear).
- **If you have enough labelled examples of the faults that matter** (hundreds, representative), supervised classification is usually better ([04_classification](../04_classification/classification_workflow.md)). Anomaly detection is for the unknown and the rare.

## 3. Conceptual foundation

### 3.1 Types of anomalies
| Type | Definition | Industrial example (cookbook process data) | Detected by |
|---|---|---|---|
| **Point** | A single observation far from the normal distribution | Sensor spike (glitch) | Univariate robust z, most multivariate methods |
| **Contextual** | Normal in general, abnormal *given the context* (regime, time, load) | Vibration 2.4 mm/s: normal at 100% load, abnormal at 60% (bearing fault) | Regime-aware models; regression residuals; conditional models |
| **Collective / slow** | Individually normal points whose pattern over time is abnormal | Fouling: efficiency decays slowly over days; sensor drift | Residual trends; EWMA/CUSUM; change detection |
| **Relationship (correlation) break** | Variables stop co-varying as physics dictates | Outlet temperature drifts while flow and inlet are unchanged | PCA-SPE, regression residuals, autoencoders |

### 3.2 Approaches, from most to least domain knowledge
1. **Rules and limits** (alarm limits, physical constraints): transparent, but static.
2. **Physics / soft-sensor residuals:** predict each variable from the others (or from a physics model). An anomaly is a large residual. This is very powerful for relationship breaks ([time_series_anomalies.md](time_series_anomalies.md)).
3. **Statistical models of normality:** robust z, Mahalanobis, PCA T²/SPE, control charts ([statistical_methods.md](statistical_methods.md)).
4. **ML density/isolation methods:** Isolation Forest, LOF, One-Class SVM, GMM, autoencoders ([ml_methods.md](ml_methods.md)).

**Novelty vs outlier detection.** In *novelty detection* you train on clean normal data and score new data (the usual industrial setup). In *outlier detection* you score a contaminated dataset in which the anomalies are mixed in. scikit-learn's LOF distinguishes the two (`novelty=True/False`).

**Contamination** is the expected fraction of anomalies. Several sklearn estimators use it *only* to set the decision threshold (`predict`). The scores (`score_samples`) do not depend on it. Choose thresholds deliberately ([evaluation_and_thresholds.md](evaluation_and_thresholds.md)).

### 3.3 The operating loop
```
define normal (period, regimes) → features → model of normality → score → threshold → alarm logic (persistence)
      ↑                                                                                          │
      └──────── feedback: operator verdicts, maintenance records, new normal behaviour ◄──────────┘
```

## 4. Assumptions and limitations
- "Normal" must be defined. Training data contaminated by undetected faults teach the model that faults are normal.
- Anomaly ≠ fault. Planned shutdowns, product changes and calibration runs are unusual but expected. Encode known events so they are not flagged.
- An anomaly detector says "unusual", not "why". Root-cause support must be designed in ([root_cause_analysis.md](root_cause_analysis.md)).
- Evaluation without labels is weak. Plan how you will obtain at least a small labelled validation set (historical incidents, injected faults, expert review).

## 5. Step-by-step procedure
1. **Frame:** what must be detected (fault types, data errors, novelty)? How fast? What does a false alarm cost (operator time, trust) compared with a miss? What alarm rate is acceptable (e.g. ≤ 1 false alarm per shift)?
2. **Understand the data:** sampling, regimes, known events, missing data ([16_industrial_sensor_data](../16_industrial_sensor_data/sensor_data_workflow.md)).
3. **Define normal:** select a reference period with no known faults (maintenance logs). Exclude planned events, or encode them as context.
4. **Build a labelled evaluation set:** historical incidents, injected synthetic faults, or expert-labelled samples. Even 20–50 events help.
5. **Choose features:** raw sensors, residuals from physics/regression, rolling statistics (for slow faults), regime indicators.
6. **Start simple:** robust z per sensor (regime-aware), then PCA-SPE / regression residuals, then ML methods ([statistical_methods.md](statistical_methods.md), [ml_methods.md](ml_methods.md)).
7. **Set thresholds and alarm logic** for the target false-alarm rate; evaluate per fault type, the detection delay and the alarms per day ([evaluation_and_thresholds.md](evaluation_and_thresholds.md)).
8. **Add root-cause support:** per-sensor contributions ([root_cause_analysis.md](root_cause_analysis.md)).
9. **Operate:** collect operator feedback, retrain on new normal data periodically, and monitor the alarm rates.

## 6. Python implementation: framing the reference period and a first detector

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cookbook_data import make_process_sensors

df = make_process_sensors()
SENSORS = ["flow_m3h", "inlet_temp_c", "outlet_temp_c", "pressure_drop_bar", "pump_current_a", "vibration_mm_s"]
days = (df.index - df.index[0]) / pd.Timedelta("1D")

# Planned shutdowns are KNOWN events: excluded from modelling (they are not anomalies)
op = df[df["regime"] != "shutdown"]
op_days = (op.index - df.index[0]) / pd.Timedelta("1D")
reference = op[op_days < 9.5]                      # maintenance logs: no faults before day 10
monitor = op[op_days >= 9.5]
print(f"reference: {len(reference)} samples; monitoring: {len(monitor)} samples; "
      f"labelled anomaly share in monitoring = {monitor['is_anomaly'].mean():.2%}")
print(monitor["fault"].value_counts())

fig, axes = plt.subplots(len(SENSORS), 1, figsize=(13, 11), sharex=True)
for ax, c in zip(axes, SENSORS):
    ax.plot(df.index, df[c], lw=0.4)
    for f, col in [("sensor_drift", "orange"), ("process_upset", "red"), ("bearing", "purple"), ("fouling", "brown")]:
        m = df["fault"] == f
        ax.scatter(df.index[m], df.loc[m, c], s=1, color=col)
    ax.set_ylabel(c, fontsize=7)
plt.tight_layout(); plt.show()
```

```python
# First detector: per-sensor robust z-score w.r.t. the REFERENCE period, regime-aware
def robust_stats(ref, cols, by="regime"):
    g = ref.groupby(by)[cols]
    return g.median(), (g.quantile(0.75) - g.quantile(0.25)) / 1.349        # IQR → robust sd

def regime_z(data, med, sd, cols, by="regime"):
    return (data[cols] - med.loc[data[by]].to_numpy()) / sd.loc[data[by]].to_numpy()

med, sd = robust_stats(reference, SENSORS)
z = regime_z(monitor, med, sd, SENSORS)
score = z.abs().max(axis=1)                                         # the most deviating sensor
alarm = score > 6
summary = monitor.assign(alarm=alarm).groupby("fault")["alarm"].mean().round(3)
print("share of samples alarmed, by (true) condition:\n", summary)
```

## 7. Interpretation
- **The time plots** show why the fault types differ in difficulty. The process upset and the spikes are obvious. The bearing fault raises vibration only modestly, and the sensor drift and fouling are slow and small in each sensor.
- **The regime-aware robust z (threshold 6)** catches all spikes, ~80% of the bearing period (vibration *for this regime* is abnormal: a contextual anomaly that a global limit would miss) and ~60% of the fouling period, with a false-alarm rate of ~0.4% of samples (`none`). It **misses the process upset**: the +6 °C inlet step is ≈ 5 robust standard deviations, just below the threshold. The threshold choice matters ([evaluation_and_thresholds.md](evaluation_and_thresholds.md)). It also **misses the sensor drift almost entirely**, because the drifting value stays inside the per-sensor normal range. Catching it requires the *relationship* between sensors ([statistical_methods.md](statistical_methods.md): PCA-SPE; [time_series_anomalies.md](time_series_anomalies.md): residuals).
- **Rule:** which fault types a method catches is governed by the *type* of anomaly (§3.1). Choose methods for the anomaly types you must detect.

## 8. Decision framework
**IF labels:**
- **none at all** → unsupervised model of normality; create an evaluation set by fault injection or expert review before trusting any threshold.
- **a few (tens)** → unsupervised scoring + use the labels for **threshold selection and method comparison** (not for training).
- **many, representative** → supervised classification (possibly with anomaly scores as features).

**IF the anomaly type is:**
- point → robust z / IQR per sensor, Isolation Forest.
- contextual (depends on load/regime) → regime-aware statistics, regression residuals given the context, conditional models.
- relationship break / sensor fault → PCA-SPE, regression/physics residuals, autoencoder reconstruction.
- slow/collective → residuals + EWMA/CUSUM, rolling-window features, change-point detection.

**IF operating regimes exist** → model normality per regime or condition on the regime. Global models treat regime changes as anomalies, or blur everything together.

**IF false alarms are the main complaint** → raise the thresholds for a target false-alarm rate, add persistence (k-of-n) logic, suppress alarms during known events, and fix the noisy sensors ([evaluation_and_thresholds.md](evaluation_and_thresholds.md)).

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Training on data containing faults | Faults learned as normal | Clean reference period from maintenance logs |
| Ignoring regimes and planned events | Floods of false alarms at regime changes | Regime-aware models; event masks |
| Using `contamination` as if it were known | Arbitrary thresholds | Threshold from the target false-alarm rate on reference data |
| One global score with no explanation | Operators cannot act | Per-sensor contributions |
| Evaluating on the same period used for training | Optimistic false-alarm rates | Separate validation period |
| Expecting one method to catch every anomaly type | Blind spots | Combine complementary detectors |

## 10. Checklist
- [ ] Detection targets, speed and the acceptable false-alarm rate defined.
- [ ] Known events and regimes identified.
- [ ] Clean reference period chosen.
- [ ] Evaluation set (historical/injected/expert) prepared.
- [ ] Methods chosen per anomaly type; simple first.
- [ ] Thresholds and alarm logic tuned for the target false-alarm rate.
- [ ] Root-cause support and feedback loop designed.

## 11. References
- Chandola, V., Banerjee, A. & Kumar, V. (2009). "Anomaly detection: A survey." *ACM Computing Surveys* 41(3), 15. https://doi.org/10.1145/1541880.1541882
- Aggarwal, C. C. (2017). *Outlier Analysis*, 2nd ed. Springer.
- scikit-learn User Guide, *Novelty and Outlier Detection*. https://scikit-learn.org/stable/modules/outlier_detection.html
- Montgomery, D. C. (2019). *Introduction to Statistical Quality Control*, 8th ed. Wiley. Control charts, MSPC.
- Pang, G., Shen, C., Cao, L. & van den Hengel, A. (2021). "Deep Learning for Anomaly Detection: A Review." *ACM Computing Surveys* 54(2). https://doi.org/10.1145/3439950
