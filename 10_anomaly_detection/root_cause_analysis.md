# Interpreting Anomalies and Supporting Root-Cause Investigation

> **Folder:** 10_anomaly_detection · **Prev:** [evaluation_and_thresholds.md](evaluation_and_thresholds.md) · **Next:** [dimensionality_reduction_workflow.md](../11_dimensionality_reduction/dimensionality_reduction_workflow.md) · **Related:** [model_interpretation.md](../06_model_improvement/model_interpretation.md), [process_monitoring.md](../16_industrial_sensor_data/process_monitoring.md)

---

## 1. Purpose
Turn "the anomaly score is high" into "**which variables, in which way, since when, and under which conditions**". This is the information an engineer needs to diagnose the cause, decide on an action and give feedback that improves the detector.

## 2. When to use it
For every alarm presented to people, and for retrospective analysis of incidents.

## 3. Conceptual foundation
| Technique | Output | Works with |
|---|---|---|
| **Per-variable robust z** (regime-aware) | Which variables are out of their normal range | Any detector, as context |
| **Contribution to SPE** | e_j² = (x_j − x̂_j)², the variables that break the correlation model | PCA monitoring |
| **Contribution to T²** | Variables driving the unusual (but consistent) operation | PCA monitoring |
| **Soft-sensor residuals** | Which physical relationship is violated | Residual-based detection |
| **Reconstruction error per feature** | Same idea as SPE for autoencoders | Autoencoders |
| **SHAP / feature attributions on the score** | Contribution of each feature to the anomaly score | Isolation Forest (TreeExplainer) and other tree models |
| **Temporal context** | Onset time (change point), trend before the alarm, recent events (maintenance, regime changes) | All |
| **Similar past cases** | Nearest neighbours among labelled past incidents | Requires an incident library |

**Smearing.** In PCA, a fault in one variable spreads into the reconstruction of the others, so contribution plots can point to neighbouring variables. Always check the top contributors in the raw time plots. Reconstruction-based contributions are better than naive ones.

**Correlation is not causation, again.** The variable with the largest contribution is where the anomaly is *visible*. That is not necessarily its cause. A fouled exchanger shows up in the outlet temperature *and* the pressure drop, and the cause is deposition. Diagnostics narrow down the search; engineering reasoning concludes it.

## 4. Assumptions and limitations
- The explanations are only as good as the normal model. If the detector is miscalibrated, the explanations mislead too.
- SHAP on Isolation Forest explains the *isolation score*, which may emphasise variables with sparse tails rather than the physics.
- Automatic root-cause classification needs labelled incidents. Without them, aim for *decision support*, not automatic diagnosis.

## 5. Step-by-step procedure
1. **For an alarm,** retrieve the window around it (e.g. −24 h … +2 h).
2. **Rank the variables** by their contribution (SPE contributions, residual z, robust z).
3. **Plot the top contributors** with their normal band (regime-aware) and the alarm time.
4. **Estimate the onset** (change point or first sustained exceedance), and note concurrent events (regime changes, maintenance, weather).
5. **Formulate hypotheses** with an engineer (sensor fault? process change? equipment degradation?) and propose checks (calibration, inspection, lab sample).
6. **Record the outcome** (true/false alarm, cause, action) in an incident library. It feeds threshold tuning and, eventually, supervised fault classification.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from cookbook_data import make_process_sensors

df = make_process_sensors()
S = ["flow_m3h", "inlet_temp_c", "outlet_temp_c", "pressure_drop_bar", "pump_current_a", "vibration_mm_s"]
op = df[df["regime"] != "shutdown"].copy()
op["day"] = (op.index - df.index[0]) / pd.Timedelta("1D")
ref, mon = op[op["day"] < 9.5], op[op["day"] >= 9.5]
sc = StandardScaler().fit(ref[S]); pca = PCA(2).fit(sc.transform(ref[S]))

def spe_contributions(d):
    X = sc.transform(d[S]); E = X - pca.inverse_transform(pca.transform(X))
    return pd.DataFrame(E ** 2, index=d.index, columns=S)

contrib_ref = spe_contributions(ref)
contrib = spe_contributions(mon)
# Normalise each variable's contribution by its typical reference level → comparable across variables
contrib_rel = contrib / contrib_ref.quantile(0.99)

# Average relative contribution per fault type: does the top variable match the physics?
print(contrib_rel.groupby(mon["fault"]).median().round(2).T)
```

```python
# Alarm report for one incident: top contributors + time plots with normal bands
def alarm_report(t_alarm, window_before="24h", window_after="2h", top=3):
    w = op.loc[t_alarm - pd.Timedelta(window_before): t_alarm + pd.Timedelta(window_after)]
    c = contrib_rel.loc[t_alarm - pd.Timedelta("1h"): t_alarm].median().sort_values(ascending=False)
    print(f"Alarm at {t_alarm} | regime: {op.loc[t_alarm, 'regime']} | top contributors:\n{c.head(top).round(2)}")
    fig, axes = plt.subplots(top, 1, figsize=(11, 2.3 * top), sharex=True)
    for ax, var in zip(np.atleast_1d(axes), c.index[:top]):
        band = ref.groupby("regime")[var].quantile([0.005, 0.995]).unstack()
        lo, hi = band.loc[w["regime"], 0.005].to_numpy(), band.loc[w["regime"], 0.995].to_numpy()
        ax.plot(w.index, w[var], lw=0.8); ax.fill_between(w.index, lo, hi, alpha=0.2, label="normal band (regime)")
        ax.axvline(t_alarm, color="r", ls="--"); ax.set_ylabel(var, fontsize=8)
    plt.tight_layout(); plt.show()
    return c

bearing_alarm = mon.index[(mon["fault"] == "bearing")][60]          # a moment inside the bearing incident
_ = alarm_report(bearing_alarm)
```

```python
# SHAP attributions for an Isolation Forest score (when IF is the deployed detector)
import shap
iso = IsolationForest(n_estimators=200, random_state=0).fit(sc.transform(ref[S]))
Xm = pd.DataFrame(sc.transform(mon[S]), columns=S, index=mon.index)
explainer = shap.TreeExplainer(iso)
sample = Xm[mon["fault"] == "process_upset"].iloc[:20]
sv = explainer.shap_values(sample)                      # contributions to the (isolation) output
print(pd.Series(-np.mean(sv, axis=0), index=S).sort_values(ascending=False).round(3))   # sign: more anomalous = positive
```

```python
# Incident library record (append to a table; feeds threshold tuning and future supervised diagnosis)
record = {"alarm_time": str(bearing_alarm), "detector": "PCA-SPE v1", "top_variables": ["vibration_mm_s"],
          "regime": op.loc[bearing_alarm, "regime"], "operator_verdict": "true alarm",
          "root_cause": "pump bearing wear (confirmed by vibration spectrum)", "action": "bearing replaced",
          "detection_delay_h": None, "notes": ""}
print(record)
```

## 7. Interpretation
- **Median relative contributions per fault type** show the strengths *and the smearing* of PCA contributions:
  - Bearing: vibration is the top contributor, as expected. Flow, current and pressure drop are also clearly elevated, even though they did not change. A 2-component PCA ties vibration to flow through the shared duty component, so an off-model vibration value spreads into the reconstruction errors of its correlated partners.
  - Sensor drift and process upset: inlet and outlet temperature receive **equal** contributions. Contributions cannot tell which one is wrong. The soft-sensor residuals in [time_series_anomalies.md](time_series_anomalies.md) can: outlet ≈ 8.6 σ, while the inlet residual does not move.
  - Fouling: the temperatures dominate, and the pressure drop is moderately elevated, consistent with the physics.
  **Lesson:** PCA contributions narrow the search down to a *group* of related variables. Physically structured residuals, redundant sensors and the raw plots pin down the variable.
- **Alarm report:** the top contributor is vibration, while the time plots show flow and current inside their normal bands. Vibration above its regime band while the duty stays normal is the signature of an equipment (mechanical) fault rather than a process change. That points the engineer towards a vibration spectrum analysis and a bearing inspection.
- **SHAP for IF on the upset:** the inlet temperature should carry the largest positive contribution. If SHAP highlights variables unrelated to the physics, the IF score is driven by sparsity artefacts, so prefer the residual-based explanations.
- **The incident library** is the bridge from unsupervised monitoring to supervised fault classification. After enough confirmed cases, train a classifier on the contribution patterns.

## 8. Decision framework
- **IF** the top contributor is a single variable with no change in the related variables → suspect a **sensor fault** (calibration, wiring). Check with a redundant sensor or a manual measurement.
- **IF** several physically linked variables deviate consistently → a **process change or upset**. Check the inputs (feed, ambient, set-points) and recent operator actions.
- **IF** a slow, monotonic deviation exists in efficiency-related variables → **degradation** (fouling, wear). Plan maintenance; estimate the remaining time ([predictive_maintenance_rul.md](../16_industrial_sensor_data/predictive_maintenance_rul.md)).
- **IF** the contributions are diffuse, with no clear pattern → a new regime or a model mismatch. Check whether this is a legitimate new operating mode and update the normal model.
- **IF** labelled incidents accumulate (≥ tens per type) → train a supervised diagnosis model on the contributions/residuals.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Presenting a score without variables | Operators cannot act; alarms are ignored | Contribution ranking + plots |
| Treating the top contributor as the cause | Wrong fixes | Physics reasoning; checks |
| Ignoring smearing in PCA contributions | Misleading variable ranking | Verify with raw plots and residuals |
| No feedback capture | Detector never improves | Incident library |
| Explaining with a different model than the one alarming | Inconsistent explanations | Explain the deployed detector |

## 10. Checklist
- [ ] Contributions available for every alarm (SPE/T²/residuals/SHAP).
- [ ] Alarm report with the top variables, regime-aware normal bands and the onset estimate.
- [ ] Concurrent events checked (regime changes, maintenance).
- [ ] Hypotheses and checks agreed with engineers.
- [ ] Outcome recorded in the incident library.

## 11. References
- Miller, P., Swanson, R. E. & Heckler, C. E. (1998). "Contribution plots: a missing link in multivariate quality control." *Applied Mathematics and Computer Science* 8(4), 775–792.
- Alcala, C. F. & Qin, S. J. (2009). "Reconstruction-based contribution for process monitoring." *Automatica* 45(7), 1593–1600. https://doi.org/10.1016/j.automatica.2009.02.027
- Westerhuis, J. A., Gurden, S. P. & Smilde, A. K. (2000). "Generalized contribution plots in multivariate statistical process monitoring." *Chemometrics and Intelligent Laboratory Systems* 51(1), 95–114.
- SHAP documentation, *TreeExplainer* (supports IsolationForest). https://shap.readthedocs.io/
- Chiang, L. H., Russell, E. L. & Braatz, R. D. (2001). *Fault Detection and Diagnosis in Industrial Systems*. Springer.
