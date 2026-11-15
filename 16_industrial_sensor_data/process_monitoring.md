# Process Monitoring and Soft Sensors: Combining Time-Series Analysis and Machine Learning

> **Folder:** 16_industrial_sensor_data · **Prev:** [predictive_maintenance_rul.md](predictive_maintenance_rul.md) · **Next:** [geospatial_workflow.md](../17_geospatial_data/geospatial_workflow.md) · **Related:** [statistical_methods.md](../10_anomaly_detection/statistical_methods.md) (MSPC details), [time_series_anomalies.md](../10_anomaly_detection/time_series_anomalies.md), [05_time_series](../05_time_series/time_series_workflow.md)

---

## 1. Purpose
Design a **monitoring system** for a continuous process (detect abnormal operation, estimate unmeasured quality variables, track efficiency), and decide how time-series methods and ML fit together in it. This recipe connects the pieces from the anomaly-detection and time-series sections into an operational architecture.

## 2. When to use it
- Continuous or semi-batch processes with many correlated sensors (heat exchangers, reactors, distillation, compressors, energy systems).
- Infrequent lab measurements of quality (purity, viscosity) that you would like to estimate online (soft sensor).
- Efficiency/KPI tracking (heat-transfer coefficient, specific energy consumption).

## 3. Conceptual foundation
**Building blocks of a monitoring system:**
| Block | Purpose | Methods |
|---|---|---|
| Data layer | Synchronised, validated sensor data at the monitoring rate | [sensor_preprocessing.md](sensor_preprocessing.md) |
| Regime/state detection | Which operating mode applies now? Steady state? | Rules on set-points/load; clustering |
| **KPI / physics indicators** | Efficiency from first principles (e.g. effectiveness, UA, specific energy) | Engineering equations |
| **Soft sensors** | Estimate slow/expensive variables from fast ones | Regression (PLS, ridge, GBM) with lagged inputs; recursive updating |
| **Multivariate monitoring** | Detect deviations from normal correlation structure | PCA T²/SPE per regime; residual-based detection |
| **Trend / drift tracking** | Slow degradation (fouling, catalyst ageing) | EWMA, trend fits, change points on KPIs/residuals |
| Alarm management | Persistence, prioritisation, suppression | [evaluation_and_thresholds.md](../10_anomaly_detection/evaluation_and_thresholds.md) |
| Diagnosis support | Contributions, residual patterns | [root_cause_analysis.md](../10_anomaly_detection/root_cause_analysis.md) |

**Soft sensors (nowcasting).** The target y (e.g. a lab-measured purity, available every 8 h with a 2 h delay) is predicted from the process variables x over the preceding window. The key issues:
- **Time alignment:** the lab sample time ≠ the analysis time. Use the process data at (and before) the *sampling* time, plus the process dead time and residence time.
- **Few labels:** lab samples are sparse, so use regularised linear models / PLS first.
- **Drift:** catalyst ageing, fouling and recalibrations. Use **adaptive** (moving-window or recursive) re-fitting, and **bias updating** with each new lab value.
- **Validation:** chronological, and without leakage from future lab values.

**Physics-informed KPIs** are often the best monitoring signals. For a heat exchanger, effectiveness ε = (T_in − T_out)/(T_in − T_coolant) declines with fouling regardless of load (if normalised for flow). A trend on such a KPI is interpretable and robust.

## 4. Assumptions and limitations
- A soft sensor extrapolates poorly outside the training conditions. Monitor its input domain (are the inputs within the training range?).
- KPIs depend on sensor accuracy. A drifting thermocouple looks like fouling. Cross-check with redundant measurements (e.g. the pressure drop rises with fouling, but not with sensor drift).
- Closed-loop control hides disturbances in the manipulated variables. Include valve positions and controller outputs as inputs.

## 5. Step-by-step procedure
1. **Define the monitoring objectives:** the faults and inefficiencies to catch; the quality variables to estimate; the response times.
2. **Data layer and regimes** (previous recipes). Build a steady-state filter.
3. **Physics KPIs** (efficiency, balances) computed per sample; plot their trends.
4. **Soft sensor:** align the lab samples to the process data (dead time); regularised model; chronological validation; bias updating.
5. **Multivariate monitoring** per regime (PCA-SPE / residuals) with calibrated limits.
6. **Trend detection** on the KPIs and residuals (EWMA, rolling slopes, change points).
7. **Integrate:** a dashboard per alarm with contributions and KPIs; operator feedback loop.
8. **Maintain:** periodic re-baselining after maintenance; drift monitoring of the inputs and residuals.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cookbook_data import make_process_sensors

df = make_process_sensors()
op = df[df["regime"] != "shutdown"].copy()
op["day"] = (op.index - df.index[0]) / pd.Timedelta("1D")
steady = op["load_pct"].diff().abs().rolling(6, min_periods=1).max() < 1          # steady-state filter

# 1. Physics KPI: heat-exchanger effectiveness, normalised for its flow dependence using the reference period
op["effectiveness"] = (op["inlet_temp_c"] - op["outlet_temp_c"]) / (op["inlet_temp_c"] - op["coolant_in_c"])
ref = op[(op["day"] < 9.5) & steady]
coef = np.polyfit(ref["flow_m3h"], ref["effectiveness"], 1)                        # expected ε as a function of flow
op["eff_ratio"] = op["effectiveness"] / np.polyval(coef, op["flow_m3h"])           # 1.0 = as clean
daily = op[steady].resample("D")["eff_ratio"].median()
dp_ratio = (op["pressure_drop_bar"] / (0.0009 * op["flow_m3h"].clip(lower=1) ** 2))[steady].resample("D").median()

fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
daily.plot(ax=ax[0], marker="o"); ax[0].set_ylabel("effectiveness / expected")
dp_ratio.plot(ax=ax[1], marker="o", color="tab:red"); ax[1].set_ylabel("ΔP / expected")
for a in ax:
    a.axvspan(pd.Timestamp("2024-03-11"), pd.Timestamp("2024-03-13"), color="orange", alpha=0.2)   # sensor drift
    a.axvspan(pd.Timestamp("2024-04-08"), df.index[-1], color="brown", alpha=0.15)                 # fouling onset
plt.tight_layout(); plt.show()
print(pd.DataFrame({"eff_ratio": daily, "dp_ratio": dp_ratio}).iloc[[5, 10, 11, 20, 39, 42, 44]].round(3))
```

```python
# 2. Soft sensor: a "lab" outlet-quality value sampled every 8 h with noise, predicted from process data.
#    (Here the synthetic quality = f(outlet temperature, flow); in practice e.g. product purity.)
rng = np.random.default_rng(0)
op["quality_true"] = 98.0 - 0.08 * (op["outlet_temp_c"] - 40) + 0.01 * (op["flow_m3h"] - 40)
lab_times = op.index[::96]                                                          # every 8 h (96 × 5 min)
lab = pd.DataFrame({"quality_lab": op.loc[lab_times, "quality_true"] + rng.normal(0, 0.05, len(lab_times))}, index=lab_times)

INPUTS = ["flow_m3h", "inlet_temp_c", "coolant_in_c", "pressure_drop_bar", "pump_current_a"]   # NOT the outlet sensor
# Features: process averages over the 30 min BEFORE each lab sample (past-only window)
win = op[INPUTS].rolling("30min").mean()
data = lab.join(win, how="inner").dropna()
cut = data.index[int(0.4 * len(data))]
train, test = data[data.index < cut], data[data.index >= cut]

models = {"ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 20))),
          "PLS(2)": make_pipeline(StandardScaler(), PLSRegression(2))}
for name, m in models.items():
    m.fit(train[INPUTS], train["quality_lab"])
    pred_static = np.ravel(m.predict(test[INPUTS]))
    # Bias updating: after each lab result, correct the next predictions by the recent mean error (EWMA)
    bias, pred_bu = 0.0, []
    for p, y_obs in zip(pred_static, test["quality_lab"]):
        pred_bu.append(p + bias)
        bias = 0.7 * bias + 0.3 * (y_obs - p)                                        # uses only PAST lab values
    pred_bu = np.array(pred_bu)
    def mae(p): return np.mean(np.abs(test["quality_lab"] - p))
    print(f"{name:7s} MAE static={mae(pred_static):.3f}  with bias updating={mae(pred_bu):.3f}")
```

```python
# 3. Input-domain monitoring for the soft sensor: are current inputs within the training envelope?
lo, hi = train[INPUTS].quantile(0.005), train[INPUTS].quantile(0.995)
outside = ((op[INPUTS] < lo) | (op[INPUTS] > hi)).any(axis=1)
print(f"share of samples with soft-sensor inputs outside the training envelope: {outside.mean():.1%}")
```

## 7. Interpretation
- **Effectiveness ratio:** ≈ 1.0 in normal operation. During the *sensor drift* (days 10–12) it drops, because the drifting outlet reading looks like worse heat transfer, while the **pressure-drop ratio stays at 1**. During *fouling* (from day 38) the effectiveness falls **and** the pressure drop rises. Cross-checking two physically independent indicators separates a **sensor fault** from a **process fault**. This is the core of monitoring with domain knowledge.
- **Soft sensor:** the static model's MAE (≈ 0.17–0.18) is about halved by bias updating (≈ 0.08). The static model degrades on the test period, because the fouling changes the outlet temperature (and hence the quality) in a way the inputs do not fully capture. **Bias updating** with past lab values recovers much of the accuracy. That is why adaptive soft sensors are standard in industry.
- **Input-domain check:** here ~55% of the samples fall outside the per-variable training envelope. The soft sensor was trained on only ~50 lab samples (16 days), and these did not cover the full range of operating conditions (regimes, ambient cycles, fouling). That is a genuine warning: many predictions are extrapolations. Remedies: collect lab samples across all regimes (a sampling plan), use a multivariate domain measure (the Mahalanobis distance or PCA-T² of the inputs) instead of per-variable ranges, and show out-of-domain predictions as "low confidence" to operators.

## 8. Decision framework
- **IF** a physics KPI exists (efficiency, balance closure) → monitor it first. It is interpretable, robust and regime-normalisable.
- **IF** two indicators disagree (KPI drop without the corroborating physical change) → suspect a sensor. Trigger a calibration check.
- **IF** quality is measured infrequently → soft sensor with past-window inputs, a regularised linear model/PLS, bias updating, and input-domain monitoring.
- **IF** the process drifts slowly (catalyst, fouling) → a moving-window or recursive re-fit for the soft sensor; trend detection on the KPIs; re-baseline after maintenance.
- **IF** there are many correlated sensors and no physics model → PCA-SPE per regime ([statistical_methods.md](../10_anomaly_detection/statistical_methods.md)).
- **IF** the dynamics matter (dead times, slow responses) → lagged inputs and windows aligned with the process residence time; dynamic PCA (augment with lags) as an extension.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Using the target-related sensor as a soft-sensor input | Trivial or circular model | Inputs causally upstream / independent |
| Aligning lab results to the analysis time | Misaligned training pairs | Sampling time + dead time |
| Static soft sensor in a drifting process | Growing bias | Bias updating / adaptive re-fit |
| KPI without flow/load normalisation | Regime changes look like degradation | Normalise using reference relationships |
| Ignoring controller outputs | Disturbances invisible | Include valve positions / MVs |
| One alarm stream for everything | Operator overload | Tiered KPIs, alarms with contributions |

## 10. Checklist
- [ ] Monitoring objectives and response times defined.
- [ ] Steady-state filter and regimes.
- [ ] Physics KPIs with load/flow normalisation; trends plotted.
- [ ] Soft sensor: aligned lab data, regularised model, chronological validation, bias updating, input-domain monitoring.
- [ ] Multivariate monitoring per regime; trend detection.
- [ ] Alarm management, diagnosis support and feedback loop.

## 11. References
- Kadlec, P., Gabrys, B. & Strandt, S. (2009). "Data-driven Soft Sensors in the process industry." *Computers & Chemical Engineering* 33(4), 795–814.
- Kadlec, P., Grbić, R. & Gabrys, B. (2011). "Review of adaptation mechanisms for data-driven soft sensors." *Computers & Chemical Engineering* 35(1), 1–24. https://doi.org/10.1016/j.compchemeng.2010.07.034
- Ku, W., Storer, R. H. & Georgakis, C. (1995). "Disturbance detection and isolation by dynamic principal component analysis." *Chemometrics and Intelligent Laboratory Systems* 30(1), 179–196.
- Qin, S. J. (2012). "Survey on data-driven industrial process monitoring and diagnosis." *Annual Reviews in Control* 36(2), 220–234. https://doi.org/10.1016/j.arcontrol.2012.09.004
- Incropera, F. P. et al. *Fundamentals of Heat and Mass Transfer*, Ch. 11 (heat-exchanger effectiveness, fouling).
