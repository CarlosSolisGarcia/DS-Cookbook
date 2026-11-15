# Industrial and Sensor Data Workflow

> **Folder:** 16_industrial_sensor_data · **Start here for sensor/IoT/process data.** · **Next:** [sensor_preprocessing.md](sensor_preprocessing.md) · **Related:** [05_time_series](../05_time_series/time_series_workflow.md), [10_anomaly_detection](../10_anomaly_detection/anomaly_detection_workflow.md), [predictive_maintenance_rul.md](predictive_maintenance_rul.md), [process_monitoring.md](process_monitoring.md)

---

## 1. Purpose
Approach multivariate sensor datasets (motors, pumps, turbines, heat exchangers, reactors, energy systems) with the right **questions about the measurement system** before modelling. Then choose the problem framing: forecasting, monitoring/anomaly detection, soft sensing, predictive maintenance/RUL, or classification of states.

## 2. When to use it
At the start of any project with time-stamped measurements from physical equipment: SCADA/historian exports, IoT platforms, condition-monitoring systems, lab plus online analysers.

## 3. Conceptual foundation

### 3.1 What makes sensor data different
| Property | Why it matters | Where handled |
|---|---|---|
| **Multiple sampling rates** (1 kHz vibration, 1 s pressure, 1 min temperature, daily lab) | Streams must be aligned and aggregated to a common time base | [sensor_preprocessing.md](sensor_preprocessing.md) |
| **Clock offsets, latency, jitter** | Misaligned causes and effects; leakage across time | [sensor_preprocessing.md](sensor_preprocessing.md) |
| **Historian compression / deadbands** | "Missing" data may mean "unchanged within the deadband" | Forward-fill rules |
| **Measurement noise and calibration** | Noise floors; offsets between sensors/sites; drift over time | [sensor_preprocessing.md](sensor_preprocessing.md) |
| **Operating regimes** (load levels, products, start-up/shutdown) | Normal behaviour depends on the regime; regime changes look like anomalies | Regime identification ([clustering_algorithms.md](../09_unsupervised_learning/clustering_algorithms.md)) |
| **Physics** (mass/energy balances, monotonic relations) | Strong priors: features, constraints and consistency checks | [18_scientific_machine_learning](../18_scientific_machine_learning/scientific_ml_workflow.md) |
| **Control loops** | Controlled variables stay at their set-points, so information moves into the manipulated variables (valve positions) | Feature choice; causal reasoning |
| **Rare failures, many units** | Labels are scarce, and units are not independent | Grouped validation; anomaly detection; RUL |
| **Censoring** | Many units have not failed yet | RUL framing ([predictive_maintenance_rul.md](predictive_maintenance_rul.md)) |

### 3.2 Typical problem framings
| Question | Framing | Recipes |
|---|---|---|
| "Will this unit fail in the next N days?" | Classification on time windows (per unit) | [04_classification](../04_classification/classification_workflow.md), [predictive_maintenance_rul.md](predictive_maintenance_rul.md) |
| "How many days until failure?" | RUL regression / survival | [predictive_maintenance_rul.md](predictive_maintenance_rul.md) |
| "Is the process behaving normally now?" | Monitoring / anomaly detection | [process_monitoring.md](process_monitoring.md), [10_anomaly_detection](../10_anomaly_detection/anomaly_detection_workflow.md) |
| "What is the (slow, expensive) quality value now?" | Soft sensor (nowcasting regression) | [process_monitoring.md](process_monitoring.md) |
| "What will the load/energy be tomorrow?" | Forecasting | [05_time_series](../05_time_series/time_series_workflow.md) |
| "Which fault type is this?" | Classification of windows (needs labelled faults) | [04_classification](../04_classification/classification_workflow.md) |

## 4. Assumptions and limitations
- Historian data have usually been compressed, interpolated or aggregated before you see them. Ask how they were exported.
- Event logs (maintenance, alarms, operator notes) are essential context and are often of low quality. Reconcile them with the sensor data.
- Data from closed-loop operation do not reveal causal effects of the controlled variables. Do not interpret them causally.

## 5. Step-by-step procedure
1. **Inventory the measurement system:** tag list, units, sensor type, sampling rate, location (P&ID), calibration history, data source (historian, PLC, IoT gateway) and export settings (compression, interpolation).
2. **Collect the context data:** maintenance work orders, alarms, operating modes and recipes, start-up/shutdown logs, lab results.
3. **Check the time base:** timezone/DST, clock offsets between systems, latency, duplicate timestamps.
4. **Align and resample** to the decision-relevant time base with the correct aggregation semantics ([sensor_preprocessing.md](sensor_preprocessing.md)).
5. **Assess the quality per tag:** flatlines (stuck sensors), out-of-range values, noise level, drift vs redundant sensors, missing patterns.
6. **Identify the operating regimes and transients;** mark the planned events.
7. **Choose the framing** (§3.2) and the validation: by **time** and by **unit/asset**.
8. **Proceed with the specific recipe** (RUL, monitoring, forecasting, classification).

## 6. Python implementation: a tag-quality report

```python
import numpy as np
import pandas as pd
from cookbook_data import make_process_sensors

df = make_process_sensors()
TAGS = ["load_pct", "flow_m3h", "inlet_temp_c", "coolant_in_c", "outlet_temp_c",
        "pressure_drop_bar", "pump_current_a", "vibration_mm_s"]
# Physical plausibility limits come from instrument datasheets / process knowledge
LIMITS = {"flow_m3h": (0, 80), "inlet_temp_c": (0, 150), "outlet_temp_c": (0, 150), "pressure_drop_bar": (0, 6),
          "pump_current_a": (0, 120), "vibration_mm_s": (0, 20), "load_pct": (0, 100), "coolant_in_c": (0, 60)}

def tag_quality(df, tags, limits, flat_window="2h"):
    rows = []
    for t in tags:
        s = df[t]
        lo, hi = limits[t]
        roll_std = s.rolling(flat_window).std()
        diffs = s.diff().abs()
        rows.append({
            "tag": t, "missing_%": 100 * s.isna().mean(),
            "out_of_range_%": 100 * ((s < lo) | (s > hi)).mean(),
            "flatline_%": 100 * (roll_std < 1e-6).mean(),                  # stuck sensor candidates
            "noise_est": 1.4826 * np.median(np.abs(diffs - np.median(diffs.dropna()))) / np.sqrt(2),  # robust step noise
            "p01": s.quantile(0.01), "p99": s.quantile(0.99),
        })
    return pd.DataFrame(rows).set_index("tag")

print(tag_quality(df, TAGS, LIMITS).round(3))
```

```python
# Context data: events and regimes over time (fraction of time per regime per day)
regime_share = pd.crosstab(df.index.floor("D"), df["regime"], normalize="index")
print(regime_share.head(8).round(2))
# Transients: samples where the load changes quickly (exclude or treat as their own regime in monitoring)
transient = df["load_pct"].diff().abs().rolling(6, min_periods=1).max() > 2
print(f"transient samples: {transient.mean():.1%}")
```

**Measurement-system inventory template (deliverable):**
```text
TAG             UNIT   RATE   SOURCE      INSTRUMENT / RANGE        CALIBRATED   NOTES
FI-101 flow     m3/h   5 min  historian   magnetic flowmeter 0–80   2024-01-12   avg of 1 s values
TI-102 inlet    °C     5 min  historian   PT100 0–150               2023-11-03
TI-103 outlet   °C     5 min  historian   PT100 0–150               2023-11-03   suspected drift (see WO-55120)
PDI-104 dP      bar    5 min  historian   dP cell 0–6
II-105 current  A      5 min  MCC         CT 0–120
VI-106 vib      mm/s   5 min  CMS (RMS)   accelerometer, RMS 10–1000 Hz   raw waveform available on request
Export settings: interpolated to 5-min grid by the historian (!), compression deadband 0.1% span
```

## 7. Interpretation
- **Missing / out-of-range:** a small fraction is expected. Out-of-range values that cluster in time point to sensor faults or export artefacts. Physically impossible values (negative flow far below zero) are errors. Values slightly below zero for flow at shutdown are instrument offsets.
- **Flatline %:** stuck sensors are a common, silent failure. Here the share is ≈ 0, because the synthetic data never flatline. In real historian data, also check for "too smooth" signals: interpolation by the historian hides missing data.
- **Noise estimate:** the robust step noise per tag gives the noise floor. Changes smaller than this cannot be detected on single samples, so they need accumulation (EWMA) or aggregation.
- **Regime shares:** strongly varying regime mixes between periods mean that comparisons across periods (and validation splits) must account for the regime.
- **Transients:** a few percent of the samples are ramps. Excluding them from "normal" models avoids inflated thresholds (see [statistical_methods.md](../10_anomaly_detection/statistical_methods.md)).

## 8. Decision framework
**IF the dataset contains multiple sensor streams:**
- **Before modelling,** check the sampling frequencies, synchronisation (clock offsets), missing measurements, export interpolation/compression and the operating regimes ([sensor_preprocessing.md](sensor_preprocessing.md)).
- **IF** the rates differ → resample to the decision time base with semantics per tag (mean/last/max/count). Compute high-frequency features (RMS, spectra) *before* downsampling.
- **IF** the clocks are not synchronised → estimate the offsets (cross-correlation of related signals), correct them, and document.
- **IF** regimes exist → identify them (rules from the load/set-points, or clustering) and condition models on them.

**Choosing the framing:**
- Failure labels per unit with censoring → RUL / survival / time-window classification ([predictive_maintenance_rul.md](predictive_maintenance_rul.md)).
- No failure labels, need for early warning → monitoring/anomaly detection.
- A slow lab measurement to be estimated online → soft sensor.

**Validation:** always split by **time**, and by **unit** when the model must generalise to new assets.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Trusting the historian export as raw data | Interpolated/compressed artefacts treated as signals | Ask for the export settings; check smoothness |
| Ignoring clock offsets between systems | Effects precede causes; leakage | Cross-correlation alignment |
| Downsampling vibration before computing features | Losing the fault frequencies | Features at the native rate |
| Mixing regimes in one normal model | False alarms; masked faults | Regime-aware models |
| Random splits across time/units | Optimistic results | Time- and unit-based validation |
| Interpreting closed-loop correlations causally | Wrong process conclusions | Account for the control structure |

## 10. Checklist
- [ ] Tag inventory (units, rates, sources, calibration, export settings).
- [ ] Context data collected (maintenance, alarms, modes, lab).
- [ ] Time base checked (timezone, offsets, latency).
- [ ] Tag-quality report (missing, range, flatline, noise).
- [ ] Regimes and transients identified; planned events marked.
- [ ] Framing chosen; validation by time and unit.

## 11. References
- Kadlec, P., Gabrys, B. & Strandt, S. (2009). "Data-driven Soft Sensors in the process industry." *Computers & Chemical Engineering* 33(4), 795–814. https://doi.org/10.1016/j.compchemeng.2008.12.012
- Lei, Y., Li, N., Guo, L., Li, N., Yan, T. & Lin, J. (2018). "Machinery health prognostics: A systematic review from data acquisition to RUL prediction." *Mechanical Systems and Signal Processing* 104, 799–834. https://doi.org/10.1016/j.ymssp.2017.11.016
- Jardine, A. K. S., Lin, D. & Banjevic, D. (2006). "A review on machinery diagnostics and prognostics implementing condition-based maintenance." *MSSP* 20(7), 1483–1510. https://doi.org/10.1016/j.ymssp.2005.09.012
- ISO 17359:2018, *Condition monitoring and diagnostics of machines — General guidelines*. ISO 10816 / 20816 (vibration severity).
- Zhang, W., Yang, D. & Wang, H. (2019). "Data-Driven Methods for Predictive Maintenance of Industrial Equipment: A Survey." *IEEE Systems Journal* 13(3), 2213–2227. https://doi.org/10.1109/JSYST.2019.2905565
