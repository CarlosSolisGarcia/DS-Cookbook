# Time-Series Anomaly Detection: Residuals, Context, Accumulation and Change Points

> **Folder:** 10_anomaly_detection · **Prev:** [ml_methods.md](ml_methods.md) · **Next:** [evaluation_and_thresholds.md](evaluation_and_thresholds.md) · **Related:** [05_time_series](../05_time_series/time_series_workflow.md), [preprocessing.md (TS outliers)](../05_time_series/preprocessing.md), [process_monitoring.md](../16_industrial_sensor_data/process_monitoring.md)

---

## 1. Purpose
Detect anomalies that exist only **in time**: values unusual for the time of day or the load, slow drifts, level shifts, and patterns that are individually normal but abnormal as a sequence. The central tool is **prediction residuals**, which compare what was measured with what normal behaviour predicts, followed by temporal accumulation and alarm logic.

## 2. When to use it
- Sensor or process time series, meter data, KPIs over time.
- Contextual anomalies (depending on load, the daily cycle or the regime) and slow faults (drift, fouling, degradation).
- Use it for online monitoring as well as for retrospective screening of historical data.

## 3. Conceptual foundation
**Residual-based detection:** anomaly score = |y_t − ŷ_t| / σ̂, where ŷ_t is a prediction from a model of normal behaviour:
- **Soft sensor / physics residual:** predict a sensor from *other simultaneous* sensors and the context (load, ambient), e.g. outlet temperature from inlet, coolant and flow. Relationship breaks and sensor faults show up immediately.
- **Forecast residual:** predict y_t from its own past (seasonal naive, ETS, ARIMA; [05_time_series](../05_time_series/forecasting_models.md)). Catches unexpected changes relative to the series' own dynamics.
- **Seasonal profile residual:** deviation from the typical value for this hour or weekday (a robust seasonal median).

**Temporal accumulation for slow faults.** A drift of 0.05σ per step is invisible point-wise. Accumulating the residuals reveals it:
- **EWMA** of the residuals (memory ~ 1/λ samples).
- **CUSUM:** detects a persistent mean shift quickly, with a tunable allowance k and decision interval h.
- **Rolling-window statistics** of the residuals (mean, slope).

**Change-point detection** locates *when* the behaviour changed (mean, variance, slope). Use it retrospectively, to segment the history into regimes or find the onset of degradation, and online with CUSUM-type statistics. Dedicated libraries (e.g. `ruptures`) implement penalised segmentation.

**Context matters.** The same value can be normal at 14:00 and abnormal at 03:00, or normal at 100% load and abnormal at 60%. Condition the normal model on the context (time features, regime, load) to detect contextual anomalies.

## 4. Assumptions and limitations
- The model of normal behaviour must be trained on normal data. An anomaly in the inputs of a soft sensor can hide the anomaly (the residual stays small if both the predictor and the target drift together).
- Residuals are autocorrelated. Calibrate the limits empirically, on reference residuals.
- EWMA/CUSUM trade detection speed against false alarms. They also keep alarming after a fault ends, until the statistic recovers (reset the CUSUM after an acknowledged alarm).

## 5. Step-by-step procedure
1. **Choose the normal model per sensor of interest:** soft sensor (from simultaneous variables + context), forecast or seasonal profile.
2. **Fit it on reference data;** compute the reference residuals and their robust scale (per regime if the noise differs).
3. **Score** the monitoring data: standardised residuals.
4. **Point alarms** on |residual z| for sudden faults.
5. **Accumulated alarms** (EWMA/CUSUM on the residuals) for slow faults, with limits from the reference quantiles.
6. **Change-point analysis** retrospectively, to date the onset of the faults found.
7. **Evaluate** the detection delay and the false alarms per day ([evaluation_and_thresholds.md](evaluation_and_thresholds.md)).

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from cookbook_data import make_process_sensors

df = make_process_sensors()
op = df[df["regime"] != "shutdown"].copy()
op["day"] = (op.index - df.index[0]) / pd.Timedelta("1D")
ref, mon = op[op["day"] < 9.5], op[op["day"] >= 9.5]

# 1. Soft sensors: each monitored variable predicted from simultaneous context variables (physics-motivated)
SOFT = {
    "outlet_temp_c": ["inlet_temp_c", "coolant_in_c", "flow_m3h"],     # heat-exchanger balance
    "pressure_drop_bar": ["flow_m3h"],                                # ΔP ∝ flow²
    "vibration_mm_s": ["flow_m3h", "pump_current_a"],                 # vibration explained by duty
}
residual_z = pd.DataFrame(index=mon.index); ref_z = pd.DataFrame(index=ref.index)
for target, inputs in SOFT.items():
    m = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(ref[inputs], ref[target])
    r_ref = ref[target] - m.predict(ref[inputs])
    scale = 1.4826 * np.median(np.abs(r_ref - np.median(r_ref)))
    ref_z[target] = r_ref / scale
    residual_z[target] = (mon[target] - m.predict(mon[inputs])) / scale
print(residual_z.groupby(mon["fault"]).median().round(2))
```

```python
# 2. Point alarms vs accumulated (EWMA) alarms on the residuals; limits = reference quantiles
def ewma(x, lam):
    out = np.empty(len(x)); out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = lam * x[i] + (1 - lam) * out[i - 1]
    return out

def alarm_rates(score_ref, score_mon, q=0.995):
    thr = np.quantile(score_ref, q)
    return (score_mon > thr).groupby(mon["fault"]).mean().round(3)

point_ref, point_mon = ref_z.abs().max(1), residual_z.abs().max(1)
lam = 0.02                                                     # memory ≈ 50 samples ≈ 4 h
ew_ref = pd.Series(np.abs(ewma(ref_z["outlet_temp_c"].to_numpy(), lam)), index=ref.index)
ew_mon = pd.Series(np.abs(ewma(np.r_[ref_z["outlet_temp_c"].to_numpy()[-300:], residual_z["outlet_temp_c"].to_numpy()], lam)[300:]),
                   index=mon.index)
print(pd.DataFrame({"point |z| (max over soft sensors)": alarm_rates(point_ref, point_mon),
                    "EWMA of outlet-temp residual": alarm_rates(ew_ref, ew_mon)}))

# The EWMA is very sensitive to small persistent biases: compare a limit 2x higher (sensitivity trade-off)
for mult in (1, 2, 3):
    thr_m = mult * np.quantile(ew_ref, 0.995)
    r = (ew_mon > thr_m).groupby(mon["fault"]).mean()
    print(f"EWMA limit x{mult}: false-alarm rate={r['none']:.3f}  sensor_drift={r['sensor_drift']:.3f}  fouling={r['fouling']:.3f}")
```

```python
# 3. Detection delay for the slow sensor drift: first sustained alarm after the fault starts
def first_sustained_alarm(alarm: pd.Series, n_consecutive=6):
    run = alarm.astype(int).rolling(n_consecutive).sum() == n_consecutive
    return run.idxmax() if run.any() else None

drift_start = mon.index[mon["fault"] == "sensor_drift"][0]
thr = np.quantile(ew_ref, 0.995)
t_alarm = first_sustained_alarm((ew_mon > thr)[drift_start:])
print("drift starts:", drift_start, "| first sustained EWMA alarm:", t_alarm,
      "| delay (h):", None if t_alarm is None else round((t_alarm - drift_start) / pd.Timedelta("1h"), 1))

fig, ax = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
ax[0].plot(mon.index, residual_z["outlet_temp_c"], lw=0.4); ax[0].set_ylabel("residual z (outlet T)")
ax[1].plot(mon.index, ew_mon, lw=0.8); ax[1].axhline(thr, color="r", ls="--"); ax[1].set_ylabel("|EWMA|")
for a in ax:
    a.axvspan(drift_start, mon.index[mon["fault"] == "sensor_drift"][-1], color="orange", alpha=0.2)
plt.tight_layout(); plt.show()
```

```python
# 4. Retrospective change-point detection (binary segmentation on the mean, least squares; no extra library)
def best_split(x, min_size=50):
    n = len(x); c = np.cumsum(x); c2 = np.cumsum(x ** 2)
    total = c2[-1] - c[-1] ** 2 / n
    best, best_i = 0.0, None
    for i in range(min_size, n - min_size):
        left = c2[i - 1] - c[i - 1] ** 2 / i
        right = (c2[-1] - c2[i - 1]) - (c[-1] - c[i - 1]) ** 2 / (n - i)
        gain = total - left - right
        if gain > best:
            best, best_i = gain, i
    return best_i, best

def binary_segmentation(x, n_cp=4, min_size=50):
    segments, cps = [(0, len(x))], []
    for _ in range(n_cp):
        cand = [(best_split(x[a:b], min_size), a, b) for a, b in segments if b - a > 2 * min_size]
        (i, gain), a, b = max(cand, key=lambda c: c[0][1])
        cps.append(a + i); segments.remove((a, b)); segments += [(a, a + i), (a + i, b)]
    return sorted(cps)

hourly = residual_z["pressure_drop_bar"].resample("1h").median().dropna()     # aggregate: less noise, faster
cps = binary_segmentation(hourly.to_numpy(), n_cp=3, min_size=12)
print("change points in the pressure-drop residual:", [str(hourly.index[i]) for i in cps])
# Dedicated libraries (e.g. `ruptures`) offer penalised segmentation (PELT) with automatic number of change points.
```

## 7. Interpretation
- **Median residual z by fault:** the soft sensors turn each fault into a clear signal on the *right* variable. The outlet-temperature residual is shifted during the sensor drift (the heat balance no longer closes). The vibration residual is high during the bearing fault (vibration is unexplained by the duty). The pressure-drop and outlet residuals shift during fouling. **The residuals also point to the root cause** ([root_cause_analysis.md](root_cause_analysis.md)).
- **Point vs EWMA alarms:** point alarms detect the spikes and the bearing fault immediately. The EWMA of the outlet residual detects the **sensor drift** (≈ 97% of its samples) and the upset, which point-wise methods partly missed, and it alarms within ~2 h of the drift onset.
- **But the EWMA false-alarm rate is far above nominal** (≈ 20% of normal samples at the reference 99.5% limit, still ≈ 11% at 3×). This is the most instructive number on the page:
  1. Part of it is **label definition**: fouling starts on day 38 but is labelled from day 40. Over half of the "false" alarms on days 37–40 are early true detections.
  2. The rest comes from **calibrating a smooth statistic on a short reference**. With a memory of ~4 h, nine reference days contain only ~50 independent EWMA values, so the 99.5% quantile underestimates normal variability. Small extrapolation errors of the soft sensor in operating conditions not seen in the reference add to it. A physics-structured linear soft sensor (inputs ΔT and flow·ΔT) reduced the rate to ≈ 10% in a test.
  3. **Remedies:** a longer and more diverse reference period; calibrating the limits on a *separate* normal validation period; persistence logic and alarm deadbands ([evaluation_and_thresholds.md](evaluation_and_thresholds.md)); physics-based soft sensors. **Sensitivity to slow faults and false alarms trade off directly.** Tune them against the operators' tolerance.
- **Detection delay:** a delay of a few hours for a drift that develops over 2 days is useful. Compare it with the operator's need (the time before the drift causes damage or off-spec product).
- **Change points:** the segmentation of the hourly pressure-drop residual should date the start of the fouling (around day 38–40) and possibly other events. Use it to put the onset on record and to segment the history.

## 8. Decision framework
| Fault/anomaly type | Recommended technique |
|---|---|
| Sensor fault, relationship break | Soft-sensor residuals (physics-motivated inputs) |
| Contextual (load/time dependent) | Condition the normal model on the context; residuals |
| Sudden spike/step | Point thresholds on residuals or robust z |
| Slow drift, degradation, fouling | EWMA/CUSUM on residuals; rolling slopes; change points |
| Unexpected change in a KPI series | Forecast residuals vs a seasonal baseline |
| Historical segmentation | Change-point detection (binary segmentation, PELT) |

- **IF** a soft sensor's inputs can also be affected by the fault → choose inputs from independent measurements, or use several soft sensors and compare their residuals.
- **IF** the residual noise differs by regime → scale the residuals per regime.
- **IF** EWMA alarms persist after a fault → reset the statistic on alarm acknowledgement, or use a moving window instead.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Thresholding raw values of contextual variables | Alarms at every load change; missed contextual faults | Residuals conditional on the context |
| Soft-sensor inputs that drift with the fault | Masked faults | Independent inputs; physics reasoning |
| Gaussian limits on autocorrelated residuals | Too many alarms | Empirical reference quantiles |
| Only point detection | Slow faults found late | Add accumulation (EWMA/CUSUM) |
| Centred windows in online detection | Look-ahead (not available online) | Trailing windows only |

## 10. Checklist
- [ ] Normal-behaviour model per key variable (soft sensor / forecast / profile).
- [ ] Residual scale from the reference (per regime if needed).
- [ ] Point and accumulated alarms with empirical limits.
- [ ] Detection delay measured for slow faults.
- [ ] Change points used to date onsets retrospectively.
- [ ] Online computations use trailing information only.

## 11. References
- Isermann, R. (2006). *Fault-Diagnosis Systems: An Introduction from Fault Detection to Fault Tolerance*. Springer. Model-based residual generation.
- Venkatasubramanian, V., Rengaswamy, R., Yin, K. & Kavuri, S. N. (2003). "A review of process fault detection and diagnosis: Part I: Quantitative model-based methods." *Computers & Chemical Engineering* 27(3), 293–311. https://doi.org/10.1016/S0098-1354(02)00160-6
- Page, E. S. (1954). "Continuous Inspection Schemes." *Biometrika* 41(1/2), 100–115 (CUSUM).
- Truong, C., Oudre, L. & Vayatis, N. (2020). "Selective review of offline change point detection methods." *Signal Processing* 167. https://doi.org/10.1016/j.sigpro.2019.107299 · `ruptures` docs: https://centre-borelli.github.io/ruptures-docs/
- Blázquez-García, A., Conde, A., Mori, U. & Lozano, J. A. (2021). "A Review on Outlier/Anomaly Detection in Time Series Data." *ACM Computing Surveys* 54(3). https://doi.org/10.1145/3444690
