# Evaluating Anomaly Detectors, Selecting Thresholds and Managing False Positives

> **Folder:** 10_anomaly_detection · **Prev:** [time_series_anomalies.md](time_series_anomalies.md) · **Next:** [root_cause_analysis.md](root_cause_analysis.md) · **Related:** [threshold_optimization.md](../04_classification/threshold_optimization.md), [classification evaluation](../04_classification/evaluation.md)

---

## 1. Purpose
Decide how good a detector is, and where to put its threshold, when labelled anomalies are scarce or absent. Then design alarm logic that operators will trust: few false alarms and timely detection of what matters.

## 2. When to use it
For every anomaly-detection deployment: when comparing detectors, choosing thresholds, and reviewing performance in operation.

## 3. Conceptual foundation

### 3.1 What to measure
| Level | Metric | Why |
|---|---|---|
| **Sample/point** | PR-AUC (average precision), precision@k, ROC-AUC | Ranking quality, where labels exist; PR-AUC because anomalies are rare |
| **Event** | Event recall: fraction of fault *events* with ≥ 1 alarm inside the event (or within a tolerance window) | Operators care about catching each incident, not every sample in it |
| **Timeliness** | Detection delay: time from the event start to the first (sustained) alarm | Early detection is the main value of monitoring |
| **Alarm burden** | False alarm *events* per day/shift on normal operation; alarm-event duration | Alarm fatigue destroys trust; there is an industrial guideline of about ≤ 1 alarm per 10 min per operator (EEMUA 191 / ISA-18.2) for all alarms combined |
| **Coverage by type** | Event recall per fault type | Reveals blind spots |

Point-wise metrics favour long events (fouling dominates the samples), while event-level metrics weight each incident equally. Report both.

### 3.2 Threshold-selection strategies
| Strategy | How | When |
|---|---|---|
| **Target false-alarm rate on normal data** | Threshold = (1 − α)-quantile of scores on a normal *validation* period | No labels; the default |
| **Contamination prior** | Threshold so that a fraction c of the data is flagged | Only if c is genuinely known; otherwise arbitrary |
| **Few labels** | Choose the threshold maximising a cost / F-β / event recall at an acceptable alarm rate on labelled validation events | Tens of labelled incidents |
| **Statistical theory** | χ², F, 3σ limits | Only if the assumptions hold (they rarely do for autocorrelated sensor data) |
| **Expert review of the top-k** | Experts label the k highest-scoring cases; estimate precision@k | Bootstrap a labelled set |

### 3.3 Alarm logic to manage false positives
- **Persistence (k-of-n):** alarm only if ≥ k of the last n samples exceed the threshold. This suppresses isolated noise and adds a little delay.
- **Hysteresis / deadband:** raise at threshold T_on, clear only below T_off < T_on. Prevents chattering.
- **Suppression during known events:** start-ups, shutdowns, maintenance, calibration.
- **Alarm grouping:** merge alarms closer than a time gap into one alarm event.
- **Severity levels:** warning vs alarm thresholds.

### 3.4 Evaluation without labels
- **Fault injection:** add realistic synthetic faults (drifts, offsets, spikes, stuck values) to normal validation data and measure their detection. It tests sensitivity, but only for the fault types you can imagine.
- **Historical incidents:** reconstruct events from maintenance logs, shift notes and alarm logs.
- **Expert review:** precision@k on the top-ranked cases.
- **Stability:** do the top-ranked anomalies change wildly with small parameter changes?

## 4. Assumptions and limitations
- Labels from maintenance logs are imprecise in time (the onset is unknown). Use tolerance windows.
- Injected faults are easier or harder than real ones depending on realism. Treat the results as indicative.
- Optimising a threshold on few events overfits. Prefer simple rules and validate them on held-out events.

## 5. Step-by-step procedure
1. **Split the time axis:** reference (fit), validation (threshold calibration), test (evaluation). Each needs normal data; validation/test need labelled or injected events.
2. **Compute the scores** for all candidate detectors.
3. **Calibrate the thresholds** for a target false-alarm rate on validation-normal data (and/or by cost on labelled validation events).
4. **Apply the alarm logic** (persistence, hysteresis, event suppression).
5. **Evaluate on test:** event recall per type, detection delay, false alarm events per day, PR-AUC.
6. **Plot the operating curve** (event recall vs false alarms per day) for several thresholds, and choose with the stakeholders.
7. **Monitor in production:** alarm rates, operator feedback (true/false), and re-calibrate periodically.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from cookbook_data import make_process_sensors

df = make_process_sensors()
S = ["flow_m3h", "inlet_temp_c", "outlet_temp_c", "pressure_drop_bar", "pump_current_a", "vibration_mm_s"]
op = df[df["regime"] != "shutdown"].copy()
op["day"] = (op.index - df.index[0]) / pd.Timedelta("1D")
ref = op[op["day"] < 7]                                        # fit
val = op[(op["day"] >= 7) & (op["day"] < 9.5)]                 # normal data for threshold calibration
test = op[op["day"] >= 9.5]                                    # evaluation (contains the faults)

sc = StandardScaler().fit(ref[S]); pca = PCA(2).fit(sc.transform(ref[S]))
def spe(d):
    X = sc.transform(d[S]); return ((X - pca.inverse_transform(pca.transform(X))) ** 2).sum(1)
s_val, s_test = spe(val), spe(test)
print("point-wise PR-AUC on test:", round(average_precision_score(test["is_anomaly"], s_test), 3),
      "| prevalence:", round(test["is_anomaly"].mean(), 3))
```

```python
# Event-level evaluation utilities
def to_events(mask: pd.Series, max_gap="15min"):
    """Group consecutive True samples (gaps < max_gap) into events: list of (start, end)."""
    t = mask.index[mask.to_numpy()]
    if len(t) == 0:
        return []
    breaks = np.where(np.diff(t) > pd.Timedelta(max_gap))[0]
    starts = np.r_[0, breaks + 1]; ends = np.r_[breaks, len(t) - 1]
    return [(t[a], t[b]) for a, b in zip(starts, ends)]

def persistence(alarm: pd.Series, k=3, n=4):
    """k-of-n rule: alarm if at least k of the last n samples are above the threshold."""
    return alarm.astype(int).rolling(n, min_periods=1).sum() >= k

def evaluate_alarms(alarm: pd.Series, data: pd.DataFrame, tolerance="1h"):
    true_events = {f: to_events(data["fault"] == f, "30min") for f in ["sensor_drift", "process_upset", "bearing", "fouling"]}
    alarm_events = to_events(alarm)
    rows = {}
    for f, evs in true_events.items():
        delays = []
        for (a, b) in evs:
            hits = alarm[(alarm.index >= a) & (alarm.index <= b + pd.Timedelta(tolerance))]
            delays.append((hits.idxmax() - a) / pd.Timedelta("1h") if hits.any() else np.nan)
        rows[f] = {"events": len(evs), "detected": int(np.sum(~np.isnan(delays))),
                   "delay_h": np.nanmedian(delays) if np.any(~np.isnan(delays)) else np.nan}
    is_fault = data["is_anomaly"].astype(bool)
    false_events = [e for e in alarm_events if not is_fault[e[0]:e[1]].any()]
    normal_days = (~is_fault).sum() * 5 / 60 / 24                      # 5-min samples → days
    return pd.DataFrame(rows).T, len(false_events) / normal_days

thr = np.quantile(s_val, 0.995)                                     # target: 0.5% of normal samples
raw = pd.Series(s_test > thr, index=test.index)
for name, alarm in [("raw threshold", raw), ("3-of-4 persistence", persistence(raw))]:
    table, fa_per_day = evaluate_alarms(alarm, test)
    print(f"\n{name}: false alarm EVENTS per normal day = {fa_per_day:.2f}")
    print(table)
```

```python
# Operating curve: event recall vs false alarm events/day for a range of thresholds (with persistence)
curve = []
for q in [0.95, 0.98, 0.99, 0.995, 0.999, 0.9995]:
    al = persistence(pd.Series(s_test > np.quantile(s_val, q), index=test.index))
    tab, fa = evaluate_alarms(al, test)
    curve.append({"quantile": q, "event_recall": tab["detected"].sum() / tab["events"].sum(), "false_alarms_per_day": fa})
curve = pd.DataFrame(curve); print(curve.round(3))
plt.plot(curve["false_alarms_per_day"], curve["event_recall"], "o-")
plt.xlabel("false alarm events per day"); plt.ylabel("event recall"); plt.title("operating curve"); plt.show()
```

```python
# Evaluation without labels: inject synthetic faults into NORMAL validation data
def inject(data, kind, sensor, start, hours, magnitude):
    d = data.copy(); idx = d.index[(d.index >= start) & (d.index < start + pd.Timedelta(hours=hours))]
    if kind == "offset":
        d.loc[idx, sensor] += magnitude
    elif kind == "drift":
        d.loc[idx, sensor] += np.linspace(0, magnitude, len(idx))
    elif kind == "stuck":
        d.loc[idx, sensor] = d.loc[idx[0], sensor]
    return d, idx

rng = np.random.default_rng(0)
results = []
for kind, mag in [("offset", 1.0), ("offset", 3.0), ("drift", 3.0), ("stuck", None)]:
    for sensor in S:
        start = val.index[rng.integers(0, len(val) - 300)]
        d, idx = inject(val, kind, sensor, start, hours=12, magnitude=(mag or 0) * val[sensor].std())
        detected = (pd.Series(spe(d), index=d.index).loc[idx] > thr).mean()
        results.append({"fault": f"{kind}{'' if mag is None else f' {mag}σ'}", "sensor": sensor, "share_alarmed": detected})
print(pd.DataFrame(results).pivot(index="sensor", columns="fault", values="share_alarmed").round(2))
```

## 7. Interpretation
- **Point-wise PR-AUC** is dominated by the long fouling period. Do not compare it across datasets with different event lengths.
- **Raw threshold vs persistence:** persistence typically cuts the false alarm *events* per day substantially. Isolated noise exceedances disappear, while real faults (sustained) are still detected, with a delay of minutes. This is usually the single most effective false-positive control.
- **Event table:** the detection delay per fault type tells you what the detector is good for. Long delays for slow faults are expected (EWMA/residual methods help, [time_series_anomalies.md](time_series_anomalies.md)). Undetected event types are blind spots, and they need another detector.
- **Raw vs persistence in this run:** false alarm events fall from ≈ 0.9 to ≈ 0.1 per normal day, while every incident is still detected. The price is the delay: the bearing fault is detected after ≈ 4.6 h instead of 1.7 h, and the slow sensor drift only after about a day (SPE is not the right tool for it).
- **Operating curve:** with only four incidents in the test period, event recall is 1.0 at every threshold. The curve then cannot discriminate, and **detection delay** becomes the informative axis. Real evaluations need more incidents (historical logs, injection). Stakeholders choose a point, e.g. "≤ 1 false alarm per day with ≥ 75% of the incidents caught". Present the trade-off; do not hide it.
- **Injection table:** shows the detector's sensitivity per sensor and fault type. Large offsets are caught. Small offsets (1σ) and stuck sensors often are not: a stuck value stays plausible. Stuck sensors need a dedicated check (zero variance over a window). Remember that injected faults are only as realistic as your injection model.

## 8. Decision framework
- **IF** there are no labels → calibrate on normal validation data for a target false-alarm rate; estimate the sensitivity with fault injection; schedule an expert review of the top alarms.
- **IF** there are a few labelled incidents → use them for threshold choice and detector comparison at the *event* level, with a held-out subset for the final check.
- **IF** operators complain about false alarms → (1) persistence/hysteresis; (2) suppress known events; (3) fix noisy or faulty sensors; (4) per-regime thresholds; (5) only then raise the thresholds.
- **IF** important faults are detected late → add accumulation (EWMA/CUSUM) or targeted residual detectors for those faults; consider warning levels.
- **IF** some fault types are never detected → add a detector designed for that anomaly type (stuck-value checks, relationship residuals).
- **IF** performance drifts in operation → recalibrate on recent normal data; retrain the normal model when the process changes (new equipment, recipes).

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Only sample-wise metrics | Long events dominate; incidents ignored | Event recall + delay |
| Threshold from the training scores | Too low → false alarms on new normal data | Separate validation-normal period |
| `contamination=0.05` "because the data has anomalies" | Arbitrary alarm volume | Target false-alarm rate |
| No persistence/hysteresis | Chattering alarms; alarm fatigue | k-of-n, deadband |
| Ignoring stuck/flatline sensors | Silent failures | Dedicated variance checks |
| Tuning the threshold on the test incidents | Optimistic evaluation | Held-out incidents |

## 10. Checklist
- [ ] Reference / validation / test periods defined.
- [ ] Thresholds calibrated on normal validation data (target FAR) or on labelled validation events.
- [ ] Alarm logic: persistence, hysteresis, event suppression.
- [ ] Event recall per type, detection delay, false alarm events per day reported.
- [ ] Operating curve discussed with operators.
- [ ] Fault injection for sensitivity where labels are missing.
- [ ] Production feedback and recalibration plan.

## 11. References
- Lavin, A. & Ahmad, S. (2015). "Evaluating Real-time Anomaly Detection Algorithms — the Numenta Anomaly Benchmark." *IEEE ICMLA*. Event windows, early-detection scoring. https://arxiv.org/abs/1510.03336
- Tatbul, N., Lee, T. J., Zdonik, S., Alam, M. & Gottschlich, J. (2018). "Precision and Recall for Time Series." *NeurIPS 31*. https://arxiv.org/abs/1803.03639
- ANSI/ISA-18.2-2016, *Management of Alarm Systems for the Process Industries*; EEMUA Publication 191, *Alarm Systems: A Guide to Design, Management and Procurement*. Alarm-rate guidelines.
- Kim, S., Choi, K., Choi, H.-S., Lee, B. & Yoon, S. (2022). "Towards a Rigorous Evaluation of Time-series Anomaly Detection." *AAAI*. On the point-adjust evaluation pitfalls. https://arxiv.org/abs/2109.05257
- Goldstein, M. & Uchida, S. (2016). "A Comparative Evaluation of Unsupervised Anomaly Detection Algorithms for Multivariate Data." *PLoS ONE* 11(4). https://doi.org/10.1371/journal.pone.0152173
