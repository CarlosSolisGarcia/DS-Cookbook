# Sensor Preprocessing: Alignment, Resampling, Noise, Calibration Drift and Regimes

> **Folder:** 16_industrial_sensor_data · **Prev:** [sensor_data_workflow.md](sensor_data_workflow.md) · **Next:** [predictive_maintenance_rul.md](predictive_maintenance_rul.md) · **Related:** [ts preprocessing.md](../05_time_series/preprocessing.md) (single-series gaps, outliers), [data_quality.md](../01_data_understanding/data_quality.md)

---

## 1. Purpose
Turn raw, multi-rate, imperfect sensor streams into a synchronised, clean, regime-annotated table at a decision-relevant time base. Along the way, extract high-frequency information (RMS, peaks, spectral bands) before it is lost to downsampling.

## 2. When to use it
After the tag inventory and quality report ([sensor_data_workflow.md](sensor_data_workflow.md)), before any modelling with several sensors.

## 3. Conceptual foundation
- **Alignment.** Each stream has its own timestamps. Common tools:
  - `resample` (bins) per stream, then join on the common grid;
  - `pd.merge_asof` (the last value at or before each target time, within a tolerance), the right tool for **"as-of"** joins of irregular streams;
  - interpolation (for states, offline only).
- **Clock offsets** between systems shift the signals relative to each other. Estimate the lag from the cross-correlation of physically related signals, or from known simultaneous events (a start-up step).
- **Aggregation semantics:** *state* variables → mean/median/last; *flows/energies* → sum or mean rate; *events* → count; *vibration* → RMS/peak/kurtosis over the window. A window should also carry its **valid-sample count**, so that sparse windows can be discarded.
- **Noise:** sensor noise (white), quantisation, and process noise. Smoothing (rolling median, EWMA) reduces noise but adds delay. Never use centred filters online.
- **Calibration offsets and drift:** an offset (constant bias) or drift (a slowly increasing bias) relative to the truth. Detect them by comparing with a **redundant or physically related** measurement (two thermocouples, an energy balance). Correct only with evidence (a calibration record). Otherwise add a feature or flag.
- **Regimes:** derive them from set-points or load (rules), or by clustering steady-state snapshots ([clustering_algorithms.md](../09_unsupervised_learning/clustering_algorithms.md)). Mark the transients separately.
- **Per-regime normalisation:** sensor levels depend on the regime (e.g. pump setting). Normalising each sensor *within* a regime (subtract the regime baseline) separates degradation from duty changes. This is essential for RUL with varying operating conditions.

## 4. Assumptions and limitations
- `merge_asof` with too large a tolerance silently carries stale values. Set a tolerance that matches the expected update rate.
- Interpolation and centred smoothing use future values. They are acceptable for offline training labels, not for online features ([data_leakage.md](../00_methodology/data_leakage.md)).
- Drift correction without a reference can remove real process changes.

## 5. Step-by-step procedure
1. **Per stream:** parse the timestamps (UTC), sort, deduplicate, apply the plausibility limits.
2. **Estimate and correct clock offsets** between the streams (cross-correlation on a related signal pair).
3. **High-frequency features** at the native rate (RMS, peak, crest factor, kurtosis; band energies from the FFT) per window.
4. **Resample** to the common grid with semantics per tag; keep the valid counts.
5. **As-of join** of irregular streams with an explicit tolerance.
6. **Gap handling:** short gaps → forward-fill within the tolerance (online) or interpolate (offline); long gaps → NaN + flag.
7. **Calibration checks** against redundant/physical references; flag the offsets/drift.
8. **Regime labels** (rules or clustering) + a transient flag; per-regime normalisation where needed.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cookbook_data import make_multirate_streams

streams = make_multirate_streams()
vib, temp, pres = (streams[k].set_index("timestamp") for k in ("vibration", "temperature", "pressure"))
for name, s in [("vibration", vib), ("temperature", temp), ("pressure", pres)]:
    dt = s.index.to_series().diff().dt.total_seconds()
    print(f"{name:12s} n={len(s):5d}  median dt={dt.median():5.1f}s  max gap={dt.max():6.1f}s")
```

```python
# 1. Clock offset: cross-correlate pressure with vibration (both follow the load) on a common 1-s grid
v1 = vib["vibration_mm_s"].resample("1s").mean().interpolate(limit=5)
p1 = pres["pressure_bar"].resample("1s").mean().interpolate(limit=15)
common = pd.concat([v1, p1], axis=1).dropna()
a = (common.iloc[:, 0] - common.iloc[:, 0].mean()).to_numpy()
b = (common.iloc[:, 1] - common.iloc[:, 1].mean()).to_numpy()
lags = np.arange(-30, 31)
xc = [np.corrcoef(a[max(0, -L):len(a) - max(0, L)], b[max(0, L):len(b) - max(0, -L)])[0, 1] for L in lags]
best_lag = int(lags[int(np.argmax(xc))])
print(f"estimated pressure clock offset: {best_lag} s (positive = pressure timestamps are late/ahead by that many seconds)")
pres_corr = pres.copy(); pres_corr.index = pres_corr.index - pd.Timedelta(seconds=best_lag)
```

```python
# 2. High-frequency features at the NATIVE rate before downsampling (per 1-minute window)
def vib_features(x):
    x = x.dropna().to_numpy()
    if len(x) < 30:                                   # require enough valid samples in the window
        return pd.Series({"vib_rms": np.nan, "vib_peak": np.nan, "vib_crest": np.nan, "vib_kurt": np.nan, "vib_n": len(x)})
    rms = np.sqrt(np.mean(x ** 2)); peak = np.max(np.abs(x))
    k = ((x - x.mean()) ** 4).mean() / (x.var() ** 2)
    return pd.Series({"vib_rms": rms, "vib_peak": peak, "vib_crest": peak / rms, "vib_kurt": k, "vib_n": len(x)})

vib_1min = vib["vibration_mm_s"].groupby(pd.Grouper(freq="1min")).apply(vib_features).unstack()
# (In real condition monitoring, the raw waveform at kHz rates gives spectral band energies via np.fft.rfft.)

# 3. Resample the states to the 1-min grid; as-of join for the irregular temperature with a tolerance
grid = vib_1min.index
pres_1min = pres_corr["pressure_bar"].resample("1min").mean().rename("pressure_bar")
# merge_asof needs identical datetime resolutions (pandas ≥ 2 can mix "us"/"ns" units): normalise with as_unit
left = pd.DataFrame({"timestamp": grid.as_unit("ns")})
right = temp.reset_index().assign(timestamp=lambda d: d["timestamp"].dt.as_unit("ns"))
temp_asof = pd.merge_asof(left, right, on="timestamp", direction="backward",
                          tolerance=pd.Timedelta("90s")).set_index("timestamp")
aligned_index = grid.as_unit("ns")
vib_1min.index, pres_1min.index = aligned_index, pres_1min.index.as_unit("ns")
aligned = vib_1min.join(pres_1min).join(temp_asof)
print(aligned.head())
print("missing per column (%):", (100 * aligned.isna().mean()).round(1).to_dict())
```

```python
# 4. Calibration drift check with a physical/redundant reference (pump fleet: bearing temperature
#    should depend on setting and vibration; a unit with a constant offset stands out in residuals)
from sklearn.linear_model import LinearRegression
from cookbook_data import make_pump_fleet
fleet = make_pump_fleet()
early = fleet[fleet["day"] <= 30].copy()                      # healthy period, little degradation
X = pd.get_dummies(early[["setting"]], drop_first=True).astype(float).assign(vib=early["vibration_rms"])
m = LinearRegression().fit(X, early["bearing_temp_c"])
early["temp_resid"] = early["bearing_temp_c"] - m.predict(X)
offsets = early.groupby("unit_id")["temp_resid"].median().sort_values()
print("units with suspected temperature-sensor offset (> 2.5 °C):", offsets[offsets > 2.5].round(1).to_dict())
```

```python
# 5. Per-regime normalisation: remove the setting-dependent baseline so degradation is comparable
def per_regime_normalise(df, cols, regime_col="setting", ref_mask=None):
    ref = df if ref_mask is None else df[ref_mask]
    base = ref.groupby(regime_col)[cols].median()
    scale = ref.groupby(regime_col)[cols].std()
    out = df.copy()
    out[cols] = (df[cols] - base.loc[df[regime_col]].to_numpy()) / scale.loc[df[regime_col]].to_numpy()
    return out

sensors = ["vibration_rms", "bearing_temp_c", "discharge_bar", "flow_m3h", "motor_current_a"]
norm = per_regime_normalise(fleet, sensors, ref_mask=fleet["day"] <= 30)
u = fleet["unit_id"] == "U000"
fig, ax = plt.subplots(1, 2, figsize=(13, 3.5))
ax[0].plot(fleet.loc[u, "day"], fleet.loc[u, "flow_m3h"], lw=0.7); ax[0].set_title("raw flow: dominated by the setting")
ax[1].plot(norm.loc[u, "day"], norm.loc[u, "flow_m3h"].rolling(7).mean(), lw=0.9); ax[1].set_title("per-setting normalised (7-day mean): degradation trend")
plt.tight_layout(); plt.show()
```

## 7. Interpretation
- **Stream summary:** vibration at 1 s with a ~3-minute gap, temperature irregular (~30 s), pressure at 10 s. Joining them naively on timestamps would produce almost no exact matches. Resampling or as-of joins are required.
- **Clock offset:** the cross-correlation peak at ≈ 8 s recovers the pressure logger's offset (true value 7 s; the resolution is limited by the 10-s pressure sampling). Without the correction, the load steps would appear in pressure 7 s before they occur in vibration. That is harmless for minute-level features, but critical for second-level causality and for event alignment.
- **Window features:** `vib_n` < 30 marks windows inside the logger gap. Such windows are set to NaN instead of producing misleading statistics from a few samples.
- **As-of temperature:** values older than the 90 s tolerance become NaN. Explicit missingness is better than stale values.
- **Calibration offsets:** units with a median residual of ≈ +4 °C in their healthy period are flagged. In the generator they really have a sensor offset. Correct them only with a calibration record. Otherwise subtract each unit's early baseline (unit-wise normalisation), so that the RUL features are offset-free.
- **Per-setting normalisation:** the raw flow jumps with the operating setting. After normalisation the slow degradation trend becomes visible. This is the key step for RUL under varying conditions.

## 8. Decision framework
| Situation | Action |
|---|---|
| Streams at different rates | High-frequency features at the native rate → resample to the common grid with semantics per tag |
| Irregular/event-driven stream | `merge_asof` with a tolerance; keep the "age of the last value" as a feature if relevant |
| Suspected clock offset | Cross-correlate related signals; correct; document |
| Noisy signal for online use | Trailing rolling median / EWMA (accept the delay); never centred |
| Sensor offset vs fleet/redundant sensor | Flag; correct only with calibration evidence; unit-wise baseline normalisation |
| Drift relative to a redundant/physical reference | Monitor the residual (EWMA); schedule a calibration; exclude or flag drifted data |
| Varying operating conditions | Regime labels; per-regime normalisation; regime as a model feature |
| Transients | Separate flag/regime; exclude from steady-state models |

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Exact-timestamp joins of multi-rate streams | Mostly NaN, or wrongly matched rows | Resample / as-of joins |
| `merge_asof` without a tolerance | Stale values carried for hours | Explicit tolerance |
| Averaging vibration to 1 min before computing features | Fault signatures lost | Native-rate features |
| Normalising with statistics from the whole history | Degradation normalised away; leakage | Reference (healthy, early) period statistics |
| "Correcting" drift without a reference | Removing real process changes | Evidence-based correction only |
| Ignoring the valid-sample count per window | Spurious features from sparse windows | Minimum count rule |
| Mixed datetime resolutions (pandas ≥ 2: `datetime64[us]` vs `[ns]`) | `merge_asof`/join errors | `.as_unit("ns")` on all time keys |

## 10. Checklist
- [ ] Per-stream cleaning (UTC, sort, dedupe, limits).
- [ ] Clock offsets estimated/corrected.
- [ ] High-frequency features at the native rate with valid counts.
- [ ] Resampling with semantics; as-of joins with tolerances.
- [ ] Gaps handled with online-safe rules; flags kept.
- [ ] Calibration/drift checks against references.
- [ ] Regimes + transients; per-regime normalisation where needed.

## 11. References
- pandas, `merge_asof`, `resample`. https://pandas.pydata.org/docs/reference/api/pandas.merge_asof.html · https://pandas.pydata.org/docs/user_guide/timeseries.html#resampling
- Randall, R. B. (2011). *Vibration-based Condition Monitoring: Industrial, Automotive and Aerospace Applications*. Wiley. Features, spectra, envelope analysis.
- Lei, Y. et al. (2018). "Machinery health prognostics: A systematic review from data acquisition to RUL prediction." *MSSP* 104, 799–834. Health indicators, operating-condition normalisation.
- Saxena, A., Goebel, K., Simon, D. & Eklund, N. (2008). "Damage propagation modeling for aircraft engine run-to-failure simulation." *PHM Conference*. The C-MAPSS dataset with multiple operating conditions. https://doi.org/10.1109/PHM.2008.4711414
