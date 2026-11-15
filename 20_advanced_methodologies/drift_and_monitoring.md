# Data Drift, Concept Drift and Model Monitoring

> **Folder:** 20_advanced_methodologies · **Prev:** [robustness_and_sensitivity.md](robustness_and_sensitivity.md) · **Next:** [transfer_learning_and_domain_adaptation.md](transfer_learning_and_domain_adaptation.md) · **Related:** [distribution_shift.md](../19_advanced_tabular_ml/distribution_shift.md) (one-off train-vs-target comparison), [evaluation_and_thresholds.md](../10_anomaly_detection/evaluation_and_thresholds.md) (alarm logic)

---

## 1. Purpose
Keep a deployed model trustworthy over time. Detect changes in the inputs (data drift) and in the input–output relationship (concept drift), track the performance once labels arrive, and define when to retrain, recalibrate or roll back.

## 2. When to use it
Any model used repeatedly in production: soft sensors, failure predictors, forecasters, classifiers on new documents and images.

## 3. Conceptual foundation
| What to monitor | Signal | Detects | Delay |
|---|---|---|---|
| **Data quality** | Missing rates, out-of-range values, flatlines, schema changes | Broken pipelines/sensors | Immediate |
| **Input drift** | Per-feature distribution distance vs the reference (PSI, KS, Wasserstein); multivariate (adversarial classifier) | Covariate shift | Immediate |
| **Prediction drift** | The distribution of the predictions/scores; alarm rates | Shifts that matter to the model | Immediate |
| **Performance** | Metrics once labels arrive (MAE, AUC, calibration, coverage) | Concept drift and the real impact | The label latency (days/months) |
| **Business KPIs** | Downtime, scrap, cost | Overall value | Long |

- **PSI (population stability index):** Σ (p_cur − p_ref) · ln(p_cur / p_ref) over bins (quantile bins of the reference). Common heuristics: < 0.1 stable, 0.1–0.25 moderate shift, > 0.25 major shift. These are conventions, not statistical tests.
- **KS test:** sensitive at large n, where it flags tiny, irrelevant shifts. Use effect sizes (the KS statistic, PSI) together with domain thresholds.
- **Concept drift** (P(y\|x) changes) is invisible in the inputs. Only performance monitoring with labels, or proxies (e.g. soft-sensor residuals vs lab values), reveals it.
- **Responses:** recalibrate (prior/threshold shifts), retrain on recent data (gradual drift), segment-specific models (new regime), or roll back/disable (broken inputs).

## 4. Assumptions and limitations
- The drift statistics depend on the binning and the window size. Calibrate the alarm thresholds on historical "normal" variation (seasonality produces drift that is benign).
- Not all drift hurts. Prioritise drift in important features (importance × drift) and in the predictions.
- Retraining on drifted data can bake in problems (e.g. sensor faults). Validate the new model before swapping.

## 5. Step-by-step procedure
1. **Freeze the reference** (the training/validation period statistics and score distributions).
2. **Define the monitoring windows** (daily/weekly) and the metrics per layer (quality, input drift, prediction drift, performance).
3. **Calibrate the alert thresholds** on historical windows (what does normal seasonal variation look like?).
4. **Dashboard + alerts;** route data-quality issues to engineering and drift/performance issues to data science.
5. **Define a retraining policy:** scheduled (e.g. monthly) and/or triggered (performance drops below X, PSI on important features > Y). Champion–challenger validation before deployment.
6. **Document every change** (model version, data range, validation results).

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from cookbook_data import load_clean_process_yield

df = load_clean_process_yield()
feats = ["temperature_c", "pressure_bar", "feed_rate_kg_h", "residence_time_min", "catalyst_age_days", "impurity_ppm",
         "catalyst", "reactor"]
df[["catalyst", "reactor"]] = df[["catalyst", "reactor"]].astype("category")
ref, prod = df.iloc[:1000], df.iloc[1000:].copy()
# Simulate production changes: from batch 1500, a new raw-material supplier raises impurity (data drift)
# and, from batch 1600, a catalyst formulation change alters the temperature response (concept drift)
prod.loc[prod.index >= 1500, "impurity_ppm"] *= 1.6
late = prod.index >= 1600
hot_excess = (prod.loc[late, "temperature_c"].fillna(180) - 180).clip(lower=0)          # missing temperature → no effect
prod.loc[late, "yield_kg"] *= np.exp(-0.05 * hot_excess)                                   # yield now drops above 180 °C
model = HistGradientBoostingRegressor(max_iter=300, categorical_features="from_dtype", random_state=0).fit(ref[feats], np.log(ref["yield_kg"]))

def psi(ref_x, cur_x, bins=10):
    ref_x, cur_x = ref_x.dropna(), cur_x.dropna()
    edges = np.unique(np.quantile(ref_x, np.linspace(0, 1, bins + 1))); edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref_x, edges)[0] / len(ref_x); q = np.histogram(cur_x, edges)[0] / len(cur_x)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))

prod["window"] = (np.arange(len(prod)) // 200)                    # monitoring windows of 200 batches
rows = []
for w, g in prod.groupby("window"):
    pred = np.exp(model.predict(g[feats]))
    rows.append({"window": w, "first_batch": g.index[0],
                 **{f"PSI {f.split('_')[0]}": psi(ref[f], g[f]) for f in ["impurity_ppm", "temperature_c", "feed_rate_kg_h"]},
                 "missing temp %": 100 * g["temperature_c"].isna().mean(),
                 "pred mean": pred.mean(), "MAE (labels arrive later)": mean_absolute_error(g["yield_kg"], pred)})
mon = pd.DataFrame(rows).set_index("window")
print(mon.round(3).to_string())
print("KS test impurity, last window vs reference:", ks_2samp(ref["impurity_ppm"].dropna(), prod.loc[prod["window"] == prod["window"].max(), "impurity_ppm"].dropna()))
```

```python
# Calibrating PSI alert thresholds on HISTORICAL normal variation (rolling windows within the reference period)
hist_psi = [psi(ref["impurity_ppm"].iloc[:500], ref["impurity_ppm"].iloc[s:s + 200]) for s in range(500, 800, 50)]
print("PSI of impurity across normal historical windows:", np.round(hist_psi, 3), "→ alert above ~", round(3 * max(hist_psi), 3))

# Retraining trigger example: MAE above the reference CV MAE + margin for 2 consecutive windows
ref_mae = mean_absolute_error(ref["yield_kg"].iloc[800:], np.exp(HistGradientBoostingRegressor(max_iter=300, categorical_features="from_dtype", random_state=0)
                              .fit(ref[feats].iloc[:800], np.log(ref["yield_kg"].iloc[:800])).predict(ref[feats].iloc[800:])))
trigger = (mon["MAE (labels arrive later)"] > 1.25 * ref_mae).rolling(2).sum() == 2
print(f"reference MAE ≈ {ref_mae:.1f}; retraining triggered at windows: {list(mon.index[trigger])}")
```

## 7. Interpretation
- **Data drift is detected immediately:** the impurity PSI rises from ≈ 0.1 (normal) to 0.33 in the window containing the supplier change (batch 1500), and to 1.2–1.5 afterwards. That is far above the alert level calibrated on historical windows (≈ 0.4). The KS test agrees (p ≈ 10⁻²³). At this sample size the KS test would also flag trivial shifts, so the PSI/KS *statistic* matters more than the p-value.
- **This data drift is harmless:** in window 2 (impurity already shifted) the MAE stays at ≈ 34, as in the reference windows. Impurity has a weak effect, and the model has seen high impurities. A drift alarm is a prompt to investigate, not a verdict.
- **Concept drift is invisible in the inputs:** from batch 1600, the temperature *response* changes (the new catalyst formulation). The **MAE jumps from ≈ 34 to ≈ 53–57**, while the temperature PSI stays at its normal level (≈ 0.04–0.08). The inputs look the same; their meaning has changed. **Only performance monitoring, with labels or label proxies, catches concept drift.**
- **Retraining trigger:** the rule "MAE > 1.25 × the reference for two consecutive windows" fires at window 4, one window after the change, without reacting to single noisy windows. The persistence-versus-delay trade-off is the same as in alarm design ([evaluation_and_thresholds.md](../10_anomaly_detection/evaluation_and_thresholds.md)). Retrain on post-change data only, and validate before swapping.

## 8. Decision framework
| Observation | Likely cause | Action |
|---|---|---|
| Missing/out-of-range spike | Pipeline or sensor failure | Fix the data; switch to a fallback; don't retrain |
| Input PSI high on an important feature, performance OK | Benign or not-yet-harmful covariate shift | Watch; check the extrapolation (applicability domain) |
| Input PSI high + performance drop | Covariate shift into poorly covered regions | Retrain with recent data (validate first) |
| Performance drop without input drift | Concept drift (process/product/firmware change) | Investigate the cause; retrain on post-change data; update the features |
| Prediction distribution shift, labels delayed | Upstream change or prevalence shift | Check the inputs; recalibrate the thresholds if a prior shift is confirmed |
| Seasonal recurring drift | Normal seasonality | Seasonal reference windows or seasonal features |

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Only monitoring input drift | Concept drift missed | Performance monitoring with labels/proxies |
| KS p-values at large n as alarms | Constant false alarms | Effect sizes + calibrated thresholds |
| Automatic retraining on any drift | Learning from faulty data | Data-quality gates; champion–challenger validation |
| No reference snapshot | Can't quantify drift | Freeze the reference statistics with the model version |
| Ignoring label latency | Late detection | Proxies (soft-sensor residuals, operator feedback) |

## 10. Checklist
- [ ] Reference statistics frozen with the model version.
- [ ] Monitoring layers: quality, input drift, prediction drift, performance, KPIs.
- [ ] Alert thresholds calibrated on historical windows.
- [ ] Retraining policy (scheduled/triggered) with validation gates.
- [ ] Change log of models and data.

## 11. References
- Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M. & Bouchachia, A. (2014). "A survey on concept drift adaptation." *ACM Computing Surveys* 46(4), 44. https://doi.org/10.1145/2523813
- Lu, J. et al. (2019). "Learning under Concept Drift: A Review." *IEEE TKDE* 31(12), 2346–2363. https://doi.org/10.1109/TKDE.2018.2876857
- Rabanser, S., Günnemann, S. & Lipton, Z. (2019). "Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift." *NeurIPS*. https://arxiv.org/abs/1810.11953
- Breck, E. et al. (2017). "The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction." *IEEE Big Data*. https://research.google/pubs/pub46555/
- Evidently AI documentation (open-source drift reports; an optional tool). https://docs.evidentlyai.com/
