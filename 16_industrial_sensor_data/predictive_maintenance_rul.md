# Predictive Maintenance: Degradation, Health Indicators and Remaining Useful Life (RUL)

> **Folder:** 16_industrial_sensor_data · **Prev:** [sensor_preprocessing.md](sensor_preprocessing.md) · **Next:** [process_monitoring.md](process_monitoring.md) · **Related:** [classification_workflow.md](../04_classification/classification_workflow.md) (failure-within-N-days framing), [ts feature_engineering.md](../05_time_series/feature_engineering.md), [cross_validation.md](../06_model_improvement/cross_validation.md)

---

## 1. Purpose
Estimate how long equipment will keep working (RUL), or how likely it is to fail soon, from its condition-monitoring history. The output should support maintenance planning: an estimate, its uncertainty, and an early enough warning.

## 2. When to use it
- Run-to-failure (or run-to-maintenance-threshold) histories exist for several similar units (pumps, bearings, turbines, batteries).
- The degradation is gradual and observable through the sensors.
- **Not** for sudden failures without precursors (no model can predict them from these signals), nor with only one or two failure histories (use physics/expert models or anomaly detection instead).

## 3. Conceptual foundation

### 3.1 Framings
| Framing | Target | Pros | Cons |
|---|---|---|---|
| **RUL regression** | Days/cycles until failure at each time point | Directly plannable | Needs failure times; early-life RUL is unpredictable (→ capping) |
| **Failure within horizon H (classification)** | 1 if failure within H days | Matches capacity/threshold decisions; robust | Choice of H; loses timing detail |
| **Survival analysis** | Hazard / survival function S(t \| x) | Handles **censoring** (units still running) correctly | More specialised tools (`lifelines`, `scikit-survival`) |
| **Health indicator (HI) + threshold** | A monotone degradation index; extrapolate to a failure threshold | Interpretable; works with few failures | Needs a good HI and a threshold definition |

**Piecewise-linear (capped) RUL.** Early in life, the sensors look the same whether 300 or 200 days remain. Training on the true RUL forces the model to guess. A common practice is to cap the target: RUL_capped = min(RUL, R_max), e.g. R_max = 100 days. Choose it from where degradation becomes visible.

**Censoring.** Units that have not failed yet (or that were maintained preventively) are *right-censored*. Dropping them biases the lifetimes downwards, and treating the censoring time as the failure time is wrong. Survival methods use them correctly.

### 3.2 Features for degradation
- **Normalise per operating condition first** ([sensor_preprocessing.md](sensor_preprocessing.md)). Otherwise duty changes dominate.
- **Unit-wise baseline:** subtract each unit's early-life median. This removes unit-to-unit and calibration offsets.
- **Past-window features per unit:** rolling means, rolling slopes (trend), rolling std, exponentially weighted means, and the cumulative operating time at a high setting. All of them must be past-only.
- **Health indicator:** a monotone combination of sensors, e.g. the first principal component of the normalised degradation features, or a regression of the capped RUL on the sensors.

### 3.3 Validation and metrics
- **Split by unit** (GroupKFold). Rows of one unit are highly correlated.
- **Mimic deployment:** evaluate on units *truncated* at random points in life ("in-service" units), predicting the RUL at their last observation, as in the NASA C-MAPSS benchmarks.
- **Metrics:** RMSE/MAE of the RUL, *especially near the end of life* (e.g. true RUL ≤ 60 days, where decisions are made). An **asymmetric score** penalises late predictions (over-estimated RUL leads to an unplanned failure) more than early ones. Also report the calibration of the intervals, if you give them.

## 4. Assumptions and limitations
- Training units must be representative of the deployed units (same equipment type, similar duty). Different failure modes may need separate models.
- Maintenance actions reset or alter degradation. Histories must be segmented at those actions.
- RUL uncertainty is large. A point estimate without an interval invites over-confidence.

## 5. Step-by-step procedure
1. **Define the failure event** (functional failure, or reaching a maintenance threshold) and the time unit (days, cycles, operating hours).
2. **Assemble the histories per unit;** mark the censored units; segment at maintenance actions.
3. **Preprocess:** per-condition normalisation, unit baselines, past-window features ([sensor_preprocessing.md](sensor_preprocessing.md)).
4. **Explore degradation:** plot the normalised sensors vs *time to failure* across units, to see which sensors trend, and from when.
5. **Choose the framing and the target** (capped RUL / failure within H / survival).
6. **Baselines:** fleet-average remaining life given age; a single-sensor HI threshold.
7. **Models:** regularised linear and GBM on the window features; GroupKFold by unit.
8. **Evaluate on truncated test units:** RMSE overall and near the end of life, an asymmetric score, and error vs true RUL.
9. **Uncertainty:** quantile models or conformal intervals ([uncertainty.md](../03_regression/uncertainty.md)), calibrated by unit groups.
10. **Decision rule:** e.g. "schedule maintenance when the predicted RUL's 10th percentile < lead time". Evaluate the cost of missed failures vs early replacements.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cookbook_data import make_pump_fleet

fleet = make_pump_fleet()
SENS = ["vibration_rms", "bearing_temp_c", "discharge_bar", "flow_m3h", "motor_current_a"]

# 1. Per-setting normalisation using early-life (healthy) data, then a unit baseline (removes sensor offsets)
early = fleet["day"] <= 30
base = fleet[early].groupby("setting")[SENS].median(); scale = fleet[early].groupby("setting")[SENS].std()
norm = fleet.copy()
norm[SENS] = (fleet[SENS] - base.loc[fleet["setting"]].to_numpy()) / scale.loc[fleet["setting"]].to_numpy()
unit_base = norm[norm["day"] <= 30].groupby("unit_id")[SENS].median()
norm[SENS] = norm[SENS] - unit_base.loc[norm["unit_id"]].to_numpy()

# 2. Past-only window features per unit
def window_features(d, w=14):
    g = d.groupby("unit_id")
    out = d[["unit_id", "day", "rul_days"]].copy()
    for c in SENS:
        out[f"{c}_mean{w}"] = g[c].transform(lambda s: s.rolling(w, min_periods=3).mean())
        out[f"{c}_ewm"] = g[c].transform(lambda s: s.ewm(span=30).mean())
        # slope of the last w days (least squares on the day index)
        out[f"{c}_slope{w}"] = g[c].transform(lambda s: s.rolling(w, min_periods=5).apply(
            lambda v: np.polyfit(np.arange(len(v)), v, 1)[0], raw=True))
    out["share_setting_B"] = g["setting"].transform(lambda s: (s == "B").expanding().mean())
    return out.dropna()

feat = window_features(norm)
FEATS = [c for c in feat.columns if c not in ("unit_id", "day", "rul_days")]
R_MAX = 100
feat["rul_capped"] = feat["rul_days"].clip(upper=R_MAX)
print(feat.shape)
```

```python
# 3. Explore degradation vs time-to-failure (aligned at failure)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
for a, c in zip(ax, ["vibration_rms_ewm", "bearing_temp_c_ewm", "flow_m3h_ewm"]):
    for u, d in feat.groupby("unit_id"):
        a.plot(-d["rul_days"], d[c], lw=0.4, alpha=0.4, color="tab:blue")
    a.set_xlabel("days before failure"); a.set_title(c)
plt.tight_layout(); plt.show()
```

```python
# 4. Deployment-like evaluation: test units truncated at a random point; predict RUL at the last observation
rng = np.random.default_rng(0)
units = feat["unit_id"].unique()
def truncate_units(df, units, rng):
    rows = []
    for u in units:
        d = df[df["unit_id"] == u]
        cut = rng.integers(int(0.3 * len(d)), len(d))          # observed until a random point in life
        rows.append(d.iloc[cut])
    return pd.DataFrame(rows)

def nasa_score(err):
    """Asymmetric PHM08 score: err = predicted − true. Late predictions (err > 0) are penalised more."""
    return np.sum(np.where(err < 0, np.exp(-err / 13) - 1, np.exp(err / 10) - 1))

models = {
    "fleet baseline (mean life − age)": None,
    "ridge (capped RUL)": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
    "GBM (capped RUL)": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=40, random_state=0),
}
results = {k: [] for k in models}
for fold, (tr_u, te_u) in enumerate(GroupKFold(5).split(units, groups=units)):
    train = feat[feat["unit_id"].isin(units[tr_u])]
    test_last = truncate_units(feat, units[te_u], rng)
    life = train.groupby("unit_id")["day"].max().mean()
    for name, fac in models.items():
        if fac is None:
            pred = np.clip(life - test_last["day"], 0, None).clip(upper=R_MAX)
        else:
            pred = fac().fit(train[FEATS], train["rul_capped"]).predict(test_last[FEATS])
        results[name].append(pd.DataFrame({"true": test_last["rul_capped"].to_numpy(),
                                           "true_uncapped": test_last["rul_days"].to_numpy(), "pred": pred}))

rows = []
for name, parts in results.items():
    r = pd.concat(parts); e = r["pred"] - r["true"]; late = r["true"] <= 60
    rows.append({"model": name, "RMSE": np.sqrt(np.mean(e ** 2)), "MAE": np.mean(np.abs(e)),
                 "RMSE (true RUL ≤ 60)": np.sqrt(np.mean(e[late] ** 2)), "late predictions %": 100 * np.mean(e > 10),
                 "NASA score": nasa_score(e)})
print(pd.DataFrame(rows).set_index("model").round(1).to_string())
```

```python
# 5. Error vs true RUL for the GBM: errors should shrink near end of life (when it matters)
r = pd.concat(results["GBM (capped RUL)"])
plt.figure(figsize=(6, 4)); plt.scatter(r["true"], r["pred"], s=10); plt.plot([0, R_MAX], [0, R_MAX], "k--")
plt.xlabel("true (capped) RUL"); plt.ylabel("predicted"); plt.title("RUL predictions on truncated test units"); plt.show()

# Simple Kaplan–Meier survival curve of unit lifetimes (no censoring here; with censoring, pass event flags)
life_times = np.sort(fleet.groupby("unit_id")["day"].max().to_numpy())
km = 1 - np.arange(1, len(life_times) + 1) / len(life_times)
print("median life (days):", np.median(life_times), "| P(survive 180 days) ≈", round(float(np.mean(life_times > 180)), 2))
# With censored units use lifelines.KaplanMeierFitter or sksurv.nonparametric.kaplan_meier_estimator.
```

## 7. Interpretation
- **Degradation plots aligned at failure:** vibration and bearing temperature rise in the last ~30–40% of life, more steeply near the end, while early life is flat. This justifies capping the RUL at ~100 days: before that point, the sensors carry little information about the remaining life.
- **Model table:** the fleet baseline uses only the age, and it is poor (RMSE ≈ 43 days) because lifetimes vary from 120 to 360 days. The sensor-based models cut the error by more than half (RMSE ≈ 18), and by about as much for **true RUL ≤ 60**, the decision-relevant region (≈ 21–22 days). Ridge and GBM tie on RMSE. GBM has a lower MAE and fewer late predictions (22% vs 35%).
- **The NASA score tells a different story:** ridge scores better (≈ 550 vs ≈ 850). The asymmetric exponential penalty is dominated by a **few very late predictions**, and GBM has some of those. Which model is "better" depends on whether you care about the typical error (MAE) or about the worst late misses (the score). Decide from the maintenance cost structure, and look at the error distribution, not only the averages.
- **Late predictions:** a model with good RMSE but many late predictions is dangerous (failures occur before the planned maintenance). If late errors dominate, bias the decision rule (use a lower quantile), or train with an asymmetric (quantile) loss.
- **Prediction vs truth:** points near the capped region (true ≈ 100) spread widely, which is expected. Near the end of life the predictions should tighten around the diagonal.
- **Kaplan–Meier:** a fleet-level reliability view (median life, survival at a given age). With censored units, use a proper survival library.

## 8. Decision framework
- **IF** there are many run-to-failure histories (≥ ~30 units) → RUL regression with capped targets, or a time-window classification; GroupKFold by unit.
- **IF** few failures but many healthy/censored units → survival analysis, or a health indicator with threshold extrapolation; anomaly detection for early warning.
- **IF** operating conditions vary → per-condition normalisation (non-negotiable).
- **IF** units have sensor offsets → unit-wise baselines.
- **IF** maintenance decisions need lead time L → evaluate whether alarms occur ≥ L before failure. Use a lower prediction quantile for the decisions.
- **IF** several failure modes exist → check whether the degradation signatures differ; use separate models or a mode classifier.
- **IF** the RUL error near the end of life is not much better than the baseline → the sensors lack precursors. Look for new signals (spectra, oil analysis) rather than tuning the model.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Random row-level CV | Same unit in train and test → severe optimism | GroupKFold by unit |
| Evaluating on full histories only | Over-weights easy early-life rows | Truncated test units; metrics near end of life |
| Uncapped RUL target | Model forced to guess early life; noisy training | Piecewise-linear cap |
| Dropping censored units | Biased (short) lifetimes | Survival methods |
| Centred/rolling-forward features | Leakage of future degradation | Past-only windows per unit |
| Ignoring maintenance resets | Mixed degradation trajectories | Segment the histories |
| Point RUL only | Over-confident planning | Quantiles / conformal intervals |

## 10. Checklist
- [ ] Failure event and time unit defined; censoring and maintenance resets handled.
- [ ] Per-condition normalisation + unit baselines.
- [ ] Past-only window features per unit.
- [ ] Degradation explored vs time-to-failure; cap chosen.
- [ ] Baselines; models with GroupKFold by unit.
- [ ] Evaluation on truncated units: RMSE/MAE near the end of life, late %, asymmetric score.
- [ ] Uncertainty and a decision rule with lead time.

## 11. References
- Saxena, A., Goebel, K., Simon, D. & Eklund, N. (2008). "Damage propagation modeling for aircraft engine run-to-failure simulation." *PHM*. The C-MAPSS data and the asymmetric scoring function. https://doi.org/10.1109/PHM.2008.4711414
- Heimes, F. O. (2008). "Recurrent neural networks for remaining useful life estimation." *PHM*. Piecewise-linear RUL target. https://doi.org/10.1109/PHM.2008.4711422
- Lei, Y. et al. (2018). "Machinery health prognostics: A systematic review from data acquisition to RUL prediction." *MSSP* 104, 799–834. https://doi.org/10.1016/j.ymssp.2017.11.016
- Si, X.-S., Wang, W., Hu, C.-H. & Zhou, D.-H. (2011). "Remaining useful life estimation — A review on the statistical data driven approaches." *EJOR* 213(1), 1–14. https://doi.org/10.1016/j.ejor.2010.11.018
- lifelines documentation (survival analysis in Python). https://lifelines.readthedocs.io/ · scikit-survival: https://scikit-survival.readthedocs.io/
- NASA Prognostics Center of Excellence, data repository. https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
