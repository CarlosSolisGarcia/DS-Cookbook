# Decision Thresholds and Their Business Implications

> **Folder:** 04_classification · **Prev:** [evaluation.md](evaluation.md) · **Next:** [calibration.md](calibration.md) · **Related:** [class_imbalance.md](class_imbalance.md), [problem_framing.md](../00_methodology/problem_framing.md)

---

## 1. Purpose
Turn model scores into decisions by choosing the threshold (or the number k of cases to act on) that best serves the business objective: minimum expected cost, a required recall, or a fixed capacity. Then estimate its performance honestly.

## 2. When to use it
Whenever predicted classes (not just scores) drive actions: alarms, inspections, interventions, accept/reject. Re-run it when the costs, capacity or prevalence change.

## 3. Conceptual foundation
**The threshold is a decision parameter, not a model parameter.** The default 0.5 is optimal only when the probabilities are calibrated and FP and FN cost the same.

**Cost-optimal threshold for calibrated probabilities.** Flag a case when the expected cost of not acting exceeds the expected cost of acting: p · C_FN > (1 − p) · C_FP, i.e.

  **t\* = C_FP / (C_FP + C_FN)**

Example: an inspection costs 1, a missed failure costs 20 → t\* = 1/21 ≈ 0.048. Flag any machine with more than about a 5% failure risk. This rule needs **calibrated** probabilities. With uncalibrated scores, find t empirically on validation data.

**Common operating-point strategies:**

| Strategy | Choose t such that… | Use when |
|---|---|---|
| Cost minimisation | the expected cost on validation data is minimal | Costs (or their ratio) are known |
| Capacity (top-k) | exactly k cases per period are flagged | Limited inspection/intervention resources |
| Recall target | recall ≥ R (e.g. 90%) with the best precision | Safety-critical misses |
| Precision target | precision ≥ P with the best recall | Each alarm is expensive or erodes trust |
| F-β maximisation | F-β is maximal | Only as a proxy when costs are unknown (state β's meaning) |

**Estimating performance at the chosen threshold.** Choosing t on the same data you report on is optimistic. Choose it on validation data (inner CV), then report it on held-out data. scikit-learn's `TunedThresholdClassifierCV` automates this. It cross-validates the threshold for a given scorer and refits the model.

## 4. Assumptions and limitations
- Costs are often uncertain. Present a **cost-sensitivity** curve (cost vs. threshold for several cost ratios) instead of a single number.
- If prevalence changes in production, both the calibrated probabilities and the precision at a fixed threshold change. Capacity-based (top-k) rules adapt automatically. Fixed thresholds do not.
- Per-period capacity rules need scores comparable across periods. Drift can make "top 20" contain many more or fewer true risks.

## 5. Step-by-step procedure
1. **Elicit costs or constraints** with the stakeholder: C_FP, C_FN (or their ratio), capacity k, minimum recall/precision.
2. **Get out-of-fold probabilities** on the training data (or a validation set).
3. **Compute the metric/cost across thresholds** and plot it (cost curve, precision/recall vs threshold).
4. **Pick the operating point** and check its sensitivity to cost assumptions.
5. **Estimate honestly:** use nested selection (`TunedThresholdClassifierCV`, or a threshold chosen on an inner split) and evaluate on held-out data.
6. **Document** the threshold, the rationale, and the expected precision/recall and alarms per period.
7. **Monitor** the alarm rate and precision in production. Re-tune when costs or prevalence move.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_recall_curve, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, TunedThresholdClassifierCV, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cookbook_data import make_equipment_failure

df = make_equipment_failure(include_leakage=False)
num = ["age_years", "load_pct", "operating_hours", "days_since_maintenance",
       "vibration_rms", "temperature_c", "error_count_7d"]
cat = ["machine_type", "site", "firmware"]
test_start = df["snapshot_date"].sort_values().unique()[-3]
train, test = df[df["snapshot_date"] < test_start], df[df["snapshot_date"] >= test_start]
X, y, groups = train[num + cat], train["failure_30d"].to_numpy(), train["machine_id"]

model = make_pipeline(ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat)]), LogisticRegression(C=0.05, max_iter=2000))
cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
p_oof = cross_val_predict(model, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]

C_FP, C_FN = 1.0, 20.0
def cost_per_case(y_true, y_pred, c_fp=C_FP, c_fn=C_FN):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return (c_fp * fp + c_fn * fn) / len(y_true)

thresholds = np.linspace(0.005, 0.5, 100)
costs = [cost_per_case(y, (p_oof >= t).astype(int)) for t in thresholds]
t_emp = thresholds[int(np.argmin(costs))]
print(f"theoretical t* = {C_FP/(C_FP+C_FN):.3f}  empirical t = {t_emp:.3f}  cost at 0.5 = "
      f"{cost_per_case(y, (p_oof>=0.5).astype(int)):.3f}  cost at t = {min(costs):.3f}")
```

```python
# Cost sensitivity: how does the optimal threshold and cost move if the cost ratio is uncertain?
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for ratio in [5, 10, 20, 40]:
    c = [cost_per_case(y, (p_oof >= t).astype(int), 1.0, ratio) for t in thresholds]
    ax[0].plot(thresholds, c, label=f"C_FN/C_FP={ratio}")
ax[0].set(xlabel="threshold", ylabel="cost per machine-month", title="cost curves"); ax[0].legend()

prec, rec, thr = precision_recall_curve(y, p_oof)
ax[1].plot(thr, prec[:-1], label="precision"); ax[1].plot(thr, rec[:-1], label="recall")
ax[1].set(xlabel="threshold", xlim=(0, 0.5), title="precision / recall vs threshold"); ax[1].legend()
plt.tight_layout(); plt.show()

# Recall-target operating point: the highest threshold that still gives recall >= 0.8
ok = rec[:-1] >= 0.8
t_rec = thr[ok].max()
print(f"recall>=0.8 → threshold {t_rec:.3f}, precision {prec[:-1][ok][thr[ok].argmax()]:.3f}, "
      f"alarm rate {(p_oof >= t_rec).mean():.3f}")
```

```python
# Honest estimate: TunedThresholdClassifierCV picks the threshold by inner CV with a cost scorer
neg_cost = make_scorer(lambda yt, yp: -cost_per_case(yt, yp))      # higher is better
# Grouped inner CV: pass the pre-computed (train, val) index pairs, so no `groups` routing is needed
inner_splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=1).split(X, y, groups))
tuned = TunedThresholdClassifierCV(model, scoring=neg_cost, cv=inner_splits,
                                   thresholds=100, response_method="predict_proba")
tuned.fit(X, y)
print(f"tuned threshold = {tuned.best_threshold_:.3f}")

# Final check on the temporal test set
y_test = test["failure_30d"].to_numpy()
pred_test = tuned.predict(test[num + cat])
print("TEST cost/case: tuned =", round(cost_per_case(y_test, pred_test), 4),
      "| default 0.5 =", round(cost_per_case(y_test, (tuned.estimator_.predict_proba(test[num + cat])[:, 1] >= 0.5).astype(int)), 4),
      "| inspect nobody =", round(cost_per_case(y_test, np.zeros_like(y_test)), 4))
```

> **Version note.** `TunedThresholdClassifierCV` and `FixedThresholdClassifier` exist in scikit-learn ≥ 1.5. Passing `groups=` to `fit` requires metadata routing (`sklearn.set_config(enable_metadata_routing=True)`). Passing explicit index splits as `cv`, as above, avoids that and works in every version.

```python
# Capacity rule: flag the top-k per month (no threshold at all, robust to prevalence/score drift)
def top_k_per_period(df_scores, k, period_col="snapshot_date", score_col="p"):
    r = df_scores.groupby(period_col)[score_col].rank(ascending=False, method="first")
    return (r <= k).astype(int)

ev = train.assign(p=p_oof)
for k in [10, 20, 40]:
    flag = top_k_per_period(ev, k)
    tp = int(((flag == 1) & (ev["failure_30d"] == 1)).sum())
    print(f"k={k:3d}/month: precision={tp/flag.sum():.3f} recall={tp/ev['failure_30d'].sum():.3f}")
```

## 7. Interpretation
- **The empirical cost-optimal threshold lies near the theoretical t\* ≈ 0.05** when the probabilities are reasonably calibrated (logistic regression usually is). A large discrepancy means miscalibration. Either calibrate ([calibration.md](calibration.md)) or rely on the empirical threshold.
- **Cost at 0.5 vs. cost at the tuned t:** with a cost ratio of 20, the 0.5 threshold misses most failures and can be worse than "inspect nobody". The tuned threshold cuts the cost substantially.
- **Cost curves:** a flat valley means the decision is robust to the exact threshold. A sharp minimum means the threshold needs care and monitoring. If the optimal threshold moves a lot with the cost ratio, the uncertainty about costs dominates. Present the options to stakeholders.
- **Capacity results:** precision falls and recall rises as k grows. This table is often the most useful artefact for planners.

## 8. Decision framework
- **IF** costs are known and the probabilities calibrated → start from t\* = C_FP/(C_FP+C_FN); confirm empirically.
- **IF** costs are unknown → show cost curves for a range of ratios, or capacity tables; let the stakeholders choose.
- **IF** capacity is fixed per period → use top-k per period rather than a fixed threshold.
- **IF** misses are unacceptable (safety) → recall-target threshold; report the resulting alarm load.
- **IF** prevalence in production differs from training → recalibrate (prior correction) or use capacity rules; re-tune periodically.
- **IF** several segments have very different costs → consider per-segment thresholds, validated per segment (watch out for small samples).

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Using 0.5 by default | Poor decisions under imbalance or asymmetric costs | Threshold from costs/capacity |
| Choosing the threshold on the test set | Optimistic reporting | Inner CV / `TunedThresholdClassifierCV` |
| Applying t\* to uncalibrated scores | Wrong operating point | Calibrate, or tune empirically |
| Ignoring alarm load | Operators overwhelmed; alerts ignored | Report alarms per period |
| Fixed threshold under drift | Alarm rate drifts | Monitor; capacity rules; re-tune |

## 10. Checklist
- [ ] Costs / capacity / recall or precision constraints elicited.
- [ ] Out-of-fold probabilities computed.
- [ ] Cost or metric curves vs threshold plotted; sensitivity to costs checked.
- [ ] Operating point chosen with a documented rationale.
- [ ] Performance at the operating point estimated on held-out data.
- [ ] Alarm load per period reported.
- [ ] Monitoring and re-tuning plan.

## 11. References
- scikit-learn User Guide, *Tuning the decision threshold for class prediction*. https://scikit-learn.org/stable/modules/classification_threshold.html
- scikit-learn, *Post-tuning the decision threshold for cost-sensitive learning*. https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html
- Elkan, C. (2001). "The Foundations of Cost-Sensitive Learning." *IJCAI*, 973–978.
- Provost, F. & Fawcett, T. (2013). *Data Science for Business*, Ch. 7 "Decision Analytic Thinking I" and Ch. 8 "Visualizing Model Performance" (profit curves).
