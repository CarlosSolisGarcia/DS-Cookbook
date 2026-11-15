# Class Imbalance Strategies

> **Folder:** 04_classification · **Prev:** [model_selection.md](model_selection.md) · **Next:** [evaluation.md](evaluation.md) · **Related:** [threshold_optimization.md](threshold_optimization.md), [calibration.md](calibration.md), [19_advanced_tabular_ml](../19_advanced_tabular_ml/) (imbalance beyond binary)

---

## 1. Purpose
Decide whether class imbalance is actually a problem for your decision and, if it is, choose the least harmful remedy: threshold tuning, class weights, under-/oversampling or SMOTE. Then verify that the remedy improves the **decision metric**, not just recall at 0.5.

## 2. When to use it
- The positive class is a small fraction (≲ 10–20%) of the data, **and** you see poor recall/precision at your operating point.
- Read it *before* applying SMOTE "because the data are imbalanced".

## 3. Conceptual foundation
Imbalance on its own is **not** a modelling error. A well-specified probabilistic model trained on imbalanced data estimates the correct (low) probabilities. The problems people attribute to imbalance usually come from three sources:
1. **The wrong metric** (accuracy) → fix the metric ([evaluation.md](evaluation.md)).
2. **The wrong threshold** (0.5 when positives have p̂ ≈ 0.05–0.3) → fix the threshold ([threshold_optimization.md](threshold_optimization.md)).
3. **Too few positive examples** to learn their patterns → this is a *data* problem. Resampling cannot create information. More positives, better features or a different framing can help.

| Remedy | What it does | Effect on probabilities | When it helps |
|---|---|---|---|
| **Threshold tuning** | Moves the decision cut-off | None (the ranking is unchanged) | Almost always the first step |
| **Class weights** (`class_weight="balanced"`) | Up-weights positives in the loss | Inflates p̂ for positives → miscalibrated | When the training algorithm otherwise ignores the minority (e.g. trees stopping early); models without good probability outputs |
| **Random undersampling** of negatives | Discards negatives | Inflated p̂ (correctable) | Very large data; speed; ensembles of undersampled models (`BalancedRandomForest`) |
| **Random oversampling** of positives | Duplicates positives | Inflated p̂ | Rarely better than weights; risk of overfitting duplicates |
| **SMOTE** | Synthesises positives by interpolating neighbours | Inflated p̂ | Occasionally for weak learners; often no gain over weights + threshold for GBMs; can create unrealistic samples |

**Prior correction.** If you trained with positives re-weighted or resampled to an effective prevalence π′ instead of the true π, you can map the scores back to calibrated probabilities:
p = (p′·π/π′) / (p′·π/π′ + (1−p′)·(1−π)/(1−π′)).
Recalibrating on validation data ([calibration.md](calibration.md)) is the practical alternative.

**Evidence.** Studies comparing these remedies (e.g. van den Goorbergh et al., 2022, for logistic regression; Elor & Averbuch-Elor, 2022, for boosting) find that rebalancing seldom improves discrimination (AUC), and that it harms calibration. Threshold choice achieves most of the claimed benefits.

## 4. Assumptions and limitations
- Any resampling must happen **inside training folds only** (`imblearn.pipeline.Pipeline`). Resampling before the split leaks synthetic copies of validation points into training.
- SMOTE interpolates in feature space. With categorical features, use `SMOTENC`. With strongly non-convex class regions it creates implausible points.
- Extreme imbalance (< 0.1%, with a handful of positives) is often better treated as anomaly detection or by changing the label horizon ([10_anomaly_detection](../10_anomaly_detection/)).

## 5. Step-by-step procedure
1. **Quantify:** prevalence overall and per segment; the absolute number of positives (the key number: 300 positives is a very different situation from 30).
2. **Fix the metric:** PR-AUC + the decision metric (cost, precision@k, recall at a fixed precision).
3. **Fit a baseline without any rebalancing**, then tune **only the threshold** on validation data.
4. **Try class weights.** Compare PR-AUC, the decision metric and the Brier score.
5. **Try resampling** (undersampling, SMOTE) **only if** steps 3–4 are unsatisfactory, inside an imblearn pipeline.
6. **Keep the simplest approach** that is within the noise of the best decision metric. If rebalancing was used and probabilities matter, recalibrate.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from cookbook_data import make_equipment_failure

df = make_equipment_failure(include_leakage=False)
num = ["age_years", "load_pct", "operating_hours", "days_since_maintenance",
       "vibration_rms", "temperature_c", "error_count_7d"]
X, y, groups = df[num], df["failure_30d"].to_numpy(), df["machine_id"]
print(f"positives: {y.sum()} of {len(y)} (prevalence {y.mean():.3f})")

def expected_cost(y_true, p, thr, c_fp=1.0, c_fn=20.0):
    """Cost per machine-month: an unnecessary inspection costs 1, a missed failure 20 (from the problem statement)."""
    pred = p >= thr
    return (c_fp * np.sum(pred & (y_true == 0)) + c_fn * np.sum(~pred & (y_true == 1))) / len(y_true)

def best_threshold(y_true, p, **kw):
    grid = np.unique(np.quantile(p, np.linspace(0, 1, 201)))
    costs = [expected_cost(y_true, p, t, **kw) for t in grid]
    return grid[int(np.argmin(costs))]

def lr(**kw):
    return LogisticRegression(max_iter=2000, **kw)

variants = {
    "logistic, no rebalancing": ImbPipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("clf", lr())]),
    "logistic, class_weight=balanced": ImbPipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                                                    ("clf", lr(class_weight="balanced"))]),
    "logistic, undersample 1:3": ImbPipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                                              ("rus", RandomUnderSampler(sampling_strategy=1/3, random_state=0)), ("clf", lr())]),
    "logistic, SMOTE": ImbPipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                                    ("smote", SMOTE(random_state=0)), ("clf", lr())]),
    "HGB, no rebalancing": ImbPipeline([("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, random_state=0))]),
    "HGB, class_weight=balanced": ImbPipeline([("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                                                                      class_weight="balanced", random_state=0))]),
}

outer = StratifiedGroupKFold(5, shuffle=True, random_state=42)
rows = []
for name, model in variants.items():
    m = {"ROC-AUC": [], "PR-AUC": [], "Brier": [], "cost@0.5": [], "cost@tuned": [], "mean p̂": []}
    for tr, va in outer.split(X, y, groups):
        # Threshold tuned on an INNER validation split of the training fold (never on the evaluation fold)
        inner_tr, inner_va = next(StratifiedGroupKFold(4, shuffle=True, random_state=1)
                                  .split(X.iloc[tr], y[tr], groups.iloc[tr]))
        model.fit(X.iloc[tr].iloc[inner_tr], y[tr][inner_tr])
        thr = best_threshold(y[tr][inner_va], model.predict_proba(X.iloc[tr].iloc[inner_va])[:, 1])
        model.fit(X.iloc[tr], y[tr])                     # refit on the full training fold
        p = model.predict_proba(X.iloc[va])[:, 1]
        m["ROC-AUC"].append(roc_auc_score(y[va], p)); m["PR-AUC"].append(average_precision_score(y[va], p))
        m["Brier"].append(brier_score_loss(y[va], p)); m["mean p̂"].append(p.mean())
        m["cost@0.5"].append(expected_cost(y[va], p, 0.5)); m["cost@tuned"].append(expected_cost(y[va], p, thr))
    rows.append({"variant": name, **{k: np.mean(v) for k, v in m.items()}})
print(pd.DataFrame(rows).set_index("variant").round(4))
print("cost of 'inspect nothing':", round(expected_cost(y, np.zeros(len(y)), 0.5), 4),
      "| cost of 'inspect everything':", round(expected_cost(y, np.ones(len(y)), 0.5), 4))
```

```python
# Prior correction: map probabilities from a model trained at effective prevalence pi_prime back to the true pi
def prior_correct(p_prime, pi, pi_prime):
    a = p_prime * pi / pi_prime
    b = (1 - p_prime) * (1 - pi) / (1 - pi_prime)
    return a / (a + b)

pi = y.mean()
print(prior_correct(np.array([0.1, 0.5, 0.9]), pi=pi, pi_prime=0.5).round(4))   # balanced-weight model → true scale
```

## 7. Interpretation
- **ROC-AUC and PR-AUC barely change** across variants. Rebalancing does not improve the *ranking*.
- **"cost@0.5" improves dramatically with class weights or resampling,** but only because these methods effectively move the threshold. Compare **cost@tuned**: with a tuned threshold the unweighted model is as good (within the noise) as the rebalanced ones.
- **Brier score and mean p̂:** the rebalanced variants have a mean p̂ far above the prevalence (≈ 0.05). Their probabilities are inflated, so they cannot be used as risks without recalibration.
- **SMOTE** typically gives no advantage here, and adds a hyperparameter (k neighbours) plus synthetic data to justify.
- **Suspicious:** a resampling method that greatly improves PR-AUC. Check whether the resampling happened outside the CV loop (leakage).

## 8. Decision framework
```
Imbalanced target
├── Is the metric appropriate (PR-AUC / cost / precision@k, not accuracy)? ── no → fix metric first
├── Tune the threshold on validation data (no rebalancing) ── good enough? → STOP
├── Does the algorithm ignore the minority (e.g. predicts all-negative, very shallow trees)? 
│     └── yes → class weights (then recalibrate if probabilities matter)
├── Very large dataset / training too slow? → undersample negatives (+ prior correction)
├── Still unsatisfactory → the problem is information, not imbalance:
│     more positive examples, better features, longer horizon, label aggregation, anomaly framing
└── SMOTE: only as a validated experiment inside imblearn Pipeline; keep only if it beats weights+threshold
```
- **IF** probabilities are used downstream → avoid rebalancing, or recalibrate afterwards.
- **IF** there are fewer than ~50 positives → favour simple models (regularised logistic), report wide uncertainty, consider anomaly detection.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| SMOTE / oversampling before splitting | Synthetic neighbours of validation points in training → optimistic | `imblearn.pipeline.Pipeline` inside CV |
| Judging rebalancing at threshold 0.5 | Credits rebalancing for a threshold effect | Compare at tuned thresholds |
| Using rebalanced probabilities as risks | Over-estimated risk | Prior correction or recalibration |
| Tuning the threshold on the evaluation fold | Optimistic cost | Inner validation split / `TunedThresholdClassifierCV` |
| Resampling the test set | Test no longer represents production | Never resample validation/test data |

## 10. Checklist
- [ ] Number of positives known (not just the %).
- [ ] Decision metric defined (cost, capacity, recall at precision).
- [ ] Unweighted model + tuned threshold evaluated first.
- [ ] Class weights / resampling compared at tuned thresholds, inside CV.
- [ ] Calibration checked after any rebalancing.
- [ ] Simplest adequate strategy selected.

## 11. References
- van den Goorbergh, R., van Smeden, M., Timmerman, D. & Van Calster, B. (2022). "The harm of class imbalance corrections for risk prediction models." *JAMIA* 29(9), 1525–1534. https://doi.org/10.1093/jamia/ocac093
- Elor, Y. & Averbuch-Elor, H. (2022). "To SMOTE, or not to SMOTE?" https://arxiv.org/abs/2201.08528
- Chawla, N. V., Bowyer, K. W., Hall, L. O. & Kegelmeyer, W. P. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." *JAIR* 16, 321–357. https://doi.org/10.1613/jair.953
- imbalanced-learn documentation (Pipeline, samplers, `BalancedRandomForestClassifier`). https://imbalanced-learn.org/stable/
- Elkan, C. (2001). "The Foundations of Cost-Sensitive Learning." *IJCAI*. Thresholds from costs; prior correction.
- scikit-learn, *Post-tuning the decision threshold for cost-sensitive learning*. https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html
