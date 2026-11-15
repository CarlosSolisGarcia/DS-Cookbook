# Classification Workflow: End-to-End Playbook

> **Folder:** 04_classification · **Start here for any categorical-target problem.** · **Next:** [baseline_models.md](baseline_models.md) · **Related:** [project_lifecycle.md](../00_methodology/project_lifecycle.md), [regression_workflow.md](../03_regression/regression_workflow.md) (parallel structure)

---

## 1. Purpose
This is the master procedure for predicting a class (binary or multiclass): failure / no failure, defect type, pass / fail, churn. It sequences the general recipes and the classification-specific decisions. Those decisions are imbalance, probability quality, thresholds and the costs of each error type.

## 2. When to use it
- The target is a category. The output is a probability, a class or a ranking.
- Binary, multiclass (exactly one of K classes) or multilabel (several labels per row; touched on only briefly here).
- Not when the class is "unusual vs. normal" and labels for the unusual cases are absent or very scarce. Then consider [10_anomaly_detection](../10_anomaly_detection/).

## 3. Conceptual foundation
A classifier usually produces a **score** or **probability** p̂(y = 1 | x). Turning that into a decision requires a **threshold**, and the threshold is a business decision, not a modelling default. Keep three things apart:

| Layer | Question | Recipe |
|---|---|---|
| **Ranking quality** | Do positives get higher scores than negatives? | ROC-AUC, PR-AUC ([evaluation.md](evaluation.md)) |
| **Probability quality** | Does p̂ = 0.2 mean a 20% chance? | Calibration, Brier score ([calibration.md](calibration.md)) |
| **Decision quality** | At the chosen threshold, are the costs minimised? | Confusion matrix, cost/capacity curves ([threshold_optimization.md](threshold_optimization.md)) |

A model can rank well, be poorly calibrated and be used at a bad threshold, all at the same time. Fix each layer separately.

**Class imbalance** affects all three layers. Accuracy becomes meaningless, PR curves become more informative than ROC curves, and the default threshold of 0.5 is rarely optimal ([class_imbalance.md](class_imbalance.md)).

## 4. Assumptions and limitations
- Labels are assumed correct. Label noise caps achievable performance ([data_quality.md](../01_data_understanding/data_quality.md)).
- Class priors in production must resemble training, or be corrected for. A changed prevalence shifts the precision and the optimal threshold even if the model is unchanged.
- Multilabel and hierarchical classification are mentioned but not covered in depth.

## 5. Step-by-step procedure

| Step | Action | Recipe | Output |
|---|---|---|---|
| 1 | Frame: classes, unit, t₀, horizon, **costs of FP and FN**, capacity | [problem_framing.md](../00_methodology/problem_framing.md) | Problem statement with a cost matrix |
| 2 | Inspect, quality, clean; audit for leakage | [01_data_understanding](../01_data_understanding/initial_inspection.md), [data_leakage.md](../00_methodology/data_leakage.md) | Clean data |
| 3 | Split: stratified + grouped / temporal | [validation_reference.md](../08_quick_references/validation_reference.md) | train / test |
| 4 | EDA: class balance, feature–class relationships | [EDA](../01_data_understanding/exploratory_data_analysis.md) + §6 below | EDA notes |
| 5 | Baselines | [baseline_models.md](baseline_models.md) | Baseline scores |
| 6 | Preprocess; candidate models (logistic → GBM) | [model_selection.md](model_selection.md) | Comparison table |
| 7 | Handle imbalance *if* it hurts the decision metric | [class_imbalance.md](class_imbalance.md) | Strategy chosen |
| 8 | Evaluate ranking + probabilities + segments | [evaluation.md](evaluation.md) | Evaluation report |
| 9 | Calibrate if probabilities are used as risks | [calibration.md](calibration.md) | Calibrated model |
| 10 | Choose the threshold from costs/capacity | [threshold_optimization.md](threshold_optimization.md) | Operating point |
| 11 | Error analysis: FP and FN profiles | [error_analysis.md](error_analysis.md) | Next actions |
| 12 | Interpret; iterate; final test once | [model_interpretation.md](../06_model_improvement/model_interpretation.md), [iterative_improvement.md](../06_model_improvement/iterative_improvement.md) | Final model + report |

## 6. Python implementation: compact end-to-end skeleton

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cookbook_data import make_equipment_failure

df = make_equipment_failure(include_leakage=False)      # outcome columns removed (see data_leakage.md)
TARGET = "failure_30d"
num = ["age_years", "load_pct", "operating_hours", "days_since_maintenance",
       "vibration_rms", "temperature_c", "error_count_7d"]
cat = ["machine_type", "site", "firmware"]

# EDA essentials for classification: class balance overall, over time and by segment
print("prevalence:", round(df[TARGET].mean(), 4))
print(df.groupby("snapshot_date")[TARGET].mean().round(3).to_dict())
print(df.groupby("machine_type")[TARGET].agg(["mean", "size"]).round(3))

# Split: the model is used on the SAME fleet in FUTURE months → temporal test (last 3 months).
# Inside training, CV grouped by machine (rows of one machine are correlated) and stratified.
test_start = df["snapshot_date"].sort_values().unique()[-3]
train, test = df[df["snapshot_date"] < test_start], df[df["snapshot_date"] >= test_start]
X, y, groups = train[num + cat], train[TARGET], train["machine_id"]
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

logit = make_pipeline(
    ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
                       ("cat", OneHotEncoder(handle_unknown="ignore"), cat)]),
    LogisticRegression(max_iter=2000))
hgb = HistGradientBoostingClassifier(categorical_features="from_dtype", learning_rate=0.05, max_depth=3,
                                     max_iter=300, random_state=0)
models = {"prior (dummy)": DummyClassifier(strategy="prior"), "logistic": logit, "hgb": hgb}
scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "brier": "neg_brier_score"}
rows = []
for name, m in models.items():
    Xm = X.astype({c: "category" for c in cat}) if name == "hgb" else X
    r = cross_validate(m, Xm, y, groups=groups, cv=cv, scoring=scoring)
    rows.append({"model": name, "ROC-AUC": r["test_roc_auc"].mean(), "PR-AUC": r["test_pr_auc"].mean(),
                 "PR-AUC std": r["test_pr_auc"].std(), "Brier": -r["test_brier"].mean()})
print(pd.DataFrame(rows).set_index("model").round(4))
```

```python
# Final test evaluation (ONCE, after choosing the model, calibration and threshold)
final = logit.fit(X, y)
p_test = final.predict_proba(test[num + cat])[:, 1]
print(f"TEST ROC-AUC={roc_auc_score(test[TARGET], p_test):.3f}  "
      f"PR-AUC={average_precision_score(test[TARGET], p_test):.3f}  "
      f"Brier={brier_score_loss(test[TARGET], p_test):.4f}  prevalence={test[TARGET].mean():.3f}")
```

## 7. Interpretation
- **The dummy "prior" model** has ROC-AUC = 0.5 and PR-AUC ≈ prevalence (≈ 0.05). These are the zero points. A PR-AUC of 0.2 is therefore **4× better than random**, even though it looks low.
- **Brier score:** the dummy's Brier is ≈ p(1−p) ≈ 0.047. A useful model must go below it.
- **Logistic ≈ HGB** here suggests mostly additive, monotonic effects (log-linear in the generating process). In that case prefer logistic: it is interpretable and usually well calibrated.
- **Prevalence changing over time or across segments** means precision and thresholds will move too. Keep that in mind for monitoring.

## 8. Decision framework
- **IF** a fixed number of cases can be acted on (capacity) → evaluate precision@k / recall@k; the threshold follows from k.
- **IF** costs per FP/FN are known → choose the threshold that minimises expected cost ([threshold_optimization.md](threshold_optimization.md)).
- **IF** probabilities are consumed downstream (risk scores, expected-cost calculations) → calibrate and report Brier/log-loss.
- **IF** prevalence < ~10% → report PR-AUC alongside ROC-AUC; ignore accuracy.
- **IF** the classes are ordered (low/medium/high severity) → consider ordinal approaches or regression on a score; at minimum, report confusion between adjacent classes separately.
- **IF** multiclass → evaluate per class (one-vs-rest metrics, confusion matrix); macro vs. weighted averages answer different questions ([evaluation.md](evaluation.md)).
- **IF** the positive class is extremely rare (tens of examples) → consider anomaly detection, a longer horizon or label aggregation ([problem_framing.md](../00_methodology/problem_framing.md)).

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Accuracy as the metric with imbalance | "95% accuracy" by predicting all negatives | PR-AUC, recall/precision at the operating point |
| Default threshold 0.5 | Suboptimal decisions | Threshold from costs/capacity |
| Resampling before splitting | Leakage, optimistic scores | Resample inside training folds only |
| Ignoring groups (machines, patients) | Optimistic CV | `StratifiedGroupKFold` |
| Interpreting uncalibrated scores as probabilities | Wrong risk estimates | Calibration check |

## 10. Checklist
- [ ] Classes, horizon and the cost of FP/FN (or capacity) defined.
- [ ] Leakage audit done (outcome columns removed).
- [ ] Split stratified and respects groups/time.
- [ ] Prevalence overall, over time and by segment known.
- [ ] Dummy and simple baselines in the comparison table.
- [ ] Ranking, probability and decision quality evaluated separately.
- [ ] Imbalance strategy justified by the decision metric (not applied by default).
- [ ] Threshold chosen on validation data from costs/capacity.
- [ ] FP/FN error analysis.
- [ ] Final test once.

## 11. References
- scikit-learn User Guide, *Model evaluation: classification metrics*. https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
- scikit-learn User Guide, *Tuning the decision threshold for class prediction*. https://scikit-learn.org/stable/modules/classification_threshold.html
- Provost, F. & Fawcett, T. (2013). *Data Science for Business*, Ch. 7–8 (expected value, ROC, cost curves).
- James, G. et al. (2023). *ISLP*, Ch. 4 "Classification". https://www.statlearning.com/
