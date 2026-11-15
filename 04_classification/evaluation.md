# Classification Evaluation: Confusion Matrix, ROC, PR and Beyond

> **Folder:** 04_classification · **Prev:** [class_imbalance.md](class_imbalance.md) · **Next:** [threshold_optimization.md](threshold_optimization.md) · **Related:** [calibration.md](calibration.md), [metrics_reference.md](../08_quick_references/metrics_reference.md), [error_analysis.md](error_analysis.md)

---

## 1. Purpose
Measure what a classifier does well and badly at three levels: ranking, probabilities and decisions at a threshold. Then compare models with the uncertainty of the estimates in view.

## 2. When to use it
At every model comparison and for the final report. Threshold *selection* is in [threshold_optimization.md](threshold_optimization.md). Probability *calibration* is in [calibration.md](calibration.md).

## 3. Conceptual foundation

### 3.1 The confusion matrix (binary, at a threshold t)

|  | Predicted negative | Predicted positive |
|---|---|---|
| **Actual negative** | TN | FP (false alarm) |
| **Actual positive** | FN (miss) | TP |

| Metric | Formula | Question it answers | Hides |
|---|---|---|---|
| Accuracy | (TP+TN)/N | Fraction correct | Useless under imbalance |
| Precision (PPV) | TP/(TP+FP) | Of the alarms, how many are real? | Missed positives |
| Recall (sensitivity, TPR) | TP/(TP+FN) | Of the real positives, how many did we catch? | False alarms |
| Specificity (TNR) | TN/(TN+FP) | Of the negatives, how many did we leave alone? | Tiny changes are meaningful when negatives are huge |
| FPR | FP/(FP+TN) = 1 − specificity | False alarm rate among negatives | — |
| F1 | 2·P·R/(P+R) | Balance of P and R | Assumes equal importance; ignores TN; threshold-dependent |
| F-β | (1+β²)PR/(β²P+R) | β > 1 favours recall | Choice of β is a cost assumption |
| Balanced accuracy | (TPR+TNR)/2 | Accuracy that is robust to imbalance | Costs |
| MCC | correlation of predicted and actual | One-number summary using all four cells | Not cost-aware |

**Precision depends on prevalence; recall and specificity do not.** The same model deployed where failures are twice as frequent will show higher precision. Remember this when comparing sites or periods.

### 3.2 Threshold-free ranking metrics
- **ROC curve:** TPR vs FPR over all thresholds. **ROC-AUC** = the probability that a random positive scores above a random negative. It is insensitive to prevalence, so it can look excellent even when precision is poor under heavy imbalance (an FPR of 1% of 100 000 negatives is 1000 false alarms).
- **Precision–Recall curve:** precision vs recall. **Average precision (AP)** is the step-wise area under it and is the recommended PR summary (scikit-learn's `average_precision_score`). The random-model level equals the prevalence. More informative than ROC when positives are rare and the costs concentrate on the positive class.
- **Partial measures:** precision@k, recall@k, or recall at a fixed precision (e.g. "recall at ≥ 50% precision"). These are what capacity-constrained decisions actually use.

### 3.3 Probability metrics
- **Log-loss** (cross-entropy): heavily penalises confident wrong probabilities.
- **Brier score:** mean (p̂ − y)². It mixes calibration and discrimination. Compare it with the prevalence baseline π(1−π).

### 3.4 Multiclass
- Per-class precision/recall/F1 from `classification_report`.
- **Macro average:** the unweighted mean over classes (rare classes count equally). **Weighted average:** weighted by support (dominated by frequent classes). **Micro average:** pooled counts (= accuracy for single-label multiclass).
- One-vs-rest ROC-AUC/AP per class; the normalised confusion matrix shows *which* classes are confused.

## 4. Assumptions and limitations
- Metrics from CV folds with few positives are noisy (PR-AUC especially). Report the std, or bootstrap CIs (grouped by entity where relevant).
- AUCs summarise all thresholds, including irrelevant ones. Always add the metric at the operating point.
- Comparisons across datasets with different prevalence are invalid for precision and PR-AUC unless the prevalence is accounted for.

## 5. Step-by-step procedure
1. **Collect out-of-fold probabilities** for every candidate on identical folds.
2. **Ranking:** ROC-AUC and PR-AUC (AP), with the std across folds; plot both curves.
3. **Operating-point metrics:** confusion matrix, precision, recall, specificity at the chosen threshold, or at the capacity k.
4. **Probabilities:** Brier, log-loss; reliability diagram ([calibration.md](calibration.md)).
5. **Segments:** metrics per site, machine type and period. Watch for prevalence differences.
6. **Uncertainty:** bootstrap the key metric (grouped by machine).
7. **Final test** once; compare with CV.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             average_precision_score, balanced_accuracy_score, brier_score_loss,
                             classification_report, confusion_matrix, log_loss, matthews_corrcoef,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cookbook_data import make_equipment_failure

df = make_equipment_failure(include_leakage=False)
num = ["age_years", "load_pct", "operating_hours", "days_since_maintenance",
       "vibration_rms", "temperature_c", "error_count_7d"]
cat = ["machine_type", "site", "firmware"]
X, y, groups = df[num + cat], df["failure_30d"].to_numpy(), df["machine_id"]
cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)

logit = make_pipeline(ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat)]), LogisticRegression(C=0.05, max_iter=2000))
hgb = HistGradientBoostingClassifier(categorical_features="from_dtype", max_depth=3, learning_rate=0.05,
                                     max_iter=300, random_state=0)
oof = {"logistic": cross_val_predict(logit, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1],
       "hgb": cross_val_predict(hgb, X.astype({c: "category" for c in cat}), y, cv=cv, groups=groups,
                                method="predict_proba")[:, 1]}

def prob_report(y, p):
    return {"ROC-AUC": roc_auc_score(y, p), "PR-AUC (AP)": average_precision_score(y, p),
            "Brier": brier_score_loss(y, p), "log-loss": log_loss(y, p),
            "Brier (prior baseline)": brier_score_loss(y, np.full_like(p, y.mean()))}
print(pd.DataFrame({k: prob_report(y, p) for k, p in oof.items()}).round(4))
```

```python
# Curves: ROC and PR side by side, with the random-model reference
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for name, p in oof.items():
    RocCurveDisplay.from_predictions(y, p, name=name, ax=ax[0])
    PrecisionRecallDisplay.from_predictions(y, p, name=name, ax=ax[1])
ax[0].plot([0, 1], [0, 1], "k--", lw=1)
ax[1].axhline(y.mean(), color="k", ls="--", lw=1, label=f"random (prevalence={y.mean():.3f})")
ax[1].legend(); plt.tight_layout(); plt.show()
```

```python
# Operating point: capacity of 20 inspections per month for ~500 machines → flag the top 4% each month
p = oof["logistic"]
df_eval = df.assign(p=p)
df_eval["flag"] = df_eval.groupby("snapshot_date")["p"].rank(ascending=False, method="first") <= 20
pred = df_eval["flag"].to_numpy().astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
print(f"precision={tp/(tp+fp):.3f} recall={tp/(tp+fn):.3f} specificity={tn/(tn+fp):.3f} "
      f"balanced_acc={balanced_accuracy_score(y, pred):.3f} MCC={matthews_corrcoef(y, pred):.3f}")
ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=["no failure", "failure"]); plt.show()
```

```python
# Segment evaluation: note how prevalence differs by segment (precision is prevalence-dependent)
def seg_metrics(g):
    if g["y"].nunique() < 2:
        return pd.Series({"n": len(g), "prevalence": g["y"].mean(), "ROC-AUC": np.nan, "AP": np.nan})
    return pd.Series({"n": len(g), "prevalence": g["y"].mean(), "ROC-AUC": roc_auc_score(g["y"], g["p"]),
                      "AP": average_precision_score(g["y"], g["p"])})
seg = df.assign(y=y, p=oof["logistic"])
for col in ["machine_type", "site", "firmware"]:
    print(seg.groupby(col).apply(seg_metrics, include_groups=False).round(3), "\n")
```

```python
# Uncertainty: group bootstrap (resample machines, not rows) for the AP difference between models
def group_bootstrap_diff(y, p1, p2, groups, metric, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    g = pd.Series(groups).astype("category")
    codes, uniq = g.cat.codes.to_numpy(), len(g.cat.categories)
    rows_by_group = pd.Series(np.arange(len(y))).groupby(codes).apply(np.array).to_list()
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([rows_by_group[i] for i in rng.integers(0, uniq, uniq)])
        if y[idx].sum() == 0:
            continue
        diffs.append(metric(y[idx], p1[idx]) - metric(y[idx], p2[idx]))
    return np.mean(diffs), np.percentile(diffs, [2.5, 97.5])

d, ci = group_bootstrap_diff(y, oof["logistic"], oof["hgb"], groups, average_precision_score)
print(f"AP(logistic) - AP(hgb) = {d:.3f}, 95% CI {ci.round(3)}")
```

```python
# Multiclass evaluation: per-class report, normalised confusion matrix, macro vs weighted
dfm = make_equipment_failure(include_leakage=True)
ym = dfm["failure_mode"]
Xm = dfm[num + cat].astype({c: "category" for c in cat})
pm = cross_val_predict(HistGradientBoostingClassifier(categorical_features="from_dtype", max_depth=3,
                                                      random_state=0), Xm, ym, cv=cv, groups=dfm["machine_id"],
                       method="predict_proba")
classes = np.sort(ym.unique())
print("macro OvR ROC-AUC:", round(roc_auc_score(ym, pm, multi_class="ovr", average="macro"), 3))
ConfusionMatrixDisplay.from_predictions(ym, classes[pm.argmax(1)], normalize="true", values_format=".2f"); plt.show()
```

## 7. Interpretation
- **ROC-AUC ≈ 0.8 with AP ≈ 0.25–0.3 at 5% prevalence** is a typical pair. The model ranks well overall, but among the top alarms a substantial share are false. Communicate this with the operating-point numbers ("of 20 inspections, about 5 find a developing failure"), not with the AUC.
- **The Brier score must beat the prior baseline.** If it barely does, the probabilities carry little information beyond the prevalence.
- **Capacity operating point:** precision@20 per month is the KPI the planner experiences. Compare it with the "oldest first" baseline ([baseline_models.md](baseline_models.md)).
- **Segments:** a segment with a higher AP but also a higher prevalence may not be "better" for the model. Compare AP/prevalence ratios, or ROC-AUC (prevalence-insensitive).
- **Group bootstrap CI including 0** means the models are not distinguishable. Choose the simpler or better-calibrated one.
- **Multiclass confusion matrix:** off-diagonal mass between related modes (bearing vs overheating) suggests the features do not separate the mechanisms. Consider the two-stage design ([model_selection.md](model_selection.md) §7).

## 8. Decision framework
| Decision context | Primary metric |
|---|---|
| Fixed capacity (inspect k per period) | Precision@k / recall@k per period |
| Known FP/FN costs | Expected cost at the cost-optimal threshold ([threshold_optimization.md](threshold_optimization.md)) |
| Minimum recall required (safety) | Precision (or FPR) at recall ≥ target |
| Probabilities consumed as risks | Log-loss / Brier + calibration |
| General model ranking, rare positives | PR-AUC (AP) with prevalence reported |
| General model ranking, balanced classes | ROC-AUC |
| Multiclass, every class matters | Macro-F1 / macro recall + confusion matrix |
| Multiclass, frequent classes matter more | Weighted-F1 / accuracy |

- **IF** ROC-AUC improves but AP does not → the improvement is in a region of the ranking that does not matter for rare positives. Check the operating point.
- **IF** fold-to-fold variation is large (few positives per fold) → use repeated CV, or report bootstrap CIs before claiming differences.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Accuracy with imbalance | Meaningless numbers | PR metrics, operating-point metrics |
| Only ROC-AUC | Over-optimistic view under rare positives | Add AP and precision@k |
| F1 as a default metric | Implicit, unjustified cost assumption; threshold-dependent | Derive the metric from costs |
| Comparing precision across segments with different prevalence | Wrong conclusions | Report prevalence; use prevalence-insensitive metrics for comparison |
| Row-level bootstrap with grouped data | CIs too narrow | Group bootstrap |
| Macro vs weighted confusion | Misleading multiclass summaries | State which average and why |

## 10. Checklist
- [ ] Out-of-fold probabilities on identical folds.
- [ ] ROC-AUC and AP (with prevalence) ± std.
- [ ] Operating-point confusion matrix and metrics.
- [ ] Brier/log-loss vs the prior baseline.
- [ ] Segment metrics with prevalence.
- [ ] Bootstrap CIs (grouped where needed) for key comparisons.
- [ ] Multiclass: per-class report, normalised confusion matrix, choice of average justified.

## 11. References
- scikit-learn User Guide, *Classification metrics*. https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
- Saito, T. & Rehmsmeier, M. (2015). "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets." *PLoS ONE* 10(3). https://doi.org/10.1371/journal.pone.0118432
- Fawcett, T. (2006). "An introduction to ROC analysis." *Pattern Recognition Letters* 27(8), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010
- Davis, J. & Goadrich, M. (2006). "The Relationship Between Precision-Recall and ROC Curves." *ICML*. https://doi.org/10.1145/1143844.1143874
- Chicco, D. & Jurman, G. (2020). "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation." *BMC Genomics* 21, 6. https://doi.org/10.1186/s12864-019-6413-7
