# Imbalance Beyond Binary Classification: Multiclass, Multilabel, Ordinal, Long-Tail and Extreme Rarity

> **Folder:** 19_advanced_tabular_ml · **Prev:** [grouped_hierarchical_data.md](grouped_hierarchical_data.md) · **Next:** [entity_resolution.md](entity_resolution.md) · **Related:** [class_imbalance.md](../04_classification/class_imbalance.md) (binary case: the principles are there), [10_anomaly_detection](../10_anomaly_detection/anomaly_detection_workflow.md)

---

## 1. Purpose
Extend the binary imbalance principles (fix the metric and the threshold first; rebalance only if it helps the decision metric) to multiclass, multilabel, ordinal and long-tailed problems, and to cases with only a handful of positives.

## 2. When to use it
Fault-mode classification with rare modes, defect taxonomies, multi-label tagging of reports, severity grades, any task where some classes have very few examples.

## 3. Conceptual foundation
| Setting | What changes | Metric | Remedies |
|---|---|---|---|
| **Multiclass imbalance** | Rare classes get low recall | Macro-F1, per-class recall, balanced accuracy | Class weights; per-class thresholds / prior adjustment on the probabilities; hierarchical (two-stage) models |
| **Multilabel** | Each label is a separate imbalanced binary problem; labels co-occur | Micro- and macro-F1, per-label AP | One-vs-rest with per-label thresholds; classifier chains for dependencies |
| **Ordinal** (grades 1–5) | Errors between distant grades are worse | MAE on grades, quadratic weighted kappa | Regression + optimised cut-points; cumulative-link models |
| **Long tail** (hundreds of classes, many with < 10 examples) | Most classes are barely learnable | Macro metrics, recall@k | Merge rare classes into "other"; hierarchies; embeddings + kNN; few-shot approaches |
| **Extreme rarity** (≤ ~30 positives) | Supervised learning is unreliable | PR-AUC with CIs, case-by-case review | Anomaly detection; longer horizons; aggregation; expert rules; collect more positives |

**Per-class decision rules:** with calibrated probabilities p_k, predicting argmax_k p_k minimises the error rate. With costs or recall targets per class, scale the probabilities (p_k · w_k) or set per-class thresholds, tuned on validation data.

**Two-stage (hierarchical) design:** stage 1 detects "abnormal vs normal" (more data per class), stage 2 classifies the type among the predicted abnormal cases. This is the remedy for the poor rare-mode recall seen in [model_selection.md (classification)](../04_classification/model_selection.md).

## 4. Assumptions and limitations
- Per-class threshold tuning with few validation examples overfits. Prefer few parameters (a single weight vector), and cross-validate.
- Ordinal approaches assume a meaningful order with roughly monotone feature effects.

## 5. Step-by-step procedure
1. **Per-class counts** (in train and validation); decide whether the rare classes are learnable at all (≥ ~20–50 examples each).
2. **Metric:** macro-F1/per-class recall (multiclass), per-label AP (multilabel), QWK/MAE (ordinal).
3. **Baseline:** a multinomial model without rebalancing; then class weights; then a probability re-weighting tuned on OOF.
4. **Structure:** two-stage models for normal-dominated problems; merge or group rare classes if justified.
5. **Ordinal:** regression on the grade + cut-points optimised on OOF; compare with plain classification.
6. **Report** per class with counts and uncertainty.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cookbook_data import make_equipment_failure

df = make_equipment_failure(include_leakage=True)
num = ["age_years", "load_pct", "operating_hours", "days_since_maintenance", "vibration_rms", "temperature_c", "error_count_7d"]
cat = ["machine_type", "site", "firmware"]
X, y, groups = df[num + cat], df["failure_mode"], df["machine_id"]
print(y.value_counts())
cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
pre = ColumnTransformer([("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                         ("c", OneHotEncoder(handle_unknown="ignore"), cat)])

def macro(p): return f1_score(y, p, average="macro")
logit = make_pipeline(pre, LogisticRegression(C=0.1, max_iter=3000))
P = cross_val_predict(logit, X, y, cv=cv, groups=groups, method="predict_proba")
classes = np.sort(y.unique())
print("multinomial logistic, argmax:     macro-F1 =", round(macro(classes[P.argmax(1)]), 3))
P_w = cross_val_predict(make_pipeline(pre, LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced")), X, y, cv=cv,
                        groups=groups, method="predict_proba")
print("class_weight='balanced', argmax:  macro-F1 =", round(macro(classes[P_w.argmax(1)]), 3))

# Probability re-weighting: one weight per class, tuned on the OOF probabilities (here: a simple grid on 'none')
best = max(((w, macro(classes[(P * np.where(classes == "none", w, 1.0)).argmax(1)])) for w in np.linspace(0.02, 1, 50)), key=lambda t: t[1])
print(f"re-weighted 'none' by {best[0]:.2f}:       macro-F1 = {best[1]:.3f}  (tune on OOF, confirm on held-out data)")
```

```python
# Two-stage design: (1) failure vs none, (2) mode among (predicted) failures
is_fail = (y != "none").astype(int)
p_fail = cross_val_predict(make_pipeline(pre, LogisticRegression(C=0.1, max_iter=3000)), X, is_fail, cv=cv, groups=groups,
                           method="predict_proba")[:, 1]
fail_idx = np.where(is_fail == 1)[0]
mode_pred = pd.Series("none", index=df.index, dtype=object)
stage2 = make_pipeline(pre, LogisticRegression(C=0.3, max_iter=3000))
p_mode = cross_val_predict(stage2, X.iloc[fail_idx], y.iloc[fail_idx], cv=StratifiedGroupKFold(5, shuffle=True, random_state=1),
                           groups=groups.iloc[fail_idx], method="predict_proba")
modes = np.sort(y.iloc[fail_idx].unique())
thr = np.quantile(p_fail, 1 - is_fail.mean() * 2)                       # flag ~2× the base rate as failures
flag = p_fail >= thr
mode_pred[flag] = "bearing"                                             # default mode for flagged rows without a stage-2 prediction
mode_pred.iloc[fail_idx[flag[fail_idx]]] = modes[p_mode[flag[fail_idx]].argmax(1)]
print("two-stage macro-F1 (approximation; stage 2 applied to the true failures flagged):", round(macro(mode_pred), 3))
print(classification_report(y, mode_pred, digits=3, zero_division=0))
```

```python
# Ordinal target: severity grade 0–3 derived from downtime; regression + optimised cut-points vs classification
grade = pd.cut(df["downtime_hours_30d"], [-1, 0, 10, 40, 1e9], labels=[0, 1, 2, 3]).astype(int)
Xo = X.astype({c: "category" for c in cat})
clf_pred = cross_val_predict(HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0), Xo, grade, cv=cv, groups=groups)
reg_raw = cross_val_predict(HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=0), Xo, grade, cv=cv, groups=groups)
def apply_cuts(z, cuts): return np.digitize(z, cuts)
def qwk(a, b): return cohen_kappa_score(a, b, weights="quadratic")
grid = [np.array(c) for c in [(0.5, 1.5, 2.5), (0.15, 0.6, 1.2), (0.1, 0.4, 0.9), (0.2, 0.5, 1.0)]]
cuts = max(grid, key=lambda c: qwk(grade, apply_cuts(reg_raw, c)))    # (use a proper optimiser on OOF in practice)
print(f"ordinal: classifier QWK={qwk(grade, clf_pred):.3f} | regression+cuts {cuts} QWK={qwk(grade, apply_cuts(reg_raw, cuts)):.3f}")
```

## 7. Interpretation
- **Class counts:** "none" 5705, bearing 165, overheating 74, electrical 56. The rare modes are at the edge of learnability.
- **Argmax of the multinomial probabilities** (macro-F1 ≈ 0.27) predicts mostly "none". **`class_weight="balanced"` does not help here** (≈ 0.26): it inflates the rare classes everywhere, including in their confusion with each other. **Re-weighting only the "none" probability** (one parameter, tuned on OOF) does better (≈ 0.31): it moves the "normal vs failure" decision boundary without distorting the comparisons between modes. Few-parameter adjustments on calibrated probabilities are the robust option.
- **The two-stage design** reaches the highest macro-F1 (≈ 0.33), and bearing recall rises to ≈ 0.43. Electrical and overheating remain poorly separated (recall < 0.1). The features simply do not distinguish them well, and no rebalancing creates that information (compare [error_analysis.md](../04_classification/error_analysis.md)). The code is a simplified sketch: flagged non-failures get a default mode. In production, stage 2 runs on *all* flagged cases.
- **Ordinal:** the classifier has QWK ≈ 0.06 (it predicts grade 0 almost always), while regression + cut-points reaches ≈ 0.22. Respecting the order and tuning the cut-points extracts much more from a rare, ordered outcome. Tune the cut-points with a proper optimiser on OOF predictions, and confirm them on held-out data.

## 8. Decision framework
- **IF** one dominant "normal" class + several rare abnormal classes → two-stage (detect, then classify), or anomaly detection for stage 1.
- **IF** costs differ per class → per-class weights/thresholds on calibrated probabilities, tuned on OOF.
- **IF** many classes with very few examples → merge them into "other"/parent classes; hierarchical labels; embeddings + kNN.
- **IF** multilabel → one-vs-rest with per-label thresholds; report micro and macro.
- **IF** ordinal → regression + cut-points or ordinal models; QWK/MAE metrics.
- **IF** ≤ ~30 positives in total → don't trust supervised metrics; review cases with experts; consider anomaly framing or collecting more data.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Accuracy / weighted-F1 only | Rare classes invisible | Macro metrics + per-class report |
| Per-class thresholds tuned on the test set | Optimistic results | Tune on OOF; confirm on held-out data |
| Treating ordinal grades as nominal | Ignoring the error severity | Ordinal methods; QWK |
| Keeping unlearnable classes separate | Noisy, unstable results | Merge or restructure |
| Resampling multilabel data naively | Distorted label co-occurrence | Per-label weighting/thresholds |

## 10. Checklist
- [ ] Per-class counts; learnability judged.
- [ ] Metrics appropriate to the setting (macro, per-label, QWK).
- [ ] Baseline → class weights → OOF-tuned re-weighting compared.
- [ ] Two-stage or hierarchical design considered.
- [ ] Ordinal treatment for ordered targets.
- [ ] Per-class reporting with counts.

## 11. References
- He, H. & Garcia, E. A. (2009). "Learning from Imbalanced Data." *IEEE TKDE* 21(9), 1263–1284. https://doi.org/10.1109/TKDE.2008.239
- Zhang, M.-L. & Zhou, Z.-H. (2014). "A Review on Multi-Label Learning Algorithms." *IEEE TKDE* 26(8), 1819–1837.
- Gutiérrez, P. A. et al. (2016). "Ordinal Regression Methods: Survey and Experimental Study." *IEEE TKDE* 28(1), 127–146.
- scikit-learn, *Multiclass and multioutput algorithms* (`OneVsRestClassifier`, `ClassifierChain`). https://scikit-learn.org/stable/modules/multiclass.html
- imbalanced-learn documentation (multiclass sampling strategies). https://imbalanced-learn.org/stable/
