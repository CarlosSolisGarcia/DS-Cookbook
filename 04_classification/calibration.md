# Probability Calibration

> **Folder:** 04_classification · **Prev:** [threshold_optimization.md](threshold_optimization.md) · **Next:** [error_analysis.md](error_analysis.md) · **Related:** [class_imbalance.md](class_imbalance.md), [20_advanced_methodologies](../20_advanced_methodologies/) (calibration across families)
>
> *Added to the original architecture: calibration is used by several other recipes (thresholds, risk scores, decision costs).*

---

## 1. Purpose
Make predicted probabilities mean what they say: among cases with p̂ ≈ 0.2, about 20% should be positive. Check calibration, and fix it when probabilities are used as risks, in expected-cost calculations, or combined across models.

## 2. When to use it
- Probabilities feed decisions (cost thresholds t\* = C_FP/(C_FP+C_FN)), risk communication, or downstream models.
- After class weights or resampling (which inflate p̂).
- With models known to be miscalibrated: RF, SVM, naive Bayes, sometimes boosting.
- **Not needed** when only the ranking matters (top-k) or when the threshold is tuned empirically on the same score.

## 3. Conceptual foundation
- **Reliability diagram:** bin the predictions; plot the mean predicted probability against the observed frequency per bin. Perfect calibration lies on the diagonal.
  - Points *below* the diagonal mean the model is **over-confident** (it predicts higher than reality).
  - An S-shaped curve means probabilities are pushed towards 0 and 1 (typical of naive Bayes and boosted trees) or towards 0.5 (typical of RF and SVM).
- **Calibration methods:**
  - **Platt / sigmoid:** fits a logistic function to the scores. Two parameters, so it works with little data. It assumes the distortion is sigmoid-shaped.
  - **Isotonic regression:** a monotone, non-parametric step function. More flexible, but it needs more data (roughly ≥ 1000 calibration points; fewer positives make it noisy) and it can overfit.
- **Brier score decomposition:** Brier = reliability − resolution + uncertainty. Calibration improves the reliability term. It cannot create resolution (discrimination).
- **Calibration does not change the ranking** when the map is strictly monotone, so ROC-AUC is essentially unchanged. Isotonic maps have flat steps that create ties, so ROC-AUC can shift slightly.

## 4. Assumptions and limitations
- The calibration data must be **separate** from the data the model was trained on (training-set probabilities are over-confident). `CalibratedClassifierCV` does this with internal CV, or you can use a held-out set with `FrozenEstimator` (scikit-learn ≥ 1.6; it replaces the deprecated `cv="prefit"`).
- Calibration is relative to a population. Under prevalence shift or drift it degrades, so monitor the observed frequency vs the mean prediction over time.
- With few positives the reliability diagram itself is noisy. Use few bins or quantile bins, and show the counts.

## 5. Step-by-step procedure
1. **Get out-of-fold probabilities** (never in-sample ones).
2. **Reliability diagram** with quantile bins + Brier/log-loss, compared with the prior baseline.
3. **Decide whether to calibrate:** is the miscalibration material for the decision (e.g. shifting t\*)?
4. **Calibrate** with sigmoid (small data) or isotonic (large data) using `CalibratedClassifierCV` inside CV, or on a held-out calibration split.
5. **Re-evaluate** on held-out data: reliability diagram, Brier, log-loss. Check that the ranking metrics are unchanged.
6. **Check calibration per segment** (site, type) and over time.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay, calibration_curve
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cookbook_data import make_equipment_failure

df = make_equipment_failure(include_leakage=False)
num = ["age_years", "load_pct", "operating_hours", "days_since_maintenance",
       "vibration_rms", "temperature_c", "error_count_7d"]
X, y, groups = df[num], df["failure_30d"].to_numpy(), df["machine_id"]
cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)

models = {
    "logistic": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000)),
    "logistic balanced": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                       LogisticRegression(max_iter=2000, class_weight="balanced")),
    "random forest": make_pipeline(SimpleImputer(strategy="median"),
                                   RandomForestClassifier(n_estimators=300, min_samples_leaf=10, n_jobs=-1, random_state=0)),
    "hgb": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, random_state=0),
}
oof = {k: cross_val_predict(m, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1] for k, m in models.items()}

fig, ax = plt.subplots(figsize=(6, 6))
for name, p in oof.items():
    CalibrationDisplay.from_predictions(y, p, n_bins=10, strategy="quantile", name=name, ax=ax)
ax.set_xlim(0, 0.6); ax.set_ylim(0, 0.6); plt.show()

print(pd.DataFrame({k: {"Brier": brier_score_loss(y, p), "log-loss": log_loss(y, p),
                        "mean p̂": p.mean(), "ROC-AUC": roc_auc_score(y, p)} for k, p in oof.items()})
      .T.assign(prevalence=y.mean()).round(4))
```

```python
# Calibrating inside CV: CalibratedClassifierCV fits the model on k-1 parts and the calibrator on the held-out part
def evaluate_calibrated(base, method, n_inner=3):
    cal = CalibratedClassifierCV(base, method=method, cv=n_inner)
    p = cross_val_predict(cal, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    return {"Brier": brier_score_loss(y, p), "log-loss": log_loss(y, p), "mean p̂": p.mean(),
            "ROC-AUC": roc_auc_score(y, p)}

rows = {}
for base_name in ["logistic balanced", "random forest"]:
    for method in ["sigmoid", "isotonic"]:
        rows[f"{base_name} + {method}"] = evaluate_calibrated(models[base_name], method)
print(pd.DataFrame(rows).T.round(4))
# NOTE: the inner cv of CalibratedClassifierCV is not grouped here (cv=3 → StratifiedKFold);
# for strict group separation, pass an explicit list of (train, test) index splits.
```

```python
# Held-out calibration with a frozen, already-fitted model (scikit-learn >= 1.6)
from sklearn.model_selection import GroupShuffleSplit
tr_idx, cal_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0).split(X, y, groups))
rf = models["random forest"].fit(X.iloc[tr_idx], y[tr_idx])
rf_cal = CalibratedClassifierCV(FrozenEstimator(rf), method="sigmoid").fit(X.iloc[cal_idx], y[cal_idx])
# rf_cal.predict_proba(new_X) now returns calibrated probabilities (evaluate them on a THIRD, untouched set)
```

```python
# Calibration per segment and over time ("calibration-in-the-large": observed vs mean predicted)
seg = df.assign(p=oof["logistic"], y=y)
print(seg.groupby("machine_type")[["y", "p"]].mean().round(4))
print(seg.groupby("snapshot_date")[["y", "p"]].mean().round(4).head(6))
```

## 7. Interpretation
- **Logistic (unweighted):** close to the diagonal, with mean p̂ ≈ prevalence. Usually no calibration is needed.
- **Logistic with `class_weight="balanced"`:** far above the diagonal (mean p̂ ≫ prevalence), with a worse Brier score and the same ROC-AUC. Its probabilities are badly inflated. Calibration fixes this almost completely.
- **Random forest:** RFs are often under-confident. Here, with `min_samples_leaf=10`, the RF is already close to calibrated, and calibration does *not* improve its Brier score (0.0424 → 0.0426). Calibration is a fix for a diagnosed problem, not an automatic improvement.
- **Isotonic vs sigmoid:** with ~300 positives, isotonic may be noisy (step artefacts). Prefer sigmoid unless isotonic clearly wins on held-out log-loss.
- **ROC-AUC unchanged after calibration** confirms that calibration only re-maps the scores.
- **Segment table:** a segment where the mean p̂ is systematically above or below the observed rate is miscalibrated locally. This hints at a missing interaction, or at prevalence differences the model does not capture.

## 8. Decision framework
- **IF** only ranking/top-k matters → skip calibration.
- **IF** probabilities drive cost thresholds or risk communication → check calibration. Calibrate if the mean p̂ deviates notably from the prevalence, or if the reliability curve departs from the diagonal in the region of the decision threshold.
- **IF** the calibration set is small (< ~1000 rows or < ~100 positives) → sigmoid. **ELSE** compare sigmoid and isotonic on held-out log-loss.
- **IF** class weights or resampling were used → recalibrate (or prior-correct).
- **IF** the prevalence in production differs → recalibrate on recent data; monitor calibration-in-the-large.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Calibrating on training data | No real correction; over-confidence remains | CV or a held-out calibration set |
| Isotonic with tiny data | Over-fitted step function | Sigmoid for small data |
| Evaluating calibration on the calibration set | Optimistic | A third, held-out set or nested CV |
| Uniform bins with rare positives | Empty or noisy bins | `strategy="quantile"`, show counts |
| Using `cv="prefit"` in new code | Deprecated in sklearn ≥ 1.6 | `FrozenEstimator` |

## 10. Checklist
- [ ] Out-of-fold reliability diagram (quantile bins) + Brier/log-loss vs the prior baseline.
- [ ] Calibration need judged against how the probabilities are used.
- [ ] Method chosen by calibration-set size; fitted on separate data.
- [ ] Re-evaluated on held-out data; ranking unchanged.
- [ ] Calibration checked per segment and over time.

## 11. References
- scikit-learn User Guide, *Probability calibration*. https://scikit-learn.org/stable/modules/calibration.html
- Niculescu-Mizil, A. & Caruana, R. (2005). "Predicting Good Probabilities with Supervised Learning." *ICML*. https://doi.org/10.1145/1102351.1102430
- Platt, J. (1999). "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods." In *Advances in Large Margin Classifiers*. MIT Press.
- Zadrozny, B. & Elkan, C. (2002). "Transforming classifier scores into accurate multiclass probability estimates." *KDD*. https://doi.org/10.1145/775047.775151
- Van Calster, B. et al. (2019). "Calibration: the Achilles heel of predictive analytics." *BMC Medicine* 17, 230. https://doi.org/10.1186/s12916-019-1466-7
