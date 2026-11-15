# Train/Test Distribution Differences: Adversarial Validation and Importance Weighting

> **Folder:** 19_advanced_tabular_ml · **Prev:** [high_cardinality_and_wide_data.md](high_cardinality_and_wide_data.md) · **Next:** [grouped_hierarchical_data.md](grouped_hierarchical_data.md) · **Related:** [drift_and_monitoring.md](../20_advanced_methodologies/drift_and_monitoring.md) (drift *over time in production*), [project_reconnaissance.md](../21_practical_projects/project_reconnaissance.md), [transfer_learning_and_domain_adaptation.md](../20_advanced_methodologies/transfer_learning_and_domain_adaptation.md)

---

## 1. Purpose
Detect whether the data you train on differ from the data you will predict on (a new site, a new period, a competition's test set), find out *which* features differ, and adapt: drop or transform the shifting features, re-weight the training data, or build a validation set that resembles the target.

## 2. When to use it
- At the reconnaissance of any project where training and deployment data come from different periods, sites, instruments or populations.
- When CV is good but the test or production performance is poor.

## 3. Conceptual foundation
| Shift type | What changes | Example | Effect |
|---|---|---|---|
| **Covariate shift** | P(x) changes, P(y\|x) stays the same | A new site operates at higher loads | Models may extrapolate; re-weighting can help |
| **Prior (label) shift** | P(y) changes | Failure rate doubles | Calibration and thresholds move ([calibration.md](../04_classification/calibration.md)) |
| **Concept shift** | P(y\|x) changes | A new firmware changes what the error codes mean | The model is wrong; it needs new labels/retraining |

- **Adversarial validation:** train a classifier to distinguish training rows (label 0) from target rows (label 1). An AUC ≈ 0.5 means no detectable covariate shift. A high AUC means a shift, and the classifier's feature importances show *which* features shift.
- **Importance weighting:** for covariate shift, weight each training row by w(x) = P_target(x) / P_train(x) ≈ p/(1 − p) × (n_train/n_target), where p is the adversarial classifier's probability of "target". Clip the extreme weights. The weighted training emphasises the target-like regions.
- **Target-like validation:** choose the validation rows with the highest p (most similar to the target) to estimate the target performance.

## 4. Assumptions and limitations
- Importance weighting fixes only covariate shift, and only where the supports overlap. It cannot extrapolate into regions with no training data.
- Concept shift cannot be detected from x alone. It needs labelled target data or monitoring of outcomes.
- A high adversarial AUC can come from harmless features (IDs, timestamps). Exclude them before interpreting.

## 5. Step-by-step procedure
1. **Assemble the training and target X** (same feature pipeline; no target column).
2. **Adversarial classifier** (GBM) with CV → AUC; feature importance (permutation).
3. **For each strongly shifting feature:** is it an ID/time artefact (drop it), a real covariate shift (keep it, consider weighting or transforming it into relative terms), or a sign of a different population (separate model)?
4. **Re-weight the training data** (clipped weights) and/or build a target-like validation set.
5. **Compare** the unweighted vs weighted models on target-like validation (or on labelled target data, if any).
6. **Prior shift:** if the target prevalence is known or estimated, adjust the calibration/threshold.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from cookbook_data import load_clean_process_yield

df = load_clean_process_yield()
feats = ["temperature_c", "pressure_bar", "feed_rate_kg_h", "residence_time_min", "catalyst_age_days", "impurity_ppm"]
# Scenario: the model is trained on reactors R1–R3 and deployed on a NEW reactor R4 operated at higher feed rates
src = df[df["reactor"] != "R4"]
tgt = df[df["reactor"] == "R4"].copy()
tgt["feed_rate_kg_h"] *= 1.35                                    # covariate shift: R4 runs harder ...
tgt["yield_kg"] *= 1.35                                          # ... and yield scales accordingly (same P(y|x) mechanism)

adv = pd.concat([src[feats].assign(is_target=0), tgt[feats].assign(is_target=1)], ignore_index=True)
clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
p = cross_val_predict(clf, adv[feats], adv["is_target"], cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
print(f"adversarial AUC = {roc_auc_score(adv['is_target'], p):.3f}")
clf.fit(adv[feats], adv["is_target"])
pi = permutation_importance(clf, adv[feats], adv["is_target"], scoring="roc_auc", n_repeats=5, random_state=0)
print(pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False).round(3))
```

```python
# Importance weights for the training rows: w = p/(1-p) · n_src/n_tgt, clipped
p_src = p[: len(src)]
w = np.clip(p_src / (1 - p_src) * len(src) / len(tgt), 0, 20)
print(f"effective sample size of the weighted training set: {w.sum() ** 2 / (w ** 2).sum():.0f} of {len(src)}")

def fit_predict(model, weights=None):
    kw = {} if weights is None else ({"sample_weight": weights} if not hasattr(model, "steps") else {"ridgecv__sample_weight": weights})
    return model.fit(src[feats], np.log(src["yield_kg"]), **kw).predict(tgt[feats])

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
def log_features(X):                                                # log of the multiplicative driver (sensible extrapolation)
    X = X.copy(); X["feed_rate_kg_h"] = np.log(X["feed_rate_kg_h"]); return X
for name, make in [("HGB", lambda: HistGradientBoostingRegressor(max_iter=300, random_state=0)),
                   ("ridge", lambda: make_pipeline(FunctionTransformer(log_features), SimpleImputer(strategy="median"), RidgeCV()))]:
    for wname, ww in [("unweighted", None), ("importance-weighted", w)]:
        pred = np.exp(fit_predict(make(), ww))
        print(f"{name:5s} {wname:20s} MAE on the new reactor: {mean_absolute_error(tgt['yield_kg'], pred):.1f}")
# Reference: in-distribution error (a random 20% of the source reactors held out), to separate "hard problem" from "shift"
from sklearn.model_selection import train_test_split
s_tr, s_te = train_test_split(src, test_size=0.2, random_state=0)
ref = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(s_tr[feats], np.log(s_tr["yield_kg"]))
print(f"HGB in-distribution MAE (held-out R1–R3 batches): {mean_absolute_error(s_te['yield_kg'], np.exp(ref.predict(s_te[feats]))):.1f}"
      f"  | mean target yield on R4: {tgt['yield_kg'].mean():.0f}")
# Extra check: what fraction of target rows lie outside the source feed-rate range? (no weighting can fix extrapolation)
print("target rows with feed rate above the source maximum:", f"{np.mean(tgt['feed_rate_kg_h'] > src['feed_rate_kg_h'].max()):.1%}")
```

## 7. Interpretation
- **Adversarial AUC ≈ 0.59, with `feed_rate_kg_h` clearly at the top of the importances:** the shift is detected *and* localised to the feature we changed. The AUC is modest because the feed-rate distribution is wide and still overlaps heavily after a +35% shift. In real data, check whether the top features are physically plausible shifts or artefacts (IDs, timestamps).
- **Effective sample size 144 of 1692:** even this mild shift concentrates the weights on a small subset of the training rows, which strongly increases the variance.
- **Importance weighting did not help here; it slightly hurt both models.** That is a common, honest outcome. With heavily overlapping supports, the unweighted model already covers the target region, and weighting mostly adds variance. Weighting pays off when the target region is under-represented but still present in training, and when the model is misspecified there.
- **Most of the error is not caused by the shift.** The in-distribution reference MAE is ≈ 58 kg (on source yields averaging ≈ 210 kg). On R4, whose yields are 1.35× larger, the MAE is ≈ 81 ≈ 58 × 1.35. The **relative** error is essentially unchanged. These features omit the catalyst/supplier effects, so the problem is hard everywhere. Diagnose what the shift costs *relative to* the in-distribution error before investing in shift corrections.
- **Support:** only ~2% of the target rows exceed the source feed-rate range. There, no re-weighting can help. Only models with sensible extrapolation (log-linear structure, physics) or target-domain data can.

## 8. Decision framework
- **IF** the adversarial AUC ≈ 0.5 → no covariate shift detected; standard validation is fine.
- **IF** the AUC is high because of IDs/time/row order → remove those features; re-check.
- **IF** a real covariate shift with overlapping support → importance weighting and/or target-like validation; compare.
- **IF** the target lies outside the training support → collect target data, use models with physically sensible extrapolation, or restrict the deployment (applicability domain).
- **IF** a prior shift (a different prevalence) → recalibrate/prior-correct; re-tune the thresholds ([class_imbalance.md](../04_classification/class_imbalance.md)).
- **IF** concept shift is suspected → get labelled target data; retrain or fine-tune; monitor ([drift_and_monitoring.md](../20_advanced_methodologies/drift_and_monitoring.md)).

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Never comparing train and target distributions | Surprise failures | Adversarial validation at recon |
| Unclipped importance weights | Huge variance | Clip; check the effective sample size |
| Expecting weighting to fix extrapolation | False confidence | Check the support overlap |
| Dropping every shifting feature | Losing real signal | Distinguish artefacts from real shifts |
| Treating concept shift as covariate shift | A persistently wrong model | Labelled target data; monitoring |

## 10. Checklist
- [ ] Adversarial validation (AUC + feature importances).
- [ ] Shifting features classified (artefact / covariate / population).
- [ ] Support overlap checked.
- [ ] Weighting and/or target-like validation compared with the unweighted model.
- [ ] Prior-shift handling (calibration/threshold) if relevant.

## 11. References
- Quiñonero-Candela, J., Sugiyama, M., Schwaighofer, A. & Lawrence, N. D. (eds.) (2009). *Dataset Shift in Machine Learning*. MIT Press.
- Shimodaira, H. (2000). "Improving predictive inference under covariate shift by weighting the log-likelihood function." *J. Statistical Planning and Inference* 90(2), 227–244.
- Sugiyama, M., Krauledat, M. & Müller, K.-R. (2007). "Covariate Shift Adaptation by Importance Weighted Cross Validation." *JMLR* 8, 985–1005.
- Moreno-Torres, J. G. et al. (2012). "A unifying view on dataset shift in classification." *Pattern Recognition* 45(1), 521–530. https://doi.org/10.1016/j.patcog.2011.06.019
- Rabanser, S., Günnemann, S. & Lipton, Z. (2019). "Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift." *NeurIPS*. https://arxiv.org/abs/1810.11953
