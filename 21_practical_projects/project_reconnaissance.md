# Competition / Project Reconnaissance

> **Folder:** 21_practical_projects · **Prev:** [kaggle_workflow.md](kaggle_workflow.md) · **Next:** [baseline_development.md](baseline_development.md) · **Related:** [initial_inspection.md](../01_data_understanding/initial_inspection.md), [data_leakage.md](../00_methodology/data_leakage.md), [distribution_shift.md](../19_advanced_tabular_ml/distribution_shift.md) (adversarial validation)

---

## 1. Purpose
Understand the problem, the metric, the data and above all **how the test set was constructed**, before writing any model code. Most competition-winning insights and most fatal mistakes come from this stage.

## 2. When to use it
Day 1 of every competition or new project. Revisit it whenever the CV and the leaderboard disagree.

## 3. Conceptual foundation
Reconnaissance answers six questions:
1. **What is predicted, for which unit, and when?** The target definition and the row granularity.
2. **How is it scored?** The exact metric, its implementation details (averaging, clipping, probability vs label) and what it rewards.
3. **What is the train/test structure?** Sizes, columns present in only one of them, ID overlaps, time ranges, group overlaps.
4. **How was the test set built?** Random, by time, by group (users, patients, sites), by geography? This dictates the validation.
5. **What could leak?** IDs correlated with the target, row order, file metadata, duplicates between train and test, features computed with future information.
6. **What is the data's origin?** The collection process, known biases, domain knowledge, external data allowed.

**Metric literacy.** Know the optimal prediction for the metric:
| Metric | Optimal prediction | Implication |
|---|---|---|
| RMSE | Conditional mean | Squared-loss models; avoid median-type predictions |
| MAE | Conditional median | `absolute_error` loss |
| RMSLE | Mean on the log1p scale | Train on log1p(y) with RMSE, then invert |
| Log-loss | Calibrated probabilities | Calibration matters; clip extremes |
| ROC-AUC | Any monotone score | Only the ranking matters; calibration is irrelevant |
| F1 / accuracy on labels | Thresholded probabilities | Threshold tuning on OOF predictions |
| MAP@k / nDCG | A good ranking at the top | Ranking losses; re-ranking |
| Quadratic weighted kappa | Ordinal agreement | Regression + optimised cut-points often beats classification |

## 4. Assumptions and limitations
- Competition descriptions can be incomplete. The forum and the data themselves are part of the documentation.
- Exploiting leaks may be against the rules or useless in real life. Identify them anyway, so they do not corrupt your validation.

## 5. Step-by-step procedure
1. **Read everything:** overview, evaluation, data description, rules, timeline; forum threads pinned by the hosts.
2. **Write the metric down** and implement it locally (verify it against a few hand calculations).
3. **Load train/test/sample submission;** compare shapes, columns, dtypes, ID formats.
4. **Compare the train and test distributions:** per-feature statistics; time ranges; group overlaps; **adversarial validation** (can a classifier tell train from test?).
5. **Target analysis:** distribution, imbalance, its relation to time/IDs.
6. **Leakage hunt:** IDs vs target, row order vs target, duplicates between train and test, suspicious features.
7. **Write the reconnaissance notes** (template in §6).

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

from cookbook_data import make_equipment_failure

# Build a toy "competition" from the cookbook data: train = first 9 months, test = last 3 months (no labels)
full = make_equipment_failure(include_leakage=True)
full.insert(0, "row_id", np.arange(len(full)))
cut = full["snapshot_date"].sort_values().unique()[-3]
train = full[full["snapshot_date"] < cut].drop(columns=["failure_mode"])
test = full[full["snapshot_date"] >= cut].drop(columns=["failure_30d", "failure_mode", "downtime_hours_30d"])
sample_sub = pd.DataFrame({"row_id": test["row_id"], "failure_30d": 0.0})
TARGET = "failure_30d"

# 3. Structure comparison
print("train", train.shape, "test", test.shape)
print("columns only in train:", sorted(set(train.columns) - set(test.columns)))
print("time ranges:", train["snapshot_date"].min().date(), "→", train["snapshot_date"].max().date(), "| test",
      test["snapshot_date"].min().date(), "→", test["snapshot_date"].max().date())
print("machines in test also in train:", f"{test['machine_id'].isin(train['machine_id']).mean():.0%}")
```

```python
# 4. Adversarial validation: can a model distinguish train rows from test rows?
FEATS = ["machine_type", "site", "age_years", "load_pct", "operating_hours", "days_since_maintenance",
         "vibration_rms", "temperature_c", "error_count_7d", "firmware"]
adv = pd.concat([train[FEATS].assign(is_test=0), test[FEATS].assign(is_test=1)], ignore_index=True)
Xa = adv[FEATS].astype({c: "category" for c in ["machine_type", "site", "firmware"]})
p = cross_val_predict(HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0), Xa, adv["is_test"],
                      cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
print(f"adversarial AUC = {roc_auc_score(adv['is_test'], p):.3f}  (0.5 = indistinguishable)")
m = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0).fit(Xa, adv["is_test"])
from sklearn.inspection import permutation_importance
pi = permutation_importance(m, Xa, adv["is_test"], scoring="roc_auc", n_repeats=3, random_state=0)
print(pd.Series(pi.importances_mean, index=FEATS).sort_values(ascending=False).head(4).round(3))
```

```python
# 6. Leakage hunt: single-feature AUC screen + ID/order checks
def single_feature_auc(df, y, cols):
    out = {}
    for c in cols:
        s = df[c]
        if s.dtype.kind in "biuf":
            a = roc_auc_score(y, s.fillna(s.median()))
            out[c] = max(a, 1 - a)
    return pd.Series(out).sort_values(ascending=False)

numeric = train.select_dtypes("number").columns.drop([TARGET])
print(single_feature_auc(train, train[TARGET], numeric).round(3).head(6))
print("target rate by row_id quintile:", train.groupby(pd.qcut(train["row_id"], 5))[TARGET].mean().round(3).tolist())
```

**Reconnaissance notes template:**
```text
RECON — <competition>
Target / unit      : failure_30d per machine-month snapshot
Metric             : ROC-AUC? log-loss? (exact implementation: ...) → optimal output: ...
Train / test       : 4500 / 1500 rows; test = last 3 months; 100% of test machines seen in train
Test construction  : TEMPORAL (dates) — same fleet, future months → validate with time-based folds (+ grouped check)
Train ≠ test       : adversarial AUC ≈ ...; driven by firmware (v3 rollout), age (time) → expect drift
Leakage            : downtime_hours_30d in train only (outcome!) → drop; row_id follows machine order → not a feature
External data      : allowed? ...
Ideas              : per-machine history features (past only), firmware effects, ...
Open questions     : ...
```

## 7. Interpretation
- **Columns only in train** (`downtime_hours_30d`): a clear outcome/leakage column. Drop it from the features. Its existence is also a hint about the label's construction.
- **Time ranges:** the test period follows the training period. The test split is **temporal**, so the validation must be time-based.
- **Machine overlap of 100%:** the test contains the same machines in later months. Per-machine history features (past-only) are legitimate and potentially strong.
- **Adversarial AUC well above 0.5:** train and test differ. The top features (`firmware`, `age_years`, which grow with time) show *why*: the v3 firmware rollout and the ageing of the fleet. Expect the models to face drift. Avoid features that only identify the period.
- **The single-feature AUC screen:** `downtime_hours_30d` reaches AUC = 1.0. It is the outcome itself (and it is absent from test anyway). Any other near-perfect feature would be a red flag. The real signals (vibration, temperature, error counts) sit at 0.7–0.74, which is typical.
- **Row order:** `row_id` follows the file order, which here is *machine by machine*. The target rate varies across row_id quintiles (≈ 0.04–0.07) because machines differ in frailty, not because of time. A model using `row_id` would learn machine identity through the back door. IDs and row order are never features. They are, however, worth a look as *clues* about how the files were assembled.

## 8. Decision framework
- **IF** the test is later in time → time-based validation; features restricted to the past; drift-aware choices.
- **IF** the test contains new groups (users, patients, sites) → GroupKFold; no group-identity features.
- **IF** the adversarial AUC ≫ 0.5 → find the drifting features. Consider dropping them, transforming them (ratios, ranks within the period), or weighting the training rows by test-likeness ([distribution_shift.md](../19_advanced_tabular_ml/distribution_shift.md)).
- **IF** a feature/ID/order predicts the target suspiciously well → treat it as leakage for validation purposes. Check the rules before any exploitation.
- **IF** the metric is ranking-based (AUC) → don't spend time on calibration. **IF** it is log-loss → calibration and clipping matter.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Skipping the evaluation page details | Optimising the wrong quantity | Implement the metric locally |
| Not checking train/test overlap and time ranges | Wrong validation scheme | Structure comparison on day 1 |
| Ignoring adversarial validation | Surprise drift | Run it early |
| Using IDs/row order as features | Leaderboard luck or failure | Leakage hunt |
| Reading only public notebooks | Inherited mistakes | Read the data + rules yourself |

## 10. Checklist
- [ ] Rules, evaluation and data pages read; forum pinned threads read.
- [ ] Metric implemented locally; optimal prediction type known.
- [ ] Train/test structure, time ranges and group overlaps compared.
- [ ] Adversarial validation done; drifting features listed.
- [ ] Leakage hunt done.
- [ ] Recon notes written.

## 11. References
- Kaufman, S. et al. (2012). "Leakage in data mining." *ACM TKDD* 6(4). https://doi.org/10.1145/2382577.2382579
- Pan, J., Pham, V., Dorairaj, M., Chen, H. & Lee, J.-Y. (2020). "Adversarial Validation Approach to Concept Drift Problem in User Targeting Automation Systems at Uber." https://arxiv.org/abs/2004.03045
- Kaggle Learn, *Data Leakage* (Intermediate ML course). https://www.kaggle.com/learn/intermediate-machine-learning
- scikit-learn, *Metrics and scoring*. https://scikit-learn.org/stable/modules/model_evaluation.html
