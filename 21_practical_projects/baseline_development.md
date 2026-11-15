# Establishing a Baseline: A Valid End-to-End Solution on Day One

> **Folder:** 21_practical_projects · **Prev:** [project_reconnaissance.md](project_reconnaissance.md) · **Next:** [competition_validation.md](competition_validation.md) · **Related:** [modeling_fundamentals.md](../00_methodology/modeling_fundamentals.md), family `baseline_models.md` recipes

---

## 1. Purpose
Build the simplest **complete** pipeline: load the data, compute features, validate, train, predict on test, write a submission and verify it. Record its local and leaderboard scores. Every later idea is then measured against this reference, and the plumbing (file formats, IDs, metric) is proven correct early.

## 2. When to use it
Immediately after reconnaissance, before any substantial feature work or tuning.

## 3. Conceptual foundation
- **Two baselines:** (1) a *trivial* one (constant/prior: the zero point of the metric); (2) a *simple model* (GBM with default parameters on the raw features, or TF-IDF + logistic for text, or seasonal naive for time series).
- **The fold assignment is frozen** and saved. All experiments use the same folds, so their OOF scores are comparable and the OOF predictions can be ensembled later.
- **OOF + test predictions** are saved for every experiment: OOF for evaluation and stacking, and test predictions averaged over the fold models (or from a refit).
- **First-submission sanity:** the local CV and the public LB should be in the same ballpark. A large discrepancy means a bug or a validation mismatch.

## 4. Assumptions and limitations
- At this stage the validation scheme is the reconnaissance's best guess. [competition_validation.md](competition_validation.md) checks it.
- Averaging the fold models' test predictions is standard. For time-based schemes, the fold models are trained on different history lengths, so a final refit on all data is often better ([final_solution.md](final_solution.md)).

## 5. Step-by-step procedure
1. **Frozen folds:** create the fold assignment matching the test construction; save it (`folds.csv`).
2. **The feature function** applied identically to train and test (one code path).
3. **`run_cv(model, features)`:** for each fold, fit → OOF predictions; predict test → average. Return the OOF score per fold and overall.
4. **Trivial baseline** + **simple-model baseline** through `run_cv`.
5. **Write the submission:** the exact columns and ID order of the sample submission. Verify the shape, the IDs, the NaNs and the value ranges.
6. **Submit;** record the CV score, the LB score and the date in the log.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from cookbook_data import make_equipment_failure

# Toy competition (see project_reconnaissance.md)
full = make_equipment_failure(include_leakage=False)
full.insert(0, "row_id", np.arange(len(full)))
cut = full["snapshot_date"].sort_values().unique()[-3]
train = full[full["snapshot_date"] < cut].reset_index(drop=True)
test = full[full["snapshot_date"] >= cut].drop(columns=["failure_30d"]).reset_index(drop=True)
sample_sub = pd.DataFrame({"row_id": test["row_id"], "failure_30d": 0.5})
TARGET, METRIC = "failure_30d", roc_auc_score

# 1. Frozen folds matching the test construction (temporal): each fold validates on later months.
#    Fold k: train on months < m_k, validate on month m_k (expanding window). Saved for reuse.
months = np.sort(train["snapshot_date"].unique())
val_months = months[-4:]                                   # 4 validation folds = the last 4 training months
folds = pd.Series(-1, index=train.index)                   # -1 = never in validation
for k, m in enumerate(val_months):
    folds[train["snapshot_date"] == m] = k
# folds.to_csv("folds.csv")

def features(df):
    """ONE code path for train and test."""
    X = df[["machine_type", "site", "firmware", "age_years", "load_pct", "operating_hours",
            "days_since_maintenance", "vibration_rms", "temperature_c", "error_count_7d"]].copy()
    return X.astype({c: "category" for c in ["machine_type", "site", "firmware"]})

def run_cv(make_model, train, test, folds, months_fold=val_months):
    oof = pd.Series(np.nan, index=train.index); test_pred = np.zeros(len(test)); scores = []
    X, Xt, y = features(train), features(test), train[TARGET]
    for k, m in enumerate(months_fold):
        tr = train["snapshot_date"] < m                    # expanding window: only the past
        va = folds == k
        model = make_model().fit(X[tr], y[tr])
        oof[va] = model.predict_proba(X[va])[:, 1]
        scores.append(METRIC(y[va], oof[va]))
        test_pred += model.predict_proba(Xt)[:, 1] / len(months_fold)
    valid = folds >= 0
    return {"fold_scores": np.round(scores, 4), "cv": METRIC(y[valid], oof[valid]), "oof": oof, "test": test_pred}

class Prior:                                               # trivial baseline: constant prevalence
    def fit(self, X, y): self.p = y.mean(); return self
    def predict_proba(self, X): return np.column_stack([1 - np.full(len(X), self.p), np.full(len(X), self.p)])

base0 = run_cv(Prior, train, test, folds)
base1 = run_cv(lambda: HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0), train, test, folds)
print("trivial:", base0["fold_scores"], round(base0["cv"], 4))
print("HGB    :", base1["fold_scores"], round(base1["cv"], 4))
```

```python
# 5. Write and VERIFY the submission
def make_submission(sample_sub, test, pred, id_col="row_id", target_col="failure_30d"):
    sub = sample_sub[[id_col]].merge(pd.DataFrame({id_col: test[id_col], target_col: pred}), on=id_col, how="left")
    assert len(sub) == len(sample_sub), "row count differs from the sample submission"
    assert sub[id_col].equals(sample_sub[id_col]), "ID order differs"
    assert sub[target_col].notna().all(), "NaN predictions"
    assert sub[target_col].between(0, 1).all(), "probabilities outside [0, 1]"
    return sub

sub = make_submission(sample_sub, test, base1["test"])
print(sub.head(), sub.shape)
# sub.to_csv(f"subs/exp001_hgb_default__cv{base1['cv']:.4f}.csv", index=False)
```

## 7. Interpretation
- **The trivial baseline** gives AUC = 0.5 in every fold (the zero point), which confirms that the metric and the folds work. Its *pooled* OOF AUC is 0.46, not 0.5. Each fold predicts a different constant (that fold's training prevalence), and pooling them creates a spurious ranking across months. **Pooled OOF metrics mix the per-fold calibration levels.** For ranking metrics with time-based folds, also report the mean of the per-fold scores, and be careful when you stack pooled OOF predictions.
- **The default HGB fold scores** show the variability across months. With few positives per month, AUC per fold varies a lot. The pooled OOF score is more stable. This variability is your noise floor for later comparisons.
- **Submission checks:** the assertions catch the most common, most painful errors (ID misalignment, NaNs, wrong scale) before you waste a submission.

## 8. Decision framework
- **IF** the CV of the simple model barely beats the trivial baseline → re-check the target, the features and the leakage removal. The signal may be weak (decide expectations accordingly).
- **IF** the LB differs strongly from the CV after the first submission → a bug (ID order, wrong column) or a validation mismatch. Investigate before anything else.
- **IF** the folds have very few positives → use fewer, larger folds or repeated schemes, and use the pooled OOF score for decisions.
- **IF** the test construction is uncertain → build two fold schemes (e.g. time-based and grouped) and track both until the LB evidence decides.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Separate feature code for train and test | Silent mismatches | One feature function |
| Re-creating folds per experiment | Incomparable scores; ensembling impossible | Frozen saved folds |
| Not saving OOF/test predictions | Can't ensemble or analyse later | Save them per experiment |
| Submitting without checks | Wasted submissions; wrong conclusions | Assertion-based verification |
| Starting with heavy tuning | No reference | Defaults first |

## 10. Checklist
- [ ] Frozen folds saved, matching the test construction.
- [ ] One feature function for train/test.
- [ ] `run_cv` returning fold scores, OOF and test predictions.
- [ ] Trivial and simple baselines logged.
- [ ] Verified submission; CV and LB recorded.

## 11. References
- scikit-learn, *Cross-validation* (`cross_val_predict` and custom splits). https://scikit-learn.org/stable/modules/cross_validation.html
- Kaggle docs, *Submitting to a competition*. https://www.kaggle.com/docs/competitions
