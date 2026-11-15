# Final Solution: Selection, Refit, Submission and Documentation

> **Folder:** 21_practical_projects · **Prev:** [advanced_improvement.md](advanced_improvement.md) · **Next:** [project_retrospective.md](project_retrospective.md) · **Related:** [reproducibility.md](../07_experimentation/reproducibility.md), [baseline_development.md](baseline_development.md) (submission checks)

---

## 1. Purpose
Choose the final model(s) robustly (not by the public leaderboard), refit them appropriately, generate and verify the submission files, and document the approach and its limitations so the solution can be reproduced and reused.

## 2. When to use it
In the last days of a competition, or at the delivery of a project.

## 3. Conceptual foundation
- **Selection by CV, not by the public LB.** The private LB (and real future data) is a different sample. The public LB is a noisy sanity check ([competition_validation.md](competition_validation.md)).
- **Two final submissions** (when allowed): (1) the **best-CV** solution; (2) a **robust** alternative, e.g. a simpler model or a broader ensemble with a good CV and consistent fold scores. It hedges against validation mismatch.
- **Refit strategy:**
  - *Average of fold models:* uses all the data across folds, with no extra fitting; standard for K-fold schemes.
  - *Full refit on all the training data:* better for time-based schemes (most recent data) and when the fold models saw much less data. Keep the hyperparameters (e.g. the number of boosting iterations scaled to the larger data size, or fixed from early stopping).
  - *Seed averaging:* refit with several seeds and average, to reduce variance.
- **Final verification:** the format, IDs, row count, value ranges, the absence of NaNs, the correlation with earlier submissions (a sudden drop in correlation signals a bug), and reproducibility from a clean run.

## 4. Assumptions and limitations
- A full refit has no direct validation. Its benefit is assumed from the data-size argument; sanity-check it by its correlation with the fold-average predictions.
- Documentation done in a rush is often incomplete. Keep the README up to date throughout the project.

## 5. Step-by-step procedure
1. **Freeze the features and code;** tag the commit.
2. **Shortlist** 2–4 candidates from the log: the best CV, the most robust (lowest fold std / best worst-fold), the simplest within the noise.
3. **Choose two submissions** (or one, if only one is allowed: the best CV, unless its folds are erratic).
4. **Refit** (fold average vs full refit vs seeds) according to the validation scheme.
5. **Generate the submission files;** run the verification checks (§6); compare them with previous submissions.
6. **Reproduce from scratch** (fresh environment, `python train.py --config final.yaml`) and confirm identical or near-identical predictions.
7. **Write the solution summary** (template below), including the limitations.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cookbook_data import make_equipment_failure

full = make_equipment_failure(include_leakage=False)
full.insert(0, "row_id", np.arange(len(full)))
cut = full["snapshot_date"].sort_values().unique()[-3]
train = full[full["snapshot_date"] < cut].reset_index(drop=True)
test = full[full["snapshot_date"] >= cut].reset_index(drop=True)
sample_sub = test[["row_id"]].assign(failure_30d=0.5)
num = ["age_years", "load_pct", "operating_hours", "days_since_maintenance", "vibration_rms", "temperature_c", "error_count_7d"]
cat = ["machine_type", "site", "firmware"]
y = train["failure_30d"]
months = np.sort(train["snapshot_date"].unique())[-4:]
def make_model(C=0.05):
    pre = ColumnTransformer([("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                             ("c", OneHotEncoder(handle_unknown="ignore"), cat)])
    return make_pipeline(pre, LogisticRegression(C=C, max_iter=2000))

# Refit strategies for a time-based scheme
fold_models = [make_model().fit(train[train["snapshot_date"] < m], y[train["snapshot_date"] < m]) for m in months]
pred_fold_avg = np.mean([m.predict_proba(test)[:, 1] for m in fold_models], axis=0)
pred_full = make_model().fit(train, y).predict_proba(test)[:, 1]
print("corr(fold-average, full refit) =", round(np.corrcoef(pred_fold_avg, pred_full)[0, 1], 4))
# (Only possible here because the toy 'test' has labels:)
print("hidden AUC fold-average:", round(roc_auc_score(test["failure_30d"], pred_fold_avg), 4),
      "| full refit:", round(roc_auc_score(test["failure_30d"], pred_full), 4))
```

```python
# Final verification checklist in code
def verify_submission(sub, sample_sub, previous=None, id_col="row_id", target="failure_30d", bounds=(0, 1)):
    checks = {
        "same columns": list(sub.columns) == list(sample_sub.columns),
        "same row count": len(sub) == len(sample_sub),
        "same ID order": sub[id_col].equals(sample_sub[id_col]),
        "no NaN": bool(sub[target].notna().all()),
        "within bounds": bool(sub[target].between(*bounds).all()),
        "not constant": sub[target].nunique() > 1,
    }
    if previous is not None:
        checks["corr with previous > 0.9"] = np.corrcoef(sub[target], previous[target])[0, 1] > 0.9
    for k, v in checks.items():
        print(f"[{'OK' if v else 'FAIL'}] {k}")
    return all(checks.values())

final = sample_sub[["row_id"]].assign(failure_30d=pred_full)
prev = sample_sub[["row_id"]].assign(failure_30d=pred_fold_avg)
assert verify_submission(final, sample_sub, previous=prev)
# final.to_csv("subs/FINAL_A_bestcv_logistic_fullrefit.csv", index=False)
```

**Solution summary template (README of the final repository):**
```text
SOLUTION SUMMARY — <competition / project>
Problem & metric      : ...
Validation            : time-based expanding folds on the last 4 training months (mimics: same fleet, future months);
                        CV–LB rank agreement checked on 8 submissions (Spearman 0.9)
Final model(s)        : A) logistic regression (C=0.05) on 7 sensor + 3 categorical features, full refit — CV 0.83
                        B) rank average of logistic + RF — CV 0.82 (robust alternative)
Key features/insights : vibration & temperature dominate; firmware v3 reduces electrical failures; history features did not help
What did not work     : SMOTE/class weights (no AUC gain); deep RF (overfits); machine-history aggregates
Limitations           : few positives per month → noisy CV (±0.03); drift from firmware rollout; label noise unknown
Reproduce             : pip install -r requirements.txt; python src/train.py --config configs/final.yaml (≈ 2 min CPU)
```

## 7. Interpretation
- **Fold-average vs full refit:** a high correlation (> 0.95) means both are similar, so either is fine. For time-based schemes the full refit includes the most recent months and is usually the better choice. In this toy example you can see it on the hidden labels, which a real competition never gives you.
- **Verification output:** all checks must pass. A low correlation with the previous good submission is a strong bug signal (shuffled IDs, wrong feature order).
- **The summary** is for your future self and for reviewers. A clear "what did not work" section is especially valuable.

## 8. Decision framework
- **IF** the best-CV model has erratic folds and a robust alternative is within the noise → submit the robust one as the primary.
- **IF** the scheme is time-based → full refit on all the data (fixed hyperparameters).
- **IF** the scheme is K-fold and the models are expensive → the fold-model average (no extra training).
- **IF** you are allowed two submissions → best CV + robust/different. Never two near-identical variants.
- **IF** it is a real project → prefer the simplest model within the noise of the best; document the monitoring and retraining plan.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Picking the final model by public-LB rank | Private-LB shake-up | Select by CV; hedge with a robust second choice |
| Last-minute untested changes | Broken submissions | Freeze and tag the code; verification checks |
| Full refit with early-stopping parameters tied to a fold | Under/over-training | Fix the iterations (scaled), or average the fold models |
| No reproducibility test | Can't rebuild the winning file | Clean-environment rerun |
| No write-up | Lost learning | The summary template |

## 10. Checklist
- [ ] Code frozen and tagged.
- [ ] Shortlist from the log; two final choices (best CV + robust).
- [ ] Refit strategy appropriate to the validation scheme; seeds averaged if useful.
- [ ] Submission verification passed; correlation with previous submissions checked.
- [ ] Clean reproduction run.
- [ ] Solution summary written, with limitations.

## 11. References
- See [reproducibility.md](../07_experimentation/reproducibility.md) §11.
- Kaggle docs, *Competitions: submissions and leaderboards*. https://www.kaggle.com/docs/competitions
