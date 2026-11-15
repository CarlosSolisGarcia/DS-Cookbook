# Systematic Experimentation in Competitions and Projects

> **Folder:** 21_practical_projects · **Prev:** [competition_validation.md](competition_validation.md) · **Next:** [advanced_improvement.md](advanced_improvement.md) · **Related:** [experiment_tracking.md](../07_experimentation/experiment_tracking.md) (log format and tooling), [iterative_improvement.md](../06_model_improvement/iterative_improvement.md) (choosing the next action)

---

## 1. Purpose
Turn the competition phase into a controlled sequence of experiments: one hypothesis at a time, compared against the current best under the frozen validation, logged with the CV, the LB and a conclusion. Generalisation is protected by not overfitting the public leaderboard.

## 2. When to use it
From the first baseline until the final week. The same discipline applies to portfolio projects (replace "LB" with "held-out test").

## 3. Conceptual foundation
- **One change per experiment** (or one coherent group of changes). Otherwise you cannot attribute the gain.
- **The comparison unit is the frozen folds.** Use paired, fold-wise comparison with the current best ([cross_validation.md](../06_model_improvement/cross_validation.md)).
- **The noise floor:** the fold std of the score, and the public-LB SE ([competition_validation.md](competition_validation.md)). Gains below them are not evidence.
- **Two scores per submission:** CV (the decision score) and public LB (a sanity check). Plot LB vs CV over time. A drift away from the diagonal signals overfitting to the LB, or a validation problem.
- **Save artefacts:** OOF predictions, test predictions, configuration, git hash. Saved OOFs make ensembling cheap later.
- **Budget the submissions:** each one should test something informative (diverse models for the CV–LB check, or the current best). Avoid submitting tiny variations to "read" the public LB.

## 4. Assumptions and limitations
- Strict one-at-a-time changes can be slow when the changes interact (features × hyperparameters). Re-tune lightly after large feature changes.
- Repeated decisions on the same folds slowly overfit them. Keep a sanity hold-out (e.g. the last time block) untouched for big decisions, if the data allow it.

## 5. Step-by-step procedure
1. **Maintain `experiments.csv`:** id, parent, hypothesis, change, CV (mean ± std, folds), LB (if submitted), runtime, conclusion.
2. **Before running:** write the hypothesis and the expected effect.
3. **Run** with the frozen folds; save the OOF/test predictions under the experiment id.
4. **Compare paired with the current best;** accept if the gain beats the noise (pre-defined rule).
5. **Submit selectively** (current best, or a diverse model); record the LB.
6. **Weekly review:** the leaderboard of experiments, the CV–LB plot, rejected ideas, the next hypotheses (from error analysis).

## 6. Python implementation

```python
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from cookbook_data import make_equipment_failure

full = make_equipment_failure(include_leakage=False)
cut = full["snapshot_date"].sort_values().unique()[-3]
train = full[full["snapshot_date"] < cut].reset_index(drop=True)
months = np.sort(train["snapshot_date"].unique())[-4:]
y = train["failure_30d"]
LOG, ARTIFACTS = [], {}

def base_features(df):
    X = df[["machine_type", "site", "firmware", "age_years", "load_pct", "operating_hours",
            "days_since_maintenance", "vibration_rms", "temperature_c", "error_count_7d"]].copy()
    return X.astype({c: "category" for c in ["machine_type", "site", "firmware"]})

def with_history(df):
    X = base_features(df)
    d = df.sort_values(["machine_id", "snapshot_date"])
    g = d.groupby("machine_id")
    X.loc[d.index, "vib_prev"] = g["vibration_rms"].shift(1)                          # past-only
    X.loc[d.index, "vib_delta"] = d["vibration_rms"] - X.loc[d.index, "vib_prev"]
    X.loc[d.index, "err_mean3"] = g["error_count_7d"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    return X

def run(exp_id, parent, hypothesis, feature_fn, params):
    t0, oof, scores = time.time(), pd.Series(np.nan, index=train.index), []
    X = feature_fn(train)
    for m in months:
        tr, va = train["snapshot_date"] < m, train["snapshot_date"] == m
        model = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0, **params).fit(X[tr], y[tr])
        oof[va] = model.predict_proba(X[va])[:, 1]; scores.append(roc_auc_score(y[va], oof[va]))
    rec = {"id": exp_id, "parent": parent, "hypothesis": hypothesis, "params": json.dumps(params),
           "cv_mean": np.mean(scores), "cv_std": np.std(scores), "folds": np.round(scores, 4).tolist(),
           "lb": None, "seconds": round(time.time() - t0, 1), "conclusion": None}
    LOG.append(rec); ARTIFACTS[exp_id] = oof                                         # save OOF for ensembling
    return rec

def decide(new, best, min_gain=None):
    d = np.array(new["folds"]) - np.array(best["folds"])
    thr = best["cv_std"] / 2 if min_gain is None else min_gain
    ok = d.mean() > thr and (d > 0).mean() >= 0.75
    new["conclusion"] = f"{'ACCEPT' if ok else 'reject'}: paired gain {d.mean():+.4f} (wins {int((d > 0).sum())}/{len(d)})"
    return new if ok else best

best = run("e001", None, "baseline defaults", base_features, {})
best = decide(run("e002", "e001", "shallower trees generalise better across months", base_features,
                  {"max_depth": 3, "learning_rate": 0.05, "max_iter": 300}), best)
best = decide(run("e003", best["id"], "machine history (past vibration, error trend) adds signal", with_history,
                  {"max_depth": 3, "learning_rate": 0.05, "max_iter": 300}), best)
print(pd.DataFrame(LOG)[["id", "parent", "hypothesis", "cv_mean", "cv_std", "conclusion"]].round(4).to_string(index=False))
print("current best:", best["id"])
```

```python
# CV vs LB tracking: once submissions are made, fill in 'lb' and watch the relationship
log = pd.DataFrame(LOG)
log.loc[log["id"] == "e001", "lb"] = 0.79      # values you would read from the leaderboard (illustrative)
log.loc[log["id"] == "e002", "lb"] = 0.80
print(log[["id", "cv_mean", "lb"]])
# Plot cv_mean vs lb over time; points drifting away from a monotone trend → validation or LB-overfitting problem.
```

## 7. Interpretation
- **The experiment table** gives the lineage of improvements and the evidence for each decision. Rejected experiments are kept on purpose. In this run, shallower trees were accepted (+0.018 AUC, better in 3 of 4 months). The machine-history features were **rejected**: an average change of −0.001, better in only 2 of 4 months, and a doubled fold std. The intuition "history must help" was not supported by this data and these features. Write that down. It saves you from re-trying the idea and makes you look for *better* history features (e.g. past failures, as in [competition_validation.md](competition_validation.md)).
- **Paired fold gains** with "wins x/4" show whether a change helps consistently across months, or only on average because of one fold.
- **CV vs LB:** once 5–10 submissions exist, a monotone relationship means the validation works. If the LB keeps improving while the CV does not, you may be fitting public-LB noise (especially when the public LB is small).

## 8. Decision framework
- **IF** a change wins on average but loses in most folds → reject it, or investigate the fold that drives it (is one month special?).
- **IF** the gain < noise but the change makes the model simpler or more robust → accept it for simplicity.
- **IF** ideas are exhausted → error analysis on the OOF predictions (which segments or months fail?) → new hypotheses ([iterative_improvement.md](../06_model_improvement/iterative_improvement.md)).
- **IF** the public LB disagrees with the CV for several diverse experiments → go back to [competition_validation.md](competition_validation.md).
- **IF** many experiments are cheap → batch them, but still log each one with its hypothesis.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Changing features and parameters together | Unattributable results | One change per experiment |
| Submitting every small tweak | Fitting public-LB noise; wasted slots | Selective submissions |
| Not saving OOF predictions | No cheap ensembling | Save per experiment |
| Moving folds or metric definitions silently | Incomparable history | Version the validation; re-run the references |
| Ignoring rejected ideas | Repeating them | Keep them in the log |

## 10. Checklist
- [ ] Log with hypothesis, change, CV ± std, folds, LB, conclusion.
- [ ] Frozen folds; paired comparisons; acceptance rule defined.
- [ ] OOF/test predictions saved per experiment.
- [ ] Selective submissions; CV–LB plot maintained.
- [ ] Weekly review of the log and next hypotheses.

## 11. References
- See [experiment_tracking.md](../07_experimentation/experiment_tracking.md) §11 (MLflow, DVC, reproducibility literature).
- Ng, A. *Machine Learning Yearning*, chapters on the development set and metrics. https://info.deeplearning.ai/machine-learning-yearning-book
- Roelofs, R. et al. (2019). "A Meta-Analysis of Overfitting in Machine Learning." *NeurIPS 32*.
