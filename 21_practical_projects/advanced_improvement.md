# Advanced Improvement: Features, Models, Ensembling, Stacking, Tuning and Validation Refinement

> **Folder:** 21_practical_projects · **Prev:** [systematic_experimentation.md](systematic_experimentation.md) · **Next:** [final_solution.md](final_solution.md) · **Related:** [ensembles_and_stacking.md](../20_advanced_methodologies/ensembles_and_stacking.md), [hyperparameter_optimization.md](../06_model_improvement/hyperparameter_optimization.md), [feature_engineering.md](../02_preprocessing/feature_engineering.md), [efficient_development.md](../20_advanced_methodologies/efficient_development.md)

---

## 1. Purpose
Use the improvement levers in the order of their typical value in competitions (features → model diversity → ensembling → tuning), while keeping validation honest and the work efficient.

## 2. When to use it
After the validation is trusted and the baseline/first experiments are logged. This is the main phase of a competition.

## 3. Conceptual foundation
| Lever | Typical value | Notes |
|---|---|---|
| **Feature engineering** (domain features, aggregates per entity/time, interactions, text/image embeddings as features) | Highest in tabular competitions | Past-only/out-of-fold computation for target-related aggregates |
| **Model diversity** (GBM + linear + NN + kNN on embeddings) | Medium–high (enables ensembling) | Diverse errors matter more than individual scores |
| **Ensembling** (averaging, rank averaging, weighted blends) | Reliable, moderate | Usually a few % of error reduction; robust to LB noise |
| **Stacking** (meta-model on OOF predictions) | Moderate | Needs clean OOFs on identical folds; leakage-prone |
| **Hyperparameter tuning** | Low–moderate | After the features stabilise; Optuna with a fixed budget |
| **Validation refinement** | Protects all of the above | More folds/repeats; shift-aware validation |
| **Post-processing** (threshold optimisation, clipping, rounding to valid values, ranking) | Metric-dependent, sometimes large | Optimise on OOF predictions, never on the LB |
| **Pseudo-labelling / semi-supervision** | Situational | Risky; validate carefully |

**Why ensembles work:** if the models' errors are not perfectly correlated, averaging reduces the variance of the error. Gains are largest when the models are individually good *and* different (different algorithms, feature sets, seeds).

**Stacking without leakage:** the meta-model must be trained on **OOF predictions** produced with the same folds for all base models. The meta-model's own evaluation needs another level of CV (or a hold-out).

## 4. Assumptions and limitations
- Competition gains from large ensembles rarely justify their complexity in production. For portfolio and real projects, prefer the simplest model within the noise of the best.
- Post-processing tuned on OOF can still overfit. Use few parameters.

## 5. Step-by-step procedure
1. **Error analysis on the current best OOF** → feature hypotheses (segments with high error, temporal patterns).
2. **Feature experiments** (logged one by one).
3. **Train diverse model families** on the best feature set with the frozen folds; save the OOF/test predictions.
4. **Ensemble:** start with a simple average or a rank average of the top diverse models; weights only if clearly better on OOF.
5. **Stacking:** a ridge/logistic meta-model on the OOF predictions (+ optionally a few raw features), evaluated by a second-level CV.
6. **Tuning** of the strongest base models with a fixed budget.
7. **Post-processing** optimised on OOF (thresholds, clipping).
8. **Robustness check:** the ensemble's score per fold/period; seed averaging.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cookbook_data import make_equipment_failure

full = make_equipment_failure(include_leakage=False)
cut = full["snapshot_date"].sort_values().unique()[-3]
train = full[full["snapshot_date"] < cut].reset_index(drop=True)
hidden = full[full["snapshot_date"] >= cut].reset_index(drop=True)
num = ["age_years", "load_pct", "operating_hours", "days_since_maintenance", "vibration_rms", "temperature_c", "error_count_7d"]
cat = ["machine_type", "site", "firmware"]
y = train["failure_30d"]
months = np.sort(train["snapshot_date"].unique())[-4:]
valid_mask = train["snapshot_date"].isin(months)

pre = ColumnTransformer([("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                         ("c", OneHotEncoder(handle_unknown="ignore"), cat)])
base_models = {
    "logistic": lambda: make_pipeline(pre, LogisticRegression(C=0.05, max_iter=2000)),
    "hgb": lambda: make_pipeline(pre, HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, random_state=0)),
    "rf": lambda: make_pipeline(pre, RandomForestClassifier(500, min_samples_leaf=10, max_features=0.5, random_state=0, n_jobs=-1)),
}

def oof_and_test(make_model):
    oof = pd.Series(np.nan, index=train.index)
    for m in months:                                            # frozen time-based folds
        tr, va = train["snapshot_date"] < m, train["snapshot_date"] == m
        oof[va] = make_model().fit(train[tr], y[tr]).predict_proba(train[va])[:, 1]
    test = make_model().fit(train, y).predict_proba(hidden)[:, 1]   # refit on all training data
    return oof, test

preds = {name: oof_and_test(mk) for name, mk in base_models.items()}
def fold_auc(p):
    return np.mean([roc_auc_score(y[train["snapshot_date"] == m], p[train["snapshot_date"] == m]) for m in months])

rows = [{"model": n, "OOF AUC (mean of folds)": fold_auc(o), "hidden AUC": roc_auc_score(hidden["failure_30d"], t)}
        for n, (o, t) in preds.items()]
O = pd.DataFrame({n: o[valid_mask] for n, (o, _) in preds.items()})
print("OOF prediction correlations (diversity):\n", O.corr(method="spearman").round(2))
```

```python
# Ensembles: simple average of probabilities, and rank average (robust to calibration differences)
def rank_avg(cols):                                        # rank within each fold/period for time-based folds
    return np.mean([rankdata(c) / len(c) for c in cols], axis=0)

oof_avg = pd.Series(np.mean([preds[n][0] for n in preds], axis=0), index=train.index)
test_avg = np.mean([preds[n][1] for n in preds], axis=0)
rows.append({"model": "average (3)", "OOF AUC (mean of folds)": fold_auc(oof_avg),
             "hidden AUC": roc_auc_score(hidden["failure_30d"], test_avg)})
test_rank = rank_avg([preds[n][1] for n in preds])
rows.append({"model": "rank average (3)", "OOF AUC (mean of folds)": np.nan,
             "hidden AUC": roc_auc_score(hidden["failure_30d"], test_rank)})
```

```python
# Stacking: logistic meta-model on OOF predictions, evaluated with a second-level time split
def logit(p):                                               # clip: tree models can output exactly 0 or 1
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
S = pd.DataFrame({n: preds[n][0] for n in preds})[valid_mask]
yS, mS = y[valid_mask], train.loc[valid_mask, "snapshot_date"]
meta_scores = []
for m in months[1:]:                                        # train the meta-model on earlier OOF months only
    tr, va = mS < m, mS == m
    meta = LogisticRegression(C=1.0).fit(logit(S[tr]), yS[tr])
    meta_scores.append(roc_auc_score(yS[va], meta.predict_proba(logit(S[va]))[:, 1]))
meta = LogisticRegression(C=1.0).fit(logit(S), yS)
T = pd.DataFrame({n: preds[n][1] for n in preds})
rows.append({"model": "stacking (logit meta)", "OOF AUC (mean of folds)": np.mean(meta_scores),
             "hidden AUC": roc_auc_score(hidden["failure_30d"], meta.predict_proba(logit(T))[:, 1])})
print(pd.DataFrame(rows).set_index("model").round(4))
print("meta-model weights:", dict(zip(S.columns, meta.coef_[0].round(2))))
```

## 7. Interpretation
- **OOF correlations ≈ 0.80–0.84:** the three families make fairly different errors, so there is diversity to exploit in principle.
- **But here no ensemble beats the best single model.** Logistic regression dominates (OOF 0.828, hidden 0.801). The tree models are clearly weaker on this small, mostly additive, rare-event problem. Averaging with weaker models dilutes the best one (average: 0.824 OOF / 0.794 hidden). Ensembling pays off when the members are *similarly strong and different*. With one dominant model, a weighted blend or the single model wins.
- **Rank averaging** (0.796 hidden) is useful for ranking metrics (AUC, MAP) when the models are calibrated differently. Here it changes little.
- **Stacking learns the dominance:** the meta-weights are logistic 0.68, RF 0.30, HGB ≈ 0. The stacked model is close to logistic but not better (0.799 hidden). With few base models and few positives, stacking mainly adds complexity and a leakage risk.
- **Hidden vs OOF:** the OOF ranking (logistic > average > stacking > RF > HGB) matches the hidden ranking closely. The time-based validation is doing its job, which is the precondition for making ensemble decisions on the OOF at all.
- **Practical rule:** in this situation submit the single best model (or a blend weighted heavily towards it), and spend the remaining effort on features that might make *other* families competitive.

## 8. Decision framework
- **IF** improvements from features have plateaued → invest in diverse models for ensembling.
- **IF** two models' OOFs correlate > 0.98 → keep one; seek diversity (different algorithm/features).
- **IF** there are ≤ 5 base models and noisy validation → a simple (rank) average; avoid learned weights.
- **IF** there are many base models and plenty of data → stacking with a regularised linear meta-model; second-level CV.
- **IF** the metric needs thresholds/labels → optimise the threshold on the OOF of the final ensemble.
- **IF** compute is limited → ensembles of cheap models (GBMs with different seeds/feature subsets) before heavy NNs ([efficient_development.md](../20_advanced_methodologies/efficient_development.md)).

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Stacking on in-fold predictions | Severe leakage | OOF predictions on identical folds |
| Optimising blend weights on the public LB | LB overfitting | Weights on OOF only, few parameters |
| Ensembling near-identical models | Complexity without gain | Check the OOF correlations |
| Tuning before the features stabilise | Wasted effort | Features first |
| Post-processing tuned on the test | Invalid results | OOF-based tuning |

## 10. Checklist
- [ ] Error analysis → feature hypotheses tested.
- [ ] Diverse base models with saved OOF/test predictions on the frozen folds.
- [ ] OOF correlation matrix checked.
- [ ] Average/rank average before stacking; stacking with second-level CV.
- [ ] Tuning with a budget; post-processing on OOF.
- [ ] Robustness per fold/period; seeds averaged.

## 11. References
- Wolpert, D. H. (1992). "Stacked generalization." *Neural Networks* 5(2), 241–259.
- Breiman, L. (1996). "Stacked regressions." *Machine Learning* 24, 49–64.
- scikit-learn User Guide, *Stacked generalization* (`StackingClassifier/Regressor`). https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization
- Kuncheva, L. I. (2014). *Combining Pattern Classifiers*, 2nd ed. Wiley.
- Kaggle competition write-ups (e.g. the solution summaries on competition discussion forums) are a rich, if informal, source of feature and ensembling ideas.
