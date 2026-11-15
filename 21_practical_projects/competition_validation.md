# Competition-Specific Validation: Reproducing the Hidden Test Locally

> **Folder:** 21_practical_projects · **Prev:** [baseline_development.md](baseline_development.md) · **Next:** [systematic_experimentation.md](systematic_experimentation.md) · **Related:** [validation_reference.md](../08_quick_references/validation_reference.md), [cross_validation.md](../06_model_improvement/cross_validation.md), [distribution_shift.md](../19_advanced_tabular_ml/distribution_shift.md)

---

## 1. Purpose
Design a local validation whose scores **rank models the same way the hidden test does**. Verify it against the leaderboard, and detect validation leakage. Everything else in a competition depends on this.

## 2. When to use it
After the first baseline submission. Revisit it whenever the CV and LB rankings diverge.

## 3. Conceptual foundation
**Reconstruct how the test was built**, then copy it:
| Test construction (inferred from recon) | Local validation |
|---|---|
| Random rows | (Stratified) K-fold |
| Later time period | Time-based folds (expanding/sliding), validating on the period(s) just before the test; same gap/horizon |
| New groups (users, patients, sites, wells) | GroupKFold / StratifiedGroupKFold on that group |
| New geographic areas | Spatial block CV ([17_geospatial_data](../17_geospatial_data/geospatial_workflow.md)) |
| Same groups, future time | Time-based folds (group features from the past allowed) |
| A different distribution (shift) | Validation weighted/selected by test-likeness (adversarial validation) |

**Public vs private leaderboard.** The public LB is usually a small random subset of the test. Its noise can be estimated: for AUC or accuracy with n_public rows, the standard error is roughly √(p(1−p)/n). Differences smaller than ~2 SE are noise.

**CV–LB correlation check.** Submit several diverse experiments (not just tiny variations). Plot LB vs CV. A good validation gives a monotone relationship. Systematic disagreement means the validation does not mimic the test.

**Validation leakage** (the CV looks better than it should):
- duplicates/near-duplicates across folds;
- group or time leakage (random folds on grouped/temporal data);
- target-derived features computed outside the folds (target encoding, aggregates);
- hyperparameters or features selected on the same folds that are reported (mild; nested CV if critical);
- preprocessing fitted on all data.

## 4. Assumptions and limitations
- With very few submissions per day, the CV–LB check is coarse. Prioritise diverse experiments for it.
- When the public LB is small, a perfectly good validation can look uncorrelated with it. Estimate the LB noise before concluding.

## 5. Step-by-step procedure
1. **List the candidate schemes** consistent with the reconnaissance (e.g. random K-fold vs grouped vs time-based).
2. **Run 3–5 diverse models/feature sets** under each scheme. Save the scores.
3. **Submit those models;** record the public LB.
4. **Compare** each scheme's ranking with the LB ranking (Spearman correlation) and their absolute agreement.
5. **Adopt the best-matching scheme;** freeze its folds.
6. **Audit for leakage:** duplicates across folds, group overlaps, feature construction inside the folds.
7. **Track** CV vs LB for every submission thereafter.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from cookbook_data import make_equipment_failure

full = make_equipment_failure(include_leakage=False)
cut = full["snapshot_date"].sort_values().unique()[-3]
train = full[full["snapshot_date"] < cut].reset_index(drop=True)
hidden = full[full["snapshot_date"] >= cut].reset_index(drop=True)      # plays the role of the private LB
num = ["age_years", "load_pct", "operating_hours", "days_since_maintenance", "vibration_rms", "temperature_c", "error_count_7d"]
cat = ["machine_type", "site", "firmware"]
y = train["failure_30d"]

# Per-machine history feature (past-only): number of failures of this machine before this snapshot
def add_history(df_all):
    df_all = df_all.sort_values(["machine_id", "snapshot_date"]).copy()
    df_all["past_failures"] = df_all.groupby("machine_id")["failure_30d"].transform(lambda s: s.shift(1).fillna(0).cumsum())
    return df_all.sort_index()
both = add_history(pd.concat([train, hidden], keys=["tr", "te"]).reset_index(level=0).rename(columns={"level_0": "part"}))
train["past_failures"] = both.loc[both["part"] == "tr", "past_failures"].to_numpy()
hidden["past_failures"] = both.loc[both["part"] == "te", "past_failures"].to_numpy()

pre = ColumnTransformer([("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num + ["past_failures"]),
                         ("c", OneHotEncoder(handle_unknown="ignore"), cat)])
candidates = {
    "logistic": lambda: make_pipeline(pre, LogisticRegression(C=0.05, max_iter=2000)),
    "logistic no-history": lambda: make_pipeline(ColumnTransformer([("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                                                                    ("c", OneHotEncoder(handle_unknown="ignore"), cat)]),
                                                 LogisticRegression(C=0.05, max_iter=2000)),
    "random forest (deep)": lambda: make_pipeline(pre, RandomForestClassifier(300, min_samples_leaf=1, random_state=0, n_jobs=-1)),
    "HGB": lambda: make_pipeline(pre, HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, random_state=0)),
}

def cv_score(make_model, scheme):
    if scheme == "random KFold":
        splits = StratifiedKFold(5, shuffle=True, random_state=0).split(train, y)
    elif scheme == "GroupKFold(machine)":
        splits = GroupKFold(5).split(train, y, train["machine_id"])
    else:                                                    # time-based: validate on each of the last 3 months
        months = np.sort(train["snapshot_date"].unique())[-3:]
        splits = [(np.where(train["snapshot_date"] < m)[0], np.where(train["snapshot_date"] == m)[0]) for m in months]
    oof_scores = []
    for tr, va in splits:
        m = make_model().fit(train.iloc[tr], y.iloc[tr])
        oof_scores.append(roc_auc_score(y.iloc[va], m.predict_proba(train.iloc[va])[:, 1]))
    return np.mean(oof_scores)

rows = []
for name, mk in candidates.items():
    row = {"model": name}
    for scheme in ["random KFold", "GroupKFold(machine)", "time-based"]:
        row[scheme] = cv_score(mk, scheme)
    row["HIDDEN TEST"] = roc_auc_score(hidden["failure_30d"], mk().fit(train, y).predict_proba(hidden)[:, 1])
    rows.append(row)
tab = pd.DataFrame(rows).set_index("model").round(4)
print(tab.to_string())
for scheme in ["random KFold", "GroupKFold(machine)", "time-based"]:
    print(f"Spearman(rank by {scheme}, rank by hidden test) = {spearmanr(tab[scheme], tab['HIDDEN TEST'])[0]:.2f}")
```

```python
# Public-LB noise: how big a difference is meaningful? (AUC standard error, Hanley & McNeil 1982)
def auc_se(auc, n_pos, n_neg):
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    return np.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg))
n_public = int(0.3 * len(hidden)); n_pos = int(hidden["failure_30d"].mean() * n_public)
print(f"public LB with {n_public} rows ({n_pos} positives): AUC SE ≈ {auc_se(0.8, n_pos, n_public - n_pos):.3f}")
```

## 7. Interpretation
- **Rank agreement alone is not enough.** With only 4 candidates, all three schemes reach Spearman = 0.8. Every scheme puts logistic first. Look at the **levels** as well:
  - For HGB, random K-fold says 0.824, GroupKFold 0.821, **time-based 0.802**, and the hidden test gives **0.786**. Only the time-based scheme anticipates that the flexible model degrades on future months (firmware drift, ageing).
  - For the deep random forest, the time-based scheme is again the most pessimistic. All schemes are within ≈ 0.02 of the hidden score.
  - The history feature (`past_failures`) adds ≈ +0.024 AUC on the hidden test. Random/grouped K-fold credit it with only ≈ +0.005. Its value comes from *accumulating* history over time, which only a scheme that respects time order can see.
  **Conclusion:** the time-based scheme mimics "same machines, future months" best. Adopt it, and track both its rank agreement and its level agreement on every submission.
- **With 4 candidates,** Spearman correlations are coarse. In practice, track 10+ *diverse* submissions.
- **Public-LB noise:** with ~450 rows and ~19 positives, the AUC SE is ≈ 0.06. Public-LB differences of 0.01–0.03 are pure noise. This is why you select by CV, and why a public/private leaderboard shake-up is common in small-test competitions.

## 8. Decision framework
- **IF** the time-based scheme agrees with the LB and the others don't → adopt it, even though it is noisier (fewer validation rows).
- **IF** no scheme agrees with the LB → check for bugs (submission alignment), for a distribution shift (adversarial validation) and for LB noise. Do not chase the public LB.
- **IF** the CV is much higher than the LB for all models → validation leakage (groups/time/duplicates/target-derived features).
- **IF** the CV is much lower than the LB → the test may be easier (different distribution) or the folds are harder (e.g. validating on unusual periods). Look at the fold-wise scores.
- **IF** the shift between train and test is strong → validate on the training rows most similar to the test (adversarial weights), and prefer robust features.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Default random K-fold for every competition | CV–LB mismatch; private-LB shake-up | Reconstruct the test construction |
| Trusting tiny public-LB gains | Overfitting public noise | Compute the LB SE; select by CV |
| Time features computed with future rows | Inflated CV | Past-only computation per entity |
| CV–LB check with near-identical models | Uninformative | Diverse candidates |
| Changing folds mid-competition without re-scoring old runs | Incomparable log | Re-run references when the scheme changes |

## 10. Checklist
- [ ] Candidate schemes derived from the recon.
- [ ] Diverse models scored under each scheme and submitted.
- [ ] CV–LB rank agreement measured; scheme chosen and frozen.
- [ ] Public-LB noise estimated.
- [ ] Leakage audit (duplicates, groups, time, target-derived features).
- [ ] CV vs LB tracked for every submission.

## 11. References
- Roberts, D. R. et al. (2017). "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure." *Ecography* 40(8), 913–929. https://doi.org/10.1111/ecog.02881
- Hanley, J. A. & McNeil, B. J. (1982). "The meaning and use of the area under a receiver operating characteristic (ROC) curve." *Radiology* 143(1), 29–36.
- Blum, A. & Hardt, M. (2015). "The Ladder: A Reliable Leaderboard for Machine Learning Competitions." *ICML*. https://arxiv.org/abs/1502.04585
- scikit-learn, *Visualizing cross-validation behavior*. https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html
