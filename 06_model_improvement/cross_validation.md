# Cross-Validation: Choosing and Using the Right Scheme

> **Folder:** 06_model_improvement · **Next:** [hyperparameter_optimization.md](hyperparameter_optimization.md) · **Related:** [validation_reference.md](../08_quick_references/validation_reference.md) (one-page summary), [ts validation](../05_time_series/validation.md), [data_leakage.md](../00_methodology/data_leakage.md)

---

## 1. Purpose
Estimate generalisation performance reliably, compare models fairly, and quantify how noisy those estimates are. To do that, the CV scheme must match the data structure and the deployment, and nested CV must be used when tuning and estimating at the same time.

## 2. When to use it
Every model comparison, every tuning run, and every claim that "A is better than B".

## 3. Conceptual foundation
- **K-fold CV:** split the data into K folds, train on K−1 and validate on 1, rotate, and average. K = 5 or 10 is typical. Larger K gives less bias (more training data per fit) but more cost, and the variance does not necessarily fall.
- **Stratification** keeps the class proportions in each fold (classification; essential for rare classes).
- **Groups:** when rows share an entity (machine, patient, batch of replicates), all of the entity's rows must be in the same fold (`GroupKFold`, `StratifiedGroupKFold`).
- **Time:** future data must not be used to predict the past (`TimeSeriesSplit`, rolling origin).
- **Repeated CV** (`RepeatedKFold`) averages over several random partitions. It reduces the variance of the estimate due to the particular split, which is useful for small n.
- **Nested CV:** an inner loop tunes the hyperparameters, an outer loop estimates the performance of the *whole procedure* (tuning included). Without nesting, the score of the best configuration is optimistically biased, because you picked the maximum of noisy estimates (Cawley & Talbot, 2010; Varma & Simon, 2006).
- **Fold-score spread:** the std across folds is a practical noise floor. Fold scores are not independent, because the training sets overlap. A simple, honest way to compare two models is to look at **paired per-fold differences** (same folds), together with domain judgement about the effect size.

## 4. Assumptions and limitations
- CV estimates the performance of the *procedure* trained on (K−1)/K of the data, under the assumption that future data come from the same distribution. It cannot detect drift unless the folds are temporal.
- Exact confidence intervals from CV are hard. Treat the fold std as an approximation; paired comparisons and repeated CV help.
- CV is expensive for big models. Then use a single, well-designed validation split (temporal/grouped), large enough to be stable.

## 5. Step-by-step procedure
1. **Identify the structure:** i.i.d.? groups? time order? class imbalance? small n? Use [validation_reference.md](../08_quick_references/validation_reference.md).
2. **Instantiate one splitter** with a fixed `random_state` and reuse it for every model.
3. **Put all preprocessing in a `Pipeline`** so it is refitted within each fold.
4. **Use `cross_validate`** with several metrics and `return_train_score=True`.
5. **Compare models on paired folds**; repeat CV if n is small.
6. **When tuning:** use nested CV for an unbiased estimate, or at least keep a final untouched test set.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, GroupKFold, KFold, RepeatedStratifiedKFold, StratifiedGroupKFold,
                                     StratifiedKFold, cross_val_score, cross_validate)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cookbook_data import make_equipment_failure

df = make_equipment_failure(include_leakage=False)
num = ["age_years", "load_pct", "operating_hours", "days_since_maintenance",
       "vibration_rms", "temperature_c", "error_count_7d"]
X, y, groups = df[num], df["failure_30d"], df["machine_id"]
logit = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
hgb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0)

# Splitters for the same data — results differ because they answer different questions
splitters = {
    "KFold (ignores classes & groups)": KFold(5, shuffle=True, random_state=0),
    "StratifiedKFold (ignores groups)": StratifiedKFold(5, shuffle=True, random_state=0),
    "StratifiedGroupKFold (correct here)": StratifiedGroupKFold(5, shuffle=True, random_state=0),
}
for name, cv in splitters.items():
    s = cross_val_score(logit, X, y, cv=cv, groups=groups, scoring="average_precision")
    pos_per_fold = [y.iloc[te].sum() for _, te in cv.split(X, y, groups)]
    print(f"{name:37s} AP={s.mean():.3f} ± {s.std():.3f}  positives/fold={pos_per_fold}")
```

```python
# Paired comparison on identical folds (repeated to reduce split noise)
cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
def repeated_scores(model, n_repeats=5):
    out = []
    for r in range(n_repeats):
        cv_r = StratifiedGroupKFold(5, shuffle=True, random_state=r)
        out.append(cross_val_score(model, X, y, cv=cv_r, groups=groups, scoring="average_precision"))
    return np.concatenate(out)

a, b = repeated_scores(logit), repeated_scores(hgb)
d = a - b
print(f"logit AP={a.mean():.3f}  hgb AP={b.mean():.3f}  paired diff={d.mean():.3f} "
      f"(std {d.std(ddof=1):.3f}; logit better in {np.mean(d > 0):.0%} of 25 folds)")
# Rule of thumb: a consistent sign across folds plus a practically relevant size → a meaningful difference.
# For formal tests, see the corrected resampled t-test (Nadeau & Bengio, 2003), which accounts for overlapping training sets.
```

```python
# Nested CV: unbiased estimate of "tune C, then fit" as a procedure
inner = StratifiedGroupKFold(3, shuffle=True, random_state=1)
outer = StratifiedGroupKFold(5, shuffle=True, random_state=2)
search = GridSearchCV(logit, {"logisticregression__C": np.logspace(-3, 2, 6)}, cv=inner, scoring="average_precision")

nested, non_nested = [], []
for tr, te in outer.split(X, y, groups):
    search.fit(X.iloc[tr], y.iloc[tr], groups=groups.iloc[tr])          # tuning sees only the outer-training fold
    non_nested.append(search.best_score_)                                # the inner score of the best config (optimistic)
    nested.append(search.score(X.iloc[te], y.iloc[te]))                  # outer score (honest)
print(f"inner best score (optimistic) = {np.mean(non_nested):.3f}   nested outer score = {np.mean(nested):.3f}")
```

## 7. Interpretation
- **Splitter comparison (this run):** all three give AP ≈ 0.26. In this dataset the features change month to month, so the model cannot "recognise" machines, and ignoring groups costs little (compare the large effect with replicates in [data_leakage.md](../00_methodology/data_leakage.md)). The visible difference is in the **positives per fold**: plain `KFold` gives 47–74, the stratified splitters ≈ 59. Uneven folds make the fold scores noisier. Use the grouped, stratified splitter anyway. It matches deployment on new machines and costs nothing. The fleet is also monitored over time, so a temporal check is needed too ([classification_workflow.md](../04_classification/classification_workflow.md)).
- **Paired differences:** logistic regression beats HGB in 80% of the 25 paired folds, with a mean AP gain of 0.025 (std 0.031). That is consistent, but the individual-fold noise is as large as the effect. The repetition is what makes the conclusion credible.
- **Nested vs inner score:** here they are essentially equal (0.262 vs 0.267). Only `C` was tuned, over 6 values, and the logistic model is insensitive to `C` in this range, so there is little selection optimism. The gap grows with the number of configurations tried and shrinks with the data size. With extensive tuning on small data it can be large. That is why nested CV (or a held-out test) is the rule, even when this particular check shows no gap.

## 8. Decision framework
| Data | Splitter |
|---|---|
| i.i.d. regression | `KFold(shuffle=True)` (5–10 folds) |
| i.i.d. classification | `StratifiedKFold(shuffle=True)` |
| Repeated entities | `GroupKFold` / `StratifiedGroupKFold` |
| Time-ordered, model used on the future | `TimeSeriesSplit` (with `gap`, `test_size`); rolling origin for forecasting |
| Small n (< ~500) | `RepeatedKFold` / `RepeatedStratifiedKFold`; consider leave-one-group-out |
| Spatial data | Spatial blocks as groups ([17_geospatial_data](../17_geospatial_data/)) |
| Tuning + honest estimate needed | Nested CV, or tune with CV and confirm on an untouched test set |
| Very large data / slow models | A single large validation split (structured as above) |

- **IF** the fold std is large relative to the differences between models → repeat the CV, increase the data per fold, or accept equivalence.
- **IF** the CV score and the test score disagree → the scheme does not mimic deployment (groups/time/drift). Fix the scheme; do not tune more.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Preprocessing outside the CV loop | Leakage | `Pipeline` |
| Different folds per model | Unfair comparisons | One splitter object, fixed seed |
| Reporting the best `best_score_` as the expected performance | Optimistic | Nested CV or a held-out test set |
| Shuffled K-fold on temporal data | Optimistic, misses drift | Temporal splitters |
| Ignoring groups | Memorisation rewarded | Group splitters |
| Treating fold scores as independent samples in a t-test | Over-confident significance | Corrected tests, or paired descriptive comparison |

## 10. Checklist
- [ ] Splitter matches the structure (strata, groups, time) and deployment.
- [ ] One seeded splitter reused across models.
- [ ] Preprocessing inside the pipeline.
- [ ] Multiple metrics, train scores recorded.
- [ ] Paired comparison (repeated if small n).
- [ ] Nested CV (or an untouched test set) for tuned procedures.

## 11. References
- scikit-learn User Guide, *Cross-validation: evaluating estimator performance* (incl. the splitter visualisation). https://scikit-learn.org/stable/modules/cross_validation.html
- scikit-learn, *Nested versus non-nested cross-validation*. https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
- Cawley, G. C. & Talbot, N. L. C. (2010). "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation." *JMLR* 11, 2079–2107. https://jmlr.org/papers/v11/cawley10a.html
- Varma, S. & Simon, R. (2006). "Bias in error estimation when using cross-validation for model selection." *BMC Bioinformatics* 7, 91. https://doi.org/10.1186/1471-2105-7-91
- Nadeau, C. & Bengio, Y. (2003). "Inference for the Generalization Error." *Machine Learning* 52, 239–281. https://doi.org/10.1023/A:1024068626366
- Bates, S., Hastie, T. & Tibshirani, R. (2024). "Cross-validation: what does it estimate and how well does it do it?" *JASA* 119(546). https://doi.org/10.1080/01621459.2023.2197686
