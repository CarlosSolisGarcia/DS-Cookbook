# Modelling Fundamentals: Baselines, Model Selection, Training and Evaluation

> **Folder:** 00_methodology · **Prev:** [data_leakage.md](data_leakage.md) · **Next:** start the data work in [initial_inspection.md](../01_data_understanding/initial_inspection.md) · **Related:** [cross_validation.md](../06_model_improvement/cross_validation.md), family workflows
>
> *Added to the original architecture. It is the general recipe for lifecycle stages 8–10 (baseline and model selection, training, evaluation). The family-specific recipes apply these ideas to regression, classification and forecasting.*

---

## 1. Purpose

**What it solves.** It sets out the shared logic of modelling: why baselines come first, how to pick candidate models, what overfitting and underfitting look like and how to diagnose them, and how to evaluate a model so the result carries over to deployment.

---

## 2. When to use it

**When to use it.** Before the first model in any project, and whenever you need to decide "more complex or not?".

---

## 3. Conceptual foundation

### 3.1 Generalisation and the three datasets

| Set | Used for | Never used for |
|---|---|---|
| **Training** | Fitting parameters (and preprocessing statistics) | Reporting performance |
| **Validation** (a hold-out, or CV folds within training) | Choosing features, models, hyperparameters, thresholds | Final reporting |
| **Test** (hold-out) | One final, unbiased estimate | Any decision |

Cross-validation turns the training data into many train/validation pairs. It gives a more stable validation estimate **and** a measure of its variability. See [cross_validation.md](../06_model_improvement/cross_validation.md).

### 3.2 Bias–variance in one paragraph

Expected error on new data = irreducible noise + bias² + variance.
- **Bias** is systematic error from a model too simple to represent the relationship (underfitting).
- **Variance** is sensitivity to the particular training sample (overfitting).

More flexible models reduce bias and increase variance. Regularisation, more data and ensembling (bagging) reduce variance. **Irreducible noise** sets a floor that no model can go below. It includes measurement error in the target, label noise and drivers you did not measure.

### 3.3 Diagnosing fit

| Pattern | Train error | Validation error | Diagnosis | Typical remedies |
|---|---|---|---|---|
| A | High | High (≈ train) | Underfitting / high bias | Better features; more flexible model; less regularisation |
| B | Low | Much higher | Overfitting / high variance | Regularise; simplify; more data; check feature leakage |
| C | Low | Slightly higher, close to the noise floor | Good fit | Stop tuning; work on data/features if more is needed |
| D | Validation **better** than train | — | Suspicious: data error, leakage in validation, wrong metric sign, or a dropout/augmentation effect | Investigate |

A **learning curve** (score vs. training-set size) tells you whether more data would help. If the validation curve is still rising at full size, it would. If it has plateaued close to the training curve, more data will not help much; you need better features.

### 3.4 Why baselines first

A baseline answers: *How much of the score is just the easy part?* Without it you cannot tell whether R² = 0.6 is impressive or trivial. Two baselines are useful:
1. **A trivial baseline** (mean, majority class, last value) sets the zero point of skill.
2. **A simple, informed baseline** (current practice, a one-feature rule, a linear model) sets the bar that complexity must beat.

### 3.5 Model selection principles (tabular data)

- There is **no universally best algorithm** (the "no free lunch" theorem). Empirically, for medium-sized tabular data, gradient-boosted trees are a very strong default. Regularised linear models remain competitive when n is small, relations are close to linear, or extrapolation and interpretability matter.
- **Compare candidates on the same folds, with the same preprocessing constraints and a similar tuning effort.** If one model gets extensive tuning and another uses defaults, the comparison is biased.
- **Select within the noise.** If several models are within about one standard error of the best, pick the simplest, the most robust or the most interpretable.

---

## 4. Assumptions and limitations

- The bias–variance decomposition is exact for squared error; for other losses it is a useful intuition rather than an identity.
- Learning curves assume that more data would come from the same distribution. Under drift, more *old* data can hurt.
- "Gradient boosting is a strong default" is an empirical regularity for medium-sized tabular data, not a law. Small data, extrapolation or strong linear structure can favour simpler models (see the cookbook's own regression and classification examples).

---

## 5. Step-by-step procedure

1. **Fix the validation scheme** ([validation_reference.md](../08_quick_references/validation_reference.md)). Fix the folds (a seeded splitter object) so that all models see identical splits.
2. **Trivial baseline.** Record its score with the same CV.
3. **Simple informed baseline.** Current practice, or a linear/logistic model with minimal preprocessing.
4. **One flexible model with sensible defaults**, e.g. `HistGradientBoosting*`.
5. **Compare train and validation scores** for each model (pattern A/B/C/D above).
6. **Learning curve** for the most promising model: would more data help?
7. **Decide the next step**:
   - The flexible model is much better than linear → non-linearity or interactions matter. Proceed to tuning and feature engineering.
   - The flexible model ≈ linear → relationships are close to linear, or the signal is limited. Prefer the simpler model; invest in features or data.
   - Both ≈ baseline → go back to framing or data ([project_lifecycle.md](project_lifecycle.md) §8).
8. **Tune** the shortlisted model(s) ([hyperparameter_optimization.md](../06_model_improvement/hyperparameter_optimization.md)).
9. **Evaluate** with the family-specific recipe: metrics with uncertainty, segments and residual or error analysis.

---

## 6. Python implementation

### 6.1 A reusable model-comparison function

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

def compare_models(models: dict, X, y, cv, scoring: dict, groups=None) -> pd.DataFrame:
    """Cross-validate several models on IDENTICAL folds and return mean ± std
    of train and validation scores. `scoring` maps display names to sklearn scorer names."""
    rows = []
    for name, model in models.items():
        res = cross_validate(model, X, y, cv=cv, groups=groups, scoring=scoring,
                             return_train_score=True, n_jobs=-1)
        row = {"model": name, "fit_time_s": res["fit_time"].mean()}
        for s in scoring:
            sign = -1 if scoring[s].startswith("neg_") else 1
            row[f"val_{s}"] = sign * res[f"test_{s}"].mean()
            row[f"val_{s}_std"] = res[f"test_{s}"].std()
            row[f"train_{s}"] = sign * res[f"train_{s}"].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
```

### 6.2 Applying it: baseline → linear → boosting (regression example)

```python
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from cookbook_data import make_process_yield

df = make_process_yield().drop_duplicates()
df["temperature_c"] = df["temperature_c"].mask(df["temperature_c"] < 0)
df = df[df["yield_kg"] > 1].sort_values("start_time")
train = df.iloc[: int(0.8 * len(df))]

num = ["temperature_c", "pressure_bar", "feed_rate_kg_h", "residence_time_min",
       "catalyst_age_days", "impurity_ppm", "ambient_humidity_pct"]
cat = ["reactor", "catalyst", "shift", "supplier"]
X, y = train[num + cat], train["yield_kg"]

linear_pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
tree_pre = ColumnTransformer([
    ("num", "passthrough", num),     # HistGradientBoosting handles NaN natively; no scaling needed
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                           encoded_missing_value=-1), cat)])

models = {
    "mean_baseline": DummyRegressor(),
    "ridge": make_pipeline(linear_pre, RidgeCV(alphas=np.logspace(-3, 3, 13))),
    "hgb_default": make_pipeline(tree_pre, HistGradientBoostingRegressor(
        categorical_features=list(range(len(num), len(num) + len(cat))), random_state=0)),
}
cv = KFold(5, shuffle=True, random_state=42)
table = compare_models(models, X, y, cv,
                       scoring={"MAE": "neg_mean_absolute_error", "R2": "r2"})
print(table.round(3))
```

### 6.3 Learning curve

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import LearningCurveDisplay

fig, ax = plt.subplots(figsize=(6, 4))
LearningCurveDisplay.from_estimator(
    models["hgb_default"], X, y, cv=cv, scoring="neg_mean_absolute_error",
    train_sizes=np.linspace(0.1, 1.0, 6), score_type="both", negate_score=True,
    score_name="MAE", ax=ax, n_jobs=-1)
ax.set_title("Learning curve — HistGradientBoosting")
plt.tight_layout(); plt.show()
```

### 6.4 Validation curve for a single complexity parameter

```python
from sklearn.model_selection import ValidationCurveDisplay

ValidationCurveDisplay.from_estimator(
    models["hgb_default"], X, y, param_name="histgradientboostingregressor__max_leaf_nodes",
    param_range=[3, 7, 15, 31, 63, 127], cv=cv, scoring="neg_mean_absolute_error",
    negate_score=True, score_name="MAE", n_jobs=-1)
plt.show()
```

---

## 7. Interpretation

- **The comparison table.** Read `val_*` first: does each step beat the previous one by more than `val_*_std`? Then compare `train_*` with `val_*` to place each model in pattern A–D.
  - *Expected here:* the baseline has R² ≈ 0. Ridge captures the main linear effects (feed rate dominates). Boosting improves further because of the temperature optimum (a non-linear effect) and its interaction with the catalyst.
  - *Suspicious:* boosting with train R² ≈ 1.0 and validation much lower means overfitting (pattern B). Gradient boosting usually shows *some* train–validation gap. Only a growing gap combined with worse validation is a problem.
- **The learning curve.** Validation MAE still falling at the largest size → more data would help. A flat curve → features or data quality are the bottleneck.
- **The validation curve.** The best value sits where the validation score peaks. On the left of the peak is underfitting, on the right is overfitting. A flat region means the parameter matters little; pick the simpler end.

---

## 8. Decision framework

**IF** the flexible model beats linear by much more than the CV std
→ Non-linearity matters. Tune the flexible model; still keep the linear model as an interpretable reference.

**IF** linear ≈ flexible
→ Prefer linear (simpler, extrapolates more sensibly, easier to explain). Invest in features: splines, interactions, domain variables.

**IF** you observe pattern B (overfitting)
→ In this order: check for leakage-like features; increase regularisation (`min_samples_leaf`, `l2_regularization`, smaller `max_depth`, larger `alpha`); reduce features; get more data.

**IF** you observe pattern A (underfitting)
→ Engineer features using domain knowledge first. Then use a more flexible model, or reduce regularisation.

**IF** several models are within one standard error
→ Choose by secondary criteria: interpretability, inference cost, robustness across segments, calibration.

**IF** the model will extrapolate (inputs in production outside the training range)
→ Tree models predict a constant outside the training range. Prefer models with physically sensible extrapolation (linear, or monotonic constraints), and flag out-of-range inputs.

---

## 9. Common mistakes

| Mistake | Consequence | Prevention |
|---|---|---|
| No baseline | Cannot judge improvement | Dummy + simple baseline in every comparison table |
| Different folds per model | Differences reflect fold luck | One seeded CV object reused |
| Tuning the favourite model only | Biased comparison | Equal tuning budget, or report tuned vs. default |
| Ignoring the train score | Cannot diagnose bias vs. variance | `return_train_score=True` |
| Declaring a winner from a difference smaller than the std | Choosing noise | Compare with fold-wise differences (see [cross_validation.md](../06_model_improvement/cross_validation.md)) |
| Concluding "the model is bad" when the data are the limit | Endless tuning | Learning curves; estimate the noise floor (e.g. repeat measurements) |

---

## 10. Checklist

- [ ] Validation scheme fixed and seeded; identical folds for all models.
- [ ] Trivial and informed baselines in the comparison table.
- [ ] Train and validation scores recorded for each model.
- [ ] Fit pattern (A/B/C/D) diagnosed.
- [ ] Learning curve checked for the shortlisted model.
- [ ] Choice made with the noise (std across folds) in mind.
- [ ] Extrapolation risk considered.

---

## 11. References

- James, G. et al. (2023). *An Introduction to Statistical Learning (Python ed.)*, Ch. 2.2 (bias–variance), Ch. 5 (resampling). https://www.statlearning.com/
- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*, Ch. 7 "Model Assessment and Selection". https://hastie.su.domains/ElemStatLearn/
- scikit-learn: *Validation curves: plotting scores to evaluate models*. https://scikit-learn.org/stable/modules/learning_curve.html
- Grinsztajn, L., Oyallon, E. & Varoquaux, G. (2022). "Why do tree-based models still outperform deep learning on typical tabular data?" *NeurIPS Datasets and Benchmarks*. https://arxiv.org/abs/2207.08815
- Wolpert, D. H. (1996). "The Lack of A Priori Distinctions Between Learning Algorithms." *Neural Computation* 8(7), 1341–1390.
