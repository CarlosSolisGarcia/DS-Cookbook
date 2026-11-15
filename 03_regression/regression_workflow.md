# Regression Workflow: End-to-End Playbook

> **Folder:** 03_regression · **Start here for any continuous-target problem.** · **Next:** [baseline_models.md](baseline_models.md) · **Related:** [project_lifecycle.md](../00_methodology/project_lifecycle.md), [modeling_fundamentals.md](../00_methodology/modeling_fundamentals.md)

---

## 1. Purpose
This is the master procedure for predicting a continuous quantity: yield, energy use, strength, concentration, cost. It sequences the general recipes (data understanding, preprocessing, validation) and the regression-specific ones (baselines, models, metrics, intervals, error analysis) into one workflow, with decision points along the way.

## 2. When to use it
- The target is numeric and its magnitude matters: predicting 10 vs 12 is better than 10 vs 30.
- Not when the target is a category, or a number used only as a category (e.g. a defect code). That is classification.
- Not for future values of a series that depend on their own past. That is forecasting ([05_time_series](../05_time_series/time_series_workflow.md)). Time-stamped rows with a per-row target are still regression, but with chronological validation.
- If the decision is binary (e.g. "below spec?"), see [problem_framing.md](../00_methodology/problem_framing.md) §6 for regress-then-threshold vs. classify.

## 3. Conceptual foundation
A regression model estimates a **functional of the conditional distribution** of y given x. The loss determines which functional:

| Loss | Estimates | Typical metric | Use when |
|---|---|---|---|
| Squared error | Conditional **mean** | RMSE / MSE | Totals and averages matter; large errors are especially costly |
| Absolute error | Conditional **median** | MAE | Robustness to outliers; "typical" error |
| Pinball / quantile (α) | Conditional **α-quantile** | Pinball loss, interval coverage | Asymmetric costs; prediction intervals |
| Poisson / Gamma / Tweedie deviance | Mean, for counts / positive skewed data | Deviance, MAE | Counts, positive continuous with variance growing with the mean |

Choosing the loss **before** choosing the model is the regression equivalent of choosing the metric from the costs ([problem_framing.md](../00_methodology/problem_framing.md)).

## 4. Assumptions and limitations
- Supervised learning assumes the training data are representative of the deployment conditions. Under drift or extrapolation this fails ([20_advanced_methodologies](../20_advanced_methodologies/)).
- Classical linear-regression assumptions (linearity, independent errors, homoscedasticity, normal errors) matter for **inference** (p-values, confidence intervals on coefficients). For **prediction** they matter only through their effect on accuracy and interval validity. Residual analysis checks both.
- The workflow assumes tabular data. For other modalities, the evaluation and error-analysis logic still applies.

## 5. Step-by-step procedure

| Step | Action | Recipe | Output |
|---|---|---|---|
| 1 | Frame: target, unit, t₀, cost of errors → loss & metric | [problem_framing.md](../00_methodology/problem_framing.md) | Problem statement |
| 2 | Inspect, assess quality, clean | [initial_inspection](../01_data_understanding/initial_inspection.md), [data_quality](../01_data_understanding/data_quality.md) | Clean data + DQ report |
| 3 | **Split** (temporal / grouped / random) and freeze the test set | [validation_reference.md](../08_quick_references/validation_reference.md) | train / test |
| 4 | EDA on train: target distribution, relationships, collinearity | [exploratory_data_analysis.md](../01_data_understanding/exploratory_data_analysis.md) | EDA notes |
| 5 | Baselines | [baseline_models.md](baseline_models.md) | Baseline scores |
| 6 | Preprocessing per model family | [02_preprocessing](../02_preprocessing/numerical_features.md) | Pipelines |
| 7 | Candidate models: regularised linear → GBM | [model_selection.md](model_selection.md) | Comparison table |
| 8 | Features from domain knowledge / error analysis | [feature_engineering.md](../02_preprocessing/feature_engineering.md) | Improved features |
| 9 | Tune shortlisted models | [hyperparameter_optimization.md](../06_model_improvement/hyperparameter_optimization.md) | Tuned configuration |
| 10 | Evaluate: metrics + uncertainty + segments | [evaluation.md](evaluation.md) | Evaluation report |
| 11 | Residual & error analysis | [error_analysis.md](error_analysis.md) | Failure modes → next actions |
| 12 | Prediction intervals if decisions need uncertainty | [uncertainty.md](uncertainty.md) | Calibrated intervals |
| 13 | Interpret | [model_interpretation.md](../06_model_improvement/model_interpretation.md) | Explanations, sanity checks |
| 14 | Iterate (steps 4–13) with tracked experiments; final test once | [iterative_improvement.md](../06_model_improvement/iterative_improvement.md) | Final model + report |

## 6. Python implementation: the compact end-to-end skeleton
Each step is expanded in its own recipe. This skeleton is the template to copy into a new project.

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from cookbook_data import load_clean_process_yield

# --- Steps 2–3: clean data, chronological split (model is used on future batches)
df = load_clean_process_yield()
cut = int(0.8 * len(df))
train, test = df.iloc[:cut], df.iloc[cut:]
TARGET = "yield_kg"

# --- Step 6–8: features + preprocessing
def domain_features(d):
    d = d.copy()
    d["log_feed_per_batch"] = np.log(d["feed_rate_kg_h"] * 8.0)
    d["log_residence"] = np.log(d["residence_time_min"])
    t_opt = d["catalyst"].map({"Pt": 190.0, "Pd": 182.0, "Ni": 195.0})
    d["temp_dev_sq"] = (d["temperature_c"] - t_opt) ** 2
    return d

lin_num = ["log_feed_per_batch", "log_residence", "temp_dev_sq", "pressure_bar",
           "catalyst_age_days", "impurity_ppm"]
raw_num = ["temperature_c", "pressure_bar", "feed_rate_kg_h", "residence_time_min",
           "catalyst_age_days", "impurity_ppm", "ambient_humidity_pct"]
cat = ["reactor", "catalyst", "shift", "supplier"]

linear = TransformedTargetRegressor(
    regressor=make_pipeline(
        FunctionTransformer(domain_features),
        ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), lin_num),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat)]),
        RidgeCV(alphas=np.logspace(-3, 3, 13))),
    func=np.log, inverse_func=np.exp)

def to_category(d):
    return d[raw_num + cat].astype({c: "category" for c in cat})

gbm = make_pipeline(FunctionTransformer(to_category),
                    HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=0))

models = {"baseline_median": DummyRegressor(strategy="median"), "ridge_domain_logy": linear, "hgb_raw": gbm}

# --- Steps 5, 7: compare under chronological CV on the training period
cv = TimeSeriesSplit(n_splits=5)
scoring = {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"}
rows = []
for name, m in models.items():
    r = cross_validate(m, train, train[TARGET], cv=cv, scoring=scoring)
    rows.append({"model": name, **{k: (-1 if k != "R2" else 1) * r[f"test_{k}"].mean() for k in scoring},
                 "MAE_std": r["test_MAE"].std()})
print(pd.DataFrame(rows).set_index("model").round(3))
```

```python
# --- Step 10: final evaluation on the frozen test set (ONCE, after all decisions)
final = models["ridge_domain_logy"].fit(train, train[TARGET])
pred = final.predict(test)
print(f"TEST  MAE={mean_absolute_error(test[TARGET], pred):.1f}  "
      f"RMSE={root_mean_squared_error(test[TARGET], pred):.1f}  R2={r2_score(test[TARGET], pred):.3f}")
```

## 7. Interpretation
- **Read the comparison table against the baseline.** R² of the median baseline is ≤ 0 by construction. A useful model reduces MAE substantially relative to it.
- **The domain-feature ridge vs. HGB on raw features:** if ridge wins, the features encode the mechanism well (see [feature_engineering.md](../02_preprocessing/feature_engineering.md)). If HGB wins, there are non-linearities or interactions left that the linear features do not capture. Error analysis tells you which.
- **Test vs. CV:** a test score within about 1–2 CV standard deviations of the CV mean is consistent. Much worse means drift between periods, or a validation scheme that does not mimic deployment.

## 8. Decision framework
- **IF** large errors are disproportionately costly → optimise and report RMSE (squared loss). **ELSE IF** robustness matters → MAE (absolute loss).
- **IF** the target is positive and right-skewed → compare a log target and a Gamma/Tweedie loss ([transformations.md](../02_preprocessing/transformations.md)).
- **IF** the data are time-ordered → `TimeSeriesSplit` for CV and a final chronological test.
- **IF** there are repeated measurements per entity → `GroupKFold`.
- **IF** n < ~200 → prefer regularised linear models / small GBMs with repeated CV; see [18_scientific_machine_learning](../18_scientific_machine_learning/) for small-data practice.
- **IF** decisions need risk ranges → add prediction intervals ([uncertainty.md](uncertainty.md)).
- **IF** the best model barely beats the baseline → go back to framing and data ([project_lifecycle.md](../00_methodology/project_lifecycle.md) §6).

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Picking R² as the only metric | Hides the error magnitude in units; not comparable across datasets | Report MAE/RMSE in units + R² |
| Random K-fold on drifting data | Optimistic CV | Chronological validation |
| Transforming the target and reporting log-scale metrics | Misleading comparisons | Metrics on the original scale |
| Skipping residual analysis | Systematic errors unnoticed | [error_analysis.md](error_analysis.md) every iteration |
| Using the test set during iteration | Optimistic final estimate | Test once |

## 10. Checklist
- [ ] Loss and metric chosen from the costs (mean vs median vs quantile).
- [ ] Clean data, documented DQ decisions.
- [ ] Split mimics deployment; test frozen.
- [ ] EDA notes with hypotheses.
- [ ] Baseline(s) recorded.
- [ ] Linear and tree-based candidates compared on identical folds.
- [ ] Features engineered and validated.
- [ ] Tuning with a fixed budget; no test usage.
- [ ] Evaluation with uncertainty and segments; residual analysis.
- [ ] Intervals if needed; interpretation sanity checks.
- [ ] Final test evaluation once; report written.

## 11. References
- James, G. et al. (2023). *ISLP*, Ch. 3 "Linear Regression", Ch. 6 "Linear Model Selection and Regularization", Ch. 8 "Tree-Based Methods". https://www.statlearning.com/
- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *ESL*, Ch. 3, 7, 10. https://hastie.su.domains/ElemStatLearn/
- scikit-learn User Guide, *Supervised learning*. https://scikit-learn.org/stable/supervised_learning.html
- Gneiting, T. (2011). "Making and Evaluating Point Forecasts." *JASA* 106(494), 746–762. On the loss-to-functional correspondence. https://doi.org/10.1198/jasa.2011.r10138
