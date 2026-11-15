# Decision Framework: The Master Map of "What Do I Do Next?"

> **Folder:** 00_methodology · **Prev:** [problem_framing.md](problem_framing.md) · **Next:** [data_leakage.md](data_leakage.md) · **Related:** every recipe's §8 (Decision framework). This file is the index of those decisions.

---

## 1. Purpose

**What it solves.** It gathers the cookbook's key decision points into one place. When you are unsure what to do next, start here: find your situation, and follow the link to the recipe that handles it.

---

## 2. When to use it

**When to use it.** At the start of a project, to pick the workflow. At every "now what?" moment. And before reporting results, as a sanity pass.

**When not to use it.** As a substitute for the recipes. Each decision here is a summary; the reasoning, code and caveats are in the linked recipe.

---

## 3. Conceptual foundation

Three principles underlie every decision in the cookbook:

1. **Evidence before action.** Do not transform, drop, resample or tune because "it is usually done". Do it because a diagnostic showed a problem, and then confirm with validation that the fix helped.
2. **Validation mimics deployment.** Every comparison must be made with a validation scheme that reproduces how the model will be used: future data, new groups, the same preprocessing fitted only on the past.
3. **Simplest adequate solution.** Add complexity (features, models, tuning) only when it beats the simpler alternative by more than the evaluation noise.

These principles cannot be broken by any rule of thumb below. When a rule and a principle conflict, the principle wins.

---

## 4. Assumptions and limitations

- The tables encode defaults that work for typical tabular, time-series and (via sections 09–21) other problems. Unusual data structures may need different choices, so always run the diagnostic first.
- Thresholds quoted in the tables (e.g. "≲ 20 levels", "< 5% missing") are prompts for investigation, not hard rules.

---

## 5. Procedure: navigating a new problem

### Step 1 — Which problem family? (detailed in [problem_framing.md](../00_methodology/problem_framing.md) §3.3)

```
Is the target observed (labelled) for historical examples?
├── No  → Unsupervised: clustering / anomaly detection (future extension)
└── Yes → Is the order of observations in time essential to the question?
          ├── Yes, I need values at FUTURE time points of the same series
          │      → FORECASTING → 05_time_series/time_series_workflow.md
          ├── Yes, rows are time-stamped events/snapshots but the target is a property of each row
          │      → TEMPORAL REGRESSION / CLASSIFICATION: tabular workflow (03 or 04)
          │        + chronological validation + lag features if useful
          └── No  → Target type?
                   ├── Continuous → 03_regression/regression_workflow.md
                   └── Categorical → 04_classification/classification_workflow.md
```

### Step 2 — Which validation scheme? (detailed in [validation_reference.md](../08_quick_references/validation_reference.md))

| Data structure | Hold-out test | Cross-validation |
|---|---|---|
| i.i.d. rows, enough data | Random (stratified for classification) | `KFold` / `StratifiedKFold` |
| Multiple rows per entity (machine, patient, customer) | Split by group | `GroupKFold` / `StratifiedGroupKFold` |
| Time-ordered, used on the future | Last period | `TimeSeriesSplit` / rolling origin |
| Time-ordered **and** grouped | Last period, possibly also unseen groups | Time-based folds + group awareness |
| Very small n (< ~200) | Use repeated CV instead of a single hold-out | `RepeatedKFold` / `RepeatedStratifiedKFold` |

### Step 3 — Which baseline? (detailed in the family `baseline_models.md`)

| Family | Trivial baseline | Simple informed baseline |
|---|---|---|
| Regression | Mean or median of the training target | Group mean (e.g. by reactor), single-variable linear model, current engineering rule |
| Classification | Majority class / class prior probability | One-feature rule, logistic regression with a few features, current business rule |
| Forecasting | Naive (last value) | Seasonal naive, moving average, drift |

### Step 4 — Run the family workflow, consulting the decision tables below as issues arise.

---

## 6. Implementation: decision tables by topic

This section is a navigable index of decisions. Each line gives the trigger, the recommended action and the recipe with details.

### 6.1 Data understanding and quality

| IF you observe… | THEN… | See |
|---|---|---|
| Duplicate rows | Determine whether they are true duplicates (export bug) or legitimate repeats; deduplicate *before* splitting | [data_quality.md](../01_data_understanding/data_quality.md) |
| Sentinel values (-999, 0 for "unknown", 9999) | Convert to `NaN`, plus a missing indicator if the sentinel is informative | [data_quality.md](../01_data_understanding/data_quality.md) |
| Missingness < ~5%, apparently random | Simple imputation inside the pipeline is usually adequate | [missing_values.md](../01_data_understanding/missing_values.md) |
| Missingness related to other variables or to the target | Add missing-indicator features; consider models that handle NaN natively; never simply drop rows | [missing_values.md](../01_data_understanding/missing_values.md) |
| Extreme values | Classify them: error, rare-but-real, or a different population. Treat each differently | [outliers.md](../01_data_understanding/outliers.md) |
| Several rows per entity | Use grouped validation; consider entity-level features | [validation_reference.md](../08_quick_references/validation_reference.md) |
| A feature almost perfectly predicts the target | Suspect leakage first | [data_leakage.md](data_leakage.md) |

### 6.2 Preprocessing and features

| IF… | THEN… | See |
|---|---|---|
| Linear / distance-based / neural model | Scale numeric features; one-hot encode categoricals | [scaling.md](../02_preprocessing/scaling.md) |
| Tree-based model | Scaling is unnecessary; ordinal or native categorical encoding is fine | [categorical_features.md](../02_preprocessing/categorical_features.md) |
| Categorical feature with many levels (> ~20–50) | Target encoding (cross-fitted), frequency encoding, or native categorical support | [categorical_features.md](../02_preprocessing/categorical_features.md) |
| Strongly skewed positive feature + linear model | Try log / Yeo-Johnson; validate | [transformations.md](../02_preprocessing/transformations.md) |
| Strongly skewed target | Consider log-target / a different loss; mind the back-transformation bias and the metric | [transformations.md](../02_preprocessing/transformations.md) |
| Domain knowledge suggests ratios, interactions, physical quantities | Engineer them explicitly; they often beat algorithm changes | [feature_engineering.md](../02_preprocessing/feature_engineering.md) |
| Highly correlated features | Matters for coefficient interpretation, rarely for prediction; consider regularisation | [feature_selection.md](../06_model_improvement/feature_selection.md) |

### 6.3 Modelling

| IF… | THEN… | See |
|---|---|---|
| No baseline yet | Build one first | family `baseline_models.md` |
| Tabular data, mixed types, non-linearities | Gradient boosting (HistGradientBoosting / LightGBM / XGBoost) is a strong default after a linear baseline | [model_selection.md (reg)](../03_regression/model_selection.md), [model_selection.md (clf)](../04_classification/model_selection.md) |
| Small n, need interpretability | Regularised linear / logistic regression, possibly with splines | same |
| Training score ≫ validation score | Overfitting: regularise, simplify, add data, check for leakage in features | [modeling_fundamentals.md](modeling_fundamentals.md) |
| Training ≈ validation, both poor | Underfitting: richer features or a more flexible model | [modeling_fundamentals.md](modeling_fundamentals.md) |
| Model choice depends on tuned hyperparameters | Use nested CV, or a fixed tuning budget per model on the same folds | [cross_validation.md](../06_model_improvement/cross_validation.md) |

### 6.4 Evaluation

| IF… | THEN… | See |
|---|---|---|
| Regression with outliers in the target | Report MAE (robust) alongside RMSE; analyse the large errors separately | [regression evaluation](../03_regression/evaluation.md) |
| Target values near zero | Avoid MAPE; use MAE, scaled errors or WAPE | [metrics_reference.md](../08_quick_references/metrics_reference.md) |
| Imbalanced classification | Use PR-AUC / precision-recall at relevant thresholds; never accuracy alone | [classification evaluation](../04_classification/evaluation.md) |
| Probabilities used as risk estimates | Check calibration | [calibration.md](../04_classification/calibration.md) |
| Business decision uses a threshold | Optimise the threshold on validation data using costs or capacity | [threshold_optimization.md](../04_classification/threshold_optimization.md) |
| Forecasting | Compare against seasonal naive; use MASE; evaluate per horizon | [ts evaluation](../05_time_series/evaluation.md) |
| Two models differ by less than the CV std | Treat them as equivalent; choose the simpler or more robust one | [cross_validation.md](../06_model_improvement/cross_validation.md) |

### 6.5 Improvement

| IF error analysis shows… | THEN the next action is… | See |
|---|---|---|
| Errors concentrated in a segment | Investigate data quality / missing features for that segment; consider segment-specific features | [iterative_improvement.md](../06_model_improvement/iterative_improvement.md) |
| Errors grow with time since training | Drift: retrain more often, add time-aware features, check for regime changes | same |
| Residuals correlated with a feature | That feature is under-used: transform it, add interactions, or use a more flexible model | [regression error analysis](../03_regression/error_analysis.md) |
| Errors look random, with variance near the label-noise level | You are near the ceiling. More tuning will not help; better data or new information might | same |

---

## 7. Interpretation: how to use these tables

The tables encode **default** actions, not laws. For each action:
1. Check that the trigger really applies (run the diagnostic).
2. Apply the action **inside a pipeline**, fitted on training folds only.
3. Compare with and without the action under the same validation scheme.
4. Keep it only if it helps beyond the noise, or if it is needed for correctness (e.g. leakage removal).

---

## 8. Decision framework: meta-decisions

**IF** two recipes give conflicting advice
→ Defer to the one closer to your deployment conditions. For example, "use all data" conflicts with "keep a temporal test set"; the temporal test set wins when you are deploying on future data.

**IF** a decision cannot be validated (e.g. too little data)
→ Prefer the conservative option (simpler model, fewer transformations) and state the uncertainty.

**IF** you have spent a long time on one stage without progress
→ Step back. Run error analysis and check whether you are working on the right stage ([project_lifecycle.md](project_lifecycle.md) §8).

---

## 9. Common mistakes

| Mistake | Consequence | Prevention |
|---|---|---|
| Applying every preprocessing step "by default" | Wasted effort, sometimes worse models (e.g. scaling for trees, SMOTE on calibrated problems) | Diagnose, then act, then validate |
| Treating rules of thumb as thresholds ("drop if VIF > 10") | Removing useful features or keeping harmful ones | Use thresholds as prompts for investigation |
| Comparing options under different validation schemes | Invalid comparisons | One frozen scheme per project phase |
| Following the table without reading the linked recipe | Missing caveats | Treat the tables as an index |

---

## 10. Checklist

- [ ] Problem family identified with the tree in §3.
- [ ] Validation scheme chosen from the table, justified by the data structure.
- [ ] Baseline chosen.
- [ ] Every preprocessing / modelling decision traceable to a diagnostic.
- [ ] Each decision validated with and without the change.

---

## 11. References

- scikit-learn: *Choosing the right estimator* (flowchart; useful, but only a starting point). https://scikit-learn.org/stable/machine_learning_map.html
- scikit-learn: *Common pitfalls and recommended practices*. https://scikit-learn.org/stable/common_pitfalls.html
- James, G., Witten, D., Hastie, T., Tibshirani, R. & Taylor, J. (2023). *An Introduction to Statistical Learning with Applications in Python*. Springer. Free PDF: https://www.statlearning.com/
- Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd ed. OTexts. https://otexts.com/fpp3/
