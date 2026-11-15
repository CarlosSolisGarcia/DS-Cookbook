# Classification Model Selection: Logistic, Trees, Ensembles, SVM and Others

> **Folder:** 04_classification · **Prev:** [baseline_models.md](baseline_models.md) · **Next:** [class_imbalance.md](class_imbalance.md) · **Related:** [regression model_selection.md](../03_regression/model_selection.md) (shared material on regularisation and GBM libraries is there), [hyperparameter_optimization.md](../06_model_improvement/hyperparameter_optimization.md)
>
> *Added to the original architecture: the classification folder lacked a model-selection recipe.*

---

## 1. Purpose
Choose and configure candidate classifiers. Understand how each one produces scores and probabilities, and how it fails. Then compare them fairly on the metrics that matter for the decision.

## 2. When to use it
After the baselines, with the preprocessing pipelines defined. Return here if error analysis points to the model family (e.g. a linear model missing an interaction).

## 3. Conceptual foundation
| Model | Decision function | Probabilities | Key hyperparameters | Use when | Watch out |
|---|---|---|---|---|---|
| **Logistic regression** | Linear in features (log-odds) | Usually well calibrated | `C` (inverse regularisation), `penalty`, `class_weight` | Baseline; interpretable odds ratios; small n | Needs scaling; non-linearity must be engineered |
| **Decision tree** | Axis-aligned rules | Coarse (leaf frequencies) | `max_depth`, `min_samples_leaf` | Explaining rules; teaching | High variance; rarely the final model |
| **Random forest** | Vote of deep bagged trees | Tend to be under-confident (pulled towards 0.5) | `n_estimators`, `max_features`, `min_samples_leaf`, `class_weight` | Robust default with little tuning | Less accurate than GBM on many tasks; calibrate if probabilities are used |
| **Gradient boosting** (HistGB, LightGBM, XGBoost) | Additive trees on log-odds | Often reasonable; check calibration | `learning_rate`, iterations + early stopping, `max_leaf_nodes`/`num_leaves`, `min_samples_leaf`, `l2_regularization` | Default for tabular data | Overfits small/noisy data without regularisation |
| **SVM** (RBF kernel) | Max-margin, kernel | **Not native**: `probability=True` uses internal CV (slow) or calibrate externally | `C`, `gamma` | Small/medium n, dense features, clear margins | O(n²)–O(n³) training; scaling essential |
| **k-NN** | Majority among neighbours | Neighbour fractions | `n_neighbors`, metric, weights | Low-dim data with meaningful distances | Curse of dimensionality; scaling essential |
| **Naive Bayes** | Product of per-feature likelihoods | Often poorly calibrated (extreme) | smoothing | Text, very high-dimensional sparse data ([12_nlp](../12_natural_language_processing/)) | Independence assumption |
| **LDA/QDA** | Gaussian class conditionals | Reasonable if the assumptions hold | shrinkage | Small n, roughly Gaussian features | Assumption-sensitive |

**Multiclass strategies.** Logistic regression (multinomial) and GBMs handle K classes natively. SVMs use one-vs-one internally. `OneVsRestClassifier` wraps any binary model.

## 4. Assumptions and limitations
- Comparisons must use the same folds (stratified and grouped as needed) and a comparable tuning budget.
- Ranking metrics (ROC/PR-AUC) do not depend on calibration, while Brier and log-loss do. Choose the metric according to how the output is used.
- `class_weight="balanced"` changes the probabilities: it inflates the positive-class probabilities, so the model is no longer calibrated to the true prevalence ([class_imbalance.md](class_imbalance.md)).

## 5. Step-by-step procedure
1. **Logistic regression** with proper preprocessing (scaling, one-hot, imputation + indicators). Tune `C` with an inner CV (`LogisticRegressionCV`) or a small grid.
2. **HistGradientBoosting** with early stopping and modest depth.
3. **Random forest** as a robust second reference (optional).
4. **SVM / kNN** only if n is moderate and there is a reason to expect them to help (e.g. low-dimensional data with geometric structure).
5. **Compare** ROC-AUC, PR-AUC and Brier (mean ± std over folds). Look at train vs. validation for over/underfitting.
6. **Shortlist**, tune ([hyperparameter_optimization.md](../06_model_improvement/hyperparameter_optimization.md)), then calibrate and threshold.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text

from cookbook_data import make_equipment_failure

df = make_equipment_failure(include_leakage=False)
num = ["age_years", "load_pct", "operating_hours", "days_since_maintenance",
       "vibration_rms", "temperature_c", "error_count_7d"]
cat = ["machine_type", "site", "firmware"]
X, y, groups = df[num + cat], df["failure_30d"], df["machine_id"]

lin_pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
tree_pre = ColumnTransformer([("num", "passthrough", num),
                              ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat)])
cat_mask = [False] * len(num) + [True] * len(cat)

models = {
    "logistic (CV over C)": make_pipeline(lin_pre, LogisticRegressionCV(Cs=10, cv=5, scoring="neg_log_loss", max_iter=3000)),
    "decision tree (depth 4)": make_pipeline(tree_pre, SimpleImputer(strategy="median"),
                                             DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=0)),
    "random forest": make_pipeline(tree_pre, SimpleImputer(strategy="median"),
                                   RandomForestClassifier(n_estimators=500, min_samples_leaf=5, max_features="sqrt",
                                                          n_jobs=-1, random_state=0)),
    "hist gradient boosting": make_pipeline(tree_pre, HistGradientBoostingClassifier(
        categorical_features=cat_mask, learning_rate=0.05, max_iter=1000, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=50, random_state=0)),
    "SVM rbf": make_pipeline(lin_pre, SVC(C=1.0, gamma="scale", probability=False, random_state=0)),
    "kNN (k=50)": make_pipeline(lin_pre, KNeighborsClassifier(n_neighbors=50)),
}
cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
rows = []
for name, m in models.items():
    scoring = {"roc": "roc_auc", "pr": "average_precision"}
    if name != "SVM rbf":
        scoring["brier"] = "neg_brier_score"     # SVC without probability=True has no predict_proba
    r = cross_validate(m, X, y, groups=groups, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1)
    rows.append({"model": name, "train_PR": r["train_pr"].mean(), "val_PR": r["test_pr"].mean(),
                 "val_PR_std": r["test_pr"].std(), "val_ROC": r["test_roc"].mean(),
                 "val_Brier": -r["test_brier"].mean() if "test_brier" in r else np.nan,
                 "fit_s": r["fit_time"].mean()})
print(pd.DataFrame(rows).set_index("model").round(4))
print("prevalence (random PR-AUC):", round(y.mean(), 4))
```

```python
# Readable rules from a shallow tree: a sanity check on what drives the predictions
tree = models["decision tree (depth 4)"].fit(X, y)
print(export_text(tree[-1], feature_names=list(num + cat), max_depth=2, show_weights=True))
```

```python
# Logistic regression coefficients as odds ratios per 1 std (interpretable reference)
logit = models["logistic (CV over C)"].fit(X, y)
names = logit[0].get_feature_names_out()
odds = pd.Series(np.exp(logit[-1].coef_[0]), index=names).sort_values()
print("chosen C:", logit[-1].C_[0].round(4))
print(odds.round(2))
```

```python
# Multiclass: failure mode (none / bearing / electrical / overheating). Multinomial logistic and HGB are native.
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict

dfm = make_equipment_failure(include_leakage=True)
ym = dfm["failure_mode"]
Xm = dfm[num + cat]
hgb_multi = make_pipeline(tree_pre, HistGradientBoostingClassifier(
    categorical_features=cat_mask, learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
    class_weight="balanced", random_state=0))
pred = cross_val_predict(hgb_multi, Xm, ym, cv=cv, groups=dfm["machine_id"])
print(classification_report(ym, pred, digits=3, zero_division=0))
```

## 7. Interpretation
- **Compare val_PR to the prevalence.** Models at 3–5× the prevalence are typical for predictive maintenance with these signals.
- **In this run logistic regression is the best model** (val PR-AUC ≈ 0.29, versus ≈ 0.21–0.24 for GBM/RF). The generating process is additive on the log-odds scale, and ~6000 rows with 5% positives (~300 events) is too little for flexible models to beat a well-specified linear one. This is typical of rare-event problems. Do not assume boosting wins. A clear GBM win would indicate interactions or thresholds.
- **Random forest with a train PR-AUC ≈ 1 and a lower validation score** is normal for deep trees. Judge it on validation. Its Brier score is often worse than GBM's or logistic's, because of under-confident probabilities.
- **SVM / kNN** rarely win on mixed-type tabular data. A large gap below GBM is expected.
- **Tree rules** (`show_weights=True` displays the class counts per node; every leaf predicts "0" at a 0.5 threshold under 5% prevalence, which is why scores, not hard classes, are what you evaluate) should agree with domain intuition (vibration, temperature, errors). Rules based on site or firmware alone suggest drift or a data artefact. Investigate.
- **Odds ratios:** e.g. 1.8 for `vibration_rms` means one std more vibration multiplies the odds of failure by 1.8, *conditional on the other features*. Note that `age_years` has an odds ratio **below 1**, even though older machines fail more often. Age acts *through* vibration and temperature, which are already in the model. In addition, old machines often lack a vibration sensor, so the missing indicator (1.35) absorbs part of the age effect. Conditional coefficients answer "holding the others fixed". That is a different question from "is age a risk factor?" ([model_interpretation.md](../06_model_improvement/model_interpretation.md)).
- **Multiclass report:** look at the per-class recall, not the accuracy (0.94 is almost entirely the "none" class). Here the failure modes are predicted poorly (recall 0–9%), even with `class_weight="balanced"`. With 56–165 examples per mode, the model cannot separate them reliably. Two practical options follow. (1) A two-stage design: first "failure vs. none" (binary, more data per class), then the mode among predicted failures. (2) Accept that the mode is uncertain and report class probabilities. The macro average (≈ 0.29) exposes this; the weighted average (≈ 0.93) hides it.

## 8. Decision framework
| Situation | First choice | Alternatives |
|---|---|---|
| Tabular, mixed types, n ≥ ~1000 | HistGB with early stopping | LightGBM/XGBoost; logistic as reference |
| Small n (< ~500) or need for interpretability | Regularised logistic (+ splines/interactions) | Shallow GBM with strong regularisation |
| Very high-dimensional sparse (text) | Linear models (logistic, linear SVM) | Naive Bayes ([12_nlp](../12_natural_language_processing/)) |
| Low-dimensional, smooth class boundaries, moderate n | SVM RBF / kNN | GBM |
| Probabilities required | Logistic or GBM, then check calibration | RF/SVM + calibration ([calibration.md](calibration.md)) |
| Known monotone effects (risk ↑ with vibration) | GBM with `monotonic_cst` | Logistic |

- **IF** logistic and GBM are within the noise → choose logistic.
- **IF** GBM wins and interpretability is needed → keep GBM, explain it with SHAP/PDP ([model_interpretation.md](../06_model_improvement/model_interpretation.md)), and keep logistic as a sanity check.
- **IF** all models perform similarly and poorly → the bottleneck is the features or labels, not the algorithm.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| `SVC(probability=True)` and trusting its probabilities | Slow; internal CV calibration can be inconsistent with `predict` | Use `decision_function` + external calibration |
| Comparing models only by ROC-AUC under heavy imbalance | Differences in the relevant region hidden | Add PR-AUC and precision@k |
| Unscaled features for logistic/SVM/kNN | Poor or unit-dependent results | Pipelines with scaling |
| Treating RF probabilities as calibrated | Wrong risk estimates | Calibration check |
| Using a single deep tree as the final model | High variance | Ensembles; trees for explanation only |

## 10. Checklist
- [ ] Logistic regression reference with tuned `C`.
- [ ] GBM with early stopping; RF optional.
- [ ] SVM/kNN only with justification.
- [ ] ROC-AUC, PR-AUC and Brier with the std across folds; train vs. validation.
- [ ] Rules/coefficients sanity-checked against domain knowledge.
- [ ] Multiclass: per-class metrics.

## 11. References
- scikit-learn User Guide: *Linear models — Logistic regression*; *Support Vector Machines*; *Nearest Neighbors*; *Decision Trees*; *Ensembles*. https://scikit-learn.org/stable/supervised_learning.html
- scikit-learn, *Multiclass and multioutput algorithms*. https://scikit-learn.org/stable/modules/multiclass.html
- Hosmer, D. W., Lemeshow, S. & Sturdivant, R. X. (2013). *Applied Logistic Regression*, 3rd ed. Wiley.
- Niculescu-Mizil, A. & Caruana, R. (2005). "Predicting Good Probabilities with Supervised Learning." *ICML*. Calibration of RF, SVM, NB and boosting. https://doi.org/10.1145/1102351.1102430
- Cortes, C. & Vapnik, V. (1995). "Support-vector networks." *Machine Learning* 20, 273–297.
