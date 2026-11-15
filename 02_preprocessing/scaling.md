# Feature Scaling

> **Folder:** 02_preprocessing · **Prev:** [transformations.md](transformations.md) · **Next:** [feature_engineering.md](feature_engineering.md) · **Related:** [numerical_features.md](numerical_features.md)

---

## 1. Purpose
Put numeric features on comparable scales **when the model is sensitive to scale**, and choose a scaler that suits the data: standard, robust, min–max or max-abs.

## 2. When to use it
| Model | Needs scaling? | Why |
|---|---|---|
| Ridge / Lasso / ElasticNet / logistic with a penalty | **Yes** | The penalty treats all coefficients equally, so their scale must be comparable |
| SVM, kNN, k-means, PCA | **Yes** | Distance- or variance-based |
| Neural networks | **Yes** | Optimisation stability |
| Unpenalised OLS | No for predictions (yes for comparable coefficients) | Predictions are scale-invariant |
| Trees, RF, gradient boosting | **No** | Splits are invariant to monotonic transforms |

## 3. Conceptual foundation
| Scaler | Transformation | Robust to outliers? | Use when |
|---|---|---|---|
| `StandardScaler` | (x − mean)/std | No | Default for linear models, SVM, NN |
| `RobustScaler` | (x − median)/IQR | Yes | Heavy tails or outliers you keep |
| `MinMaxScaler` | (x − min)/(max − min) → [0, 1] | No (min/max set by extremes) | Bounded inputs required (some NNs, image pixels) |
| `MaxAbsScaler` | x / max\|x\| | No | Sparse data (keeps zeros) |

Scaling does not change the shape of a distribution. It only shifts and stretches it. To change the shape, see [transformations.md](transformations.md).

**One-hot columns and scaling.** Standardising 0/1 dummies is not wrong, and it makes them penalised comparably to the continuous features. Leaving them unscaled is also common. Either way, apply the choice consistently.

## 4. Assumptions and limitations
- Scaler statistics are fitted, so the scaler goes inside the pipeline (fitted on the training fold).
- Values in production outside the training range are scaled beyond the usual range (e.g. > 1 for min–max). That is fine for linear models, but it is worth monitoring.
- Scaling cannot equalise the *importance* of features. Penalised coefficients are still affected by correlations and transformations.

## 5. Step-by-step procedure
1. Check whether the model needs scaling (table above).
2. Inspect distributions for heavy tails or outliers ([outliers.md](../01_data_understanding/outliers.md)).
3. Pick a scaler: `StandardScaler` by default, `RobustScaler` with outliers you are keeping.
4. Place it after imputation and transformation, before the model, inside the pipeline.
5. Verify on the training data that the transformed features have the expected location and spread.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

from cookbook_data import make_equipment_failure

df = make_equipment_failure()
num = ["age_years", "load_pct", "operating_hours", "days_since_maintenance",
       "vibration_rms", "temperature_c", "error_count_7d"]
X, y, groups = df[num], df["failure_30d"], df["machine_id"]
print(X.describe().loc[["mean", "std", "min", "max"]].round(1))   # very different scales
cv = GroupKFold(5)

for model_name, model in [("kNN (k=25)", KNeighborsClassifier(25)),
                          ("logistic C=0.1", LogisticRegression(C=0.1, max_iter=2000))]:
    for sc_name, scaler in [("none", "passthrough"), ("standard", StandardScaler()),
                            ("robust", RobustScaler()), ("minmax", MinMaxScaler())]:
        pipe = make_pipeline(SimpleImputer(strategy="median"), scaler, model)
        s = cross_val_score(pipe, X, y, cv=cv, groups=groups, scoring="roc_auc")
        print(f"{model_name:15s} scaler={sc_name:8s} ROC-AUC={s.mean():.3f} ± {s.std():.3f}")
```

## 7. Interpretation
- **kNN without scaling** is dominated by `operating_hours` and `days_since_maintenance` (scales in the hundreds). The other features barely count, so the AUC drops. Any scaler fixes this.
- **Penalised logistic regression:** here the AUC is almost unchanged without scaling. With n = 6000 the penalty at C = 0.1 is weak relative to the likelihood. With stronger penalties, fewer rows or L1 (Lasso) selection, the unscaled model penalises features *according to their units*. Coefficients and selected features then change if you switch from hours to days. Scale anyway: it makes the result independent of units, and it helps the solver converge.
- **Standard vs robust** usually gives similar results. Robust wins when a few extreme values would otherwise compress the bulk of the data.

## 8. Decision framework
- **IF** the model is distance-based or penalised → scale (Standard by default).
- **IF** you keep heavy tails/outliers → `RobustScaler`, or transform first (log) and then use `StandardScaler`.
- **IF** the data are sparse (TF-IDF, one-hot heavy) → `MaxAbsScaler` or no centring (`StandardScaler(with_mean=False)`).
- **IF** the model is tree-based → skip scaling.
- **IF** you need coefficients comparable across features → standardise and report "per 1 std" effects, or report coefficients in original units with care.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Scaling before the split | Leakage of test statistics | Scaler inside the pipeline |
| Scaling for tree models | No benefit; extra complexity | Skip |
| Expecting scaling to fix skew | Shape unchanged | Transform first |
| MinMax with outliers | The bulk squeezed into a tiny range | RobustScaler or a transformation |
| Different scaling in training and inference code | Silent prediction errors | Persist the full fitted pipeline |

## 10. Checklist
- [ ] Model's scale sensitivity checked.
- [ ] Scaler chosen with the outliers/tails in mind.
- [ ] Inside the pipeline, after imputation/transformations.
- [ ] Fitted pipeline persisted as one object for inference.

## 11. References
- scikit-learn User Guide, *Standardization, or mean removal and variance scaling*. https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
- scikit-learn, *Compare the effect of different scalers on data with outliers*. https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
- scikit-learn, *Importance of Feature Scaling*. https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html
