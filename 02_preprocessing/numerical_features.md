# Numerical Features: Preprocessing Pipeline Design

> **Folder:** 02_preprocessing · **Prev:** [exploratory_data_analysis.md](../01_data_understanding/exploratory_data_analysis.md) · **Next:** [categorical_features.md](categorical_features.md) · **Related:** [transformations.md](transformations.md), [scaling.md](scaling.md), [missing_values.md](../01_data_understanding/missing_values.md)

---

## 1. Purpose
Turn raw numeric columns into model-ready inputs **reproducibly and without leakage**. This recipe is the hub for numeric preprocessing. It shows how to assemble imputation, transformation, scaling and derived features into a single `ColumnTransformer` + `Pipeline`, and which steps each model family actually needs.

## 2. When to use it
- Every supervised tabular project, once EDA has produced the decisions about each feature.
- When moving from notebook experiments to a reusable training script.

## 3. Conceptual foundation
**Stateless vs. fitted transformations.**
- *Stateless:* a fixed formula such as `log1p(x)`, `x1/x2` or unit conversion. Safe to apply anywhere, because no data statistics are involved. Usually done in a feature function or with `FunctionTransformer`.
- *Fitted:* the transformation learns statistics from data (imputer medians, scaler means and stds, Yeo-Johnson λ, quantile bins, spline knots). It **must** be fitted on training folds only, which means it goes inside the pipeline.

**What each model family needs:**

| Step | Linear / logistic / SVM / kNN / NN | Tree ensembles (RF, GBM) |
|---|---|---|
| Imputation | Required (no NaN support) | Optional (HistGB, LightGBM, XGBoost handle NaN natively; `RandomForest` supports NaN in sklearn ≥ 1.4) |
| Scaling | Required for regularisation, distances and gradient descent | Unnecessary (splits are invariant to monotonic transforms) |
| Skew transformation | Often helpful | Unnecessary for features (monotonic invariance) |
| Non-linear expansion (splines, polynomials) | Useful to capture curvature | Unnecessary |
| Discretisation (binning) | Occasionally (step-like effects); loses information | Unnecessary |

**Derived features** (ratios, differences, physical quantities) help *both* families. They encode knowledge that would otherwise have to be learnt from data ([feature_engineering.md](feature_engineering.md)).

## 4. Assumptions and limitations
- Column roles must be stable between training and production (same names and dtypes). Use DataFrames with `set_output(transform="pandas")` to keep column names traceable.
- `ColumnTransformer` drops unlisted columns by default (`remainder="drop"`). That is safe, but it can hide the fact that a column was silently ignored.
- Monotonic-invariance for trees holds for **feature** transformations, not for the target.

## 5. Step-by-step procedure
1. **Classify each numeric column** from the EDA notes: continuous, count, bounded/percentage, cyclical (hour, angle), identifier-like (drop), or a code (treat as categorical).
2. **Decide per column:** imputation (and indicator), transformation, scaling. Use the table in §8.
3. **Group the columns by treatment** and build one sub-pipeline per group.
4. **Assemble with `ColumnTransformer`**, together with the categorical pipelines ([categorical_features.md](categorical_features.md)).
5. **Wrap with the model in a `Pipeline`.** Cross-validate the whole object.
6. **Inspect the transformed output once.** Check shape, column names, the absence of NaN and the value ranges.
7. **Compare variants** (with/without transformation, different imputations) under CV. Keep only what helps.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PowerTransformer, SplineTransformer, StandardScaler

from cookbook_data import make_process_yield

df = make_process_yield().drop_duplicates()
df["temperature_c"] = df["temperature_c"].mask(df["temperature_c"] == -999)
df = df[df["yield_kg"] > 1].sort_values("start_time").reset_index(drop=True)
train, test = df.iloc[: int(0.8 * len(df))], df.iloc[int(0.8 * len(df)):]

# Column groups decided from EDA notes
skewed_pos = ["feed_rate_kg_h", "residence_time_min", "impurity_ppm"]   # right-skewed, > 0 → log
curved = ["temperature_c"]                                             # optimum → splines (for linear models)
plain = ["pressure_bar", "catalyst_age_days", "ambient_humidity_pct"]   # roughly symmetric
cat = ["reactor", "catalyst", "shift", "supplier"]

log_pipe = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                         StandardScaler())
spline_pipe = make_pipeline(SimpleImputer(strategy="median"),
                            SplineTransformer(n_knots=5, degree=3))       # flexible curve; SplineTransformer output need not be scaled
plain_pipe = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())

linear_preprocessor = ColumnTransformer(
    [("log", log_pipe, skewed_pos),
     ("spline", spline_pipe, curved),
     ("plain", plain_pipe, plain),
     ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), cat)],
    remainder="drop", verbose_feature_names_out=True,
).set_output(transform="pandas")

Xt = linear_preprocessor.fit_transform(train)
print(Xt.shape)
print(Xt.columns.tolist()[:12], "...")
assert not Xt.isna().any().any()
print(Xt.describe().T[["mean", "std", "min", "max"]].round(2).head(8))
```

> **Note on `add_indicator=True` inside a `FunctionTransformer(np.log1p)` pipeline.** The indicator columns (0/1) are also passed through `log1p`, which maps them to 0 and log 2. They are still valid binary indicators after scaling. If that bothers you, impute and add indicators in a separate branch. The simpler version is shown here.

```python
# Wrap with a model and evaluate. Compare against a version without the log/spline treatment.
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

y_train = train["yield_kg"]
cv = TimeSeriesSplit(n_splits=5)          # chronological: data drift over time

simple_preprocessor = ColumnTransformer(
    [("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), skewed_pos + curved + plain),
     ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat)])

for name, pre in [("simple", simple_preprocessor), ("log + spline", linear_preprocessor)]:
    model = Pipeline([("pre", pre), ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13)))])
    s = cross_val_score(model, train, y_train, cv=cv, scoring="neg_mean_absolute_error")
    print(f"{name:13s} MAE = {-s.mean():.1f} ± {s.std():.1f}")
```

```python
# Tree models: much leaner preprocessing (no scaling, no transforms, native NaN + categories)
from sklearn.ensemble import HistGradientBoostingRegressor

tree_cols = skewed_pos + curved + plain + cat
hgb = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=0)
X_tree = train[tree_cols].astype({c: "category" for c in cat})
s = cross_val_score(hgb, X_tree, y_train, cv=cv, scoring="neg_mean_absolute_error")
print(f"HGB (raw numerics) MAE = {-s.mean():.1f} ± {s.std():.1f}")
```

```python
# Stateless domain features: do them in a plain function (or FunctionTransformer) BEFORE the ColumnTransformer
def add_process_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["feed_per_batch_kg"] = d["feed_rate_kg_h"] * 8.0          # known batch duration
    d["delta_T_jacket"] = d["jacket_temperature_c"] - d["temperature_c"]
    return d

feature_step = FunctionTransformer(add_process_features)       # safe: no fitted statistics
```

## 7. Interpretation
- **Transformed output check.** The scaled columns should have mean ≈ 0 and std ≈ 1 *on the training data*. On test data they will deviate slightly; that is expected and correct. Indicator columns appear as `..._missingindicator_<col>`.
- **Comparing preprocessing variants.** If log + spline improves ridge clearly, the linear model needed help with curvature and skew. If HGB on raw features beats both, the remaining gap is in interactions, which call for explicit interaction features or a tree model.
- **Suspicious:** a large improvement from a preprocessing step that should be neutral, e.g. scaling for a tree model. Look for a bug, or for leakage in that step.

## 8. Decision framework
| Column type | Linear-family treatment | Tree-family treatment |
|---|---|---|
| Continuous, roughly symmetric | Impute (median) → scale | Raw |
| Continuous, positive, right-skewed | Impute → log/log1p (or Yeo-Johnson) → scale | Raw |
| Continuous with curvature/optimum | Splines (`SplineTransformer`) or polynomial terms | Raw |
| Counts (0, 1, 2, …) | log1p or sqrt → scale; or keep as-is if small range | Raw |
| Percentages / bounded [0, 1] | Scale; logit if near the bounds and relevant | Raw |
| Cyclical (hour, day-of-year, angle) | sin/cos encoding or periodic splines ([feature_engineering.md](feature_engineering.md)) | Raw (or sin/cos to help split) |
| Numeric codes (e.g. 1 = pump A) | Treat as categorical | Treat as categorical |
| IDs, row numbers | Drop (or use as groups) | Drop |
| Many missing + informative | Impute + indicator | Native NaN |

- **IF** you are using several model families → build **separate preprocessors** per family rather than a compromise.
- **IF** a preprocessing step does not improve CV beyond the noise → remove it (simpler, less fragile).

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Preprocessing with pandas on the whole dataset, then splitting | Leakage of fitted statistics | `Pipeline` + `ColumnTransformer` |
| Same heavy preprocessing for trees and linear models | Wasted effort, harder interpretation | Separate preprocessors per family |
| Losing track of column names | Interpretation impossible | `set_output(transform="pandas")`, `get_feature_names_out()` |
| `log(x)` with zeros | `-inf` / NaN | `log1p`, or an offset justified by the measurement resolution |
| Forgetting `handle_unknown="ignore"` | Crash on unseen categories in production | Always set it |
| Columns silently dropped by `remainder="drop"` | Missing features unnoticed | Print `get_feature_names_out()` once |

## 10. Checklist
- [ ] Each numeric column classified (continuous, count, bounded, cyclical, code, ID).
- [ ] Stateless features in a function; fitted steps in the pipeline.
- [ ] Separate preprocessors for linear-family and tree-family models.
- [ ] Transformed output inspected (shape, names, NaN, ranges).
- [ ] Variants compared under the project's validation scheme.
- [ ] `handle_unknown` and missing handling defined for production.

## 11. References
- scikit-learn User Guide, *Pipelines and composite estimators* (`Pipeline`, `ColumnTransformer`). https://scikit-learn.org/stable/modules/compose.html
- scikit-learn User Guide, *Preprocessing data*. https://scikit-learn.org/stable/modules/preprocessing.html
- scikit-learn, *set_output API*. https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_set_output.html
- Kuhn, M. & Johnson, K. (2019). *Feature Engineering and Selection*, Ch. 6 "Engineering Numeric Predictors". http://www.feat.engineering/
