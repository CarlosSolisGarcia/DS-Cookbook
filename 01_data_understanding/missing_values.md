# Missing Data Strategies

> **Folder:** 01_data_understanding · **Prev:** [data_quality.md](data_quality.md) · **Next:** [outliers.md](outliers.md) · **Related:** [numerical_features.md](../02_preprocessing/numerical_features.md), [ts preprocessing](../05_time_series/preprocessing.md) (missing timestamps)

---

## 1. Purpose
Understand **why** values are missing and choose a handling strategy that (a) does not bias the model, (b) does not leak, and (c) is available identically in production.

## 2. When to use it
- Whenever any feature or the target has missing values, including disguised ones (sentinels; see [data_quality.md](data_quality.md)).
- For time series with missing **timestamps** (absent rows), see [05_time_series/preprocessing.md](../05_time_series/preprocessing.md). The problem there is different.
- Missing **target** values: rows without a target cannot be used for supervised training. Investigate *why* they are missing (censoring, for example), because the reason may bias the training set.

## 3. Conceptual foundation
**Missingness mechanisms** (Rubin, 1976):

| Mechanism | Definition | Example (cookbook data) | Consequence |
|---|---|---|---|
| **MCAR**: missing completely at random | Missingness unrelated to any value | `temperature_c` random 2% dropouts | Dropping rows loses data but does not bias; any reasonable imputation works |
| **MAR**: missing at random | Missingness depends on **observed** variables | `ambient_humidity_pct` missing mostly for reactor R4 | Imputation conditional on the observed variables (e.g. reactor) is unbiased; global-mean imputation is biased |
| **MNAR**: missing not at random | Missingness depends on the **unobserved value itself** | `impurity_ppm` missing when high (the sensor saturates) | No imputation recovers the truth. The *fact* of missingness is informative, so use a missing indicator |

You cannot prove MAR vs MNAR from the data alone. You use evidence plus domain knowledge (how the sensor or process works).

**For prediction (as opposed to inference),** the goal is not to recover the true values. It is to give the model the best usable information. Two practical facts follow:
- A **missing indicator** lets the model learn that "missing" itself predicts something. This handles MNAR pragmatically.
- **Tree models with native NaN support** (HistGradientBoosting, LightGBM, XGBoost) learn at each split which branch missing values go to. Often no imputation is needed.

## 4. Assumptions and limitations
- Imputation is a **fitted** transformation (it uses training statistics). It must go inside the pipeline ([data_leakage.md](../00_methodology/data_leakage.md)).
- Missingness patterns in production must match training. If the sensor is fixed in production, a model that learned from "missing" will behave differently.
- `IterativeImputer` (MICE-style) is still marked **experimental** in scikit-learn. It requires `from sklearn.experimental import enable_iterative_imputer`. It is slower and can overfit with many features, so use it when relationships between features are strong and the missing fraction is substantial.
- Imputation for statistical *inference* (correct standard errors) requires multiple imputation. That is beyond this cookbook's prediction focus; see van Buuren (2018).

## 5. Step-by-step procedure
1. **Quantify.** Missing % per column, per row, and the co-occurrence of missingness across columns.
2. **Find the pattern.** Is missingness associated with other features (MAR) or with time? Compare feature distributions for rows with and without missing values.
3. **Check association with the target.** Compare the target distribution between missing and observed rows. A large difference means the missingness is informative, so add an indicator.
4. **Ask the domain expert about the mechanism.** Sensor saturation, values not measured unless suspicious, a system not yet installed, and so on.
5. **Choose a strategy per column** (§8). Implement it inside a pipeline.
6. **Validate.** Compare 2–3 strategies under CV. The differences are often small; pick the simplest adequate one.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cookbook_data import make_process_yield

df = make_process_yield().drop_duplicates()
df["temperature_c"] = df["temperature_c"].mask(df["temperature_c"] == -999)
df = df[df["yield_kg"] > 1].reset_index(drop=True)

# 1. Quantify
miss = df.isna().mean().sort_values(ascending=False)
print((100 * miss[miss > 0]).round(2))

# Co-occurrence of missingness (correlation of missing indicators)
ind = df[miss[miss > 0].index].isna().astype(int)
print(ind.corr().round(2))
```

```python
# 2. MAR check: is missingness related to observed variables?
def missingness_profile(df, col, by):
    return df.groupby(by, observed=True)[col].apply(lambda s: s.isna().mean()).round(3)

print(missingness_profile(df, "ambient_humidity_pct", "reactor"))   # concentrated in R4 → MAR on reactor
print(missingness_profile(df, "impurity_ppm", "supplier").sort_values().tail(4))

# 3. Informative missingness: does the target differ between missing and observed?
for col in ["temperature_c", "impurity_ppm", "ambient_humidity_pct"]:
    g = df.groupby(df[col].isna())["yield_kg"].median()
    print(f"{col:22s} median yield observed={g.get(False):.1f}  missing={g.get(True):.1f}")
```

For `impurity_ppm` you should see a *lower* median yield when the value is missing. High impurity lowers yield, and high impurity is what goes missing (MNAR). An indicator is therefore valuable.

```python
# 4. Strategies compared inside pipelines (no leakage)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (required import)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

num = ["temperature_c", "pressure_bar", "feed_rate_kg_h", "residence_time_min",
       "catalyst_age_days", "impurity_ppm", "ambient_humidity_pct"]
cat = ["reactor", "catalyst", "shift"]
X, y = df[num + cat], np.log(df["yield_kg"])
cv = KFold(5, shuffle=True, random_state=42)

def linear_model(imputer):
    pre = ColumnTransformer([("num", make_pipeline(imputer, StandardScaler()), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    return make_pipeline(pre, RidgeCV(alphas=np.logspace(-3, 3, 13)))

strategies = {
    "median": SimpleImputer(strategy="median"),
    "median + indicator": SimpleImputer(strategy="median", add_indicator=True),
    "KNN (k=10) + indicator": KNNImputer(n_neighbors=10, add_indicator=True),
    "iterative + indicator": IterativeImputer(max_iter=10, random_state=0, add_indicator=True),
}
for name, imp in strategies.items():
    s = cross_val_score(linear_model(imp), X, y, cv=cv, scoring="neg_mean_absolute_error")
    print(f"{name:25s} MAE(log)={-s.mean():.4f} ± {s.std():.4f}")

# Native NaN handling in gradient boosting (categoricals as pandas 'category' dtype)
Xc = X.astype({c: "category" for c in cat})
hgb = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=0)
s = cross_val_score(hgb, Xc, y, cv=cv, scoring="neg_mean_absolute_error")
print(f"{'HGB native NaN':25s} MAE(log)={-s.mean():.4f} ± {s.std():.4f}")
```

> **Note on KNNImputer:** it computes distances on the *unscaled* features passed to it, so put a scaler **before** it if the scales differ. `StandardScaler` ignores NaN when fitting and keeps NaN in its output, so `make_pipeline(StandardScaler(), KNNImputer())` works. The example above keeps the simpler order for brevity. Try the scaled order as an exercise.

```python
# 5. Group-wise (MAR-aware) imputation as a simple, transparent alternative:
# impute humidity with the training-set median of the same month (humidity is seasonal).
from sklearn.base import BaseEstimator, TransformerMixin

class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Impute `col` with the training median within `group_col`; fall back to the global median."""
    def __init__(self, col, group_col):
        self.col, self.group_col = col, group_col
    def fit(self, X, y=None):
        self.medians_ = X.groupby(self.group_col, observed=True)[self.col].median()
        self.global_ = X[self.col].median()
        return self
    def transform(self, X):
        X = X.copy()
        fill = X[self.group_col].map(self.medians_).astype(float).fillna(self.global_)
        X[self.col] = X[self.col].fillna(fill)
        return X

tmp = df.assign(month=df["start_time"].dt.month)
print(GroupMedianImputer("ambient_humidity_pct", "month").fit(tmp).transform(tmp)["ambient_humidity_pct"].isna().sum())
```

## 7. Interpretation
- **The large gap between the linear pipelines and HGB** is *not* about imputation. It comes from the non-linear effects (temperature optimum, interactions) that the plain linear model cannot represent. Compare imputation strategies **within** a model family.
- **Small differences between imputation strategies** (within the CV std) are the norm for prediction. Do not over-invest; prefer the simplest.
- **A missing indicator that improves the score** means missingness is informative (MAR/MNAR). Document that the model depends on the missingness pattern, because it must remain stable in production.
- **Native NaN boosting doing as well as the imputation pipelines** is common. Imputation is then unnecessary for that model family.
- **Suspicious:** the missing indicator of a feature becoming one of the most important features. Ask whether the missingness is caused by the outcome. Example: a lab test ordered only *after* the problem was suspected. That is leakage via missingness.

## 8. Decision framework
| Situation | Recommended approach |
|---|---|
| < ~5% missing, MCAR plausible | Median (numeric) / most-frequent or "missing" category (categorical) inside the pipeline |
| Missingness informative (target differs; MNAR suspected) | Imputation **+ missing indicator** (`add_indicator=True`) |
| MAR on known variables (by reactor, by month) | Group-wise imputation, KNN or iterative imputation, or let a tree model handle it |
| Tree-based gradient boosting | Try native NaN handling first, with no imputer |
| Linear / distance / NN models | Must impute (they do not accept NaN); add indicators |
| > ~40–60% missing in a column | Ask whether the column is useful at all. Indicator only, or drop. Is it missing in production too? |
| Missing **target** | Exclude from training. Check that the exclusion does not bias the training set (e.g. censored outcomes). |
| Missingness structurally "not applicable" | Encode as a separate category/flag, not imputed |
| Time series values | Use past-only methods (forward-fill, interpolation within the training window); see [ts preprocessing](../05_time_series/preprocessing.md) |

**IF** you are unsure whether imputation helps **THEN** compare: (a) median, (b) median + indicator, (c) native NaN trees. Keep the simplest one within the noise.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| `df.fillna(df.mean())` before splitting | Leakage of test statistics | Impute inside the pipeline |
| `dropna()` on all rows | Data loss + selection bias (MNAR rows removed) | Drop only if MCAR and the loss is negligible |
| Imputing sentinels | Wrong values treated as real | Convert sentinels to NaN first |
| Mean imputation of skewed variables | Imputed value is unrepresentative | Median, or impute on a transformed scale |
| KNN imputation on unscaled features | Distances dominated by large-scale variables | Scale before KNNImputer |
| Ignoring production missingness | Model behaves differently when sensors change | Monitor missing rates; document the dependence |

## 10. Checklist
- [ ] Missing % per column and co-occurrence computed.
- [ ] Sentinels converted to NaN beforehand.
- [ ] Missingness related to other features (MAR check) and to the target (informativeness).
- [ ] Mechanism discussed with a domain expert.
- [ ] Strategy chosen per column; indicators where informative.
- [ ] Imputation inside the pipeline; 2–3 strategies compared under CV.
- [ ] Production missingness pattern documented and monitored.

## 11. References
- Rubin, D. B. (1976). "Inference and missing data." *Biometrika* 63(3), 581–592. https://doi.org/10.1093/biomet/63.3.581
- van Buuren, S. (2018). *Flexible Imputation of Missing Data*, 2nd ed. CRC Press. Free online: https://stefvanbuuren.name/fimd/ (Ch. 1–2 for the mechanisms)
- scikit-learn User Guide, *Imputation of missing values* (SimpleImputer, IterativeImputer, KNNImputer, MissingIndicator). https://scikit-learn.org/stable/modules/impute.html
- scikit-learn, *HistGradientBoosting: missing values support*. https://scikit-learn.org/stable/modules/ensemble.html#missing-values-support
- Josse, J., Prost, N., Scornet, E. & Varoquaux, G. (2019). "On the consistency of supervised learning with missing values." https://arxiv.org/abs/1902.06931
