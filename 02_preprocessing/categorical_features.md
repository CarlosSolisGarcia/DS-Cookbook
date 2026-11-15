# Categorical Variable Encoding

> **Folder:** 02_preprocessing · **Prev:** [numerical_features.md](numerical_features.md) · **Next:** [transformations.md](transformations.md) · **Related:** [data_leakage.md](../00_methodology/data_leakage.md), [19_advanced_tabular_ml](../19_advanced_tabular_ml/) (very high cardinality)

---

## 1. Purpose
Represent categorical variables (reactor, supplier, machine type, site) numerically so that models can use them **without** inventing false order, without exploding dimensionality and without leaking the target.

## 2. When to use it
Any feature whose values are labels rather than measurements, including numeric codes. Ordered categories (low/medium/high, quality grades) are a special case.

## 3. Conceptual foundation
| Encoding | How it works | Pros | Cons | Good for |
|---|---|---|---|---|
| **One-hot** | One binary column per level | No false order; interpretable coefficients | Many columns for high cardinality; rare levels are noisy | Linear models, low/medium cardinality (≲ 20–50 levels) |
| **Ordinal (integer codes)** | Level → integer | Compact | Imposes an arbitrary order (bad for linear models) | Tree models; truly ordered categories |
| **Native categorical** (HistGB, LightGBM, XGBoost, CatBoost) | The model partitions the levels into groups at each split | No expansion; handles cardinality well | Can overfit with many rare levels | Gradient boosting |
| **Target encoding** | Level → (shrunk) mean of the target for that level | One column; handles high cardinality | **Leaks** unless cross-fitted; can overfit | High cardinality, any model |
| **Frequency/count encoding** | Level → its frequency | Unsupervised, no leakage | Different levels with the same frequency collide | High cardinality, as an extra feature |
| **Grouping rare levels** | Levels with count < k → "other" | Reduces noise | Loses detail | Any, combined with the above |

**Target encoding mechanics.** For level *c*, encoding = weighted average of the level mean and the global mean. The weight increases with the level count (shrinkage), so rare levels are pulled towards the global mean. Computed naively on the training data, the encoding contains each row's own target, which leaks. scikit-learn's `TargetEncoder` uses **cross-fitting** in `fit_transform`. Each training row is encoded with statistics from the *other* folds. `transform` (for validation, test and production) uses statistics from all the training data.

## 4. Assumptions and limitations
- Unseen levels in production must be handled: `handle_unknown="ignore"` (one-hot), `unknown_value` (ordinal), or the global mean (target encoder).
- Target encoding assumes that a level's target mean is stable over time. With drift, encodings computed on old data go stale.
- When levels are **entities** that also define groups (e.g. `machine_id`), encoding them lets the model memorise entities. That is legitimate only if production predicts for the *same* entities ([data_leakage.md](../00_methodology/data_leakage.md)).

## 5. Step-by-step procedure
1. **List the categorical columns** with cardinality and level frequencies ([initial_inspection.md](../01_data_understanding/initial_inspection.md)).
2. **Clean the levels:** consistent case and whitespace, merge synonyms ("R-1" vs "R1"), decide what missing means (its own level "missing" is usually best).
3. **Check stability.** Do the levels in recent data match those in older data? Will new levels appear in production (new suppliers)?
4. **Pick the encoding by model family and cardinality** (§8).
5. **Implement inside the pipeline.** Validate against an alternative encoding.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

from cookbook_data import make_process_yield

df = make_process_yield().drop_duplicates()
df["temperature_c"] = df["temperature_c"].mask(df["temperature_c"] == -999)
df = df[df["yield_kg"] > 1].reset_index(drop=True)

cat = ["reactor", "catalyst", "shift", "supplier"]
num = ["temperature_c", "pressure_bar", "feed_rate_kg_h", "residence_time_min", "catalyst_age_days", "impurity_ppm"]
X, y = df[cat + num], np.log(df["yield_kg"])

print(df[cat].nunique())
print(df["supplier"].value_counts().tail(5))     # rare levels
```

```python
num_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
encoders = {
    "one-hot": OneHotEncoder(handle_unknown="ignore"),
    "one-hot, rare→other": OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30),
    "target (cross-fitted)": TargetEncoder(target_type="continuous", random_state=0),
    "ordinal (wrong for linear)": OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
}
cv = KFold(5, shuffle=True, random_state=42)
for name, enc in encoders.items():
    pre = ColumnTransformer([("num", num_pipe, num), ("cat", enc, cat)])
    model = make_pipeline(pre, RidgeCV(alphas=np.logspace(-3, 3, 13)))
    s = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
    print(f"ridge + {name:28s} MAE(log) = {-s.mean():.4f} ± {s.std():.4f}")
```

```python
# Gradient boosting with native categorical support: just use pandas 'category' dtype
Xc = X.astype({c: "category" for c in cat})
hgb = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=0)
s = cross_val_score(hgb, Xc, y, cv=cv, scoring="neg_mean_absolute_error")
print(f"HGB native categorical  MAE(log) = {-s.mean():.4f} ± {s.std():.4f}")

# LightGBM equivalent (also reads pandas 'category' dtype):
# import lightgbm as lgb; lgb.LGBMRegressor(min_data_per_group=50, cat_smooth=10).fit(Xc, y)
```

```python
# Demonstration: why NAIVE target encoding leaks
rng = np.random.default_rng(0)
noise_id = pd.Series(rng.integers(0, 500, len(df)).astype(str), name="noise_id")  # random high-card ID, no signal
y_arr = y.to_numpy()

naive_te = noise_id.map(pd.Series(y_arr).groupby(noise_id).mean())      # includes each row's own target
print("corr(naive TE of pure noise, y) =", round(np.corrcoef(naive_te, y_arr)[0, 1], 3))   # spuriously high

te = TargetEncoder(target_type="continuous", random_state=0)
cross_fitted = te.fit_transform(noise_id.to_frame(), y_arr).ravel()        # cross-fitting in fit_transform
print("corr(cross-fitted TE, y)      =", round(np.corrcoef(cross_fitted, y_arr)[0, 1], 3))  # ≈ 0
# NOTE: te.fit(X, y).transform(X) on the SAME rows would NOT cross-fit; use fit_transform for training data.
```

## 7. Interpretation
- **Low-cardinality features (reactor, catalyst, shift):** one-hot and target encoding perform similarly. One-hot keeps coefficients interpretable ("R3 reduces log-yield by …").
- **Ordinal encoding with a linear model** imposes R1 < R2 < R3 < R4 as if the levels were equally spaced numbers. Any good result is a coincidence of level order. Avoid it unless the categories really are ordered.
- **Native categorical HGB** is a strong default for boosting.
- **The naive-TE demonstration:** a pure-noise ID looks predictive, which is the leakage signature. Cross-fitting removes it.
- **Suspicious:** a high-cardinality ID-like column becoming the top feature after encoding. Is the model memorising entities? Check grouped CV.

## 8. Decision framework
| Situation | Recommendation |
|---|---|
| Linear/logistic model, ≤ ~20 levels | One-hot (`handle_unknown="ignore"`); group rare levels with `min_frequency` |
| Linear model, many levels (> ~50) | `TargetEncoder` (cross-fitted), possibly + frequency encoding |
| Gradient boosting, any cardinality | Native categorical (pandas `category` dtype); compare with `TargetEncoder` if cardinality is very high and levels are rare |
| Random forest | Ordinal encoding is acceptable (trees can isolate levels with several splits); one-hot for low cardinality |
| Truly ordered categories (grades, sizes) | `OrdinalEncoder(categories=[[...ordered...]])` |
| New levels expected in production | Encoders with unknown handling; monitor the rate of unseen levels |
| Entity IDs (machine_id, customer_id) | Usually **do not** encode; use them as groups. If production predicts on the same entities, target encoding with grouped CV can be legitimate |
| Category ⇄ a hierarchy (site → region) | Encode several levels of the hierarchy; the coarser levels help the rare fine levels ([19_advanced_tabular_ml](../19_advanced_tabular_ml/)) |

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Target encoding computed on the full data | Severe leakage, especially for rare levels | `TargetEncoder.fit_transform` in the pipeline |
| `te.fit(X,y).transform(X)` on training data | No cross-fitting → overfit | Use `fit_transform` for training rows (the pipeline does this) |
| Ordinal codes in linear models | Arbitrary order imposed | One-hot or target encoding |
| One-hot with thousands of levels | Huge sparse matrices, overfitting | Group rare levels, target or native encoding |
| Inconsistent level spelling | Same category split into several | Clean the levels first |
| `pd.get_dummies` separately on train and test | Column mismatch | `OneHotEncoder` in the pipeline |

## 10. Checklist
- [ ] Cardinality and level frequencies inspected; levels cleaned.
- [ ] Missing as an explicit level where meaningful.
- [ ] Encoding chosen by model family and cardinality.
- [ ] Unseen levels handled.
- [ ] Target encoding (if any) cross-fitted inside the pipeline.
- [ ] Entity identifiers treated as groups, not features (unless justified).
- [ ] Alternative encoding compared under CV.

## 11. References
- scikit-learn User Guide, *Encoding categorical features* and *Target Encoder*. https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features
- scikit-learn, *Categorical Feature Support in Gradient Boosting*. https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_categorical.html
- Micci-Barreca, D. (2001). "A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems." *ACM SIGKDD Explorations* 3(1), 27–32. https://doi.org/10.1145/507533.507538
- Pargent, F., Pfisterer, F., Thomas, J. & Bischl, B. (2022). "Regularized target encoding outperforms traditional methods in supervised machine learning with high cardinality features." *Computational Statistics* 37, 2671–2692. https://doi.org/10.1007/s00180-022-01207-6
- LightGBM docs, *Categorical Feature Support*. https://lightgbm.readthedocs.io/en/stable/Advanced-Topics.html#categorical-feature-support
