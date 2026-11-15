# High-Cardinality Categoricals, Extremely Wide Data and Small-Data Tabular ML

> **Folder:** 19_advanced_tabular_ml · **Next:** [distribution_shift.md](distribution_shift.md) · **Related:** [categorical_features.md](../02_preprocessing/categorical_features.md) (encoding basics: not repeated), [high_dimensional_data.md](../11_dimensionality_reduction/high_dimensional_data.md) (p ≫ n), [small_data_and_uncertainty.md](../18_scientific_machine_learning/small_data_and_uncertainty.md)

---

## 1. Purpose
Handle the tabular situations the basic recipes do not cover well: categorical variables with thousands of levels (part numbers, suppliers, sensor tags, postcodes), datasets with thousands of columns, and very small datasets.

## 2. When to use it
- A categorical feature has more than ~100 levels, or levels appear/disappear over time.
- p (columns) is in the thousands (wide sensor sets, aggregated logs, one-hot explosions, descriptor libraries).
- n is below a few hundred rows.

## 3. Conceptual foundation
**High-cardinality options** (beyond one-hot/target encoding):
| Technique | Idea | Pros | Cons |
|---|---|---|---|
| Cross-fitted **target encoding** (`TargetEncoder`) | Shrunk level means | Strong, compact | Leakage if not cross-fitted; drift-sensitive |
| **Count/frequency encoding** | The level's frequency | Unsupervised; handles new levels | Collisions |
| **Feature hashing** (`FeatureHasher`) | Hash the levels into m buckets | Fixed memory; streaming; new levels ok | Collisions; no interpretability |
| **Hierarchical encoding** | Use the parent levels (part family → part number) | Rare levels borrow strength | Needs a hierarchy |
| **Native categorical GBM** (LightGBM/CatBoost/HistGB) | Optimal partitions of the levels | Little preprocessing | Overfitting on rare levels → use `min_data_per_group`, `cat_smooth`, or HistGB's limits |
| **Entity embeddings** (NN) | Learned dense vectors per level | Captures similarity between levels | Needs data and an NN pipeline |

**Wide data (p ≫ n or p in the thousands):** regularised linear models, feature subsampling in GBMs (`colsample_bytree` / `max_features`), variance and correlation filtering *inside CV*, grouping columns by source and aggregating them (e.g. statistics of a sensor group), PCA/PLS ([high_dimensional_data.md](../11_dimensionality_reduction/high_dimensional_data.md)).

**Small data (n < ~300):** strong priors (domain features, monotonic constraints), simple models (regularised linear, shallow trees, GP), repeated or nested CV, and an uncertainty report. Avoid broad hyperparameter searches.

## 4. Assumptions and limitations
- Target and frequency encodings assume a level's meaning is stable over time. New parts or suppliers break them. Monitor the unseen-level rate.
- Hashing collisions are random. Their cost grows as the number of buckets falls relative to the active levels.

## 5. Step-by-step procedure
1. **Profile the cardinality:** levels, the frequency distribution (long tail?), the share of rare levels, the rate of new levels over time.
2. **Choose the encodings** (§8) and implement them inside a pipeline; compare 2–3 options with CV respecting time if the levels drift.
3. **For wide data:** remove constant/duplicate columns; group related columns; regularise; check the stability.
4. **For small data:** few, physically motivated features; simple models; repeated CV; report the spread.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder

# Spare-part lead times: 3000 orders, 800 part numbers (long tail) nested in 40 part families
rng = np.random.default_rng(0)
n, n_parts, n_fam = 3000, 800, 40
fam_of_part = rng.integers(0, n_fam, n_parts)
fam_effect, part_effect = rng.normal(0, 8, n_fam), rng.normal(0, 4, n_parts)
part = rng.choice(n_parts, n, p=(w := rng.pareto(1.2, n_parts) + 1) / w.sum())        # long-tailed popularity
qty = rng.lognormal(1, 0.8, n)
y = 20 + fam_effect[fam_of_part[part]] + part_effect[part] + 3 * np.log(qty) + rng.normal(0, 3, n)
df = pd.DataFrame({"part": "P" + pd.Series(part).astype(str), "family": "F" + pd.Series(fam_of_part[part]).astype(str),
                   "log_qty": np.log(qty), "lead_time_days": y})
counts = df["part"].value_counts()
print(f"{df['part'].nunique()} part numbers; {np.mean(counts < 5):.0%} of the parts have < 5 orders")

cv = KFold(5, shuffle=True, random_state=0)
def score(model, cols):
    s = -cross_val_score(model, df[cols], df["lead_time_days"], cv=cv, scoring="neg_mean_absolute_error")
    return f"{s.mean():.2f} ± {s.std():.2f}"

ridge = lambda pre: make_pipeline(pre, RidgeCV(alphas=np.logspace(-2, 3, 20)))
num = ["log_qty"]
variants = {
    "one-hot part": (ColumnTransformer([("p", OneHotEncoder(handle_unknown="ignore"), ["part"]), ("n", "passthrough", num)]), ["part"] + num),
    "target-encoded part": (ColumnTransformer([("p", TargetEncoder(random_state=0), ["part"]), ("n", "passthrough", num)]), ["part"] + num),
    "target-encoded part + family (hierarchy)": (ColumnTransformer([("p", TargetEncoder(random_state=0), ["part", "family"]),
                                                                   ("n", "passthrough", num)]), ["part", "family"] + num),
    "hashed part (256 buckets)": (ColumnTransformer([("p", make_pipeline(FunctionTransformer(lambda X: X.astype(str).to_numpy().tolist()),
                                                                         FeatureHasher(256, input_type="string")), ["part"]),
                                                     ("n", "passthrough", num)]), ["part"] + num),
}
for name, (pre, cols) in variants.items():
    print(f"ridge + {name:42s} MAE = {score(ridge(pre), cols)}")
hgb_df = df.assign(part=df["part"].astype("category"), family=df["family"].astype("category"))
try:                                                  # scikit-learn's HGB supports at most 255 categories per feature
    HistGradientBoostingRegressor(categorical_features="from_dtype").fit(hgb_df[["part", "family", "log_qty"]], df["lead_time_days"])
except ValueError as e:
    print("HGB:", str(e)[:90], "...")
import lightgbm as lgb                                # LightGBM's native categorical handling has no such limit
lgbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, min_data_per_group=20, cat_smooth=20, verbose=-1, random_state=0)
s = -cross_val_score(lgbm, hgb_df[["part", "family", "log_qty"]], df["lead_time_days"], cv=cv, scoring="neg_mean_absolute_error")
print(f"LightGBM native categorical (part + family)         MAE = {s.mean():.2f} ± {s.std():.2f}")
hgb_te = make_pipeline(ColumnTransformer([("p", TargetEncoder(random_state=0), ["part"]), ("rest", "passthrough", ["family", "log_qty"])]),
                       HistGradientBoostingRegressor(categorical_features=[1], random_state=0))
s = -cross_val_score(hgb_te, hgb_df[["part", "family", "log_qty"]], df["lead_time_days"], cv=cv, scoring="neg_mean_absolute_error")
print(f"HGB with target-encoded part + native family        MAE = {s.mean():.2f} ± {s.std():.2f}")
```

```python
# Wide data: 2000 columns, mostly irrelevant/redundant — regularisation + column-group aggregation
p_wide = 2000
W = rng.normal(size=(400, p_wide)); groups_of_cols = np.repeat(np.arange(100), 20)        # 100 sensor groups × 20 channels
latent = rng.normal(size=(400, 100)); W += 2 * latent[:, groups_of_cols]                  # channels within a group correlate
y_w = latent[:, 0] - latent[:, 1] + 0.5 * latent[:, 2] + rng.normal(0, 0.5, 400)
cvw = KFold(5, shuffle=True, random_state=0)
r = lambda X: -cross_val_score(make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 30))), X, y_w, cv=cvw,
                                scoring="neg_mean_absolute_error").mean()
W_grp = np.column_stack([W[:, groups_of_cols == g].mean(1) for g in range(100)])           # domain aggregation: group means
print(f"wide ridge on 2000 raw columns: MAE={r(W):.3f} | ridge on 100 group means: MAE={r(W_grp):.3f} | baseline std={y_w.std():.3f}")
```

## 7. Interpretation
- **Long tail:** 83% of the 592 part numbers have fewer than 5 orders. Any per-part estimate is noisy for them.
- **The hierarchy is what helps:** target-encoding the part alone (MAE ≈ 3.21) is slightly *worse* than one-hot + ridge (3.09). Rare parts get shrunk towards the *global* mean. Adding the **family** level (2.98) lets rare parts borrow their family's mean. That is the best linear option, and it is exactly what multilevel/hierarchical models do ([grouped_hierarchical_data.md](grouped_hierarchical_data.md)).
- **Hashing into 256 buckets** is clearly worse (4.52): ~590 levels collide in 256 buckets. Use more buckets (≥ several × the active levels) when hashing.
- **scikit-learn's HistGradientBoosting refuses > 255 categories** (categories are binned). LightGBM's native categorical handling works (3.43 here, regularised by `min_data_per_group`/`cat_smooth`). The best tree option is HGB with the **target-encoded part + native family** (2.94), the best result overall.
- **Wide data:** aggregating the 20 correlated channels of each group into a group mean gives a slightly lower MAE (0.495 vs 0.509) with 20× fewer features. The domain structure helps both accuracy and interpretability, even when ridge already handles the raw width reasonably well.

## 8. Decision framework
| Situation | Recommendation |
|---|---|
| > 100 levels, stable meaning | Cross-fitted target encoding (+ frequency); native categorical GBM |
| Natural hierarchy (family → part, region → site) | Encode several levels; parents help the rare children |
| New levels appear constantly / streaming | Hashing + frequency; monitor the unseen-level rate |
| Levels with semantic similarity (part descriptions) | Text embeddings of the descriptions as features ([embeddings_and_similarity.md](../12_natural_language_processing/embeddings_and_similarity.md)) |
| Thousands of correlated columns from known groups | Aggregate per group (domain) → model; or PLS/ridge |
| Thousands of heterogeneous columns | Filters inside CV + regularisation; GBM with column subsampling |
| n < 300 | Few domain features; regularised/simple models; repeated CV; uncertainty |

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Target encoding without cross-fitting | Leakage, especially for rare levels | `TargetEncoder.fit_transform` in the pipeline |
| One-hot with thousands of levels for trees | Memory blow-up, overfitting | Native categorical or encodings |
| Ignoring level drift over time | Stale encodings | Temporal validation; monitor new levels |
| Filtering wide columns on all data | Selection leakage | Filters inside CV |
| Complex models on tiny n | High-variance, irreproducible results | Simple models + strong priors |

## 10. Checklist
- [ ] Cardinality, tail and new-level rate profiled.
- [ ] 2–3 encodings compared (incl. hierarchy if available) under appropriate CV.
- [ ] Wide data: constant/duplicate removal; group aggregation; regularisation; selection inside CV.
- [ ] Small data: simple models, repeated CV, uncertainty reported.

## 11. References
- scikit-learn, `TargetEncoder`, `FeatureHasher`. https://scikit-learn.org/stable/modules/preprocessing.html#target-encoder · https://scikit-learn.org/stable/modules/feature_extraction.html#feature-hashing
- Pargent, F. et al. (2022). "Regularized target encoding outperforms traditional methods in supervised machine learning with high cardinality features." *Computational Statistics* 37. https://doi.org/10.1007/s00180-022-01207-6
- Weinberger, K. et al. (2009). "Feature Hashing for Large Scale Multitask Learning." *ICML*. https://arxiv.org/abs/0902.2206
- Guo, C. & Berkhahn, F. (2016). "Entity Embeddings of Categorical Variables." https://arxiv.org/abs/1604.06737
- Prokhorenkova, L. et al. (2018). "CatBoost: unbiased boosting with categorical features." *NeurIPS*. https://arxiv.org/abs/1706.09516
