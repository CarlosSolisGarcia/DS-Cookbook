# Feature Engineering (Tabular)

> **Folder:** 02_preprocessing · **Prev:** [scaling.md](scaling.md) · **Next:** [regression_workflow.md](../03_regression/regression_workflow.md) / [classification_workflow.md](../04_classification/classification_workflow.md) · **Related:** [ts feature_engineering.md](../05_time_series/feature_engineering.md), [feature_selection.md](../06_model_improvement/feature_selection.md), [18_scientific_machine_learning](../18_scientific_machine_learning/)

---

## 1. Purpose
Create features that make the relationship between the inputs and the target **easier to learn**, by encoding domain knowledge, aggregates, interactions and time context. In tabular problems, good features usually matter more than the choice of algorithm.

## 2. When to use it
- After EDA has suggested hypotheses (curvature, interactions, ratios, group effects).
- During error analysis, when residuals reveal a pattern a feature could explain.
- Less needed when the data are huge and the model is very flexible. Even then, domain features reduce the data needed and improve robustness.

## 3. Conceptual foundation
**Families of engineered features:**

| Family | Examples (cookbook data) | Why it helps |
|---|---|---|
| **Domain / physical** | Feed per batch = feed rate × duration; ΔT = jacket − reactor; residence time × temperature (severity); Arrhenius-like term exp(−Ea/RT) | Encodes known mechanisms; better extrapolation |
| **Ratios and differences** | Yield per kg feed; load / rated capacity; error count per operating hour | Removes scale effects; normalises by exposure |
| **Interactions** | Temperature × catalyst (catalyst-specific optimum); distance from the optimum per catalyst | Linear models cannot learn them otherwise |
| **Non-linear expansions** | Splines, polynomials, binning | Curvature for linear models |
| **Group aggregates** | Supplier mean impurity; machine's past failure count | Context from related rows; **leakage risk** |
| **Temporal context** | Days since maintenance; rolling vibration mean over 7 days; catalyst age | Captures degradation and state ([ts feature_engineering.md](../05_time_series/feature_engineering.md)) |
| **Date parts and cyclical** | Month, day of week, hour as sin/cos | Seasonality and calendar effects |
| **Missingness and quality flags** | `impurity_missing`, `sensor_replaced` | Informative missingness |

**Leakage in aggregates.** A group aggregate (e.g. "mean yield of this supplier") computed on all the data uses the target of validation and test rows. Aggregates of the **target** must be computed within the training folds (cross-fitted, like `TargetEncoder`). Aggregates of **features** over *past* data only are safe if computed as-of t₀.

## 4. Assumptions and limitations
- Each engineered feature must be computable at prediction time in production (same inputs, same latency).
- Many speculative features increase the risk of overfitting and of selecting noise. Prefer hypothesis-driven features, and validate them in groups.
- Physically motivated features often help linear models a lot and tree models a little. Trees approximate them, but less efficiently.

## 5. Step-by-step procedure
1. **Start from the EDA notes and domain knowledge.** List candidate features, each with a one-line hypothesis ("yield depends on distance from a catalyst-specific temperature optimum").
2. **Check availability at t₀** for each input involved.
3. **Implement** as pure functions (stateless) or as fitted transformers (if they need training statistics).
4. **Add the features in coherent groups** (e.g. "temperature features", "supplier context") and compare CV results against the current best.
5. **Keep a group** if it improves validation beyond the noise or improves robustness/interpretability. Otherwise drop it.
6. **Re-run error analysis.** Did the targeted errors decrease?

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from cookbook_data import make_process_yield

df = make_process_yield().drop_duplicates()
df["temperature_c"] = df["temperature_c"].mask(df["temperature_c"] == -999)
df = df[df["yield_kg"] > 1].sort_values("start_time").reset_index(drop=True)
train = df.iloc[: int(0.8 * len(df))].copy()
y = np.log(train["yield_kg"])

def domain_features(d: pd.DataFrame) -> pd.DataFrame:
    """Stateless, physically motivated features (computable at batch start)."""
    d = d.copy()
    d["log_feed_per_batch"] = np.log(d["feed_rate_kg_h"] * 8.0)          # multiplicative scale driver
    d["log_residence"] = np.log(d["residence_time_min"])
    # Distance from a catalyst-specific optimum: the optimum is a HYPOTHESIS from EDA
    # (in reality from lab data or vendor sheets).
    t_opt = d["catalyst"].map({"Pt": 190.0, "Pd": 182.0, "Ni": 195.0})
    d["temp_dev_sq"] = (d["temperature_c"] - t_opt) ** 2
    d["impurity_missing"] = d["impurity_ppm"].isna().astype(int)        # MNAR flag
    d["hour"] = d["start_time"].dt.hour
    d["hour_sin"], d["hour_cos"] = np.sin(2 * np.pi * d["hour"] / 24), np.cos(2 * np.pi * d["hour"] / 24)
    return d

base_num = ["temperature_c", "pressure_bar", "feed_rate_kg_h", "residence_time_min", "catalyst_age_days", "impurity_ppm"]
eng_num = ["pressure_bar", "log_feed_per_batch", "log_residence", "temp_dev_sq",
           "catalyst_age_days", "impurity_ppm", "impurity_missing"]
cat = ["reactor", "catalyst", "shift", "supplier"]

def ridge_on(num_cols):
    pre = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
                             ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat)])
    return make_pipeline(FunctionTransformer(domain_features), pre, RidgeCV(alphas=np.logspace(-3, 3, 13)))

cv = TimeSeriesSplit(5)
for name, cols in [("raw features", base_num), ("domain features", eng_num)]:
    s = cross_val_score(ridge_on(cols), train, y, cv=cv, scoring="neg_mean_absolute_error")
    print(f"ridge, {name:16s} MAE(log) = {-s.mean():.4f} ± {s.std():.4f}")

hgb_cols = base_num + cat
Xh = train[hgb_cols].astype({c: "category" for c in cat})
s = cross_val_score(HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=0),
                    Xh, y, cv=cv, scoring="neg_mean_absolute_error")
print(f"HGB,   raw features      MAE(log) = {-s.mean():.4f} ± {s.std():.4f}")
```

```python
# Group aggregates of FEATURES computed as-of t0 (past data only): safe
def supplier_history(d: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling mean impurity of the previous `window` batches from the same supplier (excludes current batch)."""
    d = d.sort_values("start_time")
    return (d.groupby("supplier")["impurity_ppm"]
              .transform(lambda s: s.shift(1).rolling(window, min_periods=3).mean()))

train["supplier_impurity_hist"] = supplier_history(train)
print(train[["start_time", "supplier", "impurity_ppm", "supplier_impurity_hist"]].dropna().head())

# Aggregates of the TARGET must be cross-fitted (use TargetEncoder) or strictly past-only, e.g.:
train["supplier_yield_hist"] = (train.sort_values("start_time").groupby("supplier")["yield_kg"]
                                .transform(lambda s: s.shift(1).expanding(min_periods=5).mean()))
```

```python
# Automatic interaction expansion for linear models (use sparingly, with regularisation)
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
interactions = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             PolynomialFeatures(degree=2, interaction_only=False, include_bias=False))
print(interactions.fit_transform(train[base_num]).shape)   # 6 → 27 columns: grows quadratically
```

## 7. Interpretation
- **Domain features closing the gap between ridge and HGB** is the typical outcome when the features encode the true mechanism. In this run, ridge with domain features (MAE(log) ≈ 0.12) **beats** gradient boosting on raw features (≈ 0.21) and the raw-feature ridge (≈ 0.43). The linear model becomes competitive, *and* it is interpretable and extrapolates better. This is the strongest argument in the cookbook for investing in domain knowledge before algorithms.
- **A feature group that improves CV within the noise:** it is not proven. Keep it only for interpretability or robustness reasons.
- **A feature that improves CV hugely:** check for leakage, especially aggregates involving the target.
- **Suspicious:** a past-only feature computed with `shift(1)` that still dominates the importance. Confirm that the sort order is correct and that no future row has slipped in.

## 8. Decision framework
| Signal | Feature idea |
|---|---|
| Relationship is multiplicative / scale-driven | Log of drivers; ratios (per unit feed, per hour) |
| Optimum / U shape | Squared deviation from a known/estimated optimum; splines |
| Effect differs by category | Interaction, or a category-specific term |
| Residuals trend with time | Age/degradation features; time since an event; regime flags |
| Errors concentrated in entities | Entity history features (past-only); hierarchical encodings |
| Many weak raw signals of one phenomenon | Combine them (an average, the first principal component, a physics index) |
| Linear model underfits but data are small | Engineer features rather than switching to a flexible model |

- **IF** a feature requires information available only after t₀ → do not use it.
- **IF** a feature is a target aggregate → cross-fit it or make it strictly past-only.
- **IF** you are tempted to generate hundreds of automatic features → do so only with strong regularisation and nested evaluation, and expect diminishing returns.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Target aggregates on the full data | Leakage | Cross-fitting or past-only windows |
| Features computed differently in training and production | Training–serving skew | One feature function used by both |
| Mass automatic feature generation | Overfitting, slow, uninterpretable | Hypothesis-driven groups |
| Ignoring units / physics | Missed the strongest features | Talk to domain experts; dimensional analysis |
| Evaluating each feature individually on the test set | Selection leakage | Evaluate groups with CV |

## 10. Checklist
- [ ] Candidate features listed, each with a hypothesis.
- [ ] Availability at t₀ checked for every input.
- [ ] Stateless features as functions; fitted ones as transformers in the pipeline.
- [ ] Target-based aggregates cross-fitted or past-only.
- [ ] Feature groups evaluated against the current best under CV.
- [ ] Error analysis repeated after adding features.

## 11. References
- Kuhn, M. & Johnson, K. (2019). *Feature Engineering and Selection*. http://www.feat.engineering/ (Ch. 5 categorical, Ch. 6 numeric, Ch. 7 interactions)
- Zheng, A. & Casari, A. (2018). *Feature Engineering for Machine Learning*. O'Reilly.
- scikit-learn, *Time-related feature engineering* (cyclical encoding, splines). https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html
- scikit-learn User Guide, *Polynomial and spline features*. https://scikit-learn.org/stable/modules/preprocessing.html#generating-polynomial-features
