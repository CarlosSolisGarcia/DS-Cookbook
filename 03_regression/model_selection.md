# Regression Model Selection: Linear, Regularised, Trees and Boosting

> **Folder:** 03_regression · **Prev:** [baseline_models.md](baseline_models.md) · **Next:** [evaluation.md](evaluation.md) · **Related:** [modeling_fundamentals.md](../00_methodology/modeling_fundamentals.md), [hyperparameter_optimization.md](../06_model_improvement/hyperparameter_optimization.md), [04 model_selection](../04_classification/model_selection.md) (the classification counterpart)

---

## 1. Purpose
Choose candidate model families for a regression problem, understand what each one assumes and how it fails, configure it sensibly, and compare the candidates fairly.

## 2. When to use it
After the baselines, once preprocessing pipelines exist. Return here when error analysis suggests the model family is the bottleneck (e.g. a linear model with structured residuals).

## 3. Conceptual foundation

### 3.1 Families at a glance
| Family | Captures | Key hyperparameters | Strengths | Weaknesses |
|---|---|---|---|---|
| **OLS** (`LinearRegression`) | Additive linear effects | — | Interpretable; extrapolates linearly | Unstable with collinearity; no non-linearity unless engineered |
| **Ridge** (L2) | Same, coefficients shrunk | `alpha` | Stable under collinearity; good with many correlated features | Keeps all features |
| **Lasso** (L1) | Same, sparse | `alpha` | Feature selection | Arbitrary choice among correlated features; unstable selection |
| **ElasticNet** | L1 + L2 | `alpha`, `l1_ratio` | Sparse *and* stable with correlated groups | Two parameters |
| **Huber / quantile linear** | Robust or quantile linear | `epsilon` / `quantile` | Robust to outliers | Linear |
| **Splines / GAM-like** (`SplineTransformer` + Ridge) | Smooth non-linear additive effects | knots, `alpha` | Interpretable curves | Interactions must be explicit |
| **Decision tree** | Piecewise-constant, interactions | `max_depth`, `min_samples_leaf` | Readable rules | High variance; step-like predictions |
| **Random Forest** | Averaged deep trees (bagging) | `n_estimators`, `max_features`, `min_samples_leaf` | Robust defaults; little tuning | Larger models; weaker than GBM on many tabular tasks; no extrapolation |
| **Gradient boosting** (HistGB, LightGBM, XGBoost) | Additive sequence of shallow trees | `learning_rate`, `max_iter`/`n_estimators`, `max_leaf_nodes`/`num_leaves`, `min_samples_leaf`, regularisation | State of the art on many tabular problems; NaN + categorical handling | Needs tuning and early stopping; no extrapolation |
| **kNN, SVR, Gaussian process** | Local similarity / kernels | `k`; `C`, `gamma`; kernel | GP: uncertainty, great for small smooth data | Scale poorly with n; GP needs kernel design ([18_scientific_ml](../18_scientific_machine_learning/)) |

### 3.2 Regularisation intuition
Ridge minimises squared error + α·Σβ². With correlated features (temperature and jacket temperature), OLS can assign +50 to one and −49 to the other. Ridge shares the effect between them in a stable way. Lasso (α·Σ|β|) pushes some coefficients exactly to zero. The penalty is meaningful only if the features are **scaled** ([scaling.md](../02_preprocessing/scaling.md)).

### 3.3 Bagging vs boosting
- **Random forest** averages many deep, decorrelated trees, which reduces variance. More trees never hurt accuracy; they only cost time.
- **Gradient boosting** adds shallow trees sequentially, each fitting the residual errors of the ensemble so far, which reduces bias. Too many iterations overfit, so control the number with **early stopping** or CV. A lower learning rate needs more iterations, and usually generalises slightly better.

### 3.4 Which GBM library?
| Library | When to prefer it |
|---|---|
| `HistGradientBoostingRegressor` (scikit-learn) | Default: no extra dependency, native NaN and categorical support, early stopping, monotonic constraints, quantile loss |
| LightGBM | Large data (fast), many tuning options, `num_leaves`-based growth; very common in competitions |
| XGBoost | Mature, GPU support, a wide range of objectives |
| CatBoost (not used here) | Many categorical features (ordered target statistics) |

The three main libraries usually perform similarly when tuned. Tuning and features matter more than the library.

## 4. Assumptions and limitations
- **Tree-based models do not extrapolate.** Outside the training range they predict roughly the value at the edge of the range. Linear models extrapolate linearly, which may or may not be physically correct.
- Coefficient interpretation in linear models requires care with collinearity and scaling ([model_interpretation.md](../06_model_improvement/model_interpretation.md)).
- Defaults are reasonable for HGB/RF. For LightGBM/XGBoost the defaults are less conservative (e.g. a high number of iterations without early stopping).

## 5. Step-by-step procedure
1. **Start linear** (RidgeCV), with the preprocessing from [numerical_features.md](../02_preprocessing/numerical_features.md). It is an interpretable reference.
2. **Add a GBM with defaults and early stopping** (HistGB). It tests how much non-linearity and interaction is left.
3. **Optionally add a random forest**, a robust reference that needs little tuning.
4. **Compare on identical folds** with train and validation scores ([modeling_fundamentals.md](../00_methodology/modeling_fundamentals.md) §6.1).
5. **Diagnose** each model's fit (under/overfit) and look at its residuals ([error_analysis.md](error_analysis.md)).
6. **Shortlist 1–2 models**, then tune them with a fixed budget ([hyperparameter_optimization.md](../06_model_improvement/hyperparameter_optimization.md)).
7. **Consider constraints:** monotonicity (physics), extrapolation needs, inference latency, interpretability.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from cookbook_data import load_clean_process_yield

df = load_clean_process_yield()
train = df.iloc[: int(0.8 * len(df))]
num = ["temperature_c", "jacket_temperature_c", "pressure_bar", "feed_rate_kg_h", "residence_time_min",
       "catalyst_age_days", "impurity_ppm", "feed_purity_pct", "ambient_humidity_pct"]
cat = ["reactor", "catalyst", "shift", "supplier"]
X, y = train[num + cat], np.log(train["yield_kg"])     # log target (see transformations.md)

lin_pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), cat)])
tree_pre = ColumnTransformer([
    ("num", "passthrough", num),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), cat)])
cat_idx = list(range(len(num), len(num) + len(cat)))      # positions of categoricals after tree_pre

models = {
    "ridge": make_pipeline(lin_pre, RidgeCV(alphas=np.logspace(-3, 3, 25))),
    "lasso": make_pipeline(lin_pre, LassoCV(alphas=np.logspace(-5, -1, 40), cv=5, random_state=0, max_iter=20000)),
    "elasticnet": make_pipeline(lin_pre, ElasticNetCV(l1_ratio=[.2, .5, .8], cv=5, random_state=0, max_iter=20000)),
    "random_forest": make_pipeline(
        tree_pre, SimpleImputer(strategy="median"),     # RF in sklearn ≥1.4 accepts NaN, imputing keeps it portable
        RandomForestRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=0)),
    "hgb_early_stop": make_pipeline(tree_pre, HistGradientBoostingRegressor(
        categorical_features=cat_idx, learning_rate=0.05, max_iter=2000, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=50, random_state=0)),
}
cv = TimeSeriesSplit(5)
rows = []
for name, m in models.items():
    r = cross_validate(m, X, y, cv=cv, scoring="neg_mean_absolute_error", return_train_score=True, n_jobs=-1)
    rows.append({"model": name, "train_MAE": -r["train_score"].mean(), "val_MAE": -r["test_score"].mean(),
                 "val_std": r["test_score"].std(), "fit_s": r["fit_time"].mean()})
print(pd.DataFrame(rows).set_index("model").round(4))
```

```python
# Collinearity in action: OLS vs ridge coefficients for the two near-duplicate temperatures
from sklearn.linear_model import LinearRegression
Z = train[["temperature_c", "jacket_temperature_c", "pressure_bar"]].dropna()
yz = np.log(train.loc[Z.index, "yield_kg"])
Zs = (Z - Z.mean()) / Z.std()
for name, m in [("OLS", LinearRegression()), ("Ridge α=10", RidgeCV(alphas=[10.0]))]:
    coefs = []
    for seed in range(5):                                  # refit on bootstrap samples → coefficient stability
        idx = np.random.default_rng(seed).choice(len(Zs), len(Zs))
        coefs.append(m.fit(Zs.iloc[idx], yz.iloc[idx]).coef_)
    print(name, np.round(np.array(coefs), 3).tolist())
```

```python
# Lasso path: which features survive? (unstable among correlated features: compare across folds)
lasso = models["lasso"].fit(X, y)
names = lasso[0].get_feature_names_out()
coef = pd.Series(lasso[-1].coef_, index=names)
print("alpha:", round(lasso[-1].alpha_, 5), " non-zero:", (coef != 0).sum(), "of", len(coef))
print(coef[coef != 0].sort_values(key=abs, ascending=False).head(10).round(3))
```

```python
# LightGBM / XGBoost equivalents (same pipeline position as HGB). Use early stopping on a validation split.
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split

Xc = X.astype({c: "category" for c in cat})
X_tr, X_va, y_tr, y_va = train_test_split(Xc, y, test_size=0.2, shuffle=False)   # chronological inner split

lgbm = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                         random_state=0, verbose=-1)
lgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="l1",
         callbacks=[lgb.early_stopping(100, verbose=False)])
print("LightGBM best iteration:", lgbm.best_iteration_)

xgbm = xgb.XGBRegressor(n_estimators=3000, learning_rate=0.03, max_depth=6, subsample=0.8,
                        colsample_bytree=0.8, enable_categorical=True, tree_method="hist",
                        early_stopping_rounds=100, eval_metric="mae", random_state=0)
xgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
print("XGBoost best iteration:", xgbm.best_iteration)
```

```python
# Physics-informed constraint: yield should not DECREASE with residence time (all else equal)
cons = {"residence_time_min": 1}                      # +1 increasing, -1 decreasing, 0 none
hgb_mono = HistGradientBoostingRegressor(categorical_features="from_dtype",
                                         monotonic_cst=cons, random_state=0)
r = cross_validate(hgb_mono, Xc, y, cv=cv, scoring="neg_mean_absolute_error")
print("HGB with monotonic constraint, val MAE(log):", round(-r["test_score"].mean(), 4))
```

> **Mistake to avoid:** when using early stopping for the final model, the number of iterations is chosen on an internal validation split. In CV, `HistGradientBoostingRegressor(early_stopping=True)` does this inside each fold automatically. For LightGBM/XGBoost, either use their CV utilities or fix `n_estimators` to the value found and refit on all the training data.

## 7. Interpretation
- **Linear variants (ridge ≈ lasso ≈ elastic net)** usually give similar validation errors. Choose by purpose: ridge for stability, lasso for sparsity.
- **GBM ≪ linear on raw features** means non-linearity and interactions matter. With the domain features from [feature_engineering.md](../02_preprocessing/feature_engineering.md), the gap shrinks or reverses.
- **Random forest's train MAE is much lower than its validation MAE.** That is normal for RF (deep trees). Judge RF by its validation score.
- **Coefficient stability:** OLS coefficients for the two temperatures swing wildly across bootstrap samples, while ridge's stay stable. Collinearity harms *interpretation*, not necessarily prediction.
- **Lasso keeps one of a pair of correlated features** and drops the other. Which one it keeps can change with the data sample, so do not read the selection as "the other one doesn't matter".
- **Monotonic constraint:** if the score is unchanged or slightly better, keep the constraint. You gain physical plausibility and robustness at no cost.

## 8. Decision framework
| Situation | Recommended first choice | Then try |
|---|---|---|
| Small n (< ~500), few features | Ridge (+ splines/domain features) | GBM with strong regularisation; Gaussian process |
| Medium/large n, mixed types, non-linear | HistGB with early stopping | LightGBM/XGBoost tuned; ensemble with a linear model |
| p ≫ n or many correlated features | Ridge / ElasticNet | PLS, PCA + ridge ([11_dimensionality_reduction](../11_dimensionality_reduction/)) |
| Interpretability required | Ridge with engineered features; GAM-like splines | GBM + SHAP as a check |
| Extrapolation beyond the training range expected | Linear / physics-based model | Hybrid: physics + ML residual ([18_scientific_ml](../18_scientific_machine_learning/)) |
| Known monotone physics | GBM with `monotonic_cst` | — |
| Heavy outliers in the target | Huber / `loss="absolute_error"` | Quantile models |

- **IF** GBM and linear are within noise → choose linear.
- **IF** GBM overfits (a growing train–val gap with worse val) → lower `max_leaf_nodes`, raise `min_samples_leaf`, add `l2_regularization`, lower `learning_rate` with early stopping.
- **IF** LightGBM/XGBoost does not beat HistGB after tuning → keep HistGB (fewer dependencies).

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Unscaled features with ridge/lasso | Penalty depends on units | Scale in the pipeline |
| GBM without early stopping or iteration tuning | Over/underfitting | `early_stopping=True` or tune `max_iter` |
| Interpreting lasso's zeroed features as irrelevant | Wrong scientific conclusions | Check stability; prefer ridge or grouped analysis |
| Comparing a tuned GBM to a default linear model | Biased comparison | Equal tuning effort |
| Using tree models for extrapolation | Flat, wrong predictions outside the range | Check the production input ranges |
| Ordinal-encoding categories for linear models | Arbitrary order | One-hot / target encoding |

## 10. Checklist
- [ ] Linear reference model with proper preprocessing.
- [ ] GBM with early stopping; RF optional.
- [ ] Identical folds; train and validation scores recorded.
- [ ] Fit diagnosis per model; residuals checked.
- [ ] Constraints considered (monotonicity, extrapolation, latency, interpretability).
- [ ] Shortlist justified by validation differences beyond the noise.

## 11. References
- scikit-learn User Guide, *Linear Models* (Ridge, Lasso, ElasticNet, Huber, Quantile). https://scikit-learn.org/stable/modules/linear_model.html
- scikit-learn User Guide, *Ensembles: Gradient boosting, random forests, bagging…* (HistGradientBoosting, monotonic constraints, early stopping). https://scikit-learn.org/stable/modules/ensemble.html
- Breiman, L. (2001). "Random Forests." *Machine Learning* 45, 5–32. https://doi.org/10.1023/A:1010933404324
- Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine." *Annals of Statistics* 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451
- Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *KDD*. https://doi.org/10.1145/2939672.2939785 · Docs: https://xgboost.readthedocs.io/
- Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *NeurIPS 30*. · Docs, *Parameters Tuning*: https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html
- Zou, H. & Hastie, T. (2005). "Regularization and variable selection via the elastic net." *JRSS B* 67(2), 301–320.
