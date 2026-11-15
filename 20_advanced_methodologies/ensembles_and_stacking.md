# Ensemble Learning and Stacking (scikit-learn)

> **Folder:** 20_advanced_methodologies · **Prev:** [transfer_learning_and_domain_adaptation.md](transfer_learning_and_domain_adaptation.md) · **Next:** [efficient_development.md](efficient_development.md) · **Related:** [advanced_improvement.md](../21_practical_projects/advanced_improvement.md) (OOF blending in competitions), [model_selection.md (regression)](../03_regression/model_selection.md) (bagging vs boosting)

---

## 1. Purpose
Combine models to reduce error and variance: bagging, boosting (already the core of GBMs), voting/averaging and stacking. Know when the gain is worth the extra complexity.

## 2. When to use it
When several reasonable models exist with different error patterns, when you need robustness to seeds/splits, and in competitions. In production, use it when the gain exceeds the cost of maintaining several models.

## 3. Conceptual foundation
| Technique | How | Reduces | scikit-learn |
|---|---|---|---|
| **Bagging** | The same model on bootstrap samples, averaged | Variance | `BaggingRegressor`, `RandomForest*` |
| **Boosting** | Models fitted sequentially on the residuals | Bias (and variance with shrinkage) | `HistGradientBoosting*`, `AdaBoost*` |
| **Seed averaging** | The same model with different random seeds | Variance of stochastic models | Loop + mean |
| **Voting / averaging** | Average the predictions/probabilities of different models | Variance; idiosyncratic errors | `VotingRegressor/Classifier` |
| **Stacking** | A meta-model learns how to combine the base predictions (fitted on OOF predictions) | Both, if the bases are complementary | `StackingRegressor/Classifier` |

**Why averaging works:** for m models with error variance σ² and pairwise error correlation ρ, the averaged error variance is ρσ² + (1 − ρ)σ²/m. Gains require ρ < 1 (diversity), and they saturate quickly as m grows.

**Stacking without leakage:** `StackingRegressor` trains the final estimator on **cross-validated** predictions of the base estimators (the `cv` parameter), then refits the bases on all data. Keep the meta-model simple (ridge/logistic, positive weights).

## 4. Assumptions and limitations
- If one model dominates and the others are much weaker, an ensemble can be worse than the best single model ([advanced_improvement.md](../21_practical_projects/advanced_improvement.md) showed this).
- Stacking adds a CV layer (compute) and is harder to explain and maintain.
- With time series/groups, use the same structured CV for the stacking folds.

## 5. Step-by-step procedure
1. Build 3–5 **diverse** base models (linear, GBM, RF, kNN/SVR on scaled features…), each reasonably tuned.
2. Check the correlation of their OOF errors (diversity).
3. Try a simple average (`VotingRegressor`) and stacking (`StackingRegressor` with `RidgeCV`) under the same outer CV.
4. Compare with the best single model (paired folds). Keep the ensemble only if it is clearly better.
5. Add seed averaging for stochastic models when the variance matters.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from cookbook_data import load_clean_process_yield

df = load_clean_process_yield()
def domain(d):
    d = d.copy(); d["log_feed"] = np.log(d["feed_rate_kg_h"]); d["log_res"] = np.log(d["residence_time_min"])
    d["temp_dev_sq"] = (d["temperature_c"] - d["catalyst"].map({"Pt": 190.0, "Pd": 182.0, "Ni": 195.0})) ** 2
    return d
D = domain(df)
num = ["log_feed", "log_res", "temp_dev_sq", "pressure_bar", "catalyst_age_days", "impurity_ppm", "temperature_c"]
cat = ["reactor", "catalyst", "supplier"]
X, y = D[num + cat], np.log(D["yield_kg"])
pre = ColumnTransformer([("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                         ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), cat)])
base = [("ridge", make_pipeline(pre, RidgeCV(alphas=np.logspace(-3, 3, 13)))),
        ("hgb", make_pipeline(pre, HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0))),
        ("rf", make_pipeline(pre, RandomForestRegressor(300, min_samples_leaf=3, max_features=0.5, random_state=0, n_jobs=-1))),
        ("knn", make_pipeline(pre, KNeighborsRegressor(15, weights="distance")))]
cv = KFold(5, shuffle=True, random_state=42)
oof = pd.DataFrame({n: cross_val_predict(m, X, y, cv=cv) for n, m in base})
err = oof.sub(y.to_numpy(), axis=0)
print("OOF MAE per model:", err.abs().mean().round(4).to_dict())
print("correlation of OOF errors (diversity):\n", err.corr().round(2))
```

```python
# Averaging vs stacking vs the best single model (same outer folds)
models = {name: m for name, m in base}
models["average (voting)"] = VotingRegressor(base)
models["stacking (ridge meta, inner cv=5)"] = StackingRegressor(base, final_estimator=RidgeCV(), cv=5, n_jobs=-1)
scores = {n: -cross_val_score(m, X, y, cv=cv, scoring="neg_mean_absolute_error") for n, m in models.items()}
print(pd.DataFrame({n: [s.mean(), s.std()] for n, s in scores.items()}, index=["MAE(log)", "std"]).T.round(4))
best_single = min(["ridge", "hgb", "rf", "knn"], key=lambda n: scores[n].mean())
for ens in ["average (voting)", "stacking (ridge meta, inner cv=5)"]:
    d = scores[best_single] - scores[ens]
    print(f"{ens} vs {best_single}: paired gain {d.mean():+.4f}, better in {(d > 0).sum()}/5 folds")
stack = models["stacking (ridge meta, inner cv=5)"].fit(X, y)
print("meta-model weights:", dict(zip([n for n, _ in base], stack.final_estimator_.coef_.round(3))))
```

```python
# Seed averaging for a stochastic model (variance reduction)
def rf(seed): return make_pipeline(pre, RandomForestRegressor(100, min_samples_leaf=3, max_features=0.5, random_state=seed, n_jobs=-1))
single = [-cross_val_score(rf(s), X, y, cv=cv, scoring="neg_mean_absolute_error").mean() for s in range(3)]
avg = VotingRegressor([(f"rf{s}", rf(s)) for s in range(3)])
print(f"RF single-seed MAE: {np.round(single, 4)} | 3-seed average: {-cross_val_score(avg, X, y, cv=cv, scoring='neg_mean_absolute_error').mean():.4f}")
```

## 7. Interpretation
- **Diversity:** ridge (domain features) and HGB are similarly strong (MAE(log) ≈ 0.126 vs 0.121), and their errors correlate only ≈ 0.62. That is the ideal setting for combining. RF and HGB correlate ≈ 0.80 (both trees). kNN is clearly weaker.
- **Plain averaging is slightly *worse* than HGB alone** (0.123 vs 0.121; better in only 1 of 5 folds). The weak kNN and RF members dilute the two good ones.
- **Stacking wins clearly:** 0.107, better than HGB in **all 5 folds** (a gain of 0.014 ≈ 12%). The meta-model puts ≈ 0.5 on ridge and ≈ 0.56 on HGB. It has learned that the linear "physics" model and the trees are complementary.
- **The negative weight on RF (−0.2)** means the meta-model uses RF as a *correction* term against HGB (they are correlated). It is legitimate under CV, but it is a sign of a flexible meta-model. With noisier data, constrain it (`LinearRegression(positive=True)` as the `final_estimator`) or drop redundant members.
- **Seed averaging:** single-seed RF MAEs vary (0.159–0.161), and the 3-seed average sits at the low end (≈ 0.159). Cheap variance reduction.

## 8. Decision framework
- **IF** the base models are similarly strong and their errors correlate < ~0.9 → averaging or stacking is likely to help.
- **IF** one model dominates → use it (or stacking, which will mostly select it). Avoid naive averaging.
- **IF** production simplicity matters → keep a single model unless the ensemble gain is clearly beyond the noise and worth the cost.
- **IF** temporal/grouped data → the stacking `cv` must be a temporal/grouped splitter (pass an iterable of splits).
- **IF** probabilities are needed (classification) → `StackingClassifier(stack_method="predict_proba")`; check the calibration.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Stacking on in-sample predictions | Leakage, an overconfident meta-model | `StackingRegressor` (internal CV) or manual OOF |
| Ensembling near-duplicates | Cost without gain | Check the error correlations |
| A complex meta-model | Overfitting the combination | Ridge/logistic, few parameters |
| Comparing ensemble vs single on different folds | Invalid conclusions | Same outer CV, paired comparison |

## 10. Checklist
- [ ] 3–5 diverse, reasonably tuned base models.
- [ ] OOF error correlations checked.
- [ ] Averaging and stacking vs the best single model on paired folds.
- [ ] Meta-weights inspected; complexity justified.
- [ ] Seed averaging for stochastic models.

## 11. References
- scikit-learn User Guide, *Ensembles* (bagging, voting, stacking). https://scikit-learn.org/stable/modules/ensemble.html
- Breiman, L. (1996). "Bagging Predictors." *Machine Learning* 24, 123–140.
- Wolpert, D. H. (1992). "Stacked generalization." *Neural Networks* 5(2), 241–259.
- Dietterich, T. G. (2000). "Ensemble Methods in Machine Learning." *MCS*, LNCS 1857.
- Zhou, Z.-H. (2012). *Ensemble Methods: Foundations and Algorithms*. CRC Press.
