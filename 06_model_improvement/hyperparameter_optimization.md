# Hyperparameter Optimisation

> **Folder:** 06_model_improvement · **Prev:** [cross_validation.md](cross_validation.md) · **Next:** [feature_selection.md](feature_selection.md) · **Related:** [regression model_selection.md](../03_regression/model_selection.md), [classification model_selection.md](../04_classification/model_selection.md), [20_advanced_methodologies](../20_advanced_methodologies/) (efficient development with limited compute)

---

## 1. Purpose
Find good hyperparameter values efficiently, without overfitting the validation data, and know when tuning is no longer worth the effort.

## 2. When to use it
After a model family has been shortlisted and the features are reasonably mature. Tuning is usually a **smaller** lever than features and data. Do it after those, and repeat lightly when the features change substantially.

## 3. Conceptual foundation
| Strategy | How | Strengths | Weaknesses |
|---|---|---|---|
| **Manual / informed defaults** | Expert starting values, a few changes | Cheap; builds intuition | Not systematic |
| **Grid search** (`GridSearchCV`) | Every combination of listed values | Exhaustive on a small grid; reproducible | Cost grows exponentially; wastes trials on unimportant parameters |
| **Random search** (`RandomizedSearchCV`) | Sample from distributions | Much more efficient than grid when only a few parameters matter (Bergstra & Bengio, 2012) | Unguided |
| **Successive halving** (`HalvingRandomSearchCV`) | Many configs on little data/iterations; keep the best fraction | Very efficient for large spaces | Early rounds are noisy; the resource must be meaningful (n_samples, n_iter) |
| **Bayesian / TPE** (Optuna) | Model the score as a function of the hyperparameters; sample promising regions; prune bad trials | Efficient for expensive models with 3–10 parameters | Extra dependency; sequential |

**What matters most for gradient boosting:** learning rate × number of iterations (use early stopping), tree complexity (`max_leaf_nodes`/`num_leaves`, `max_depth`), the minimum leaf size (`min_samples_leaf`/`min_child_samples`), and regularisation (`l2_regularization`/`reg_lambda`). Row/column subsampling is secondary. For **random forests**: `max_features` and `min_samples_leaf`. For **regularised linear models**: α/C on a log scale.

**Search spaces on log scales** for scale parameters (learning rate, C, α, regularisation). Uniform sampling between 0.001 and 1 would spend 90% of the trials above 0.1.

## 4. Assumptions and limitations
- Every tuned score is optimistic (the max of noisy estimates). Use nested CV or a held-out test to report ([cross_validation.md](cross_validation.md)).
- Gains below the CV noise are not real. Stop when the improvement over the last N trials is smaller than the fold std.
- Tuning on a validation scheme that does not mimic deployment (random folds on temporal data) optimises the wrong thing.

## 5. Step-by-step procedure
1. **Fix the validation scheme and the metric** (the same as for model comparison).
2. **Start from good defaults** and record their score (the "untuned" reference).
3. **Define the search space:** 3–6 important parameters, log scales where appropriate, sensible bounds.
4. **Choose a strategy by budget:** random search (≈ 30–60 trials) as the default; Optuna for expensive models; halving for huge spaces.
5. **Run with a fixed seed**, logging every trial ([experiment_tracking.md](../07_experimentation/experiment_tracking.md)).
6. **Inspect the results:** which parameters matter? Are the best values at the edges of the space (then extend it)? How large is the gain vs the untuned reference, relative to the fold std?
7. **Refit the best configuration** on all the training data (`refit=True`); evaluate once on the test set.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV

from cookbook_data import load_clean_process_yield

df = load_clean_process_yield()
train = df.iloc[: int(0.8 * len(df))]
num = ["temperature_c", "pressure_bar", "feed_rate_kg_h", "residence_time_min", "catalyst_age_days", "impurity_ppm"]
cat = ["reactor", "catalyst", "shift", "supplier"]
X = train[num + cat].astype({c: "category" for c in cat})
y = np.log(train["yield_kg"])
cv = TimeSeriesSplit(5)

base = HistGradientBoostingRegressor(categorical_features="from_dtype", early_stopping=True,
                                     validation_fraction=0.15, n_iter_no_change=30, max_iter=2000, random_state=0)
ref = cross_val_score(base, X, y, cv=cv, scoring="neg_mean_absolute_error")
print(f"defaults: MAE(log)={-ref.mean():.4f} ± {ref.std():.4f}")

space = {
    "learning_rate": loguniform(0.01, 0.3),
    "max_leaf_nodes": randint(4, 64),
    "min_samples_leaf": randint(5, 80),
    "l2_regularization": loguniform(1e-3, 10),
    "max_features": [0.5, 0.8, 1.0],           # feature subsampling per split (sklearn >= 1.4)
}
rs = RandomizedSearchCV(base, space, n_iter=30, cv=cv, scoring="neg_mean_absolute_error",
                        random_state=42, n_jobs=-1, return_train_score=True)
rs.fit(X, y)
print(f"random search best: MAE(log)={-rs.best_score_:.4f}  params={rs.best_params_}")

res = pd.DataFrame(rs.cv_results_)
cols = ["param_learning_rate", "param_max_leaf_nodes", "param_min_samples_leaf",
        "param_l2_regularization", "mean_test_score", "std_test_score", "mean_train_score"]
print(res.sort_values("rank_test_score")[cols].head(8).round(4))
```

```python
# Which parameters matter? Spearman correlation between each parameter and the score across trials
imp = {c: res[c].astype(float).corr(res["mean_test_score"], method="spearman")
       for c in cols[:4]}
print(pd.Series(imp).round(2))
```

```python
# Successive halving: many configurations, resource = number of training samples
hs = HalvingRandomSearchCV(base, space, n_candidates=60, factor=3, resource="n_samples",
                           min_resources=200, cv=cv, scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1)
hs.fit(X, y)
print(f"halving best: MAE(log)={-hs.best_score_:.4f}, iterations={hs.n_iterations_}, "
      f"candidates per round={hs.n_candidates_}")
```

```python
# Optuna (TPE sampler + pruning of bad trials after the first folds)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = dict(learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                  max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 4, 64, log=True),
                  min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 80),
                  l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10, log=True))
    model = base.set_params(**params)
    scores = []
    for i, (tr, va) in enumerate(cv.split(X)):
        model.fit(X.iloc[tr], y.iloc[tr])
        scores.append(np.mean(np.abs(y.iloc[va] - model.predict(X.iloc[va]))))
        trial.report(np.mean(scores), i)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return np.mean(scores)

study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42),
                            pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
study.optimize(objective, n_trials=30)
print(f"optuna best MAE(log)={study.best_value:.4f}  params={study.best_params}")
print("pruned trials:", sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials))
```

## 7. Interpretation
- **Gain vs defaults:** random search improves MAE(log) from 0.213 to 0.202, but the fold std is ≈ 0.04. **That gain is not distinguishable from noise.** This is the normal outcome: HistGB defaults are good, and tuning is a small lever compared with features. Compare with [feature_engineering.md](../02_preprocessing/feature_engineering.md), where domain features cut the error almost in half.
- **Top-trials table:** many configurations lie within the noise of the best, so the landscape is flat. Pick a simpler or more regularised configuration among them (more robust).
- **Parameter–score correlations:** here `min_samples_leaf` is strongly negatively correlated with the score (smaller leaves are better) and `max_leaf_nodes` positively. Focus the next search on these. Best values at the boundary of the space (e.g. `min_samples_leaf=5`, the lower bound) mean the bounds should be extended.
- **Optuna** found a configuration with **very small trees** (`max_leaf_nodes=4`) and a better score (0.178). Many weak, shallow trees suit this mostly additive problem. Confirm such a result with full CV and a different seed before trusting it: with 30 noisy trials, the "best" is partly luck.
- **Halving's `best_score_` (≈ 0.32) is not comparable** with the others. It is measured in the last round, on the resource used there (here a subset of the samples, with fewer training rows per fold). Refit the halving winner and evaluate it with the full CV before comparing. Also, subsampling `n_samples` interacts badly with `TimeSeriesSplit` (it shortens the history). For temporal data, prefer `resource="max_iter"`, or plain random search.
- **Train vs test scores of the trials:** configurations with a much better train score and no better test score are overfitting. Prefer the regularised end.

## 8. Decision framework
- **IF** fewer than ~3 parameters → grid search over log-spaced values (it is interpretable).
- **IF** 3–8 parameters, model fits in seconds → random search with 30–60 trials.
- **IF** fits are expensive (minutes) → Optuna with pruning, or halving; reduce the CV folds (3) during the search, then confirm the best configuration with full CV.
- **IF** the tuning gain is below the noise → stop tuning; work on features or data.
- **IF** the dataset is small → tune few parameters, prefer stronger regularisation, and use nested CV for reporting.
- **IF** using GBMs → always tune with early stopping (or tune the number of iterations) together with the learning rate.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Tuning on the test set | Optimistic final score | Tune on CV; test once |
| Linear search space for scale parameters | Wasted trials | Log-uniform |
| Huge grids | Cost explosion; overfitting the validation | Random/Bayesian with a budget |
| Ignoring the fold std | Chasing noise | Compare with the std |
| Tuning before the features stabilise | Re-tuning repeatedly | Features first, tuning later |
| Different CV for tuning and evaluation | Mismatched objectives | One scheme |

## 10. Checklist
- [ ] Validation scheme and metric fixed; default score recorded.
- [ ] 3–6 important parameters; log scales; sensible bounds.
- [ ] Strategy chosen by budget; seeded; trials logged.
- [ ] Gain vs defaults compared with the fold std.
- [ ] Boundary hits checked; parameter importance inspected.
- [ ] Best configuration refitted; reported via nested CV or test.

## 11. References
- scikit-learn User Guide, *Tuning the hyper-parameters of an estimator* (Grid, Randomized, Successive Halving). https://scikit-learn.org/stable/modules/grid_search.html
- Bergstra, J. & Bengio, Y. (2012). "Random Search for Hyper-Parameter Optimization." *JMLR* 13, 281–305. https://jmlr.org/papers/v13/bergstra12a.html
- Akiba, T., Sano, S., Yanase, T., Ohta, T. & Koyama, M. (2019). "Optuna: A Next-generation Hyperparameter Optimization Framework." *KDD*. https://doi.org/10.1145/3292500.3330701 · Docs: https://optuna.readthedocs.io/
- Jamieson, K. & Talwalkar, A. (2016). "Non-stochastic Best Arm Identification and Hyperparameter Optimization." *AISTATS*. Successive halving.
- Probst, P., Boulesteix, A.-L. & Bischl, B. (2019). "Tunability: Importance of Hyperparameters of Machine Learning Algorithms." *JMLR* 20(53). https://jmlr.org/papers/v20/18-444.html
- LightGBM docs, *Parameters Tuning*. https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html
