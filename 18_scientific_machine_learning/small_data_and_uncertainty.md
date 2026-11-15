# Small-Data Modelling with Uncertainty: Grouped CV, Gaussian Processes, Weighting and Applicability Domain

> **Folder:** 18_scientific_machine_learning · **Prev:** [scientific_ml_workflow.md](scientific_ml_workflow.md) · **Next:** [surrogate_and_physics_informed.md](surrogate_and_physics_informed.md) · **Related:** [uncertainty.md](../03_regression/uncertainty.md), [cross_validation.md](../06_model_improvement/cross_validation.md), [high_dimensional_data.md](../11_dimensionality_reduction/high_dimensional_data.md)

---

## 1. Purpose
Build trustworthy models from tens to hundreds of experiments, for regression of continuous responses and classification of experimental outcomes. Use validation that respects replicates, models whose complexity matches the data, explicit measurement uncertainty, prediction intervals, and a notion of where the model can be trusted (the applicability domain).

## 2. When to use it
Experimental datasets in chemistry, chemical engineering, materials, energy and environment, where every data point is expensive.

## 3. Conceptual foundation
- **Variance dominates with small n.** The choice of model and validation matters more than tuning. Prefer models with strong, sensible inductive biases (linear in physically transformed features, or a smooth GP).
- **Gaussian process regression (GPR):** a distribution over smooth functions, defined by a kernel (RBF/Matérn for smoothness; `WhiteKernel` for noise). Advantages for small data: calibrated-ish predictive uncertainty that grows away from the data, automatic smoothness from marginal-likelihood optimisation, and good interpolation. Limits: O(n³) cost (fine below ~10⁴ points), and it is sensitive to feature scaling and kernel choice.
- **Replicates:** use them to estimate the noise variance (the noise floor), and keep them together in CV (GroupKFold / Leave-One-Group-Out).
- **Measurement uncertainty:** weight the observations by 1/σ² (heteroscedastic noise). In GPR, use per-point noise via `alpha` = σ²_i.
- **Applicability domain (AD):** the region of feature space supported by the training data. Simple AD measures are the distance to the nearest training points (kNN distance in scaled space), leverage (linear models), and the GP predictive standard deviation.
- **Classifying outcomes** (success/failure, pass/fail) from few experiments: logistic regression with few features, or GP classification. Report the probabilities with uncertainty, and use Leave-One-Group-Out evaluation.

## 4. Assumptions and limitations
- With 20–30 conditions, every CV estimate is noisy. Report the spread (repeated grouped CV) and avoid fine-grained model comparisons.
- GP intervals assume the kernel is appropriate. They can be over-confident with mis-specified smoothness or non-stationarity.
- AD measures are heuristics. "Inside the domain" does not guarantee accuracy.

## 5. Step-by-step procedure
1. **Noise floor** from the replicates ([scientific_ml_workflow.md](scientific_ml_workflow.md)).
2. **Features:** physically transformed, scaled; few in number.
3. **Candidates:** mean baseline; linear on mechanistic features; quadratic response surface (ridge); GPR (Matérn + White); a small GBM (for comparison only).
4. **Validation:** Leave-One-Group-Out (per condition) or repeated GroupKFold; RMSE compared with the noise floor; plus an extrapolation test (leave-region-out, e.g. the highest temperatures held out).
5. **Uncertainty:** GP predictive SD or bootstrap ensembles; check the coverage on the held-out groups.
6. **Applicability domain:** the kNN distance to the training set; flag queries beyond the threshold.
7. **Outcome classification** (if relevant): a logistic/GP classifier with grouped CV; calibration check.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.dummy import DummyRegressor

from cookbook_data import make_catalyst_experiments

df = make_catalyst_experiments()
df = df[df["failed_run"] == 0].copy()
df["inv_T"] = 1000 / df["temperature_K"]; df["ln_P"] = np.log(df["pressure_bar"]); df["ln_load"] = np.log(df["catalyst_loading_pct"])
df["ratio_dev_sq"] = (df["h2_co_ratio"] - 2) ** 2               # rate optimum near H2/CO = 2 (domain knowledge)
feats = ["inv_T", "ln_P", "ln_load", "h2_co_ratio", "ratio_dev_sq"]
X, y, groups = df[feats], df["conversion_pct"], df["condition_id"]
noise_floor = np.sqrt(np.nanmean(df.groupby("condition_id")["conversion_pct"].std() ** 2))

kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(len(feats)), nu=2.5) + WhiteKernel(noise_level=1.0)
# A truly MECHANISTIC model: first-order kinetics X = 1 − exp(−k τ) with Arrhenius/power-law k, i.e.
# ln(−ln(1 − X)) is LINEAR in 1/T, ln P, ln(loading) and (ratio − 2)². Fit that line, transform back.
from sklearn.compose import TransformedTargetRegressor
to_lnk = lambda c: np.log(-np.log(1 - np.clip(c, 0, 99.5) / 100))
from_lnk = lambda z: 100 * (1 - np.exp(-np.exp(z)))
mechanistic = TransformedTargetRegressor(make_pipeline(StandardScaler(), LinearRegression()),
                                         func=to_lnk, inverse_func=from_lnk, check_inverse=False)
models = {
    "mean baseline": DummyRegressor(),
    "MECHANISTIC (linearised kinetics)": mechanistic,
    "linear (mechanistic features)": make_pipeline(StandardScaler(), LinearRegression()),
    "quadratic response surface (ridge)": make_pipeline(StandardScaler(), PolynomialFeatures(2), RidgeCV(alphas=np.logspace(-3, 3, 20))),
    "Gaussian process (Matérn + noise)": make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0)),
    "small GBM": HistGradientBoostingRegressor(max_iter=200, max_depth=2, learning_rate=0.05, min_samples_leaf=5, random_state=0),
}
logo = LeaveOneGroupOut()
rows = []
for name, m in models.items():
    p = cross_val_predict(m, X, y, cv=logo, groups=groups)
    rows.append({"model": name, "LOGO RMSE": np.sqrt(np.mean((y - p) ** 2))})
print(pd.DataFrame(rows).set_index("model").round(2), f"\nnoise floor (replicate SD) ≈ {noise_floor:.2f}")
```

```python
# Extrapolation test: hold out the 25% HOTTEST conditions (outside the training temperature range)
cond_T = df.groupby("condition_id")["temperature_K"].first()
hot = cond_T[cond_T >= cond_T.quantile(0.75)].index
tr, te = ~df["condition_id"].isin(hot), df["condition_id"].isin(hot)
for name in ["MECHANISTIC (linearised kinetics)", "linear (mechanistic features)", "Gaussian process (Matérn + noise)", "small GBM"]:
    m = models[name].fit(X[tr], y[tr]); p = m.predict(X[te])
    print(f"extrapolation to hotter conditions — {name:34s} RMSE = {np.sqrt(np.mean((y[te] - p) ** 2)):.2f}")
```

```python
# GP predictive uncertainty: coverage of 95% intervals on held-out GROUPS, and growth of uncertainty with distance
gp = models["Gaussian process (Matérn + noise)"]
inside, sds = [], []
for tr_i, te_i in logo.split(X, y, groups):
    gp.fit(X.iloc[tr_i], y.iloc[tr_i])
    mu, sd = gp.predict(X.iloc[te_i], return_std=True)
    inside += list(np.abs(y.iloc[te_i] - mu) <= 1.96 * sd); sds += list(sd)
print(f"95% GP interval coverage on held-out conditions: {np.mean(inside):.2f}  (mean predictive SD {np.mean(sds):.2f})")
gp.fit(X, y)
print("fitted kernel:", gp[-1].kernel_)

# Applicability domain: kNN distance in scaled feature space; compare a typical query with an extrapolated one
from sklearn.neighbors import NearestNeighbors
X_cond = X.drop_duplicates()                                        # one row per CONDITION (replicates share X!)
sc = StandardScaler().fit(X_cond); nn = NearestNeighbors(n_neighbors=2).fit(sc.transform(X_cond))
d_train = nn.kneighbors(sc.transform(X_cond))[0][:, 1]              # distance to the nearest OTHER condition
thr = np.quantile(d_train, 0.95)
queries = pd.DataFrame({"inv_T": [1000 / 560, 1000 / 650], "ln_P": [np.log(25), np.log(25)],
                        "ln_load": [np.log(3), np.log(3)], "h2_co_ratio": [2.0, 2.0], "ratio_dev_sq": [0.0, 0.0]},
                       index=["T=560 K (inside)", "T=650 K (outside)"])
d_q = nn.kneighbors(sc.transform(queries), n_neighbors=1)[0][:, 0]
mu_q, sd_q = gp.predict(queries, return_std=True)
out_of_range = ((queries < X_cond.min()) | (queries > X_cond.max())).any(axis=1)      # simple per-feature range check
print(pd.DataFrame({"kNN distance": d_q.round(2), "kNN threshold": round(thr, 2), "kNN says in domain": d_q <= thr,
                    "outside training ranges": out_of_range.to_numpy(), "GP mean": mu_q.round(1), "GP sd": sd_q.round(1),
                    "mechanistic": mechanistic.fit(X, y).predict(queries).round(1)}, index=queries.index).T)
```

```python
# Weighted fitting with known per-measurement uncertainty (e.g. a less precise analyser on some days)
rng = np.random.default_rng(0)
sigma = np.where(rng.random(len(df)) < 0.3, 6.0, 1.5)                  # 30% of runs measured with a worse method
y_noisy = y + rng.normal(0, np.sqrt(np.maximum(sigma ** 2 - 1.5 ** 2, 0)))
for name, w in [("unweighted", None), ("weighted 1/σ²", 1 / sigma ** 2)]:
    preds = np.zeros(len(df))
    for tr_i, te_i in logo.split(X, y, groups):
        kw = {} if w is None else {"linearregression__sample_weight": w[tr_i]}   # routed to the pipeline step
        preds[te_i] = mechanistic.fit(X.iloc[tr_i], y_noisy.iloc[tr_i], **kw).predict(X.iloc[te_i])
    print(f"mechanistic, {name:14s} RMSE vs the precise measurements = {np.sqrt(np.mean((y - preds) ** 2)):.2f}")
```

```python
# Classification of experimental outcomes: "target conversion ≥ 60% reached?" with grouped CV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
success = (df.groupby("condition_id")["conversion_pct"].transform("mean") >= 60).astype(int)
clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0))
p = cross_val_predict(clf, X, success, cv=GroupKFold(6), groups=groups, method="predict_proba")[:, 1]
print(f"success classifier (grouped CV): ROC-AUC={roc_auc_score(success, p):.3f}  Brier={brier_score_loss(success, p):.3f} "
      f"(prior Brier={brier_score_loss(success, np.full(len(p), success.mean())):.3f})")
```

## 7. Interpretation
- **LOGO RMSE vs the noise floor (≈ 2.0):** the **mechanistic model reaches ≈ 2.1, essentially the noise floor**. Its structure (first-order kinetics with an Arrhenius-type rate) is correct, so only the measurement noise remains. The GP (≈ 3.3) is the best *generic* model. The quadratic response surface (≈ 4.5) and a linear model on the same features (≈ 5.4) are mis-specified: conversion saturates, so it is not linear in 1/T. The small GBM (≈ 10) needs far more data. A model *below* the noise floor would signal leakage.
- **Extrapolation to hotter conditions:** mechanistic ≈ 2.1, GP ≈ 6.9, linear ≈ 8.7, GBM ≈ 27. The GBM predicts flat values beyond the training range, and the GP reverts towards its mean. Only the correct physical structure extrapolates. **For extrapolation, physics-informed structure beats flexibility.** "Mechanistic features in a linear model" is not the same as a mechanistic model: the functional form matters.
- **GP coverage (≈ 0.93 for nominal 95%):** good calibration for interpolation. The fitted kernel shows the learned length-scales (short = an important, rapidly varying factor; the very long one for `h2_co_ratio` says that factor is handled by `ratio_dev_sq`) and the noise level.
- **Applicability domain:** the domain must be computed on unique *conditions*, because replicates share identical inputs. Otherwise every nearest-neighbour distance is 0. Even then, the **kNN-distance criterion does not flag the 650 K query**: 24 points in 5 dimensions are sparse, so typical nearest-neighbour distances are large, and stepping beyond the range in one variable adds little distance. The **per-feature range check** flags it immediately, and the GP SD grows ≈ 6×. Use several AD signals.
- **Physical plausibility:** at 650 K the GP predicts a conversion above 100%, which is physically impossible. The mechanistic model saturates below 100% by construction. Constraints matter most exactly where the data run out ([surrogate_and_physics_informed.md](surrogate_and_physics_informed.md)).
- **Weighting:** with 30% of the runs measured on a less precise instrument, weighting by 1/σ² cuts the mechanistic model's error substantially. When σ is known, use it.
- **Outcome classification:** with grouped CV the logistic classifier separates the conditions reaching 60% conversion (AUC ≈ 0.91; Brier 0.11 vs the prior's 0.24), so its probabilities are informative.

## 8. Decision framework
- **IF** n < 30 conditions → mechanistic/linear models + LOGO CV; a GP if the response is smooth; avoid tuning many hyperparameters.
- **IF** replicates exist → group them in CV; use their spread as the noise floor and as the GP noise prior.
- **IF** σ_i is known → weighted fitting (`sample_weight`) or a GP with per-point `alpha`.
- **IF** you must predict outside the data range → physics-based features/hybrid models; AD flags; report extrapolations explicitly.
- **IF** the decision needs uncertainty (which experiment next?) → GP or ensembles; check the coverage on held-out groups.
- **IF** the outcome is pass/fail → logistic regression with few physical features; grouped CV; calibration check.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Row-level CV with replicates | Optimistic errors | Leave-One-Group-Out |
| Tuning many hyperparameters on 30 points | Overfitting the validation | Few candidates; nested CV if tuning |
| Unscaled features in a GP | Bad length-scale fits | Standardise inside the pipeline |
| Trusting GP intervals in extrapolation | Over/under-confidence | AD checks; physics |
| Ignoring known σ | Noisy points dominate | Weighting |
| Reporting a single CV number | Hidden variance | Spread across folds/repeats |

## 10. Checklist
- [ ] Noise floor from the replicates; failed runs handled.
- [ ] Few physical features; scaled.
- [ ] Baseline, mechanistic, response-surface and GP candidates; LOGO CV.
- [ ] An extrapolation test.
- [ ] Interval coverage on held-out groups.
- [ ] Applicability-domain check for new queries.
- [ ] Weighted fitting if σ is known; outcome classification with grouped CV if relevant.

## 11. References
- Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. Free: http://gaussianprocess.org/gpml/
- scikit-learn User Guide, *Gaussian Processes*. https://scikit-learn.org/stable/modules/gaussian_process.html
- Netzeva, T. I. et al. (2005). "Current status of methods for defining the applicability domain of (quantitative) structure–activity relationships." *ATLA* 33(2), 155–173.
- Sheridan, R. P. (2013). "Time-split cross-validation as a method for estimating the goodness of prospective prediction." *J. Chem. Inf. Model.* 53(4), 783–790. https://doi.org/10.1021/ci400084k
- Box, G. E. P. & Draper, N. R. (2007). *Response Surfaces, Mixtures, and Ridge Analyses*, 2nd ed. Wiley.
