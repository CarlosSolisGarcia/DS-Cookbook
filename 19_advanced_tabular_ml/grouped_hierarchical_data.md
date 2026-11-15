# Grouped Observations and Hierarchical Data: Group-Aware Validation, Group Features and Mixed-Effects Models

> **Folder:** 19_advanced_tabular_ml · **Prev:** [distribution_shift.md](distribution_shift.md) · **Next:** [imbalance_beyond_binary.md](imbalance_beyond_binary.md) · **Related:** [data_leakage.md](../00_methodology/data_leakage.md) (group leakage), [cross_validation.md](../06_model_improvement/cross_validation.md), [high_cardinality_and_wide_data.md](high_cardinality_and_wide_data.md)

---

## 1. Purpose
Model data in which rows are nested in groups (measurements in batches, machines in sites, students in schools, patients in hospitals, wells in fields). Validate in the way the predictions will be used (new groups vs known groups), exploit group structure as features, and use partial pooling (mixed-effects/hierarchical models) when groups are small.

## 2. When to use it
Whenever there is an entity ID with several rows per entity, or nested entity levels (site → line → machine).

## 3. Conceptual foundation
- **Two prediction targets:** (a) new rows from **known** groups (the same machines next month), where group identity/history features are legitimate; (b) rows from **new** groups (a new machine or site), where only group-independent features generalise, and the validation must hold out whole groups.
- **Pooling spectrum:**
  - *No pooling:* a separate model or intercept per group. Noisy for small groups.
  - *Complete pooling:* ignore the groups. Biased when the groups differ.
  - *Partial pooling* (mixed effects / random effects / hierarchical Bayes): group effects are shrunk towards the overall mean, more strongly for small groups. This is the statistically principled middle ground, and it is equivalent in spirit to cross-fitted target encoding with shrinkage.
- **Mixed-effects model:** y_ij = x_ijᵀβ + u_j + ε_ij with u_j ~ N(0, σ_u²) (random intercepts; random slopes possible). The intraclass correlation ICC = σ_u² / (σ_u² + σ_ε²) measures how much of the variance lies between groups.
- **Group-level features:** group means/counts of features (computed out of fold or from past data), group size, deviations of the row from its group mean (the "within-group" signal).

## 4. Assumptions and limitations
- Mixed models (statsmodels `MixedLM`) are linear and assume normal random effects. For non-linear relations, combine GBMs with group features, or use GPBoost-like approaches (an extension).
- Group-mean features computed on all rows leak target information when they include the target. Feature means are fine. Target means must be cross-fitted.
- Only a few groups (< ~10) make the random-effect variance estimates unreliable.

## 5. Step-by-step procedure
1. **Identify the group levels** and decide the deployment target (known vs new groups).
2. **Quantify the clustering:** the ICC of the target; the group sizes.
3. **Validation:** GroupKFold for new groups; time-based within groups for known groups.
4. **Features:** group-level feature aggregates; within-group deviations; cross-fitted group target encodings (known-group use).
5. **Models:** a pooled GBM/linear model with group features; a mixed-effects model for interpretable partial pooling; compare them under the relevant validation.
6. **Report** performance separately for small and large groups.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict

# Lab data: product purity measured on samples from 60 production batches (3–30 samples per batch)
rng = np.random.default_rng(0)
n_b = 60; sizes = rng.integers(3, 31, n_b)
batch_eff = rng.normal(0, 1.2, n_b)                                  # between-batch variation (raw material, operator)
rows = []
for b in range(n_b):
    T = rng.normal(80, 4, sizes[b]); t = rng.uniform(1, 5, sizes[b])
    purity = 95 + 0.15 * (T - 80) + 0.6 * np.log(t) + batch_eff[b] + rng.normal(0, 0.8, sizes[b])
    rows.append(pd.DataFrame({"batch": f"B{b:02d}", "temp": T, "time_h": t, "purity": purity}))
df = pd.concat(rows, ignore_index=True)
df["log_time"] = np.log(df["time_h"])

# ICC from a random-intercept model (no predictors)
null = smf.mixedlm("purity ~ 1", df, groups=df["batch"]).fit()
s2_u, s2_e = float(null.cov_re.iloc[0, 0]), null.scale
print(f"ICC = {s2_u / (s2_u + s2_e):.2f}  (share of the variance between batches)")
```

```python
# Mixed model vs pooled OLS: coefficients and the shrinkage of batch effects
mixed = smf.mixedlm("purity ~ temp + log_time", df, groups=df["batch"]).fit()
pooled = smf.ols("purity ~ temp + log_time", df).fit()
print(pd.DataFrame({"pooled OLS": pooled.params, "mixed effects": mixed.fe_params}).round(3))
print("SE of temp — pooled:", round(pooled.bse["temp"], 4), "| mixed:", round(mixed.bse["temp"], 4))
re = pd.Series({k: v.iloc[0] for k, v in mixed.random_effects.items()})
raw = df.groupby("batch").apply(lambda g: (g["purity"] - mixed.predict(g)).mean(), include_groups=False)   # unpooled batch means
comp = pd.DataFrame({"size": df.groupby("batch").size(), "unpooled": raw, "partial pooling (BLUP)": re, "true": batch_eff})
print(comp.sort_values("size").iloc[[0, 1, 2, -3, -2, -1]].round(2))
```

```python
# Validation for NEW batches vs KNOWN batches, and the value of group features
X = df[["temp", "log_time"]].copy()
X["batch_mean_temp"] = df.groupby("batch")["temp"].transform("mean")               # feature aggregate (no target)
X["temp_dev"] = df["temp"] - X["batch_mean_temp"]                                  # within-batch deviation
gbm = HistGradientBoostingRegressor(max_iter=200, random_state=0)
for name, cv, groups in [("random KFold (known batches)", KFold(5, shuffle=True, random_state=0), None),
                         ("GroupKFold (new batches)", GroupKFold(5), df["batch"])]:
    p = cross_val_predict(gbm, X, df["purity"], cv=cv, groups=groups)
    print(f"{name:30s} GBM RMSE = {np.sqrt(np.mean((df['purity'] - p) ** 2)):.3f}")

# Mixed model on new batches: the random effect is unknown → predict with the fixed effects only
rmse_fixed = []
for tr, te in GroupKFold(5).split(df, groups=df["batch"]):
    m = smf.mixedlm("purity ~ temp + log_time", df.iloc[tr], groups=df["batch"].iloc[tr]).fit()
    rmse_fixed.append(np.sqrt(np.mean((df["purity"].iloc[te] - m.predict(df.iloc[te])) ** 2)))
print(f"mixed model, new batches (fixed effects only) RMSE = {np.mean(rmse_fixed):.3f} | "
      f"null-model SD = {np.sqrt(s2_u + s2_e):.3f} | between-batch SD σ_u = {np.sqrt(s2_u):.3f}")
```

## 7. Interpretation
- **ICC ≈ 0.56:** more than half of the variance lies between batches. Ignoring the grouping misrepresents the information in the data, and random CV will be optimistic for new batches.
- **Coefficients and standard errors:** the fixed effects (temperature ≈ 0.15, log time ≈ 0.56) are similar in pooled OLS and the mixed model. Here the mixed model's SE for temperature is *smaller* (0.006 vs 0.011). Temperature varies *within* batches, so removing the between-batch noise sharpens the estimate. For predictors that vary mainly *between* groups (e.g. a raw-material property per batch), the opposite happens: pooled OLS then treats correlated samples as independent and is over-confident. Either way, the mixed model's SEs are the honest ones.
- **Partial pooling (BLUP):** the batch effects of the 3-sample batches are shrunk towards 0 compared with their unpooled means (e.g. 0.76 → 0.66). The 30-sample batches keep their estimates. Shrinkage is strongest where the evidence is weakest.
- **Known vs new batches:** random KFold gives GBM RMSE ≈ 1.09, because it learns each batch's offset from its sibling samples. GroupKFold (new batches) gives ≈ 1.91. **The mixed model's fixed effects do better for new batches (≈ 1.47)** than the GBM with batch-aggregate features. Those features let the GBM fit batch idiosyncrasies that do not transfer. The remaining error is dominated by the unknown batch effect (σ_u ≈ 1.2). Only batch-level covariates (raw-material lot analyses) could reduce it. **The validation must match the use**, and with strong group effects, a simple partially pooled model can beat a flexible one.

## 8. Decision framework
| Situation | Approach |
|---|---|
| Predict for new groups | GroupKFold; group-independent features; mixed model fixed effects or GBM; collect group-level covariates |
| Predict for known groups (future rows) | Time-based validation within groups; group identity/history features (cross-fitted/past-only) |
| Many small groups | Partial pooling (mixed effects, shrunk target encoding) |
| Need inference on the effects (which factor matters?) | Mixed-effects model (correct SEs); random slopes if effects vary by group |
| Nested levels (site → line → machine) | Nested random effects or multi-level encodings |
| Non-linear effects + groups | GBM with group aggregates / encodings; compare with a mixed model |

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Random CV with grouped rows (for new-group use) | Optimistic errors | GroupKFold |
| Group target means computed on all rows | Leakage | Cross-fitting or past-only |
| OLS standard errors with clustered data | Over-confident inference | Mixed models / cluster-robust SEs |
| A separate model per small group | High variance | Partial pooling |
| Ignoring that new groups lack the random effect | Unrealistic accuracy claims | Evaluate with fixed effects only |

## 10. Checklist
- [ ] Group levels identified; the deployment target (known vs new groups) decided.
- [ ] ICC and group sizes computed.
- [ ] Validation matches the deployment.
- [ ] Group aggregates / within-group deviations / cross-fitted encodings as appropriate.
- [ ] Pooled vs mixed vs GBM compared; results by group size.

## 11. References
- Gelman, A. & Hill, J. (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.
- statsmodels, *Linear Mixed Effects Models* (`MixedLM`). https://www.statsmodels.org/stable/mixed_linear.html
- Bates, D., Mächler, M., Bolker, B. & Walker, S. (2015). "Fitting Linear Mixed-Effects Models Using lme4." *J. Statistical Software* 67(1). https://doi.org/10.18637/jss.v067.i01
- Sigrist, F. (2022). "Gaussian Process Boosting." *JMLR* 23(232) (GPBoost: trees + random effects). https://jmlr.org/papers/v23/20-322.html
