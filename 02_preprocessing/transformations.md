# Transformations of Features and Targets

> **Folder:** 02_preprocessing · **Prev:** [categorical_features.md](categorical_features.md) · **Next:** [scaling.md](scaling.md) · **Related:** [numerical_features.md](numerical_features.md), [regression evaluation](../03_regression/evaluation.md), [ts preprocessing](../05_time_series/preprocessing.md)

---

## 1. Purpose
Decide **whether and how** to transform a feature or the target (log, Box-Cox, Yeo-Johnson, square root, quantile), what you gain from it, and what it costs. The cost side matters most for the target, because a transformation changes what the model optimises.

## 2. When to use it
- Features: when using linear, distance-based or neural models and a feature is skewed or has a curved relationship with the target.
- Target: when the target is strongly skewed and positive, when errors are multiplicative (they grow with the level), or when residuals are heteroscedastic.
- Usually unnecessary for **features** of tree models.

## 3. Conceptual foundation
| Transformation | Formula | Domain | Use when |
|---|---|---|---|
| log | ln(x) | x > 0 | Multiplicative effects; right skew; quantities spanning orders of magnitude (concentrations, flows) |
| log1p | ln(1 + x) | x ≥ 0 | Counts or quantities with zeros |
| sqrt | √x | x ≥ 0 | Counts (variance ∝ mean), milder skew |
| Box-Cox | (x^λ − 1)/λ | x > 0 | Data-driven power; λ = 0 ≡ log |
| Yeo-Johnson | Box-Cox-like, extended | any real | Skewed data that include zero or negative values |
| Quantile → normal/uniform | rank-based | any | Very irregular distributions; destroys the distance structure. Use sparingly |
| logit | ln(p/(1−p)) | 0 < p < 1 | Proportions near the bounds |

**Why log helps linear models.** If yield ≈ feed × efficiency × noise (multiplicative), then log(yield) = log(feed) + log(efficiency) + log(noise) is additive. That is exactly the structure a linear model represents. As a chemical engineer you will recognise the Arrhenius and power-law linearisations.

**Target transformation: the back-transformation issue.** Suppose the model minimises squared error on log(y). It then predicts E[log y | x]. Back-transforming gives exp(E[log y]), which is the **conditional median** under log-normal errors, *not* the mean. It systematically under-predicts the mean by a factor of about exp(σ²/2). This matters when you need unbiased totals (e.g. the sum of predicted yields for production planning). Options:
- Accept median predictions and evaluate with MAE. MAE is minimised by the median, so the two are consistent.
- Apply a smearing correction (Duan, 1983): multiply by the mean of exp(residuals) computed on the training data.
- Model the original scale with a suitable loss instead, e.g. Poisson/Gamma/Tweedie deviance in `HistGradientBoostingRegressor(loss="gamma")` or `TweedieRegressor`.

`TransformedTargetRegressor` applies the transform when fitting and the inverse when predicting. Your metrics are then computed on the **original** scale, which is what stakeholders care about.

## 4. Assumptions and limitations
- Transformations change the *meaning* of coefficients (a coefficient on log x is an elasticity) and of the loss (squared error on log y ≈ relative error).
- Box-Cox/Yeo-Johnson λ is fitted, so it goes in the pipeline.
- A transformation cannot fix a missing variable or a wrong functional form. It only changes the scale.
- Normality of the **features** is *not* an assumption of linear regression. What matters is linearity of the relationship and, for inference, the properties of the residuals.

## 5. Step-by-step procedure
1. **Diagnose the need.** Skewness, multiplicative structure (spread grows with the level), a curved relationship in the binned plot ([EDA](../01_data_understanding/exploratory_data_analysis.md)), a funnel shape in the residuals ([regression error_analysis.md](../03_regression/error_analysis.md)).
2. **Prefer interpretable transformations** (log, sqrt) justified by physics or domain reasoning over data-driven ones.
3. **For the target,** decide what you want to predict: the mean (e.g. for totals) or the median/typical value. Choose the metric to match.
4. **Implement** with `FunctionTransformer` / `PowerTransformer` (features) or `TransformedTargetRegressor` (target).
5. **Compare** against the untransformed version under CV, with metrics on the original scale.
6. **Check the residuals** after the transformation: is the heteroscedasticity reduced?

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PowerTransformer, StandardScaler

from cookbook_data import make_process_yield

df = make_process_yield().drop_duplicates()
df["temperature_c"] = df["temperature_c"].mask(df["temperature_c"] == -999)
df = df[df["yield_kg"] > 1].reset_index(drop=True)
num = ["temperature_c", "pressure_bar", "feed_rate_kg_h", "residence_time_min", "catalyst_age_days", "impurity_ppm"]
cat = ["reactor", "catalyst", "shift"]
X, y = df[num + cat], df["yield_kg"]

# Feature transformation: log of skewed positive features vs. Yeo-Johnson vs. none
def pre(num_tf):
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), num_tf, StandardScaler()), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])

variants = {
    "no transform": FunctionTransformer(),                                    # identity
    "log1p": FunctionTransformer(np.log1p),
    "yeo-johnson": PowerTransformer(method="yeo-johnson", standardize=False),
}
cv = KFold(5, shuffle=True, random_state=42)
scoring = {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error"}
for name, tf in variants.items():
    base = make_pipeline(pre(tf), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    for tgt_name, model in [("raw y", base),
                            ("log y", TransformedTargetRegressor(regressor=base, func=np.log, inverse_func=np.exp))]:
        r = cross_validate(model, X, y, cv=cv, scoring=scoring)
        print(f"features={name:13s} target={tgt_name:6s} "
              f"MAE={-r['test_MAE'].mean():6.1f}  RMSE={-r['test_RMSE'].mean():6.1f}")
```

```python
# Back-transformation bias: mean of predictions vs mean of truth on held-out folds
from sklearn.model_selection import cross_val_predict

base = make_pipeline(pre(FunctionTransformer(np.log1p)), RidgeCV(alphas=np.logspace(-3, 3, 13)))
log_model = TransformedTargetRegressor(regressor=base, func=np.log, inverse_func=np.exp)
pred_log = cross_val_predict(log_model, X, y, cv=cv)
pred_raw = cross_val_predict(base, X, y, cv=cv)
print(f"mean(y)={y.mean():.1f}  mean(pred, raw-y model)={pred_raw.mean():.1f}  "
      f"mean(pred, log-y model)={pred_log.mean():.1f}  (log model ≈ median → biased low for totals)")

# Duan smearing correction, estimated on TRAINING residuals (here, illustrated on one split)
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
base.fit(X_tr, np.log(y_tr))
smear = np.mean(np.exp(np.log(y_tr) - base.predict(X_tr)))
pred_corrected = np.exp(base.predict(X_te)) * smear
print(f"smearing factor={smear:.4f}; test mean(y)={y_te.mean():.1f}, "
      f"uncorrected={np.exp(base.predict(X_te)).mean():.1f}, corrected={pred_corrected.mean():.1f}")
```

```python
# Alternative: keep the original scale and use a loss suited to positive, skewed targets
Xc = X.astype({c: "category" for c in cat})
for loss in ["squared_error", "gamma", "absolute_error"]:
    m = HistGradientBoostingRegressor(loss=loss, categorical_features="from_dtype", random_state=0)
    r = cross_validate(m, Xc, y, cv=cv, scoring=scoring)
    print(f"HGB loss={loss:14s} MAE={-r['test_MAE'].mean():6.1f}  RMSE={-r['test_RMSE'].mean():6.1f}")
```

## 7. Interpretation
- **What this run shows (an honest, instructive result):** neither the feature transformations nor the log target improve this ridge model. The log target even makes MAE worse (≈ 59 → 68–71). That seems to contradict the multiplicative structure of the data. The reason is that the model is **misspecified** in a more important way: temperature acts through a catalyst-specific optimum, and a linear term cannot represent it. A transformation fixes only the *scale* of a relationship, and it cannot fix a missing functional form. Once the right features are present ([feature_engineering.md](feature_engineering.md) gets ridge on log(y) to an MAE(log) of ≈ 0.12, better than gradient boosting), the log target becomes the natural choice. **Lesson:** treat a transformation as a hypothesis, validate it, and revisit it after feature engineering.
- **Feature transforms for ridge:** Yeo-Johnson often lands close to log. When it does, prefer log, which is interpretable.
- **Log target:** its predictions are median-type. They may lower MAE when the model is well specified, and they bias totals downwards. Here the mean of the predictions (≈ 180) is below the mean of y (≈ 202). That is the back-transformation bias.
- **Smearing** brings the mean back in line. Use it when sums or averages of the predictions matter.
- **HGB:** loss choice mostly changes *which* errors are emphasised. `absolute_error` targets the median (best MAE), and `squared_error` targets the mean (best RMSE). `gamma` suits positive targets with variance growing with the mean.
- **Suspicious:** a transformation improving one metric dramatically while worsening the other. You probably changed the estimand (mean vs median) rather than improved the model.

## 8. Decision framework
**IF** the target is positive, right-skewed and the spread grows with the level
→ Try (a) a log target via `TransformedTargetRegressor` and (b) a gamma/Poisson/Tweedie loss on the original scale. Compare both with metrics on the original scale. If totals matter, apply smearing to (a) or prefer (b).

**IF** the target contains zeros
→ Use log1p (a change of interpretation), a Tweedie loss, or a two-part model (P(y > 0) × E[y | y > 0]).

**IF** a feature is skewed and the model is linear
→ log/log1p if it is positive and physically multiplicative. Otherwise Yeo-Johnson. Validate.

**IF** the model is tree-based
→ Do not transform features. Consider the target transformation or loss only.

**IF** the stakeholder needs interpretable coefficients
→ Prefer log (elasticities) over Box-Cox with an arbitrary λ.

**IF** the transformation does not improve CV
→ Do not use it. "Normalising distributions" is not a goal in itself.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Transforming to achieve "normal features" | Wasted effort; no gain | Transform for linearity or for the loss, not for normality |
| Reporting metrics on the log scale | Not comparable with untransformed models; misleading to stakeholders | `TransformedTargetRegressor`; metrics on the original scale |
| Ignoring the back-transformation bias | Totals under-predicted | Smearing correction or a mean-targeting loss |
| Box-Cox λ fitted on all data | Leakage (minor, but real) | `PowerTransformer` in the pipeline |
| log of values ≤ 0 | NaN / −inf | log1p, Yeo-Johnson, or check the data |
| Quantile transform everywhere | Destroys meaningful distances; overfits small data | Use only for very irregular distributions |

## 10. Checklist
- [ ] Need diagnosed (skew, multiplicative structure, curvature, heteroscedastic residuals).
- [ ] Interpretable, domain-justified transformation preferred.
- [ ] For the target: decision on mean vs median; metric aligned.
- [ ] Implemented in the pipeline / `TransformedTargetRegressor`.
- [ ] Compared with the untransformed version; metrics on the original scale.
- [ ] Back-transformation bias checked when totals matter.

## 11. References
- Box, G. E. P. & Cox, D. R. (1964). "An Analysis of Transformations." *JRSS B* 26(2), 211–252.
- Yeo, I.-K. & Johnson, R. A. (2000). "A new family of power transformations to improve normality or symmetry." *Biometrika* 87(4), 954–959.
- Duan, N. (1983). "Smearing Estimate: A Nonparametric Retransformation Method." *JASA* 78(383), 605–610.
- scikit-learn, *Effect of transforming the targets in regression model*. https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html
- scikit-learn User Guide, *Non-linear transformation* (PowerTransformer, QuantileTransformer). https://scikit-learn.org/stable/modules/preprocessing.html#non-linear-transformation
- scikit-learn, *Generalized Linear Models* (Tweedie, Poisson, Gamma). https://scikit-learn.org/stable/modules/linear_model.html#generalized-linear-models
