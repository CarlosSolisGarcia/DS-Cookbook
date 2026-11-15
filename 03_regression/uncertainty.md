# Prediction Intervals and Uncertainty Estimation (Regression)

> **Folder:** 03_regression · **Prev:** [evaluation.md](evaluation.md) · **Next:** [error_analysis.md](error_analysis.md) · **Related:** [ts evaluation](../05_time_series/evaluation.md) (forecast intervals), [20_advanced_methodologies](../20_advanced_methodologies/) (broader uncertainty methods)
>
> *Added to the original architecture: prediction intervals are a distinct decision tool and deserve their own recipe.*

---

## 1. Purpose
Go beyond a point prediction: give a **range** that contains the true value with a stated probability (e.g. 90%), and check that the stated probability actually holds (coverage).

## 2. When to use it
- Decisions depend on risk: "Will this batch fall below spec?", "How much buffer stock?", "Is this measurement surprising?"
- Communicating reliability to stakeholders.
- Not needed when only rankings or average performance matter.

## 3. Conceptual foundation
- **Confidence interval vs. prediction interval.** A *confidence* interval describes uncertainty about the **mean** prediction, i.e. about the model parameters. A *prediction* interval describes where a **new observation** will fall. It also includes the noise, so it is much wider. Stakeholders almost always need prediction intervals.
- **Aleatoric vs. epistemic uncertainty.** Aleatoric is inherent noise (measurement error, unmodelled variation); it does not shrink with more data. Epistemic is uncertainty from limited data or model knowledge; it is large in sparsely sampled regions and under extrapolation.
- **Marginal vs. conditional coverage.** *Marginal* coverage means 90% of all intervals contain the truth, on average over all inputs. *Conditional* coverage means 90% for every kind of input. Conformal methods guarantee marginal coverage. Good conditional behaviour needs adaptive widths (quantile models, CQR).

**Methods covered:**

| Method | Idea | Adapts width to x? | Guarantee | Cost |
|---|---|---|---|---|
| **Quantile regression** (GBM `loss="quantile"`) | Fit the α/2 and 1−α/2 quantiles directly | Yes | None (may under-cover) | 2 extra models |
| **Split conformal** (absolute residuals) | Calibrate a constant width on held-out residuals | No | Marginal coverage ≥ 1−α (exchangeable data) | Calibration split |
| **Conformalised quantile regression (CQR)** | Quantile model + conformal correction | Yes | Marginal coverage ≥ 1−α | Calibration split |
| **Residual-based, on the log scale** | Constant width in log space → multiplicative intervals | Yes (proportional) | Approximate | Cheap |
| **Bootstrap / ensembles** | Variance of predictions across resampled models | Partly (epistemic) | None | Many fits |
| **Bayesian / GP** | Posterior predictive | Yes | Model-based | See [18_scientific_ml](../18_scientific_machine_learning/) |

## 4. Assumptions and limitations
- Conformal guarantees require **exchangeability** of the calibration and test data. Under drift (time-ordered data) coverage can fail. Calibrate on the most recent data, and **monitor coverage**.
- The guarantee is marginal. Some segments can be under-covered while others are over-covered, so check coverage per segment.
- Quantile GBMs can produce crossing quantiles (lower > upper). Sort them, or use CQR.

## 5. Step-by-step procedure
1. **Choose the nominal coverage** (e.g. 90%) from the decision's risk tolerance.
2. **Split the training data** into a proper-training part and a **calibration** part. For temporal data, use the most recent block for calibration.
3. **Fit the interval method** on the proper-training data, then calibrate on the calibration data (conformal).
4. **Evaluate on held-out data:** empirical coverage (fraction inside), mean width, coverage per segment.
5. **Compare methods.** Prefer the one with the narrowest intervals at the target coverage. Prefer adaptive widths when the noise varies with x.
6. **In production, monitor coverage** over time.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from cookbook_data import load_clean_process_yield

df = load_clean_process_yield()
num = ["temperature_c", "pressure_bar", "feed_rate_kg_h", "residence_time_min", "catalyst_age_days", "impurity_ppm"]
cat = ["reactor", "catalyst", "shift", "supplier"]
X = df[num + cat].astype({c: "category" for c in cat})
y = df["yield_kg"].to_numpy()

# Chronological split: proper train (60%) | calibration (20%) | test (20%)
n = len(df); i1, i2 = int(0.6 * n), int(0.8 * n)
X_tr, y_tr = X.iloc[:i1], y[:i1]
X_cal, y_cal = X.iloc[i1:i2], y[i1:i2]
X_te, y_te = X.iloc[i2:], y[i2:]
alpha = 0.10                                                  # target 90% coverage

def gbm(**kw):
    return HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=0,
                                         learning_rate=0.05, max_iter=400, **kw)

def coverage_width(lo, hi, y):
    return {"coverage": np.mean((y >= lo) & (y <= hi)), "mean_width": np.mean(hi - lo)}
```

```python
# (a) Quantile regression: two GBMs for the 5% and 95% quantiles (+ median for the point prediction)
q_lo = gbm(loss="quantile", quantile=alpha / 2).fit(X_tr, y_tr)
q_hi = gbm(loss="quantile", quantile=1 - alpha / 2).fit(X_tr, y_tr)
lo_q, hi_q = q_lo.predict(X_te), q_hi.predict(X_te)
lo_q, hi_q = np.minimum(lo_q, hi_q), np.maximum(lo_q, hi_q)   # guard against crossing
print("quantile GBM      ", coverage_width(lo_q, hi_q, y_te))
```

```python
# (b) Split conformal with absolute residuals: constant width
point = gbm().fit(X_tr, y_tr)
scores = np.abs(y_cal - point.predict(X_cal))                 # nonconformity scores on calibration data
k = int(np.ceil((len(scores) + 1) * (1 - alpha)))             # finite-sample corrected quantile index
qhat = np.sort(scores)[min(k, len(scores)) - 1]
p_te = point.predict(X_te)
print("split conformal   ", coverage_width(p_te - qhat, p_te + qhat, y_te))

# (c) Conformal on the LOG scale: multiplicative intervals (width ∝ level), suits multiplicative noise
point_log = gbm().fit(X_tr, np.log(y_tr))
s_log = np.abs(np.log(y_cal) - point_log.predict(X_cal))
q_log = np.sort(s_log)[min(k, len(s_log)) - 1]
pl = point_log.predict(X_te)
print("log-scale conformal", coverage_width(np.exp(pl - q_log), np.exp(pl + q_log), y_te))
```

```python
# (d) Conformalised Quantile Regression (CQR, Romano et al. 2019): adaptive AND calibrated
lo_cal, hi_cal = q_lo.predict(X_cal), q_hi.predict(X_cal)
cqr_scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)       # how far outside the band each cal point is
q_cqr = np.sort(cqr_scores)[min(k, len(cqr_scores)) - 1]
lo_c, hi_c = lo_q - q_cqr, hi_q + q_cqr
print("CQR               ", coverage_width(lo_c, hi_c, y_te))
```

```python
# Conditional coverage check: per segment and per predicted-level band
seg = pd.DataFrame({"reactor": X_te["reactor"].to_numpy(), "y": y_te,
                    "inside_cqr": (y_te >= lo_c) & (y_te <= hi_c),
                    "inside_const": (y_te >= p_te - qhat) & (y_te <= p_te + qhat),
                    "level": pd.qcut(p_te, 4, labels=["low", "mid-low", "mid-high", "high"])})
print(seg.groupby("level", observed=True)[["inside_cqr", "inside_const"]].mean().round(3))
print(seg.groupby("reactor", observed=True)[["inside_cqr", "inside_const"]].mean().round(3))
```

> **Library option:** MAPIE (https://mapie.readthedocs.io/) implements split, cross- and time-series conformal methods. Its API changed substantially in v1 (2025), so check the current docs. The manual code above shows exactly what such libraries do.

## 7. Interpretation
- **Quantile GBM coverage below 90%** is common. Quantile models fitted on training data tend to be over-confident, and drift in the test period makes it worse. Use conformal calibration.
- **Split conformal** reaches about 90% marginal coverage, but with a constant width. It over-covers low-yield batches and under-covers high-yield ones (see the level table), because the noise is multiplicative.
- **Log-scale conformal and CQR** adapt the width to the level. In this run their average width is similar to the constant-width interval, but the coverage across predicted-level bands is much more uniform (CQR ≈ 0.89–0.98 vs. 0.83–1.00 for the constant width). Same average cost, better conditional behaviour.
- **Coverage below the target on the test set even after conformal:** the exchangeability assumption failed (drift between the calibration and test periods). Recalibrate more frequently, or use a time-series conformal method.
- A **very wide** interval is honest information: the inputs do not determine the output precisely. Do not "fix" it by shrinking the intervals.

## 8. Decision framework
- **IF** you need a guaranteed average coverage → split conformal or CQR.
- **IF** the noise grows with the level or varies across inputs (heteroscedastic) → CQR, or conformal on a transformed scale.
- **IF** the data are time-ordered → calibrate on the most recent block; monitor rolling coverage; consider adaptive conformal methods.
- **IF** costs are asymmetric (e.g. under-prediction is costly) → use a single quantile model at the relevant quantile, not a symmetric interval.
- **IF** you have very little data → conformal calibration sets become tiny and the interval quantile is noisy. Consider cross-conformal (CV+/jackknife+) or Bayesian/GP models.
- **IF** inputs are outside the training domain → no interval method is reliable. Detect out-of-domain inputs separately ([10_anomaly_detection](../10_anomaly_detection/)).

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Reporting confidence intervals as prediction intervals | Far too narrow ranges | Prediction intervals include the noise |
| Calibrating on training residuals | Under-coverage (training residuals are too small) | Separate calibration set |
| Checking only marginal coverage | Segments with bad coverage unnoticed | Coverage per segment/level |
| Ignoring drift | Coverage decays in production | Monitor coverage; recalibrate |
| Using quantile models without calibration | Over-confident intervals | CQR |

## 10. Checklist
- [ ] Nominal coverage chosen from the decision's risk tolerance.
- [ ] Calibration split separate from the training data (most recent block for temporal data).
- [ ] Coverage and width evaluated on held-out data.
- [ ] Conditional coverage checked (segments, levels).
- [ ] Method chosen: narrowest at the target coverage, adaptive if heteroscedastic.
- [ ] Coverage monitoring planned for production.

## 11. References
- Angelopoulos, A. N. & Bates, S. (2023). "Conformal Prediction: A Gentle Introduction." *Foundations and Trends in ML* 16(4). https://arxiv.org/abs/2107.07511
- Romano, Y., Patterson, E. & Candès, E. (2019). "Conformalized Quantile Regression." *NeurIPS 32*. https://arxiv.org/abs/1905.03222
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. & Wasserman, L. (2018). "Distribution-Free Predictive Inference for Regression." *JASA* 113(523). https://doi.org/10.1080/01621459.2017.1307116
- scikit-learn, *Prediction Intervals for Gradient Boosting Regression*. https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html
- Koenker, R. & Bassett, G. (1978). "Regression Quantiles." *Econometrica* 46(1), 33–50.
- MAPIE documentation. https://mapie.readthedocs.io/
