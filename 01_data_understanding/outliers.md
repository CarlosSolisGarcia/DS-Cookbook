# Outlier Detection and Treatment

> **Folder:** 01_data_understanding · **Prev:** [missing_values.md](missing_values.md) · **Next:** [exploratory_data_analysis.md](exploratory_data_analysis.md) · **Related:** [data_quality.md](data_quality.md), [transformations.md](../02_preprocessing/transformations.md), [10_anomaly_detection](../10_anomaly_detection/) (when the outliers *are* the target)

---

## 1. Purpose
Find unusual observations, decide **what they are** (an error, a rare but real event, or a different population), and treat each kind appropriately, so that the model is neither distorted by errors nor blind to legitimate extremes.

## 2. When to use it
- During data-quality assessment and EDA, before choosing transformations and losses.
- During error analysis: large residuals often point to outliers in the inputs or the target.
- **Not this recipe** when detecting unusual events *is* the goal (fault detection, fraud). Then outliers are the signal; see [10_anomaly_detection](../10_anomaly_detection/).

## 3. Conceptual foundation
"Outlier" is a statement about a point relative to a **model of normality**. It is not a property of the point alone. Three kinds:

| Kind | Example | Treatment |
|---|---|---|
| **Error** (measurement, recording, unit) | −999 temperature; yield in tonnes instead of kg | Correct, or set to NaN / drop ([data_quality.md](data_quality.md)) |
| **Rare but real extreme** | A genuinely high-impurity raw-material lot | **Keep.** Make the model robust (robust loss, transformation, trees) |
| **Different population / regime** | Start-up batches, maintenance runs, a pilot reactor | Model separately, add a regime feature, or exclude **by definition** of the problem scope |

**Univariate vs. multivariate outliers.** A point can be ordinary on each axis and unusual jointly. Example: a normal temperature with a normal jacket temperature, but a jacket–reactor difference that is physically implausible. Univariate rules miss these. Use consistency rules or multivariate methods.

**Common detection rules:**
- **IQR rule (Tukey fences):** outside [Q1 − k·IQR, Q3 + k·IQR], with k = 1.5 ("outside") or 3 ("far out"). Robust, and makes no normality assumption, but flags many points in skewed data.
- **Robust z-score:** |x − median| / (1.4826 · MAD) > 3.5. Robust analogue of the z-score; assumes rough symmetry.
- **Plain z-score > 3:** non-robust. The outliers inflate the std they are measured against (masking). Avoid it as a primary tool.
- **Multivariate:** robust Mahalanobis distance (`MinCovDet`), Isolation Forest, LOF. See [10_anomaly_detection](../10_anomaly_detection/).
- **Model-based:** large residuals from a reasonable model ([regression error_analysis.md](../03_regression/error_analysis.md)).

## 4. Assumptions and limitations
- No statistical rule can tell an error from a real extreme. Domain knowledge and physical limits decide.
- In skewed distributions, IQR and z rules flag the natural tail. Apply them on a transformed scale (log) or interpret them with care.
- Removing outliers from the **test** set makes the evaluation optimistic. Production data will contain them.
- Outlier treatment that uses data statistics (e.g. winsorising at the training 1st/99th percentiles) is a fitted transformation, so it belongs in the pipeline.

## 5. Step-by-step procedure
1. **Visualise first.** Box plots, histograms on linear and log scales, and scatter plots against related variables and against time.
2. **Apply physical and validity limits.** This was mostly done in [data_quality.md](data_quality.md). Impossible values are errors.
3. **Flag statistical outliers** with robust rules (IQR on the appropriate scale, robust z) and with consistency rules (for multivariate cases).
4. **Inspect the flagged rows in full.** Look at the other columns, the timestamp and the batch notes. Do they share a reactor, a supplier, a time window?
5. **Classify** each group as error / real extreme / different population, with a domain expert where possible.
6. **Treat** according to the classification (§8).
7. **Assess impact.** Compare model results with and without the treatment, under CV, and check how much the treatment changes the conclusions.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cookbook_data import make_process_yield

df = make_process_yield().drop_duplicates()
df["temperature_c"] = df["temperature_c"].mask(df["temperature_c"] == -999)

def iqr_flags(s: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)

def robust_z(s: pd.Series) -> pd.Series:
    mad = (s - s.median()).abs().median()
    return (s - s.median()) / (1.4826 * mad)

summary = []
for col in ["temperature_c", "pressure_bar", "feed_rate_kg_h", "impurity_ppm", "yield_kg"]:
    s = df[col].dropna()
    summary.append({"col": col, "skew": round(s.skew(), 2),
                    "iqr_1.5": int(iqr_flags(s).sum()),
                    "iqr_3": int(iqr_flags(s, 3).sum()),
                    "iqr_1.5_on_log": int(iqr_flags(np.log(s.clip(lower=1e-9))).sum()),
                    "robust_z>3.5": int((robust_z(s).abs() > 3.5).sum())})
print(pd.DataFrame(summary).to_string(index=False))
```

```python
# Visualise: linear vs log scale, and against time
fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
df["yield_kg"].plot.hist(bins=60, ax=axes[0], title="yield_kg (linear)")
np.log10(df["yield_kg"]).plot.hist(bins=60, ax=axes[1], title="log10(yield_kg)")
axes[2].scatter(df["start_time"], df["yield_kg"], s=4, alpha=0.5)
axes[2].set_yscale("log"); axes[2].set_title("yield over time (log)")
plt.tight_layout(); plt.show()
```

```python
# Multivariate consistency: reactor vs. jacket temperature should differ by ~6 °C
diff = df["jacket_temperature_c"] - df["temperature_c"]
flag = robust_z(diff.dropna()).abs() > 3.5
print("inconsistent temperature pairs:", int(flag.sum()))

# Robust Mahalanobis distance on a few related process variables
from sklearn.covariance import MinCovDet
cols = ["temperature_c", "pressure_bar", "residence_time_min", "feed_rate_kg_h"]
Z = df[cols].dropna()
mcd = MinCovDet(random_state=0).fit(Z)
d2 = mcd.mahalanobis(Z)                      # squared robust distances
from scipy.stats import chi2
thr = chi2.ppf(0.999, df=len(cols))          # under multivariate normality ~ chi2(p)
print("multivariate outliers:", int((d2 > thr).sum()), "of", len(Z))
```

```python
# Inspect flagged rows in full context: do they share something?
flag_rows = df.loc[iqr_flags(np.log(df["yield_kg"]), k=3)]
print(flag_rows[["batch_id", "start_time", "reactor", "supplier", "feed_rate_kg_h", "yield_kg"]])
# Here: tiny yields (~0.1–0.6 kg) with normal inputs → unit errors, not process events.
```

```python
# Treatments that belong INSIDE a pipeline (fitted on training data only)
from sklearn.base import BaseEstimator, TransformerMixin

class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to [q_low, q_high] quantiles learned on the training data."""
    def __init__(self, q_low=0.01, q_high=0.99):
        self.q_low, self.q_high = q_low, q_high
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.low_, self.high_ = X.quantile(self.q_low), X.quantile(self.q_high)
        return self
    def transform(self, X):
        return pd.DataFrame(X).clip(self.low_, self.high_, axis=1)

w = Winsorizer().fit(df[["impurity_ppm"]])
print(w.low_.round(1).to_dict(), w.high_.round(1).to_dict())
```

Robust alternatives to removing points:
- **Robust loss:** `HuberRegressor`, or `HistGradientBoostingRegressor(loss="absolute_error")` / `loss="quantile"`.
- **Transformation:** log or Yeo-Johnson ([transformations.md](../02_preprocessing/transformations.md)) pulls in the long tails.
- **Tree models:** split points depend only on the order of a feature's values, so extreme feature values have limited influence. Extreme **target** values still affect squared-error trees.

## 7. Interpretation
- **IQR on the linear scale flags many points for skewed variables** (`impurity_ppm`, `feed_rate_kg_h`), but few on the log scale. That indicates a natural long tail, not errors.
- **A tight cluster far from the bulk** (the log10 yield cluster near −0.5 to 0) is typical of unit or recording errors.
- **Outliers concentrated in time** suggest an event: a calibration, a start-up, a sensor failure. Check the logbook.
- **Outliers concentrated in a category** (one supplier, one reactor) may be a *legitimate* population. Removing them would hide a real effect.
- **Mahalanobis χ² thresholds assume normality.** With skewed variables, treat the flags as candidates for inspection, not a verdict.

## 8. Decision framework
```
Flagged point
├── Physically impossible / violates validity rule? ──► ERROR → correct if unambiguous, else NaN/drop (+ log)
├── Explained by a known event (start-up, maintenance, test run)?
│     ├── Event is out of scope for the use case ──► exclude BY RULE (apply the same rule in production)
│     └── Event occurs in production too ──► keep; add an event/regime feature
└── Plausible extreme of the same process ──► KEEP
      ├── Linear model distorted? → robust loss (Huber), transformation, or winsorise inside the pipeline
      ├── Squared-error metric dominated by a few points? → also report MAE / quantile errors
      └── Tree model → usually robust to extreme feature values; check target extremes
```
- **IF** you cannot decide whether a point is an error → keep it, flag it, and test the sensitivity of the results (with vs. without).
- **IF** removing a handful of points changes the conclusions → the conclusions are fragile. Report that.
- **IF** the outliers are what the stakeholder cares about (failures, off-spec batches) → do not remove them. They may be the problem itself (classification of off-spec, anomaly detection).

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Automatically dropping |z| > 3 | Removes legitimate tails; masking makes the rule unreliable | Robust rules + classification by kind |
| Removing outliers from the test set | Optimistic evaluation | Treat test data as production data |
| Applying IQR to skewed data on the linear scale | Many false flags | Check skewness; apply on the log scale |
| Winsorising using full-data quantiles | Leakage | Fit inside the pipeline |
| Removing a rare category's points as "outliers" | Biased model for that segment | Check concentration by category first |
| Ignoring multivariate inconsistencies | Errors pass univariate checks | Physical consistency rules, robust Mahalanobis |

## 10. Checklist
- [ ] Distributions plotted on linear and log scales, and over time.
- [ ] Validity limits applied (errors handled).
- [ ] Robust univariate flags computed on the right scale.
- [ ] Multivariate / consistency checks run for related variables.
- [ ] Flagged rows inspected in full context.
- [ ] Each flagged group classified (error / real / different population).
- [ ] Treatment chosen per class; fitted treatments inside the pipeline.
- [ ] Sensitivity of the results to the treatment checked.
- [ ] Test set not cleaned beyond what production cleaning would do.

## 11. References
- Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley. Origin of the box plot and the fences.
- Iglewicz, B. & Hoaglin, D. (1993). *How to Detect and Handle Outliers*. ASQC Quality Press. The modified z-score with the 3.5 threshold.
- Rousseeuw, P. J. & Van Driessen, K. (1999). "A Fast Algorithm for the Minimum Covariance Determinant Estimator." *Technometrics* 41(3), 212–223.
- NIST/SEMATECH *e-Handbook of Statistical Methods*, §1.3.5.17 "Detection of Outliers". https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
- scikit-learn, *Robust covariance estimation and Mahalanobis distances relevance*. https://scikit-learn.org/stable/auto_examples/covariance/plot_mahalanobis_distances.html
- scikit-learn, *HuberRegressor*. https://scikit-learn.org/stable/modules/linear_model.html#huber-regression
