# Uncertainty Estimation Across Problem Families: Ensembles, Bootstrap and Conformal Prediction Sets

> **Folder:** 20_advanced_methodologies · **Next:** [robustness_and_sensitivity.md](robustness_and_sensitivity.md) · **Related:** [uncertainty.md](../03_regression/uncertainty.md) (regression intervals, CQR: not repeated), [calibration.md](../04_classification/calibration.md), [small_data_and_uncertainty.md](../18_scientific_machine_learning/small_data_and_uncertainty.md) (GPs)

---

## 1. Purpose
Attach honest uncertainty to predictions in every problem family. That means telling apart the noise in the data (aleatoric uncertainty) from the model's lack of knowledge (epistemic uncertainty), producing calibrated **prediction sets** for classification, and choosing methods that work for the model at hand.

## 2. When to use it
Decisions with risk (reject a part? schedule maintenance? trust this soft-sensor value?), active learning / experiment selection, out-of-domain detection, and reporting to stakeholders.

## 3. Conceptual foundation
| Method | Captures | Works with | Notes |
|---|---|---|---|
| Quantile / distributional models | Aleatoric | Regression (GBM quantile loss, NGBoost-type) | No epistemic uncertainty |
| **Conformal prediction** | Total, with a coverage guarantee (marginal) | Any model; regression intervals, **classification sets** | Needs exchangeability; split or cross-conformal |
| **Ensembles** (seeds, bootstrap/bagging, different models) | Mostly epistemic (disagreement) | Any | Simple, strong; deep ensembles for NNs |
| Gaussian processes / Bayesian models | Both | Small data | [small_data_and_uncertainty.md](../18_scientific_machine_learning/small_data_and_uncertainty.md) |
| MC dropout | Approximate epistemic | Neural networks | Cheap but often under-dispersed |
| Calibration (Platt/isotonic) | Probability reliability | Classifiers | Not sets; [calibration.md](../04_classification/calibration.md) |

**Conformal classification (split, LAC/"score" method):** on a calibration set compute the scores s_i = 1 − p̂(y_i | x_i). Let q̂ be the ⌈(n+1)(1−α)⌉/n quantile. For a new x, the prediction set is {k : 1 − p̂(k | x) ≤ q̂}. It contains the true class with probability ≥ 1 − α on average. Its size adapts: a single class when the model is confident, several classes when it is ambiguous, and a large set flags hard cases.

**Epistemic uncertainty as an out-of-domain signal:** in principle, ensemble disagreement grows for inputs unlike the training data. In practice this holds for diverse members (and for GPs), but **not** for tree ensembles in extrapolation (see §7). Do not rely on it alone.

## 4. Assumptions and limitations
- Conformal guarantees are marginal and require exchangeable calibration/test data. Under drift, recalibrate on recent data ([drift_and_monitoring.md](drift_and_monitoring.md)).
- Ensemble spread is not calibrated by itself. Scale it with a calibration set, or combine it with conformal methods.
- Small calibration sets make q̂ noisy. Use a few hundred examples, or cross-conformal methods.

## 5. Step-by-step procedure
1. Decide what the decision needs: an interval, a set, a probability, or an out-of-domain flag.
2. Split train / calibration / test (respecting time/groups).
3. Fit the model on train. Compute the conformal scores on calibration. Build the sets/intervals for test.
4. Evaluate the **coverage** and the **set size/width**, overall and per segment/class.
5. For epistemic uncertainty: train an ensemble (5–10 members), compute the spread, and check that it grows out of domain.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cookbook_data import make_equipment_failure, load_clean_process_yield

# --- Conformal prediction SETS for multiclass failure-mode classification (failures only) ---
df = make_equipment_failure(include_leakage=True)
df = df[df["failure_mode"] != "none"]
num = ["age_years", "load_pct", "operating_hours", "days_since_maintenance", "vibration_rms", "temperature_c", "error_count_7d"]
cat = ["machine_type", "site", "firmware"]
X, y = df[num + cat], df["failure_mode"].to_numpy()
X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)
X_cal, X_te, y_cal, y_te = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=0)
clf = make_pipeline(ColumnTransformer([("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                                       ("c", OneHotEncoder(handle_unknown="ignore"), cat)]),
                    LogisticRegression(C=0.5, max_iter=3000)).fit(X_tr, y_tr)
classes = clf.classes_; ci = {c: k for k, c in enumerate(classes)}
alpha = 0.10
s_cal = 1 - clf.predict_proba(X_cal)[np.arange(len(y_cal)), [ci[c] for c in y_cal]]
q = np.quantile(s_cal, min(1.0, np.ceil((len(s_cal) + 1) * (1 - alpha)) / len(s_cal)), method="higher")
P_te = clf.predict_proba(X_te)
sets = P_te >= 1 - q                                              # boolean matrix: class k is in the set
covered = sets[np.arange(len(y_te)), [ci[c] for c in y_te]]
print(f"top-1 accuracy: {np.mean(classes[P_te.argmax(1)] == y_te):.2f} | 90% conformal sets: coverage={covered.mean():.2f}, "
      f"mean set size={sets.sum(1).mean():.2f} of {len(classes)}")
print(pd.DataFrame({"true": y_te, "set size": sets.sum(1)}).groupby("true")["set size"].mean().round(2))
```

```python
# --- Ensemble disagreement as epistemic uncertainty (regression), in vs out of the training domain ---
d = load_clean_process_yield()
feats = ["temperature_c", "pressure_bar", "feed_rate_kg_h", "residence_time_min", "catalyst_age_days"]
train = d[d["feed_rate_kg_h"] < d["feed_rate_kg_h"].quantile(0.8)]          # train only on lower feed rates
test_in = d[(d["feed_rate_kg_h"] < d["feed_rate_kg_h"].quantile(0.8))].sample(200, random_state=0)
test_out = d[d["feed_rate_kg_h"] >= d["feed_rate_kg_h"].quantile(0.95)]     # extrapolation in feed rate
members = []
for seed in range(8):                                                       # bootstrap ensemble
    b = train.sample(frac=1.0, replace=True, random_state=seed)
    members.append(HistGradientBoostingRegressor(max_iter=200, random_state=seed).fit(b[feats], np.log(b["yield_kg"])))
def ens(Xd):
    P = np.column_stack([m.predict(Xd[feats]) for m in members]); return P.mean(1), P.std(1)
for name, t in [("in-domain", test_in), ("out-of-domain (high feed)", test_out)]:
    mu, sd = ens(t)
    print(f"{name:26s} ensemble SD (log scale) = {sd.mean():.3f} | abs error (log) = {np.mean(np.abs(np.log(t['yield_kg']) - mu)):.3f}")
```

## 7. Interpretation
- **Conformal sets are honest about a weak model:** top-1 accuracy is only ≈ 0.51 among three failure modes. The 90% sets reach their coverage (0.97 here, at or above the target, as guaranteed) by containing ≈ 2.6 of the 3 classes on average. The set size *is* the information: this classifier can rarely narrow the mode down. For a maintenance planner, "inspect two or three subsystems" is the truthful answer. A better model would show up as smaller sets at the same coverage.
- **Ensemble spread fails as an extrapolation signal here, and that is the lesson.** Out of domain (feed rates above the training range) the absolute error more than doubles (0.21 → 0.54 on the log scale), yet the bootstrap-ensemble SD is *lower* (0.19 → 0.17). Every tree member predicts the same flat value beyond the last split, so they agree perfectly while all being wrong. **Ensembles of similar models share their blind spots.** Detect extrapolation in the *input space* (range checks, the kNN distance to the training data, [small_data_and_uncertainty.md](../18_scientific_machine_learning/small_data_and_uncertainty.md)), or use models whose uncertainty grows away from the data (GPs, diverse ensembles mixing linear and tree members).

## 8. Decision framework
- **IF** you need a guaranteed coverage for intervals/sets → conformal (split; CQR for regression).
- **IF** you need to detect extrapolation / unfamiliar inputs → ensemble disagreement + applicability-domain checks; GPs for small data.
- **IF** probabilities feed into decisions → calibration first; conformal sets for "which classes are plausible".
- **IF** the data drift → recalibrate the conformal quantile on recent data; monitor the coverage.
- **IF** neural networks → deep ensembles (best), or MC dropout (cheap).

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Calibrating on training predictions | Under-coverage | A separate calibration split |
| Reporting coverage without the set size/width | Trivial solutions (always all classes) look perfect | Report both |
| Treating ensemble SD as calibrated, or as an OOD detector for trees | Over-confidence exactly where errors are largest | Input-space domain checks; calibrate; conformal |
| Ignoring per-class/segment coverage | Some groups under-covered | Conditional checks |

## 10. Checklist
- [ ] Uncertainty type needed (interval/set/probability/OOD flag) defined.
- [ ] Train/calibration/test split respecting structure.
- [ ] Coverage and size/width overall and per segment.
- [ ] Epistemic signal (ensemble/GP) checked in and out of domain.
- [ ] Recalibration plan under drift.

## 11. References
- Angelopoulos, A. N. & Bates, S. (2023). "Conformal Prediction: A Gentle Introduction." https://arxiv.org/abs/2107.07511
- Sadinle, M., Lei, J. & Wasserman, L. (2019). "Least Ambiguous Set-Valued Classifiers With Bounded Error Levels." *JASA* 114(525), 223–234.
- Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." *NeurIPS*. https://arxiv.org/abs/1612.01474
- Gal, Y. & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation." *ICML*. https://arxiv.org/abs/1506.02142
- Hüllermeier, E. & Waegeman, W. (2021). "Aleatoric and epistemic uncertainty in machine learning: an introduction to concepts and methods." *Machine Learning* 110, 457–506. https://doi.org/10.1007/s10994-021-05946-3
- MAPIE documentation. https://mapie.readthedocs.io/
