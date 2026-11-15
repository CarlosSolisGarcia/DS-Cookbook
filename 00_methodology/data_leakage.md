# Data Leakage: Recognising and Preventing It

> **Folder:** 00_methodology · **Prev:** [decision_framework.md](decision_framework.md) · **Next:** [modeling_fundamentals.md](modeling_fundamentals.md) · **Related:** [problem_framing.md](problem_framing.md), [cross_validation.md](../06_model_improvement/cross_validation.md), [ts validation](../05_time_series/validation.md)
>
> *Added to the original architecture because leakage cuts across every stage and family. The other recipes link here instead of repeating this material.*

---

## 1. Purpose

**What it solves.** Leakage happens when information that will not be available at prediction time reaches the model during training or evaluation. The result is an optimistic score that collapses in production. This recipe lists the forms leakage takes, the ways to detect them and a prevention protocol.

---

## 2. When to use it

**When to use it.** Read it once carefully, then come back here whenever a result looks too good, whenever CV and production disagree, and before any final evaluation.

---

## 3. Conceptual foundation

Leakage is any violation of this rule: **the model, its features, its preprocessing and its evaluation may use only information that would be available at the moment of prediction, for the unit being predicted.**

### 3.1 Taxonomy

| Type | What happens | Example (cookbook datasets) |
|---|---|---|
| **Target leakage** (feature leakage) | A feature is a consequence or proxy of the target | `qc_purity_pct` is measured *after* the batch and correlates with yield; `downtime_hours_30d` is an outcome of failure |
| **Temporal leakage** (look-ahead) | Features or training rows come from the future relative to the prediction | Rolling mean centred on t (includes t+1…); random K-fold on drifting data; filling gaps with later values |
| **Train–test contamination** (preprocessing leakage) | Statistics computed on the full data before splitting | Scaling, imputation, target encoding or feature selection fitted on all rows |
| **Group leakage** (entity leakage) | The same entity appears in training and test; the model memorises the entity | The same `machine_id` in both folds; replicate measurements of one sample split apart |
| **Duplicate leakage** | Identical or near-identical rows on both sides of the split | The 10 duplicated batches in `make_process_yield` |
| **Selection / tuning leakage** | Test data influence modelling choices | Choosing features, thresholds or hyperparameters after looking at test performance |
| **Label-definition leakage** | The label's construction uses information after t₀ in a way that also shapes the features | A "failure" flag back-filled using later diagnoses that also updated sensor records |

### 3.2 Why it is so common

Leakage is rarely visible in the code. It lives in the **semantics** of the data: when a column was recorded, by whom, and whether it was updated later. That is why the prediction-time contract from [problem_framing.md](problem_framing.md) is the main defence. Code-level fixes, such as pipelines, catch only the train–test contamination type.

---

## 4. Assumptions and limitations

- The prediction-time contract assumes you can find out *when* each column is created. That knowledge comes from the data owners and system documentation, not from the data itself.
- Code-level defences (pipelines, splitters) only catch preprocessing, split and group leakage. Target and temporal leakage in the *semantics* of the columns need human audit.
- Some leakage is subtle and only shows up in production (a column back-filled after the fact). Monitoring the production performance is the final safeguard.

---

## 5. Step-by-step procedure

### Step 1 — Audit every column's timing
For each column ask: *When is this value created? Can it be updated after creation? Would it exist, with this value, at t₀ in production?* Record the answer in a data dictionary (see [initial_inspection.md](../01_data_understanding/initial_inspection.md)).

### Step 2 — Screen for suspiciously strong single-feature predictors
Fit a quick univariate model per feature (or compute a univariate AUC / R²). Any single feature that is nearly perfect on a hard problem deserves a timing audit.

### Step 3 — Choose the split according to structure
Use time order and groups as described in [validation_reference.md](../08_quick_references/validation_reference.md). Deduplicate before splitting.

### Step 4 — Put all data-dependent preprocessing inside a `Pipeline`
Imputers, scalers, encoders, feature selectors and resamplers (SMOTE) must be fitted on training folds only. A `Pipeline` passed to `cross_validate` does this automatically.

### Step 5 — Separate validation from test
Tune and select on validation folds. Evaluate on the test set once.

### Step 6 — Run the "too good to be true" test
Compare the score with domain expectations and with the baseline. Big unexplained jumps call for investigation, not celebration.

---

## 6. Python implementation

### 6.1 Detecting target leakage with univariate screening

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

from cookbook_data import make_equipment_failure

df = make_equipment_failure()
y = df["failure_30d"]
groups = df["machine_id"]

def univariate_auc(df, y, groups, exclude):
    """Cross-validated AUC of a shallow tree on each numeric column alone.
    A very high value for a single column is a leakage alarm, not a success."""
    out = {}
    for col in df.select_dtypes("number").columns.difference(exclude):
        model = make_pipeline(SimpleImputer(strategy="median"),
                              DecisionTreeClassifier(max_depth=3, random_state=0))
        out[col] = cross_val_score(model, df[[col]], y, cv=GroupKFold(5),
                                   groups=groups, scoring="roc_auc").mean()
    return pd.Series(out).sort_values(ascending=False)

print(univariate_auc(df, y, groups, exclude=["failure_30d"]).round(3))
# downtime_hours_30d ≈ 0.98 → an outcome of the failure; it must be removed.
```

### 6.2 Preprocessing leakage: wrong vs. right

The effect is largest with **supervised** preprocessing such as feature selection or target encoding. The classic demonstration uses pure noise:

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

rng = np.random.default_rng(0)
X_noise = rng.normal(size=(200, 5000))       # 5000 features of pure noise
y_noise = rng.integers(0, 2, 200)            # random labels: true AUC = 0.5
cv = StratifiedKFold(5, shuffle=True, random_state=0)

# WRONG: select features on all data, then cross-validate
X_sel = SelectKBest(f_classif, k=20).fit_transform(X_noise, y_noise)
wrong = cross_val_score(LogisticRegression(max_iter=1000), X_sel, y_noise, cv=cv, scoring="roc_auc")

# RIGHT: selection is a pipeline step, refitted inside every training fold
pipe = Pipeline([("select", SelectKBest(f_classif, k=20)),
                 ("clf", LogisticRegression(max_iter=1000))])
right = cross_val_score(pipe, X_noise, y_noise, cv=cv, scoring="roc_auc")
print(f"leaky AUC = {wrong.mean():.2f}   honest AUC = {right.mean():.2f}")
# leaky ≈ 0.9+ on pure noise; honest ≈ 0.5
```

Unsupervised steps (scaling, median imputation) leak much less in practice, but put them in the pipeline anyway. Doing so costs nothing and the code then matches what happens in deployment.

### 6.3 Group leakage

The clearest case is **experimental replicates**. Several measurements of the same formulation share the same inputs and a formulation-specific effect that the inputs do not explain:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

rng = np.random.default_rng(0)
n_form, n_rep = 80, 5                                   # 80 formulations × 5 replicate measurements
comp = rng.uniform(0, 1, (n_form, 4))                   # composition of each formulation
form_effect = rng.normal(0, 1.0, n_form)                # formulation effect NOT explained by composition
y_form = 2 * comp[:, 0] - comp[:, 1] ** 2 + form_effect
X_rep = np.repeat(comp, n_rep, axis=0) + rng.normal(0, 0.01, (n_form * n_rep, 4))  # tiny dosing noise
y_rep = np.repeat(y_form, n_rep) + rng.normal(0, 0.2, n_form * n_rep)              # measurement noise
g_rep = np.repeat(np.arange(n_form), n_rep)

rf = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)
r2_random = cross_val_score(rf, X_rep, y_rep, cv=KFold(5, shuffle=True, random_state=0), scoring="r2")
r2_group = cross_val_score(rf, X_rep, y_rep, cv=GroupKFold(5), groups=g_rep, scoring="r2")
print(f"R² random K-fold = {r2_random.mean():.2f}   grouped by formulation = {r2_group.mean():.2f}")
# ≈ 0.92 vs ≈ 0.0: under random K-fold the model "predicts" a replicate from its siblings.
# For a NEW formulation it has almost no skill, because composition explains little here.
```

Always run the same comparison on your own data. Here is the machine-fleet version:

```python
from sklearn.ensemble import RandomForestClassifier

features = ["age_years", "load_pct", "operating_hours", "days_since_maintenance",
            "vibration_rms", "temperature_c", "error_count_7d"]
rf_clf = make_pipeline(SimpleImputer(strategy="median"),
                       RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1))
for name, cv_, g in [("random K-fold", KFold(5, shuffle=True, random_state=0), None),
                     ("GroupKFold", GroupKFold(5), groups)]:
    s = cross_val_score(rf_clf, df[features], y, cv=cv_, groups=g, scoring="average_precision")
    print(f"PR-AUC {name:13s}: {s.mean():.3f} ± {s.std():.3f}")
# In this dataset the gap is small: the features change month to month, so machines are hard to
# "recognise". The size of the gap is an empirical question. Measure it rather than assume it.
```

### 6.4 Temporal leakage in rolling features

```python
s = pd.Series(np.arange(10, dtype=float))
leaky = s.rolling(3, center=True).mean()     # uses t-1, t, t+1   ← LEAKS the future
safe = s.shift(1).rolling(3).mean()          # uses t-3, t-2, t-1 ← only the past
print(pd.DataFrame({"s": s, "leaky_centered": leaky, "safe_past_only": safe}))
```

---

## 7. Interpretation

- **Univariate screen.** Real predictors on hard problems typically have univariate AUC of 0.55–0.75. A single feature at 0.95+ is almost always leakage, or a trivial re-encoding of the target.
- **Random vs. grouped CV gap.** A large gap (as in the replicate example) means the model exploits entity identity. If production predicts on **new** entities, trust the grouped estimate. If it predicts on the same entities in the future, use a temporal split, and entity features may then be legitimate.
- **Leaky vs. honest pipeline gap.** Any noticeable gap shows that preprocessing decisions were adapting to validation data.
- **CV good, production bad.** Classic leakage or drift. Re-audit column timing first, then the split.

---

## 8. Decision framework

**IF** a feature's value can be updated after t₀ (status fields, "last modified" values, aggregated totals)
→ Use a historical snapshot ("as-of" join) with the value at t₀, or drop the feature.

**IF** a feature is a near-perfect predictor
→ Audit its timing with the data owner before using it. If it is legitimate (e.g. a physical law), confirm with domain experts and keep it.

**IF** the data have repeated measures per entity and the model will see new entities
→ Use `GroupKFold` / `StratifiedGroupKFold`.

**IF** the data are time-ordered and the model is used on the future
→ Use a chronological split, and build features with past-only windows (`shift` before `rolling`).

**IF** you need target encoding
→ Use cross-fitting (`sklearn.preprocessing.TargetEncoder` does this in `fit_transform`), inside the pipeline.

**IF** you looked at the test set to make a decision
→ The test estimate is now optimistic. Report it as such, or obtain fresh test data.

---

## 9. Common mistakes

| Mistake | Consequence | Prevention |
|---|---|---|
| `scaler.fit(X)` on all data before `train_test_split` | Mild contamination; becomes a habit that grows into serious leaks | Use `Pipeline` always |
| Feature selection / target encoding outside CV | Severe optimism, even on noise | Put supervised steps in the pipeline |
| `df.fillna(method="bfill")` on a time series | Future values used to fill the past | Forward-fill or interpolate with past data only, inside the training window |
| Centred rolling windows | Uses future values | `shift(1)` before rolling |
| Random split with duplicated rows | Memorisation is rewarded | Deduplicate before splitting; check near-duplicates |
| Oversampling before splitting | Synthetic copies of validation points end up in training | Resample inside the training fold (`imblearn.pipeline.Pipeline`) |
| Using "date of label" features | Directly encodes the label | Audit timestamps relative to t₀ |

---

## 10. Checklist

- [ ] Every column has a documented creation time relative to t₀.
- [ ] Post-outcome columns removed (`LEAKAGE_COLUMNS` in the cookbook data).
- [ ] Univariate leakage screen run; strong predictors audited.
- [ ] Duplicates removed before splitting.
- [ ] Split respects time and groups as in deployment.
- [ ] All fitted preprocessing inside `Pipeline` / `ColumnTransformer`.
- [ ] Resampling only inside training folds.
- [ ] Rolling / lag features use past-only windows.
- [ ] Test set used once, after all decisions.

---

## 11. References

- Kaufman, S., Rosset, S., Perlich, C. & Stitelman, O. (2012). "Leakage in data mining: Formulation, detection, and avoidance." *ACM TKDD* 6(4), 1–21. https://doi.org/10.1145/2382577.2382579
- Kapoor, S. & Narayanan, A. (2023). "Leakage and the reproducibility crisis in machine-learning-based science." *Patterns* 4(9). https://doi.org/10.1016/j.patter.2023.100804
- scikit-learn: *Common pitfalls — Data leakage*. https://scikit-learn.org/stable/common_pitfalls.html#data-leakage
- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd ed., §7.10.2 "The Wrong and Right Way to Do Cross-validation". https://hastie.su.domains/ElemStatLearn/
- scikit-learn: *Target Encoder's internal cross fitting*. https://scikit-learn.org/stable/modules/preprocessing.html#target-encoder
