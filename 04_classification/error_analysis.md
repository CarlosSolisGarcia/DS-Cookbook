# Classification Error Analysis: False Positives, False Negatives and Segments

> **Folder:** 04_classification · **Prev:** [calibration.md](calibration.md) · **Next:** [iterative_improvement.md](../06_model_improvement/iterative_improvement.md) · **Related:** [regression error_analysis.md](../03_regression/error_analysis.md), [evaluation.md](evaluation.md)

---

## 1. Purpose
Understand **which** cases the classifier gets wrong and **why**, so that you can choose the next improvement: features, labels, data, segment treatment or threshold.

## 2. When to use it
After each meaningful evaluation, on out-of-fold predictions at the chosen operating point. Use it as well when stakeholders report "the model flags the wrong machines".

## 3. Conceptual foundation
Errors come in two types, and each has different typical causes:

| Error | Possible causes | Typical investigation |
|---|---|---|
| **False negatives** (missed failures) | Failure mode without a precursor in the features; sudden failures; label noise (failure logged late or with the wrong date); a sensor missing for that machine | Compare FN with TP: which features differ? Which failure modes dominate the FNs? |
| **False positives** (false alarms) | Genuinely high risk without failure (prevented by chance or by maintenance); label noise (failure not logged); a sensor artefact | Compare FP with TN; check maintenance logs: was an intervention done? |
| **Confident errors** (p̂ ≥ 0.5 with y = 0, or p̂ ≈ 0.01 with y = 1) | Data errors, leakage reversal, rare regimes | Inspect them individually |

Two further views:
- **Slices:** performance by segment (site, type, firmware, age band, period). A model can be good overall and fail in a slice.
- **Score distributions:** histograms of p̂ for positives vs negatives. Overlap shows the region where the features cannot separate the classes.

## 4. Assumptions and limitations
- Labels are themselves uncertain. Some "errors" are label errors. Look at the raw records before concluding that the model is wrong.
- Slices with few positives produce noisy metrics. Show counts and CIs.
- In predictive maintenance, an intervention after an alarm can prevent the failure, so the label becomes 0 and the alarm looks like an FP. This is a **feedback loop**. Track interventions ([problem_framing.md](../00_methodology/problem_framing.md)).

## 5. Step-by-step procedure
1. **Collect OOF probabilities** and the predicted class at the operating point, with all metadata.
2. **Score distributions** of positives vs negatives.
3. **Slice table:** AP, recall and precision per segment, with prevalence and counts.
4. **FN profile:** compare the feature distributions of FN vs TP. For multiclass sources (failure mode), tabulate the modes among the FNs.
5. **FP profile:** compare FP vs TN; check external information (maintenance logs).
6. **Inspect the most confident errors** (top 10 FP by p̂, bottom 10 FN by p̂).
7. **Formulate hypotheses → actions** (§8). Prioritise by cost impact (C_FN × #FN, C_FP × #FP).

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cookbook_data import make_equipment_failure

df = make_equipment_failure(include_leakage=True)     # failure_mode kept ONLY for analysis, not as a feature
num = ["age_years", "load_pct", "operating_hours", "days_since_maintenance",
       "vibration_rms", "temperature_c", "error_count_7d"]
cat = ["machine_type", "site", "firmware"]
y, groups = df["failure_30d"].to_numpy(), df["machine_id"]
model = make_pipeline(ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat)]), LogisticRegression(C=0.05, max_iter=2000))
cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
df["p"] = cross_val_predict(model, df[num + cat], y, cv=cv, groups=groups, method="predict_proba")[:, 1]
THR = 0.05                                          # cost-optimal threshold from threshold_optimization.md
df["pred"] = (df["p"] >= THR).astype(int)
df["outcome"] = np.select([(df.pred == 1) & (y == 1), (df.pred == 1) & (y == 0), (df.pred == 0) & (y == 1)],
                          ["TP", "FP", "FN"], "TN")
print(df["outcome"].value_counts())
```

```python
# Score distributions: where do the classes overlap?
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.hist(df.loc[y == 0, "p"], bins=60, alpha=0.6, density=True, label="no failure")
ax.hist(df.loc[y == 1, "p"], bins=60, alpha=0.6, density=True, label="failure")
ax.axvline(THR, color="k", ls="--", label="threshold"); ax.set_xlim(0, 0.5); ax.legend(); plt.show()
```

```python
# Slice table with counts (beware small slices)
def slice_table(d, col):
    def f(g):
        yy = (g["failure_30d"] == 1)
        tp, fn, fp = ((g.pred == 1) & yy).sum(), ((g.pred == 0) & yy).sum(), ((g.pred == 1) & ~yy).sum()
        return pd.Series({"n": len(g), "positives": int(yy.sum()), "prevalence": yy.mean(),
                          "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
                          "AP": average_precision_score(yy, g["p"]) if yy.any() and (~yy).any() else np.nan})
    return d.groupby(col, observed=True).apply(f, include_groups=False).round(3)

df["age_band"] = pd.cut(df["age_years"], [0, 5, 10, 15, 25])
df["vib_sensor"] = np.where(df["vibration_rms"].isna(), "missing", "present")
for col in ["machine_type", "age_band", "vib_sensor", "firmware"]:
    print(slice_table(df, col), "\n")
```

```python
# FN profile: which failure modes are missed? How do FN differ from TP?
pos = df[df["failure_30d"] == 1]
print(pd.crosstab(pos["failure_mode"], pos["outcome"], normalize="index").round(2))
print(pos.groupby("outcome")[num].median().T.round(2))     # FN vs TP medians

# FP profile: FP vs TN medians (what makes non-failing machines look risky?)
neg = df[df["failure_30d"] == 0]
print(neg.groupby("outcome")[num].median().T.round(2))
```

```python
# Most confident errors: inspect individually (in a real project: join maintenance logs, sensor plots)
cols = ["machine_id", "snapshot_date", "machine_type", "vibration_rms", "temperature_c", "error_count_7d", "p", "failure_mode"]
print("Confident false alarms:\n", df[df.outcome == "FP"].nlargest(5, "p")[cols].round(3))
print("Confident misses:\n", df[df.outcome == "FN"].nsmallest(5, "p")[cols].round(3))
```

## 7. Interpretation
What this run shows (check it against your own output):
- **Operating point (t = 0.05):** recall ≈ 74%, precision ≈ 13%, about 28% of machine-months flagged. With a cost ratio of 20 : 1 this is cost-optimal, but the alarm load may be unacceptable to the planner. Discuss capacity ([threshold_optimization.md](threshold_optimization.md)).
- **Score overlap is large.** Many failures have a low p̂. The features carry only partial information, which is typical of predictive maintenance.
- **Slices:** type-L machines have clearly lower recall (≈ 58%) and AP. Positives there are rarer (3.5%), and their failures are less often preceded by high load or temperature. Firmware v3 also has lower recall: the update removed part of the electrical failure mechanism, and the remaining failures are harder to anticipate. **Actions:** type-specific features or interactions (type × temperature); check whether v3 changed what the error counters measure.
- **`vib_sensor = missing`:** recall is similar to machines with a sensor. The missing indicator and age partly compensate. In real fleets this slice is often much worse. Always check it, because it tells you where *new sensors* would pay off.
- **Failure modes among the FNs** are spread evenly (≈ 25% of each mode missed). No single mode is being ignored. If one mode dominated the FNs, it would call for mode-specific features.
- **FN vs TP medians:** FNs have vibration, temperature and load close to normal values. These are "silent" failures that the current signals do not precede. **Action:** this is an information limit. Look for new signals (current draw, oil analysis, acoustic data) rather than more modelling.
- **FP vs TN medians:** FPs look genuinely risky (higher vibration, longer since maintenance, more errors). They are candidates for prevented failures (maintenance after the snapshot), label noise, or simply the randomness of failure. **Action:** join the maintenance logs. If interventions explain them, consider redefining the label as "failure *or* corrective maintenance".

## 8. Decision framework
| Finding | Likely cause | Action |
|---|---|---|
| A failure mode dominates the FNs | No precursor features for that mode | Domain features for the mode; separate model; new sensors |
| A slice with low recall and missing sensors | Missing information | Data collection; proxy features; separate reporting |
| Confident FPs with plausible risk | Prevented failures / label noise | Join intervention data; redefine the label; feedback-loop aware evaluation |
| Errors concentrated in a recent period | Drift (firmware change, new operating regime) | Time-aware validation; retraining; drift features |
| Errors random, large score overlap | Information ceiling | More informative data, not more modelling |
| Slice precision differs but recall is similar | Prevalence differs by slice | Per-slice thresholds only if costs differ; report the prevalence |
| Confident errors that look like data errors | Data quality | Fix the data; add validation rules ([data_quality.md](../01_data_understanding/data_quality.md)) |

Prioritise by **cost impact**: C_FN × (#FN in the pattern) vs C_FP × (#FP in the pattern).

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Analysing in-sample predictions | Errors hidden | OOF predictions |
| Treating all FPs as model failures | Missing prevented failures / label noise | Join intervention data |
| Over-reading small slices | Chasing noise | Counts + CIs; require repeated evidence |
| Fixing everything with a threshold change | Moves errors between FP and FN without reducing them | Threshold is for costs; features/data for errors |
| Using outcome columns (e.g. failure_mode) as features after seeing them in error analysis | Leakage | Analysis-only columns kept separate |

## 10. Checklist
- [ ] OOF probabilities + operating-point predictions with metadata.
- [ ] Score distributions of the classes.
- [ ] Slice table with counts and prevalence.
- [ ] FN profile (modes, features) and FP profile (vs TN, external logs).
- [ ] Confident errors inspected individually.
- [ ] Hypotheses → actions, prioritised by cost impact.
- [ ] Next experiment logged.

## 11. References
- scikit-learn User Guide, *Classification metrics*. https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
- Chung, Y., Kraska, T., Polyzotis, N., Tae, K. H. & Whang, S. E. (2019). "Slice Finder: Automated Data Slicing for Model Validation." *ICDE*. https://arxiv.org/abs/1807.06068
- Northcutt, C., Jiang, L. & Chuang, I. (2021). "Confident Learning: Estimating Uncertainty in Dataset Labels." *JAIR* 70. https://doi.org/10.1613/jair.1.12125
- Ng, A. *Machine Learning Yearning*, error-analysis chapters. https://info.deeplearning.ai/machine-learning-yearning-book
