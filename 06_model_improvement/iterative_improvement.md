# Iterative Improvement: Choosing the Next Best Action

> **Folder:** 06_model_improvement · **Prev:** [model_interpretation.md](model_interpretation.md) · **Next:** [experiment_tracking.md](../07_experimentation/experiment_tracking.md) · **Related:** [project_lifecycle.md](../00_methodology/project_lifecycle.md) §8, [regression error_analysis.md](../03_regression/error_analysis.md), [classification error_analysis.md](../04_classification/error_analysis.md), [ts evaluation](../05_time_series/evaluation.md)

---

## 1. Purpose
Turn evaluation and error-analysis findings into a **prioritised backlog** of improvement actions, run them as controlled experiments, and decide when to stop. This is the general recipe for the improvement loop, applicable to every problem family.

## 2. When to use it
After the first end-to-end result, and after every major evaluation round.

## 3. Conceptual foundation
**Sources of error and their remedies** (in typical order of value in tabular projects):

| Source | Symptoms | Lever |
|---|---|---|
| **Problem framing** | Good metrics, no business value; wrong target/horizon | Re-frame ([problem_framing.md](../00_methodology/problem_framing.md)) |
| **Validation** | CV ≠ test/production | Fix the scheme ([cross_validation.md](cross_validation.md)) |
| **Data quality / labels** | Confident errors on obviously wrong rows; error ≈ label noise | Clean, relabel, audit ([data_quality.md](../01_data_understanding/data_quality.md)) |
| **Missing information** | Errors concentrated where a known driver is not measured; large score overlap | New data sources, sensors, joins |
| **Features** | Structured residuals vs features/time/segments | Domain features ([feature_engineering.md](../02_preprocessing/feature_engineering.md)) |
| **Model family / capacity** | Under/overfitting patterns | Different family, regularisation ([modeling_fundamentals.md](../00_methodology/modeling_fundamentals.md)) |
| **Hyperparameters** | Small gains, landscape mapping | Tuning ([hyperparameter_optimization.md](hyperparameter_optimization.md)) |
| **Decision layer** | Good ranking, poor decisions | Thresholds, calibration ([threshold_optimization.md](../04_classification/threshold_optimization.md)) |

**Estimating the ceiling.** Before investing, estimate how much improvement is possible:
- **Irreducible noise:** repeated measurements, a duplicate-label agreement rate, or lab measurement uncertainty give the noise floor. For yield measured with ±3% precision, no model can do much better than that.
- **Human or expert performance** (if relevant) and the best performance reported for similar problems.
- **Learning curves** ([modeling_fundamentals.md](../00_methodology/modeling_fundamentals.md) §6.3): would more data help?

**Prioritisation:** expected gain × probability of success / cost. The expected gain comes from error analysis (the error mass of the targeted pattern). The cost includes the time to implement *and* the long-term maintenance.

## 4. Assumptions and limitations
- Each iteration that looks at validation results slightly overfits the validation scheme. Many iterations on a small validation set lead to optimistic estimates. That is why the test set stays untouched until the end, and why fresh data are worth getting periodically.
- Improvements that are smaller than the noise are not improvements.

## 5. Step-by-step procedure
1. **Establish the current best** (the score with its CV std) and record it in the experiment log.
2. **Run error analysis** (the family recipe). List the patterns with their error mass.
3. **Write hypotheses → actions** as backlog items (template in §6), each with an expected gain, cost and evidence.
4. **Pick the top item.** Change **one thing** (or one coherent group) per experiment.
5. **Run the experiment** with the same validation scheme. Log the configuration, the results and the conclusion.
6. **Accept the change** if it beats the current best by more than the noise (paired folds) or is needed for correctness/robustness. Otherwise reject it and record why.
7. **Re-run error analysis** after accepted changes. The error landscape shifts.
8. **Check the stopping criteria** (§8). At the end, evaluate on the test set once and write the report.

## 6. Python implementation: a lightweight improvement loop

```python
import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from cookbook_data import load_clean_process_yield

df = load_clean_process_yield()
train = df.iloc[: int(0.8 * len(df))]
y = train["yield_kg"]
cv = TimeSeriesSplit(5)

@dataclass
class BacklogItem:
    hypothesis: str
    action: str
    evidence: str
    expected_gain: str
    cost: str
    status: str = "open"
    result: dict = field(default_factory=dict)

backlog = [
    BacklogItem("Temperature acts via a catalyst-specific optimum", "add (T - T_opt[catalyst])^2",
                "residual trend vs temperature & catalyst (error_analysis.md)", "large", "low"),
    BacklogItem("Yield scales multiplicatively with feed", "log target + log(feed)",
                "funnel-shaped residuals; Breusch–Pagan p << 0.05", "medium", "low"),
    BacklogItem("Supplier lots differ in quality", "one-hot supplier (rare → other)",
                "residual by supplier range ≫ se", "small", "low"),
]

def features(d, use_opt=False, use_logfeed=False):
    d = d.copy()
    if use_opt:
        d["temp_dev_sq"] = (d["temperature_c"] - d["catalyst"].map({"Pt": 190.0, "Pd": 182.0, "Ni": 195.0})) ** 2
    if use_logfeed:
        d["log_feed"] = np.log(d["feed_rate_kg_h"])
    return d

def build(use_opt=False, use_logfeed=False, use_supplier=False, log_target=False):
    num = ["temperature_c", "pressure_bar", "residence_time_min", "catalyst_age_days", "impurity_ppm"]
    num += ["temp_dev_sq"] if use_opt else []
    num += ["log_feed"] if use_logfeed else ["feed_rate_kg_h"]
    cat = ["reactor", "catalyst", "shift"] + (["supplier"] if use_supplier else [])
    pipe = make_pipeline(
        FunctionTransformer(features, kw_args={"use_opt": use_opt, "use_logfeed": use_logfeed}),
        ColumnTransformer([("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                           ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat)]),
        RidgeCV(alphas=np.logspace(-3, 3, 13)))
    return TransformedTargetRegressor(pipe, func=np.log, inverse_func=np.exp) if log_target else pipe

def run(config, name, log):
    t0 = time.time()
    s = -cross_val_score(build(**config), train, y, cv=cv, scoring="neg_mean_absolute_error")
    entry = {"name": name, "config": config, "MAE": s.mean(), "MAE_std": s.std(), "fold_MAE": s.round(2).tolist(),
             "seconds": round(time.time() - t0, 2)}
    log.append(entry)
    return entry

log = []
current = {"use_opt": False, "use_logfeed": False, "use_supplier": False, "log_target": False}
best = run(current, "baseline linear", log)
steps = [("use_opt", backlog[0]), (("use_logfeed", "log_target"), backlog[1]), ("use_supplier", backlog[2])]
for keys, item in steps:
    trial = dict(current, **{k: True for k in (keys if isinstance(keys, tuple) else (keys,))})
    r = run(trial, item.action, log)
    diffs = np.array(best["fold_MAE"]) - np.array(r["fold_MAE"])     # paired per-fold improvement
    accepted = diffs.mean() > best["MAE_std"] / 2 and (diffs > 0).mean() >= 0.8
    item.status, item.result = ("accepted" if accepted else "rejected"), {"MAE": round(r["MAE"], 2),
                                                                         "paired_gain": round(diffs.mean(), 2)}
    if accepted:
        current, best = trial, r

print(pd.DataFrame(log)[["name", "MAE", "MAE_std"]].round(2))
print(pd.DataFrame([asdict(b) for b in backlog])[["action", "status", "result"]])
# json.dump(log, open("reports/experiment_log.json", "w"), indent=2, default=str)
```

## 7. Interpretation
- **The log table** shows the trajectory: MAE 58 → 33 (catalyst-specific optimum, the largest error pattern) → 26 (log target + log feed: multiplicative structure) → 23 (supplier). All three were accepted. The supplier gain (≈ 3.7 kg, consistent across folds) is larger than its "small" prior estimate in the backlog. Evidence beats expectations, which is why each item is tested rather than assumed. Compare the final ≈ 23 kg with the regression workflow's best (≈ 22 kg): the loop has rediscovered the domain features step by step.
- **The acceptance rule** used here (a mean paired gain larger than half the fold std, with improvement in ≥ 80% of folds) is a pragmatic convention, not a statistical test. Set your own rule upfront and keep it fixed.
- **Rejected items are valuable.** They tell you where the remaining error is *not*. Record them so you do not repeat them.
- **When the remaining residuals look like noise** and the error approaches the measurement uncertainty, the modelling loop is done. Further gains require better data.

## 8. Decision framework
**What to do next (ordered checks):**
1. **IF** the CV and test/production disagree → fix validation first.
2. **IF** error analysis shows obvious data/label errors → fix the data.
3. **IF** errors concentrate in a pattern that a known driver explains → engineer that feature, or acquire that data.
4. **IF** under/overfitting is diagnosed → adjust the model family/regularisation.
5. **IF** only noise-level structure remains → stop modelling. Consider better measurements, more data (if the learning curve still rises) or reframing.
6. **IF** the ranking is good but decisions are poor → calibration/threshold.
7. **IF** you are tuning for gains below the noise → stop.

**Stopping criteria (choose upfront):** the target performance is reached; the last k accepted experiments each gained less than the noise; the remaining error is close to the measurement noise; or the time/budget is exhausted. Document which one applied.

**Going backwards:** see [project_lifecycle.md](../00_methodology/project_lifecycle.md) §8. Error analysis tells you which earlier stage to revisit.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Changing many things at once | Can't attribute gains | One change (or one coherent group) per experiment |
| No log of rejected ideas | Repeated work | Log everything |
| Accepting noise-level gains | Complexity without benefit | Pre-defined acceptance rule with paired folds |
| Tuning before fixing data/features | Small gains, wasted time | Follow the ordered checks |
| Endless iteration on one validation set | Validation overfitting | Untouched test set; fresh data |
| Ignoring the cost of complexity | Fragile production systems | Include maintenance cost in prioritisation |

## 10. Checklist
- [ ] Current best recorded with its noise level.
- [ ] Error analysis → backlog with hypotheses, evidence, gain, cost.
- [ ] One change per experiment; same validation.
- [ ] Pre-defined acceptance rule (paired comparison).
- [ ] Accepted and rejected results logged.
- [ ] Error analysis repeated after accepted changes.
- [ ] Stopping criterion checked; test evaluated once at the end.

## 11. References
- Ng, A. *Machine Learning Yearning* (error analysis, bias/variance, prioritisation). https://info.deeplearning.ai/machine-learning-yearning-book
- Sculley, D. et al. (2015). "Hidden Technical Debt in Machine Learning Systems." *NeurIPS 28*.
- Recht, B., Roelofs, R., Schmidt, L. & Shankar, V. (2019). "Do ImageNet Classifiers Generalize to ImageNet?" *ICML*. On adaptive overfitting to a reused test set. https://arxiv.org/abs/1902.10811
- Google, *Rules of Machine Learning*, Phase II–III. https://developers.google.com/machine-learning/guides/rules-of-ml
