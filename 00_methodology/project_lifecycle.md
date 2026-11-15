# Project Lifecycle: How a Data Science Project Actually Moves

> **Folder:** 00_methodology · **Read first.** · **Next:** [problem_framing.md](problem_framing.md) · **Related:** [decision_framework.md](decision_framework.md), [data_leakage.md](data_leakage.md), [modeling_fundamentals.md](modeling_fundamentals.md)

---

## 1. Purpose

**What it solves.** It gives you a map of the whole project, so at any moment you know which stage you are in, what that stage must produce, and when the evidence says you should go back a stage.

---

## 2. When to use it

**When to use it.** At the start of every project (to plan), and whenever you feel stuck or are tempted to "just try another model" (to work out which stage the problem really belongs to).

**When it is not needed.** For a one-off descriptive question ("what was the mean downtime last quarter?") you do not need the full lifecycle. Use [exploratory_data_analysis.md](../01_data_understanding/exploratory_data_analysis.md) directly.

---

## 3. Conceptual foundation

### 3.1 The stages and their deliverables

| # | Stage | Key question | Deliverable (artefact you should be able to show) | Recipe |
|---|---|---|---|---|
| 1 | Problem framing | What decision will the model support, and how will we know it helps? | One-page problem statement: target, unit of prediction, prediction time, metric, baseline, constraints | [problem_framing.md](problem_framing.md) |
| 2 | Data acquisition & initial inspection | What data exists, and what does one row mean? | Data inventory + schema + row-meaning statement | [initial_inspection.md](../01_data_understanding/initial_inspection.md) |
| 3 | Data quality assessment | Can these data be trusted for this purpose? | Data-quality report (issues, severity, decision per issue) | [data_quality.md](../01_data_understanding/data_quality.md) |
| 4 | Split strategy | How will we estimate future performance honestly? | Frozen test set + validation scheme, decided **before** modelling | [validation_reference.md](../08_quick_references/validation_reference.md) |
| 5 | Exploratory data analysis | What structure, signal and traps are in the data? | EDA notes: hypotheses, candidate features, risks | [exploratory_data_analysis.md](../01_data_understanding/exploratory_data_analysis.md) |
| 6 | Cleaning & preprocessing | Which transformations does the model need, fitted on training data only? | A scikit-learn `Pipeline` / `ColumnTransformer` | [02_preprocessing/](../02_preprocessing/) |
| 7 | Feature engineering | Which domain knowledge can we encode? | Feature functions + rationale | [feature_engineering.md](../02_preprocessing/feature_engineering.md) |
| 8 | Baseline & model selection | What does "good" mean relative to a trivial solution? | Baseline score + shortlist of candidate model families | [modeling_fundamentals.md](modeling_fundamentals.md) |
| 9 | Training & tuning | What is the best configuration under honest validation? | CV results table, chosen configuration | [hyperparameter_optimization.md](../06_model_improvement/hyperparameter_optimization.md) |
| 10 | Evaluation | How good is it, for whom, and how sure are we? | Metric(s) with uncertainty, segment breakdown, comparison with the baseline | family `evaluation.md` |
| 11 | Error analysis | Where and why does it fail? | Ranked list of failure modes, each with a hypothesis | family `error_analysis.md` |
| 12 | Experiment tracking | What did we try, and what happened? | Experiment log | [experiment_tracking.md](../07_experimentation/experiment_tracking.md) |
| 13 | Iterative improvement | What is the highest-value next action? | Prioritised improvement backlog | [iterative_improvement.md](../06_model_improvement/iterative_improvement.md) |
| — | Reproducibility (all the time) | Could someone else rerun it and get the same result? | Versioned code, data, configuration and environment | [reproducibility.md](../07_experimentation/reproducibility.md) |

Stage 4 (the split strategy) sits **before** EDA. This is deliberate. Any decision you make after looking at the test data, including outlier rules, feature ideas or thresholds, makes the test estimate optimistic. Decide the split early, put the test set away, and do EDA on the training portion.

### 3.2 It is a loop, not a line

```
            ┌──────────────────────────────────────────────────────────┐
            ▼                                                          │
  Framing ──► Data & quality ──► Split ──► EDA ──► Preprocess/FE ──► Model ──► Evaluate ──► Error analysis
     ▲              ▲                      ▲            ▲              ▲                        │
     │              │                      │            │              │                        │
     └──────────────┴──────────────────────┴────────────┴──────────────┴────────────────────────┘
                         error analysis tells you WHICH arrow to take back
```

The most common failure in practice is going back to the wrong stage. Typical examples are tuning hyperparameters when the target is mislabelled, or adding features when the validation scheme leaks. The table in §8 maps each symptom to the stage that most likely caused it.

### 3.3 Three kinds of iteration

1. **Inner loop (minutes):** change preprocessing, a feature or a hyperparameter, then re-run cross-validation. Use the validation data only.
2. **Middle loop (hours to days):** error analysis shows a systematic failure, so you return to EDA or feature engineering, or collect new data.
3. **Outer loop (days to weeks):** the evaluation shows the framing was wrong. The metric may not reflect the business cost, the prediction time may be unrealistic, or the target may be ill-defined. You return to stage 1.

Keep the outer loop cheap by getting to a first end-to-end result quickly: baseline, simple model, honest evaluation. You find framing mistakes only by evaluating something.

---

## 4. Assumptions and limitations

- The lifecycle assumes a supervised prediction problem with a defined target. For exploratory or unsupervised work, use the same loop with different evaluation criteria (see [09_unsupervised_learning](../09_unsupervised_learning/) and [10_anomaly_detection](../10_anomaly_detection/)).
- Time estimates ("day 1–3") assume a dataset that already exists. Data acquisition can dominate the schedule.
- Deployment and monitoring (MLOps) are only touched on here; see [20_advanced_methodologies](../20_advanced_methodologies/).

---

## 5. Step-by-step procedure

1. **Write the problem statement** ([problem_framing.md](problem_framing.md)). If you cannot fill in "prediction time" and "decision supported", stop: modelling is premature.
2. **Inventory and inspect the data.** Write one sentence describing what a row represents. Many later bugs come from getting this wrong, for example rows that are snapshots rather than events, or several rows per entity.
3. **Assess quality.** For each issue, record its severity and a decision (fix, flag, drop, ask the data owner).
4. **Decide the split and validation scheme.** Is the data temporal? Grouped? Is it small? Use [validation_reference.md](../08_quick_references/validation_reference.md). Freeze the test set.
5. **EDA on training data.** Produce hypotheses, not just plots.
6. **Build the simplest end-to-end pipeline:** baseline, then one simple model, then evaluation. Do this early (day 1–3 of a typical project).
7. **Error analysis**, then pick the next action from the improvement backlog.
8. **Iterate** with tracked experiments until one of the stopping conditions below is met.
9. **Final evaluation on the test set, once.** Report it with uncertainty, segment results and known limitations.

**Stopping conditions (decide them in advance):**
- The target performance from the problem statement is reached.
- The improvement over the last N experiments is smaller than the evaluation noise (see the CV standard deviation in [cross_validation.md](../06_model_improvement/cross_validation.md)).
- The remaining errors are irreducible with the available data, e.g. label noise or a missing causal driver. More modelling will not help; more or better data might.

---

## 6. Python implementation: a project skeleton

A minimal, reusable layout. Details are in [reproducibility.md](../07_experimentation/reproducibility.md).

```text
my-project/
├── README.md               # problem statement + how to run
├── data/
│   ├── raw/                # immutable, never edited by hand
│   ├── interim/            # cleaned, intermediate
│   └── processed/          # model-ready (or generated on the fly)
├── notebooks/              # exploration only; numbered: 01_eda.ipynb …
├── src/
│   ├── data.py             # loading + cleaning functions
│   ├── features.py         # feature engineering functions
│   ├── pipeline.py         # sklearn Pipeline construction
│   ├── evaluate.py         # metrics, plots, segment analysis
│   └── train.py            # entry point: config -> trained model + metrics
├── configs/                # YAML/JSON experiment configs
├── reports/                # figures, model cards, data-quality report
└── requirements.txt        # pinned versions
```

A minimal end-to-end run on the cookbook regression example. This is the "day 1" result that every later experiment is compared against.

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cookbook_data import make_process_yield

df = make_process_yield()

# Stage 3 decisions (minimal): remove exact duplicates, treat sentinel as missing,
# drop rows with an obviously wrong unit (yield < 1 kg). Leakage column excluded.
df = df.drop_duplicates()
df["temperature_c"] = df["temperature_c"].mask(df["temperature_c"] < 0)
df = df[df["yield_kg"] > 1]

# Stage 4: chronological hold-out (last 20% of batches), because the model
# will be used on FUTURE batches and the process drifts (catalyst ageing).
df = df.sort_values("start_time")
cut = int(len(df) * 0.8)
train, test = df.iloc[:cut], df.iloc[cut:]   # `test` is not touched again until the end

target = "yield_kg"
num = ["temperature_c", "pressure_bar", "feed_rate_kg_h", "residence_time_min",
       "catalyst_age_days", "impurity_ppm", "ambient_humidity_pct"]
cat = ["reactor", "catalyst", "shift", "supplier"]
X_train, y_train = train[num + cat], train[target]

pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
])
models = {
    "baseline_mean": DummyRegressor(strategy="mean"),
    "ridge": make_pipeline(pre, RidgeCV(alphas=np.logspace(-3, 3, 13))),
}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    res = cross_validate(model, X_train, y_train, cv=cv,
                         scoring=["neg_mean_absolute_error", "r2"])
    print(f"{name:15s} MAE={-res['test_neg_mean_absolute_error'].mean():6.1f} "
          f"± {res['test_neg_mean_absolute_error'].std():.1f}   R2={res['test_r2'].mean():.3f}")
```

> **Note on the CV choice above.** Shuffled K-fold *inside* the training period is a reasonable first pass for i.i.d.-like batch data. Because this process drifts, you should later confirm the result with a time-ordered scheme (`TimeSeriesSplit`); see [cross_validation.md](../06_model_improvement/cross_validation.md). Starting simple and then checking the assumption is the cookbook's general pattern.

---

## 7. Interpretation

- **The baseline is the reference point.** Ridge MAE much lower than the baseline MAE means there is learnable signal. If a reasonable model barely beats the baseline, suspect the framing, the features or the data quality before blaming the algorithm.
- **The CV standard deviation is your noise floor.** If a later idea improves MAE by less than about one CV standard deviation, treat the improvement as unproven.
- **The first end-to-end result is supposed to be imperfect.** It exists to expose problems: leakage (a suspiciously high score), mis-framing (a metric that does not move with what the stakeholders care about) or data issues.

---

## 8. Decision framework: when to go back, and where

| Symptom observed | Most likely root cause | Go back to |
|---|---|---|
| Score is suspiciously good (e.g. R² > 0.99, AUC > 0.99) on a hard real problem | Leakage: target proxy, future information, duplicates across splits | [data_leakage.md](data_leakage.md), Split |
| CV score is good but the test / production score is much worse | Validation does not mimic deployment (random split on temporal or grouped data), or drift | Split, [validation_reference.md](../08_quick_references/validation_reference.md) |
| Strong models barely beat the baseline | Weak signal, wrong target definition, key driver not measured | Framing, Data acquisition |
| Errors concentrate in a segment (one site, one supplier, a time window) | Data-quality issue or missing feature for that segment | Data quality, EDA, Feature engineering |
| Training error ≈ validation error, both high | Underfitting: model too simple or features insufficient | Feature engineering, Model selection |
| Training error ≪ validation error | Overfitting: too complex, too little data, leakage in training features | Regularisation, Model selection, more data |
| Stakeholders do not trust or use the predictions | Metric or output format does not match the decision | Framing |
| Results cannot be reproduced a week later | Unpinned randomness, data or environment | [reproducibility.md](../07_experimentation/reproducibility.md) |
| Residuals show structure (trend vs. time, funnel shape) | Missing variable, wrong transformation, heteroscedasticity | EDA, Transformations, family `error_analysis.md` |

**General rules:**
- **IF** you are unsure which stage is responsible, **THEN** do error analysis first. It is cheaper than guessing.
- **IF** the problem may be in the data, **THEN** look at raw rows, not just aggregated statistics.
- **IF** you change the validation scheme, **THEN** recompute all previous results under the new scheme before comparing.

---

## 9. Common mistakes

| Mistake | Consequence | Prevention |
|---|---|---|
| Starting with a complex model | No reference point; cannot tell whether complexity helps | Always run a baseline and a simple model first |
| EDA on the full dataset, including the test set | Decisions are tuned to the test set, so the estimate is optimistic | Split before EDA |
| Treating the lifecycle as linear | Effort goes into polishing a model built on a wrong framing | Get to an end-to-end result early; schedule review points |
| Iterating without a log | Repeated experiments, no idea which change helped | [experiment_tracking.md](../07_experimentation/experiment_tracking.md) from day 1 |
| Peeking at the test set repeatedly | The test set becomes another validation set | Use the test set once; if you must reuse it, report that honestly |
| No stopping criterion | Endless tuning for noise-level gains | Define target performance and the noise floor upfront |

---

## 10. Checklist

- [ ] Problem statement written (target, unit, prediction time, decision, metric, baseline).
- [ ] One sentence describing what a row represents.
- [ ] Data-quality issues listed, each with a decision.
- [ ] Split strategy chosen to mimic deployment; test set frozen **before** EDA.
- [ ] Baseline result recorded.
- [ ] Simple end-to-end pipeline running.
- [ ] Experiment log started.
- [ ] Stopping criteria defined.
- [ ] Error analysis done before each major iteration.
- [ ] Final test evaluation done once, reported with uncertainty and limitations.

---

## 11. References

- Chapman, P. et al. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*. The classic process model. Note that it, too, is iterative.
- Sculley, D. et al. (2015). "Hidden Technical Debt in Machine Learning Systems." *NeurIPS 28*. https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html
- scikit-learn: *Common pitfalls and recommended practices*. https://scikit-learn.org/stable/common_pitfalls.html
- Kuhn, M. & Johnson, K. (2019). *Feature Engineering and Selection*, Ch. 1 ("Introduction", the modelling process). http://www.feat.engineering/
- Zinkevich, M. *Rules of Machine Learning: Best Practices for ML Engineering* (Google). https://developers.google.com/machine-learning/guides/rules-of-ml
