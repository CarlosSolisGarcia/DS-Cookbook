# Problem Framing and Business Understanding

> **Folder:** 00_methodology · **Prev:** [project_lifecycle.md](project_lifecycle.md) · **Next:** [decision_framework.md](decision_framework.md) · **Related:** [data_leakage.md](data_leakage.md), [metrics_reference.md](../08_quick_references/metrics_reference.md)

---

## 1. Purpose

**What it solves.** It turns a vague request ("can we use ML to improve yield?") into a precisely specified prediction task. That task can be validated honestly and it supports a real decision.

**What we want to find out.**
- Which **decision** the prediction supports, and who makes it.
- What exactly is being predicted (the **target**), for which **unit** (batch, machine-month, site-day) and **when** (the prediction time).
- Which data are **available at prediction time**.
- How success will be measured, both technically (the metric) and in business terms (cost, savings, risk).
- What the **baseline** is: the current process or a trivial rule.

---

## 2. When to use it

**When to use it.** Always, before touching data. Return to it whenever evaluation results do not translate into value.

**When it is less critical.** Well-defined benchmark exercises (a Kaggle-style task with a given metric), although even then, defining the prediction time carefully protects against leakage.

---

## 3. Conceptual foundation

### 3.1 Prediction vs. explanation vs. causal inference

| Goal | Question | What you optimise | Typical tools | Cookbook coverage |
|---|---|---|---|---|
| **Prediction** | "What will the yield of this batch be?" | Out-of-sample error | ML models, CV | Main focus |
| **Explanation / description** | "Which variables are associated with yield?" | Interpretability, stability of estimates | Linear models, PDP, SHAP | [model_interpretation.md](../06_model_improvement/model_interpretation.md) |
| **Causal inference / intervention** | "If we raise temperature by 5 °C, how will yield change?" | Unbiased effect estimates | Designed experiments (DoE), causal inference methods | Not covered (future extension) |

A predictive model can be excellent while its feature importances say **nothing causal**. If the stakeholder wants to *act on the inputs* ("which knob should we turn?"), you have a causal question. Say so explicitly. As a chemical engineer you already know the best tool for that question: a designed experiment.

### 3.2 Prediction time and the information set

The **prediction time** (t₀) is the moment the prediction is made. Every feature must be known at t₀, in the form it will have in production. This single concept prevents most leakage (see [data_leakage.md](data_leakage.md)).

```
   past data ───────────────► t₀ (predict) ───────── horizon ─────────► outcome observed
   features must come from here               nothing from here may enter the features
```

Examples:
- *Yield prediction at batch start:* recipe settings and raw-material analysis are known. The average reactor temperature during the batch is **not** known. It becomes usable only if you predict mid-batch, which is a different problem.
- *Failure in the next 30 days:* sensor readings up to the snapshot date are known. Downtime in the next 30 days is the outcome itself.

### 3.3 Choosing the problem type

| If the target is… | And the decision needs… | Problem family |
|---|---|---|
| A continuous quantity (kg, °C, €) | A number, or a number with an interval | Regression → [03_regression](../03_regression/) |
| A category (fail / no fail, failure mode) | A class or a probability, plus an action threshold | Classification → [04_classification](../04_classification/) |
| A future value of a time-indexed quantity | Values at future times (horizon h) | Forecasting → [05_time_series](../05_time_series/) |
| A rare deviation without reliable labels | A flag for unusual behaviour | Anomaly detection → [10_anomaly_detection](../10_anomaly_detection/) (see also [time_series_understanding.md](../05_time_series/time_series_understanding.md) §3.4) |

The same business question can often be framed several ways, and the framing changes the data you need. "Predict machine failure" could be:
- binary classification: will it fail within 30 days?
- regression: what is the remaining useful life?
- survival / remaining-useful-life analysis: what is the time-to-failure distribution, with censoring? (see [predictive_maintenance_rul.md](../16_industrial_sensor_data/predictive_maintenance_rul.md))

Choose the framing whose output maps most directly onto the decision.

### 3.4 Metrics come from costs

A metric is a proxy for the cost of errors. Work out the cost structure first, then pick the metric (see [metrics_reference.md](../08_quick_references/metrics_reference.md)):
- Are over- and under-predictions equally costly? If not, consider asymmetric metrics or quantile predictions.
- Is a large error disproportionately worse than two small ones? If so, RMSE is a better match than MAE.
- In classification: what does a false positive cost (an unnecessary inspection) compared with a false negative (an unplanned breakdown)?

---

## 4. Assumptions and limitations

- The framing assumes stakeholders can describe the decision and the costs of errors. When they cannot, use ranges or scenarios, and revisit after a first model exists.
- A good framing does not guarantee that the data contain the needed signal. That is found out in stages 2–10.
- Causal questions need designs (experiments, causal inference) that this cookbook does not cover in depth.

---

## 5. Step-by-step procedure

### Step 1 — Identify the decision and the user
- **Ask:** Who will use the output? What will they do differently? How often? What do they do today?
- **Why:** If no decision changes, the model has no value, however accurate it is.
- **Expected result:** One sentence, e.g. "The maintenance planner uses a weekly ranked list of machines to schedule up to 20 inspections."
- **Red flag:** "We want insights" with no decision attached. Push back gently, or reframe the work as an EDA or reporting project.

### Step 2 — Define the unit of analysis and the target precisely
- **Ask:** What is one row (one prediction)? How is the target measured? Who labels it, and how reliable is the label? Is it a direct measurement or a proxy?
- **Why:** An ambiguous target produces label noise that no model can fix.
- **Techniques:** Write the target as an SQL-like definition, e.g. `failure_30d = 1 if an unplanned work order of type 'breakdown' is opened within 30 days after snapshot_date`.
- **Problems to identify:** Targets that change definition over time, censored targets (the 30-day window not yet complete for recent rows), targets that are themselves model outputs.

### Step 3 — Define the prediction time and horizon
- **Ask:** When is the prediction made? How far ahead? How quickly are new data available (latency)?
- **Expected result:** "Predictions are generated every Monday 06:00 using data up to Sunday 23:59. The horizon is 30 days."
- **Problem:** Sensor data that arrive with a 2-day lag. In the training set they look available, but in production they are not.

### Step 4 — Inventory candidate inputs and check availability at t₀
For each candidate input, record: source, update frequency, latency, whether it is known at t₀, and whether it is affected by the action the model triggers (feedback loops).

### Step 5 — Define success: metric, baseline, target performance
- **Technical metric:** primary plus secondary (e.g. MAE primary; P90 absolute error secondary).
- **Baseline:** the current practice (operator rule of thumb, schedule-based maintenance) and a trivial model (mean, last value, majority class).
- **Target performance:** what level of improvement over the baseline would justify deployment? This defines your stopping criterion.

### Step 6 — List constraints and risks
Latency, interpretability requirements (regulated domains), fairness, cost of data collection, maintenance burden.

### Step 7 — Write the one-page problem statement
Use the template in §6 and get it confirmed by the stakeholder.

---

## 6. Implementation: problem statement template

Keep this as `README.md` or `reports/problem_statement.md` in your project. A filled-in example for the cookbook's predictive-maintenance dataset:

```text
PROBLEM STATEMENT — Predictive maintenance (v1, 2026-01-15)

Decision supported : Weekly selection of up to 20 machines (of ~500) for preventive inspection.
User               : Maintenance planner, every Monday.
Unit of prediction : machine × month snapshot.
Target             : failure_30d = 1 if an unplanned breakdown work order is opened within
                     30 days after snapshot_date; else 0. Label source: CMMS work orders.
                     Known label issues: ~3% of breakdowns logged as 'planned' (label noise).
Prediction time    : snapshot_date (month end), data available up to that date.
Horizon            : 30 days.
Inputs at t0       : machine type/site/age, sensor aggregates (vibration, temperature, load),
                     error counts last 7 days, days since maintenance, firmware.
NOT available at t0: downtime_hours_30d, failure_mode (outcomes!).
Problem type       : Binary classification (ranking + probability). Multiclass failure mode later.
Primary metric     : Precision@20 per week (capacity-constrained) + PR-AUC.
Secondary          : Recall at the chosen threshold; Brier score (probability quality).
Baseline           : (a) current rule: inspect the 20 oldest machines; (b) DummyClassifier.
Success criterion  : ≥ 2× the precision@20 of the age rule, stable across sites.
Validation         : Grouped by machine_id; final test = last 3 months (temporal).
Constraints        : Explanations required per flagged machine (planner trust).
Risks              : Feedback loop — inspected machines fail less, which alters future labels.
```

A small helper to check, in code, that the declared features are a subset of what is allowed at t₀. It can be reused in every project:

```python
import pandas as pd
from cookbook_data import make_equipment_failure

df = make_equipment_failure()

# Declare the information set explicitly; this is your contract with the deployment team.
AVAILABLE_AT_T0 = {
    "machine_type", "site", "age_years", "load_pct", "operating_hours",
    "days_since_maintenance", "vibration_rms", "temperature_c", "error_count_7d", "firmware",
}
ID_COLS = {"machine_id", "snapshot_date"}
TARGET = "failure_30d"

def check_feature_contract(columns, available, ids, target):
    """Raise if a feature is used that is not declared as available at prediction time."""
    features = set(columns) - ids - {target}
    undeclared = sorted(features - available)
    if undeclared:
        raise ValueError(f"Columns not declared as available at t0: {undeclared}")
    return sorted(features)

try:
    check_feature_contract(df.columns, AVAILABLE_AT_T0, ID_COLS, TARGET)
except ValueError as e:
    print("Contract violation ->", e)   # flags downtime_hours_30d and failure_mode

features = check_feature_contract(
    df.drop(columns=["downtime_hours_30d", "failure_mode"]).columns,
    AVAILABLE_AT_T0, ID_COLS, TARGET,
)
print(features)
```

---

## 7. Interpretation

How to tell if the framing is good:
- **Good framing:** you can describe the exact data available at t₀, the action taken from the output, and a baseline you have to beat. Different stakeholders describe the target the same way.
- **Warning signs:**
  - The target is available only long after the decision (e.g. 6-month outcomes for a weekly decision). Then you need a faster proxy target or a different framing.
  - The positive class is extremely rare (e.g. < 0.1%) and you have very few positive examples. Consider anomaly detection, a longer horizon, or aggregation to a higher level.
  - Stakeholders want feature importances to decide interventions. That is a causal question, so set expectations.
  - Your best feature is something only measured *after* the event. You are about to leak.

---

## 8. Decision framework

**IF** the stakeholder asks "why does X happen?" or "what should we change?"
→ Make clear that predictive models show **associations**. Recommend designed experiments or causal methods for intervention decisions. Use interpretation tools only to *generate* hypotheses.

**IF** the decision has a capacity constraint (inspect 20 machines, call 100 customers)
→ Frame the task as **ranking**. Use metrics such as precision@k or recall@k rather than accuracy at 0.5. Thresholds follow from capacity ([threshold_optimization.md](../04_classification/threshold_optimization.md)).

**IF** the error costs are asymmetric
→ Use a cost-weighted metric, quantile regression (regression), or a cost-based threshold (classification).

**IF** the target is continuous but the decision is binary (e.g. "is yield below spec?")
→ Compare two framings: (a) regression, then threshold the prediction; (b) classify directly. (a) keeps more information and lets the spec change later. (b) can be better when the region near the threshold matters most. Validate both on the decision metric.

**IF** the data are time-ordered and the model will be used on future data
→ Validate chronologically, even if the task is not "forecasting" ([validation_reference.md](../08_quick_references/validation_reference.md)).

**IF** labels are noisy or inconsistently defined
→ Fix the labelling process first, or quantify the noise. The achievable performance is capped by label quality.

**IF** no baseline exists ("nobody does this today")
→ Build a heuristic baseline with domain experts (e.g. "flag machines whose vibration is above the 90th percentile"). It is often surprisingly strong and it makes the value of ML measurable.

---

## 9. Common mistakes

| Mistake | Consequence | Prevention |
|---|---|---|
| Starting from the data ("what can we predict with this?") | Technically correct but useless models | Start from the decision |
| Undefined prediction time | Leakage; the model fails in production | Write t₀ in the problem statement; enforce the feature contract |
| Choosing accuracy / R² by default | A metric that does not reflect cost (e.g. 95% accuracy with 5% positives is trivial) | Derive the metric from error costs |
| Ignoring the current process as a baseline | Cannot show value; may be worse than what exists | Always benchmark against current practice |
| Interpreting the predictive model causally | Wrong interventions | Separate predictive and causal claims explicitly |
| Ignoring feedback loops | Model degrades because its actions change the data | Document actions taken; consider holding out a control group |
| Too-ambitious first framing | Months without a result | Start with the simplest useful framing; extend later |

---

## 10. Checklist

- [ ] The decision, the user and the frequency of use are written down.
- [ ] Unit of prediction (what one row is) defined.
- [ ] Target defined operationally, with known label issues.
- [ ] Prediction time t₀, horizon and data latency defined.
- [ ] Every candidate feature checked for availability at t₀.
- [ ] Problem type chosen; alternatives considered.
- [ ] Primary and secondary metrics derived from error costs.
- [ ] Baselines defined (current practice + trivial).
- [ ] Success criterion and stopping criterion agreed.
- [ ] Constraints and risks (interpretability, feedback loops, drift) listed.
- [ ] Problem statement reviewed by a stakeholder.

---

## 11. References

- Provost, F. & Fawcett, T. (2013). *Data Science for Business*. O'Reilly. Ch. 2 ("Business Problems and Data Science Solutions") and Ch. 7 ("Decision Analytic Thinking": expected value, costs).
- Shmueli, G. (2010). "To Explain or to Predict?" *Statistical Science* 25(3), 289–310. https://doi.org/10.1214/10-STS330
- Kaufman, S., Rosset, S., Perlich, C. & Stitelman, O. (2012). "Leakage in data mining: Formulation, detection, and avoidance." *ACM TKDD* 6(4). https://doi.org/10.1145/2382577.2382579
- Montgomery, D. C. *Design and Analysis of Experiments* (Wiley). For when the question is causal and you can run experiments.
- Google, *Rules of Machine Learning*, Rules #1–#3 (on launching without ML first and choosing metrics). https://developers.google.com/machine-learning/guides/rules-of-ml
