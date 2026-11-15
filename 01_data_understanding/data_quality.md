# Data Understanding and Quality Assessment

> **Folder:** 01_data_understanding · **Prev:** [initial_inspection.md](initial_inspection.md) · **Next:** [missing_values.md](missing_values.md) · **Related:** [outliers.md](outliers.md), [data_leakage.md](../00_methodology/data_leakage.md)

---

## 1. Purpose
Decide whether the data can be trusted **for the stated purpose**, and record a decision for each quality problem: fix, flag, drop or escalate. The output is a short *data-quality report* that later steps (cleaning, preprocessing, evaluation) refer back to.

## 2. When to use it
- After initial inspection, before EDA conclusions and before modelling.
- On each new data delivery. Turn the checks into automated assertions once you know them.
- Lighter touch for curated benchmark data, but still check duplicates, ranges and the label distribution.

## 3. Conceptual foundation
Data quality has several dimensions. Check each one deliberately:

| Dimension | Question | Example check |
|---|---|---|
| **Validity** | Do values respect type, range and format rules? | Temperature in [0, 400] °C; categories in the allowed set |
| **Completeness** | Is anything missing (cells, rows, periods)? | Missing fraction per column; gaps in the time series |
| **Uniqueness** | Are entities recorded once? | Duplicate keys, near-duplicates |
| **Consistency** | Do related fields agree? | `end_time ≥ start_time`; jacket temperature ≈ reactor temperature + offset |
| **Accuracy** | Do values reflect reality? | Comparison with a reference or lab value; unit errors |
| **Timeliness** | Are values recorded when we think they are? | Lag between event and record; later back-filling (a leakage risk) |
| **Label quality** | Is the target correct and consistently defined? | Agreement between labellers; definition changes over time |

Quality is **fitness for purpose**. A 2% sensor dropout may be irrelevant for monthly averages and critical for anomaly detection.

## 4. Assumptions and limitations
- Rules require domain knowledge: physical limits, process constraints and business logic. Generic statistics cannot tell you that 999 is a sentinel. A process engineer can.
- Accuracy usually cannot be verified from the data alone. Seek an external reference (lab samples, calibration records).
- Fixing data changes the distribution. Every fix must be reproducible code, never manual edits of the raw file.

## 5. Step-by-step procedure
1. **Define validity rules per column** (type, range, allowed set, format) with the domain expert. Write them as code (§6).
2. **Run the checks** and quantify violations (count, %, which rows).
3. **Look for sentinel and placeholder values.** Repeated "round" or impossible values such as −999, 0, 9999, 1900-01-01 are often codes for "missing" or "not applicable".
4. **Check uniqueness**: exact duplicates and duplicate keys with differing content.
5. **Check cross-field consistency** using physical and process relations (mass balances, ordering of timestamps, known correlations).
6. **Check for unit and scale errors.** Look for clusters of values offset by factors of 10, 1000, 60 or 3600 (kg vs t, min vs s, °F vs °C).
7. **Check stability over time.** Per-period means and missingness of key columns. Look for step changes from calibrations, sensor replacements or software updates.
8. **Assess label quality**: the distribution, its changes over time, and agreement with an independent source where possible.
9. **Decide per issue** (§8) and write the data-quality report.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from cookbook_data import make_process_yield

df = make_process_yield()

# 1. Validity rules as data: (column, rule description, function returning a boolean mask of VIOLATIONS)
RULES = [
    ("temperature_c",     "in [100, 300] °C",   lambda d: ~d["temperature_c"].between(100, 300) & d["temperature_c"].notna()),
    ("pressure_bar",      "in [0, 40] bar",     lambda d: ~d["pressure_bar"].between(0, 40)),
    ("yield_kg",          "> 1 kg",             lambda d: d["yield_kg"] <= 1),
    ("reactor",           "in {R1..R4}",        lambda d: ~d["reactor"].isin(["R1", "R2", "R3", "R4"])),
    ("jacket_vs_reactor", "jacket - reactor in [0, 15] °C",
     lambda d: ~(d["jacket_temperature_c"] - d["temperature_c"]).between(0, 15) & d["temperature_c"].between(100, 300)),
]

def run_rules(df, rules):
    rows = []
    for col, desc, fn in rules:
        mask = fn(df)
        rows.append({"check": col, "rule": desc, "n_violations": int(mask.sum()),
                     "pct": round(100 * mask.mean(), 2),
                     "examples": df.loc[mask, col if col in df else df.columns[0]].head(3).tolist()})
    return pd.DataFrame(rows)

print(run_rules(df, RULES).to_string())
```

```python
# 2. Sentinel detection: values that repeat suspiciously often at the extremes
def suspicious_repeats(s: pd.Series, top=3):
    vc = s.value_counts()
    return vc[vc > max(3, 0.002 * len(s))].head(top)

for col in df.select_dtypes("number"):
    rep = suspicious_repeats(df[col])
    if len(rep) and col not in ("catalyst_age_days",):   # integer-valued columns repeat legitimately
        print(col, rep.to_dict())
```

```python
# 3. Duplicates: exact vs. key-only
exact = df.duplicated(keep=False)
key_only = df.duplicated("batch_id", keep=False) & ~exact
print("exact duplicate rows:", exact.sum(), " same key, different content:", key_only.sum())

# 4. Unit-error detection: inspect the log10 of the target; separate clusters reveal scale errors
log_y = np.log10(df["yield_kg"].clip(lower=1e-6))
print(pd.cut(log_y, bins=[-7, -2, -1, 0, 1, 2, 3, 4]).value_counts().sort_index())
```

```python
# 5. Stability over time: per-month mean and missing fraction of key columns
monthly = df.set_index("start_time").resample("ME")
stability = pd.DataFrame({
    "temp_mean": monthly["temperature_c"].apply(lambda s: s[s > 0].mean()),
    "humidity_missing": monthly["ambient_humidity_pct"].apply(lambda s: s.isna().mean()),
    "yield_median": monthly["yield_kg"].median(),
    "n": monthly.size(),
})
print(stability.round(2).head(12))
```

```python
# 6. Apply decisions as reproducible code (never edit raw files by hand)
def clean_process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, documented fixes. Each line corresponds to an entry in the data-quality report."""
    out = df.drop_duplicates().copy()                                     # DQ-01 export duplicates
    out["temperature_c"] = out["temperature_c"].mask(out["temperature_c"] == -999)  # DQ-02 sentinel → NaN
    out = out[out["yield_kg"] > 1]                                         # DQ-03 unit errors (5 rows, can't be verified) → drop
    return out.reset_index(drop=True)

clean = clean_process_data(df)
print(len(df), "→", len(clean))
```

A data-quality report can be a simple table kept in `reports/data_quality.md`:

| ID | Column | Issue | Evidence | Severity | Decision | Owner |
|---|---|---|---|---|---|---|
| DQ-01 | all | 10 exact duplicate rows | `duplicated()` | Medium (leakage across splits) | Drop | DS |
| DQ-02 | temperature_c | −999 sentinel (≈0.5%) | repeated value outside physical range | High | → NaN, then impute in the pipeline | DS |
| DQ-03 | yield_kg | 5 values ≈ 1000× too small | log10 histogram cluster | High | Drop (can't confirm unit); ask the data owner | DS + process eng. |
| DQ-04 | impurity_ppm | Missing more often when high (sensor saturation) | see [missing_values.md](missing_values.md) | Medium | Missing indicator + imputation | DS |
| DQ-05 | qc_purity_pct | Measured after the batch | data dictionary | Critical | Exclude from features | DS |

## 7. Interpretation
- **Rule violations of ~0%** on most rules is expected. Investigate a rule with a few violations by looking at the actual rows. A rule with many violations usually means the rule is wrong, not the data. Revisit it with the expert.
- **Repeated extreme values** (−999 appearing about 10 times) are almost certainly codes.
- **Separate clusters in the log-scale histogram** of a positive quantity mean scale or unit errors. Smooth long tails are more likely real (handle them in [outliers.md](outliers.md)).
- **Step changes in monthly means** may be real process changes, which matter for validation (drift), or instrument changes, which may call for a correction or a "regime" feature.
- **Missingness changing over time** usually comes from sensors added or removed, or from logging changes. It is often MAR on time; see [missing_values.md](missing_values.md).

## 8. Decision framework
| Issue type | IF… | THEN… |
|---|---|---|
| Exact duplicates | Generated by the export/join process | Drop before splitting |
| | Legitimate repeats (same settings, separate runs) | Keep, but treat them as a **group** in validation |
| Sentinel | The code means "missing" | → NaN; add a missing indicator if missingness may be informative |
| | The code means "not applicable" (e.g. 0 = no catalyst) | Create an explicit category/flag; do not impute |
| Out-of-range value | Physically impossible | → NaN (then impute) or drop the row if critical fields are affected |
| | Possible but rare | Keep; see [outliers.md](outliers.md) |
| Unit errors | Correction is unambiguous (exact factor, confirmed by the owner) | Correct in code, documenting the rule |
| | Ambiguous | Drop, or NaN plus a flag |
| Target errors | Few rows | Drop from training; **also consider them in test** (production will contain them too → monitoring) |
| Instrument change / recalibration | Known date | Add a regime feature, or correct using the calibration offset; validate across the change |
| Label definition changed | Known date | Restrict to a consistent period, or harmonise the definition |

**General rule:** fix in code, and keep the raw data immutable. Prefer *flag and keep* over *silently drop* when unsure. Dropped rows bias the dataset, so record how many rows each decision removes.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Imputing sentinels as if they were real values | Strongly biased features | Detect and convert sentinels first |
| Dropping all rows with any issue | Selection bias; may remove the informative cases (e.g. failures) | Decide per issue; quantify the rows lost per rule |
| Fixing data in Excel | Irreproducible; the error recurs on the next delivery | Cleaning functions in `src/data.py` |
| Cleaning differently for train and test | Train/test inconsistency | One cleaning function applied to all data; *fitted* steps go in the pipeline |
| Assuming the label is correct | A noise ceiling you can't explain | Audit labels; estimate noise ([iterative_improvement.md](../06_model_improvement/iterative_improvement.md)) |
| Over-cleaning the test set | Optimistic evaluation | Test data should look like production data |

## 10. Checklist
- [ ] Validity rules defined with a domain expert and coded.
- [ ] Sentinel/placeholder values identified and converted.
- [ ] Exact and key duplicates handled.
- [ ] Cross-field consistency checked.
- [ ] Unit and scale errors checked (log histograms).
- [ ] Stability over time checked (means, missingness).
- [ ] Label quality assessed.
- [ ] Every issue in the data-quality report with severity and decision.
- [ ] Cleaning implemented as a deterministic function; raw data untouched.

## 11. References
- Wang, R. Y. & Strong, D. M. (1996). "Beyond Accuracy: What Data Quality Means to Data Consumers." *J. Management Information Systems* 12(4), 5–33. https://doi.org/10.1080/07421222.1996.11518099
- Breck, E., Polyzotis, N., Roy, S., Whang, S. & Zinkevich, M. (2019). "Data Validation for Machine Learning." *Proceedings of MLSys*. https://mlsys.org/Conferences/2019/doc/2019/167.pdf
- Pandera (dataframe validation library; useful when the rules become many). https://pandera.readthedocs.io/
- Northcutt, C., Athalye, A. & Mueller, J. (2021). "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks." *NeurIPS Datasets and Benchmarks*. https://arxiv.org/abs/2103.14749
