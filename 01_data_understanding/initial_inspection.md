# Data Acquisition and Initial Inspection

> **Folder:** 01_data_understanding · **Prev:** [modeling_fundamentals.md](../00_methodology/modeling_fundamentals.md) · **Next:** [data_quality.md](data_quality.md) · **Related:** [problem_framing.md](../00_methodology/problem_framing.md), [data_leakage.md](../00_methodology/data_leakage.md)

---

## 1. Purpose
Get the data loaded correctly and work out **what you actually have** before analysing anything. That means its size, its schema, the data types, what one row means, the keys, the time coverage and how the parts relate to each other. Most silent bugs in DS projects start here: wrong dtypes, mis-parsed dates, an unexpected row granularity, joins that duplicate rows.

## 2. When to use it
- Always, as the first contact with any new dataset or a new data delivery (a new extract may differ from the last one).
- Re-run it when a pipeline breaks or results change unexpectedly after a data refresh.
- Not needed in depth when the data are a well-documented, previously validated version you already know. Still run the automated summary.

## 3. Conceptual foundation
- **Granularity (unit of observation).** What does one row represent: a batch, a machine-month, a sensor reading? It must match the *unit of prediction* from the problem statement. If it does not, you will need to aggregate or join.
- **Keys.** The column(s) that uniquely identify a row. If the declared key is not unique, either the granularity is not what you think or there are duplicates.
- **Tidy data** (Wickham, 2014): each variable a column, each observation a row, each type of observational unit a table. Models need one tidy table at the prediction granularity.
- **Data dictionary.** For each column: meaning, unit, type, source, creation time relative to t₀, allowed values. It doubles as your leakage audit trail.
- **Joins change row counts.** A one-to-many join silently duplicates rows. Always compare row counts before and after.

## 4. Assumptions and limitations
- This recipe assumes tabular data in files or a database. Other modalities (images, text, graphs) have their own inspection steps in their sections.
- Automated profiling summarises, but it does not interpret. Meaning comes from the data owner and from domain knowledge.
- pandas loads data into memory. For files larger than about 30–50% of RAM, read only the columns you need, choose dtypes explicitly, or use chunked reading, Parquet with column selection, or Polars/DuckDB.

## 5. Step-by-step procedure
1. **Record provenance.** Note the source, the extraction query or file, the date and the version/hash. *Why:* you must be able to reproduce the dataset ([reproducibility.md](../07_experimentation/reproducibility.md)).
2. **Load with explicit parsing.** Parse dates, set dtypes for identifiers (string, not int: leading zeros matter), and define the NA tokens. *Problem to spot:* numeric columns read as `object` or `str` mean there are non-numeric tokens ("n/a", "<LOD", "1,234").
3. **Shape and memory.** Number of rows and columns. Does it match what the data owner told you?
4. **Schema review.** dtype per column, number of unique values and missing fraction. *Ask:* Is each type what it should be? Are there categoricals disguised as numbers (codes such as 1/2/3), or numbers disguised as strings?
5. **Determine granularity and keys.** Test the uniqueness of candidate keys. *Ask:* What is one row? Are there several rows per entity?
6. **Time coverage.** Min and max timestamps, row counts over time. *Spot:* gaps, sudden volume changes (a system migration), the timezone.
7. **Look at raw rows.** Head, tail and a random sample. *Why:* summary statistics hide formatting problems and odd records.
8. **Draft the data dictionary.** Include the "available at t₀?" column.
9. **Validate joins** if you combine tables. Check row counts and key uniqueness on both sides.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from cookbook_data import make_process_yield

# In a real project you would load from file. Note the explicit parsing:
# df = pd.read_csv("batches.csv", parse_dates=["start_time"],
#                  dtype={"batch_id": "string", "reactor": "category"},
#                  na_values=["", "NA", "n/a", "-"])
df = make_process_yield()

def schema_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, missing %, unique count, example values."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "pct_missing": (100 * df.isna().mean()).round(2),
        "n_unique": df.nunique(dropna=True),
        "example": [df[c].dropna().iloc[:3].tolist() for c in df.columns],
    })

print(df.shape, f"{df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(schema_summary(df).to_string())
```

```python
def check_key(df: pd.DataFrame, key: list[str]) -> None:
    """Report whether `key` uniquely identifies rows, and show offending rows."""
    dup = df.duplicated(subset=key, keep=False)
    print(f"key={key}: unique={not dup.any()}  duplicated rows={dup.sum()}")
    if dup.any():
        print(df.loc[dup].sort_values(key).head(6).to_string())

check_key(df, ["batch_id"])          # NOT unique → duplicates to investigate (data_quality.md)

# Time coverage and volume over time
print(df["start_time"].agg(["min", "max"]))
print(df.set_index("start_time").resample("QE").size())   # rows per quarter
```

```python
# Categorical overview: levels and frequencies (look for typos, rare levels, unexpected codes)
for col in df.select_dtypes(include=["object", "string", "category"]).columns.drop("batch_id"):
    print(col, df[col].value_counts(dropna=False).to_dict())

# Raw rows: always look at some
print(df.sample(5, random_state=0).T)
```

```python
# Safe join pattern: validate the cardinality explicitly
suppliers = pd.DataFrame({"supplier": [f"S{i:02d}" for i in range(1, 16)],
                          "country": np.random.default_rng(0).choice(["ES", "DE", "FR"], 15)})
n_before = len(df)
merged = df.merge(suppliers, on="supplier", how="left", validate="many_to_one")  # raises if suppliers has duplicate keys
assert len(merged) == n_before, "join changed the row count!"
print(merged["country"].isna().sum(), "rows without supplier match")
```

```python
# Data-dictionary skeleton: fill the semantic columns with the data owner
dictionary = schema_summary(df)[["dtype", "pct_missing", "n_unique"]].assign(
    description="", unit="", source="", available_at_t0="", notes="")
dictionary.loc["qc_purity_pct", ["available_at_t0", "notes"]] = ["NO", "measured after batch → leakage"]
dictionary.loc["yield_kg", ["unit", "notes"]] = ["kg", "TARGET"]
print(dictionary.head(8))
# dictionary.to_csv("reports/data_dictionary.csv")
```

## 7. Interpretation
- **Expected:** the dtypes match the meanings, the key is unique, the time coverage matches what the data owner said, and the categorical levels look clean.
- **Anomalies in this dataset:** `batch_id` is not unique (10 duplicated rows). The minimum of `temperature_c` is −999 (a sentinel; see the summary). Some `yield_kg` values are around 0.1–0.5, three orders of magnitude below normal (a unit error). All three go to [data_quality.md](data_quality.md).
- **Suspicious patterns in general:**
  - A numeric column with very few unique values is probably a code or category.
  - An `object` column that should be numeric contains string tokens.
  - A sudden change in rows per period suggests a system change or a filter in the extraction.
  - A missing fraction that is exactly the same across several columns suggests those columns come from the same source or join.
  - Identifiers stored as floats (e.g. `1.0`, `2.0`) were int columns that contained NaN.

## 8. Decision framework
- **IF** the granularity ≠ the unit of prediction → aggregate (e.g. sensor readings → machine-month features) or redefine the unit. Document the choice.
- **IF** the key is not unique → determine whether rows are exact duplicates (remove them), versions/updates (keep the version valid at t₀) or a sign of a different granularity (re-think).
- **IF** the data have multiple rows per entity → note it now. It dictates grouped validation later.
- **IF** the data are larger than memory → load the needed columns only, downcast dtypes, or use Parquet/DuckDB/Polars. Do not sample before understanding the structure; sampling can break groups and time order.
- **IF** a join changes the row count → stop. Find the duplicated keys before continuing.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Letting pandas guess dtypes | IDs lose leading zeros; dates as strings; codes as numbers | Explicit `dtype=`, `parse_dates=` |
| Not checking key uniqueness | Duplicates leak across splits; wrong aggregates | `check_key` on every table |
| Joins without `validate=` | Silent row multiplication | `merge(..., validate="many_to_one")` + row-count assert |
| Ignoring timezones / DST | Misaligned features, duplicated or missing hours | Store UTC; convert explicitly (`tz_localize`, `tz_convert`) |
| Looking only at `describe()` | Formatting problems invisible | Look at raw rows |
| No provenance record | Irreproducible results | Log the source, query, date and hash |

## 10. Checklist
- [ ] Source, extraction date and version recorded.
- [ ] Loaded with explicit dtypes, date parsing and NA tokens.
- [ ] Shape and memory checked against expectations.
- [ ] Schema summary reviewed; type mismatches noted.
- [ ] Row meaning written in one sentence; key uniqueness tested.
- [ ] Time coverage and volume over time checked.
- [ ] Raw rows inspected (head, tail, random sample).
- [ ] Data dictionary drafted, including "available at t₀".
- [ ] Joins validated (cardinality + row counts).

## 11. References
- Wickham, H. (2014). "Tidy Data." *Journal of Statistical Software* 59(10). https://doi.org/10.18637/jss.v059.i10
- pandas User Guide, *IO tools* (`read_csv` parameters: `dtype`, `parse_dates`, `na_values`, `usecols`). https://pandas.pydata.org/docs/user_guide/io.html
- pandas User Guide, *Merge, join, concatenate* (the `validate` argument). https://pandas.pydata.org/docs/user_guide/merging.html
- pandas User Guide, *Scaling to large datasets*. https://pandas.pydata.org/docs/user_guide/scale.html
