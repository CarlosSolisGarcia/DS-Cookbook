# Python Code Patterns

> **Folder:** 08_quick_references · Copy-paste patterns used throughout the cookbook, each with the mistake it prevents. Versions: see `requirements.txt` (scikit-learn ≥ 1.5, pandas ≥ 2.2).

---

## 1. Leakage-safe preprocessing + model (`ColumnTransformer` + `Pipeline`)

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def make_linear_pipeline(num, cat, model):
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                              OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                                            sparse_output=False)), cat),
    ]).set_output(transform="pandas")       # keep column names (pandas output requires dense one-hot)
    return make_pipeline(pre, model)
# Prevents: fitting imputers/scalers on validation/test data; crashes on unseen categories.
```

## 2. Native-categorical gradient boosting (no encoding pipeline needed)

```python
from sklearn.ensemble import HistGradientBoostingClassifier

def to_model_frame(df, num, cat):
    return df[num + cat].astype({c: "category" for c in cat})

hgb = HistGradientBoostingClassifier(categorical_features="from_dtype", early_stopping=True, random_state=0)
# Prevents: one-hot explosions; manual NaN handling (HGB handles NaN natively).
```

## 3. Cross-validated evaluation with several metrics and train scores

```python
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

def cv_report(model, X, y, cv, scoring, groups=None):
    """Mean and std of every train/validation score; rows like 'test_ap', 'train_ap'."""
    r = cross_validate(model, X, y, cv=cv, groups=groups, scoring=scoring, return_train_score=True, n_jobs=-1)
    rows = {k: (r[k].mean(), r[k].std()) for k in r if k.startswith(("test_", "train_"))}
    return pd.DataFrame(rows, index=["mean", "std"]).T

from cookbook_data import make_equipment_failure
df = make_equipment_failure(include_leakage=False)
num = ["age_years", "load_pct", "vibration_rms", "temperature_c", "error_count_7d"]; cat = ["machine_type", "site"]
print(cv_report(make_linear_pipeline(num, cat, LogisticRegression(max_iter=2000)), df[num + cat], df["failure_30d"],
                StratifiedGroupKFold(5, shuffle=True, random_state=0),
                {"ap": "average_precision", "roc": "roc_auc"}, groups=df["machine_id"]).round(3))
# Prevents: comparing models on different folds; missing the train/validation gap.
```

## 4. Out-of-fold predictions for error analysis

```python
from sklearn.model_selection import cross_val_predict
oof = cross_val_predict(make_linear_pipeline(num, cat, LogisticRegression(max_iter=2000)), df[num + cat],
                        df["failure_30d"], cv=StratifiedGroupKFold(5, shuffle=True, random_state=0),
                        groups=df["machine_id"], method="predict_proba")[:, 1]
df_err = df.assign(p=oof)
# Prevents: error analysis on in-sample predictions. NOTE: cross_val_predict requires a partition
# (every row predicted once). For TimeSeriesSplit, loop over the folds manually (see 03_regression/evaluation.md).
```

## 5. Custom transformer skeleton (fitted statistics → inside the pipeline)

```python
from sklearn.base import BaseEstimator, TransformerMixin

class ClipToTrainQuantiles(BaseEstimator, TransformerMixin):
    def __init__(self, low=0.01, high=0.99):
        self.low, self.high = low, high
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lo_, self.hi_ = X.quantile(self.low), X.quantile(self.high)
        self.feature_names_in_ = X.columns.to_numpy()
        return self
    def transform(self, X):
        return pd.DataFrame(X).clip(self.lo_, self.hi_, axis=1)
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_
# Rules: learn in fit (attributes end with "_"), never in transform; __init__ only stores the parameters.
```

## 6. Time-safe feature creation

```python
s = pd.Series(np.arange(30.0), index=pd.date_range("2024-01-01", periods=30, freq="D"))
features = pd.DataFrame({
    "lag_1": s.shift(1),
    "lag_7": s.shift(7),
    "roll7_mean": s.shift(1).rolling(7).mean(),          # shift BEFORE rolling
    "ewm_past": s.shift(1).ewm(span=7).mean(),
    "dow": s.index.dayofweek,
})
# Per entity: df.groupby("machine_id")["vibration"].transform(lambda x: x.shift(1).rolling(7).mean())
# Prevents: look-ahead leakage through windows.
```

## 7. Reproducibility basics

```python
SEED = 42
rng = np.random.default_rng(SEED)             # local generator, not np.random.seed
# pass random_state=SEED to every splitter / model / sampler
import sklearn, platform
env = {"sklearn": sklearn.__version__, "pandas": pd.__version__, "python": platform.python_version()}
```

## 8. Persist and reload the whole pipeline

```python
import joblib, tempfile, os
model = make_linear_pipeline(num, cat, LogisticRegression(max_iter=2000)).fit(df[num + cat], df["failure_30d"])
path = os.path.join(tempfile.mkdtemp(), "model.joblib")
joblib.dump(model, path)
assert np.allclose(joblib.load(path).predict_proba(df[num + cat])[:, 1], model.predict_proba(df[num + cat])[:, 1])
# Prevents: re-implementing preprocessing at inference (training–serving skew).
```

## 9. pandas pitfalls (pandas ≥ 2 / 3)
| Pitfall | Pattern |
|---|---|
| Chained assignment (`df[df.a > 0]["b"] = 1`) silently fails under Copy-on-Write | `df.loc[df.a > 0, "b"] = 1` |
| `.to_numpy()` arrays can be read-only views | `np.array(df, copy=True)` before in-place edits |
| Frequency aliases changed | `"h"` (hour), `"min"`, `"ME"` (month end), `"QE"`, `"YE"` |
| Default string dtype is `str` (pandas 3) | Check dtypes with `select_dtypes(include=["object", "string", "category"])` |
| `groupby(...).apply` including the grouping columns (deprecated) | `.apply(f, include_groups=False)` |
| Joins duplicating rows | `merge(..., validate="many_to_one")` + row-count assert |

## 10. References
- scikit-learn, *Pipelines and composite estimators*. https://scikit-learn.org/stable/modules/compose.html
- scikit-learn, *Developing scikit-learn estimators*. https://scikit-learn.org/stable/developers/develop.html
- pandas, *Copy-on-Write*. https://pandas.pydata.org/docs/user_guide/copy_on_write.html
- pandas, *Offset aliases*. https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases
