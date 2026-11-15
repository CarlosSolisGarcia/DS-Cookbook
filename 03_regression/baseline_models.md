# Regression Baselines

> **Folder:** 03_regression · **Prev:** [regression_workflow.md](regression_workflow.md) · **Next:** [model_selection.md](model_selection.md) · **Related:** [modeling_fundamentals.md](../00_methodology/modeling_fundamentals.md) §3.4

---

## 1. Purpose
Establish the score that any serious model must beat, and calibrate what counts as "good" for this specific problem.

## 2. When to use it
Always, before any non-trivial model, and in every comparison table thereafter.

## 3. Conceptual foundation
| Baseline | Prediction | What it tells you |
|---|---|---|
| **Mean** (`DummyRegressor("mean")`) | Training mean | Zero point for RMSE/R² (R² = 0 by definition on the training data) |
| **Median** (`DummyRegressor("median")`) | Training median | Zero point for MAE |
| **Group mean / median** | Mean per category (e.g. per reactor) | How much the obvious grouping alone explains |
| **One-variable model** | Linear fit on the dominant driver | How much the main driver alone explains |
| **Domain rule / current practice** | E.g. "yield = 0.7 × feed" from the process design | The bar that matters to stakeholders |
| **Last value / previous batch** | Value of the previous similar entity | For temporally ordered rows; tests persistence |

A good baseline set includes **one trivial** and **one informed** baseline.

## 4. Assumptions and limitations
- Baselines are evaluated with the *same* validation scheme and metric as the models.
- An "informed" baseline built from EDA on the training data must not use test data, otherwise it is optimistic too.

## 5. Step-by-step procedure
1. Compute the trivial baselines (mean, median) under the project's CV.
2. Compute the group-mean baseline for the 1–2 most obvious categories.
3. Fit a one-variable linear model on the dominant driver from EDA.
4. Implement the domain or current-practice rule if one exists (ask stakeholders).
5. Record all of them in the experiment log ([experiment_tracking.md](../07_experimentation/experiment_tracking.md)).

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_validate

from cookbook_data import load_clean_process_yield

df = load_clean_process_yield()
train = df.iloc[: int(0.8 * len(df))]
y = train["yield_kg"]

class GroupMeanRegressor(BaseEstimator, RegressorMixin):
    """Predict the training mean (or median) of the target within each group; global value for unseen groups."""
    def __init__(self, group_col: str, stat: str = "mean"):
        self.group_col, self.stat = group_col, stat
    def fit(self, X, y):
        s = pd.Series(np.asarray(y), index=X.index)
        self.table_ = s.groupby(X[self.group_col]).agg(self.stat)
        self.global_ = s.agg(self.stat)
        return self
    def predict(self, X):
        return X[self.group_col].map(self.table_).astype(float).fillna(self.global_).to_numpy()

class RatioRule(BaseEstimator, RegressorMixin):
    """Domain rule: yield = k × feed per batch, with k estimated as the median ratio on training data."""
    def fit(self, X, y):
        self.k_ = np.median(np.asarray(y) / (X["feed_rate_kg_h"] * 8.0))
        return self
    def predict(self, X):
        return self.k_ * X["feed_rate_kg_h"].to_numpy() * 8.0

one_var = ColumnTransformer([("feed", "passthrough", ["feed_rate_kg_h"])])
from sklearn.pipeline import make_pipeline
baselines = {
    "mean": DummyRegressor(strategy="mean"),
    "median": DummyRegressor(strategy="median"),
    "reactor mean": GroupMeanRegressor("reactor"),
    "linear(feed only)": make_pipeline(one_var, LinearRegression()),
    "rule: k × feed": RatioRule(),
}
cv = TimeSeriesSplit(5)
scoring = {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"}
rows = []
for name, m in baselines.items():
    r = cross_validate(m, train, y, cv=cv, scoring=scoring)
    rows.append({"baseline": name, "MAE": -r["test_MAE"].mean(), "RMSE": -r["test_RMSE"].mean(),
                 "R2": r["test_R2"].mean()})
print(pd.DataFrame(rows).set_index("baseline").round(3))
```

## 7. Interpretation
- **Mean/median baselines have R² ≤ 0 under CV.** Slightly negative values are normal, because the fold means differ from the training means.
- **Reactor mean ≈ mean:** the reactor alone explains little.
- **"k × feed" or linear(feed) already reach a large R²:** most of the variance comes from throughput. This is crucial context. A model with R² = 0.85 is less impressive than it sounds if feed alone gives 0.6. Report the **improvement over the informed baseline**, e.g. MAE reduction in % relative to the "k × feed" rule.
- **Suspicious:** a complex model only marginally better than the domain rule. The remaining variance may be noise, or the drivers may not be measured.

## 8. Decision framework
- **IF** an informed baseline is already close to the target performance → consider deploying the simple rule, or a lightly corrected version. ML adds maintenance cost.
- **IF** the trivial and informed baselines are close → no single dominant driver. Expect gains from combining features and from flexible models.
- **IF** there is a strong dominant driver → consider modelling the *residual* or the ratio (yield per feed) so the model focuses on the rest. Check that the metric is still on the original target.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Only a trivial baseline | Inflated sense of model value | Always add an informed baseline |
| Baseline evaluated differently (no CV, other metric) | Invalid comparison | Same CV object and scorers |
| Ignoring current practice | Can't demonstrate business value | Ask stakeholders for their rule |

## 10. Checklist
- [ ] Mean and median baselines.
- [ ] Group baseline for the obvious categories.
- [ ] Dominant-driver baseline.
- [ ] Domain/current-practice baseline (if one exists).
- [ ] All on the same CV and metrics; logged.

## 11. References
- scikit-learn, `DummyRegressor`. https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators
- scikit-learn, *Developing scikit-learn estimators* (for custom baselines such as `GroupMeanRegressor`). https://scikit-learn.org/stable/developers/develop.html
