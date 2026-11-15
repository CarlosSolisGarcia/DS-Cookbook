# Exploratory Data Analysis (EDA)

> **Folder:** 01_data_understanding · **Prev:** [outliers.md](outliers.md) · **Next:** [numerical_features.md](../02_preprocessing/numerical_features.md) · **Related:** [visualization_reference.md](../08_quick_references/visualization_reference.md), [ts exploratory_analysis.md](../05_time_series/exploratory_analysis.md), [regression_workflow.md](../03_regression/regression_workflow.md), [classification_workflow.md](../04_classification/classification_workflow.md)

---

## 1. Purpose
Build an understanding of the data that leads to **concrete modelling decisions**: which transformations, which features, which validation risks and which segments need special attention. Good EDA ends with a written list of hypotheses and decisions, not a gallery of plots.

## 2. When to use it
- After data-quality assessment, on the **training portion only** (split first; see [project_lifecycle.md](../00_methodology/project_lifecycle.md)).
- Again during error analysis, focused on the segments where the model fails.
- For time series, use this recipe for the non-temporal aspects and [05_time_series/exploratory_analysis.md](../05_time_series/exploratory_analysis.md) for the temporal structure.

## 3. Conceptual foundation
EDA answers five groups of questions:

| Question group | Typical tools | Leads to decisions about… |
|---|---|---|
| 1. **Target**: distribution, range, skew, balance, stability over time | Histogram (linear and log), class counts, target over time | Transformations, loss/metric, imbalance strategy, validation |
| 2. **Features (univariate)**: distribution, rare levels, constant columns | Histograms, value counts | Transformations, encoding, dropping useless columns |
| 3. **Feature–target relationships**: shape, strength, monotonicity | Scatter + smoother, binned means, box plots by category, mutual information | Feature engineering, model family (linear vs flexible) |
| 4. **Feature–feature relationships**: correlation, redundancy, interactions | Correlation matrix (Spearman), VIF, pair plots, conditional plots | Multicollinearity handling, interactions |
| 5. **Structure**: groups, time, segments, drift | Counts by entity, plots against time, comparisons of distributions | Validation scheme, segment evaluation |

**Correlation ≠ dependence.** Pearson r measures *linear* association. A U-shaped relationship can have r ≈ 0. Spearman ρ captures monotonic relationships; mutual information captures any dependence, but it is noisy for small n. Always **look** at the plot.

**Marginal ≠ conditional.** A feature can look unrelated to the target on its own and still matter after conditioning on another variable. Here, temperature matters through its distance from a *catalyst-specific* optimum. The reverse also happens: a marginal relationship can be entirely due to a confounder (Simpson's paradox).

## 4. Assumptions and limitations
- EDA findings are **hypotheses**, not conclusions. With many plots, some patterns will appear by chance (multiple comparisons). Confirm the important ones with validation.
- Doing EDA on the full dataset, test set included, contaminates later choices. Use the training portion.
- Statistical tests (normality, independence) at large n reject tiny, irrelevant deviations, and at small n they miss large ones. Prefer effect sizes and plots to p-values.

## 5. Step-by-step procedure
1. **Split** off the test set (temporal, grouped or random, as decided). Work on `train` only.
2. **Target analysis.** Histogram (linear and log), quantiles, skewness, and its behaviour over time and by key categories. *Ask:* Is it skewed (consider a transformation and the metric)? Bounded? Are there spikes at specific values (rounding, defaults)? Is it stable over time?
3. **Univariate features.** Distributions, rare levels, near-constant columns. *Spot:* columns with one dominant value (low information), and unexpected modes (mixed regimes).
4. **Feature–target relationships.**
   - Numeric: scatter with a LOWESS smoother or binned means. *Ask:* linear? monotonic? threshold? optimum (U shape)?
   - Categorical: target distribution per level (box or violin plots) with the counts shown. *Ask:* Are the differences large compared with the within-level spread? Is a level rare?
5. **Feature–feature relationships.** Spearman correlation heatmap; VIF if you plan a linear model with interpretable coefficients. *Ask:* Which features are redundant? Do they measure the same physical quantity?
6. **Interactions and conditioning.** Plot feature vs. target coloured by a key category. *Ask:* Does the relationship change across groups?
7. **Structure and drift.** Rows per entity, and target/features over time. *Ask:* Is a grouped or temporal validation needed? (This should already have been decided; EDA can confirm or contradict it.)
8. **Write down hypotheses and decisions** (template in §6).

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cookbook_data import make_process_yield

df = make_process_yield().drop_duplicates()
df["temperature_c"] = df["temperature_c"].mask(df["temperature_c"] == -999)
df = df[df["yield_kg"] > 1].sort_values("start_time").reset_index(drop=True)
train = df.iloc[: int(0.8 * len(df))].copy()        # EDA on training data only
target = "yield_kg"
num = ["temperature_c", "jacket_temperature_c", "pressure_bar", "feed_rate_kg_h",
       "residence_time_min", "catalyst_age_days", "ambient_humidity_pct",
       "impurity_ppm", "feed_purity_pct"]
cat = ["reactor", "supplier", "catalyst", "shift"]
```

### 6.1 Target

```python
y = train[target]
print(y.describe(percentiles=[.01, .05, .5, .95, .99]).round(1))
print("skewness:", round(y.skew(), 2), "  skewness of log:", round(np.log(y).skew(), 2))

fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
sns.histplot(y, bins=50, ax=ax[0]).set_title("target")
sns.histplot(np.log(y), bins=50, ax=ax[1]).set_title("log(target)")
y.rolling(100).median().plot(ax=ax[2], title="rolling median of target (100 batches)")
plt.tight_layout(); plt.show()
```

### 6.2 Feature–target relationships (numeric): binned means beat raw scatter for noisy data

```python
def binned_relationship(df, x, y, bins=15):
    """Mean and 95% CI of y within quantile bins of x: shows shape (linear / threshold / optimum)."""
    b = pd.qcut(df[x], q=bins, duplicates="drop")
    g = df.groupby(b, observed=True)[y].agg(["mean", "std", "count"])
    g["ci"] = 1.96 * g["std"] / np.sqrt(g["count"])
    g["x_mid"] = df.groupby(b, observed=True)[x].median()
    return g

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, col in zip(axes.ravel(), [c for c in num if c != "jacket_temperature_c"]):
    g = binned_relationship(train, col, target)
    ax.errorbar(g["x_mid"], g["mean"], yerr=g["ci"], fmt="o-", ms=3)
    ax.set_title(col, fontsize=9)
plt.tight_layout(); plt.show()
```

### 6.3 Conditioning on a category: discovering interactions

```python
g = sns.lmplot(data=train, x="temperature_c", y=target, hue="catalyst", lowess=True,
               scatter_kws={"s": 5, "alpha": 0.3}, height=4, aspect=1.6)
plt.show()
# Normalising the target by the dominant driver (feed rate) often reveals the weaker effects:
train["yield_per_feed"] = train[target] / train["feed_rate_kg_h"]
sns.lmplot(data=train, x="temperature_c", y="yield_per_feed", hue="catalyst", lowess=True,
           scatter_kws={"s": 5, "alpha": 0.3}, height=4, aspect=1.6)
plt.show()
```

### 6.4 Categorical features vs target (with counts)

```python
def category_effect(df, col, y):
    g = df.groupby(col, observed=True)[y].agg(["count", "median", "mean", "std"])
    return g.sort_values("median")

for col in cat:
    print(category_effect(train, col, "yield_per_feed").round(2), "\n")
```

### 6.5 Feature–feature: correlation and VIF

```python
corr = train[num + [target]].corr(method="spearman")
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, annot_kws={"size": 7})
plt.title("Spearman correlation (training data)"); plt.tight_layout(); plt.show()

from statsmodels.stats.outliers_influence import variance_inflation_factor
Z = train[num].dropna()
Z = (Z - Z.mean()) / Z.std()
Z.insert(0, "const", 1.0)       # VIF must be computed with an intercept
vif = pd.Series([variance_inflation_factor(Z.values, i) for i in range(1, Z.shape[1])], index=num)
print(vif.sort_values(ascending=False).round(1))
```

### 6.6 Model-agnostic dependence: mutual information

```python
from sklearn.feature_selection import mutual_info_regression
Zmi = train[num].fillna(train[num].median())
mi = mutual_info_regression(Zmi, np.log(train[target]), random_state=0)
print(pd.Series(mi, index=num).sort_values(ascending=False).round(3))
```

### 6.7 EDA notes template (the real deliverable)

```text
EDA NOTES — process yield (training period, n=1600)
Target   : right-skewed (skew≈0.9); log(target) ≈ symmetric → try a log target; MAE on the original scale.
           Rolling median declines and recovers in a saw-tooth pattern → catalyst ageing; validate chronologically.
Drivers  : feed_rate dominant (≈ proportional) → consider modelling yield_per_feed or keeping feed as a feature.
           temperature: inverted-U, optimum differs by catalyst → interaction temperature×catalyst, or a tree model.
           residence_time: increasing, concave → log transform for the linear model.
           catalyst_age: negative, roughly linear.
Redundant: jacket_temperature ≈ temperature (ρ≈0.98), feed_purity ≈ −impurity → VIF high; affects linear
           coefficients, not tree predictions. Keep one of each for linear models.
Segments : reactor R3 lower yield per feed; supplier effects small but visible (15 levels, some rare).
Risks    : impurity missing when high (MNAR) → indicator; humidity missing mostly in R4 (MAR).
Next     : baseline → ridge with log target + engineered terms → gradient boosting; compare under TimeSeriesSplit.
```

## 7. Interpretation
- **Target skew ≈ 0.9 and log-skew ≈ 0.** A log transform symmetrises the target. That is useful for linear models and squared losses, but it changes what the model optimises; see [transformations.md](../02_preprocessing/transformations.md).
- **The binned-mean plots** show the *shape* of each relationship. A flat line in the *marginal* plot does not mean "irrelevant" (see the conditioning plot). A U shape tells you a linear term is insufficient.
- **The conditioning plot** shows temperature optima shifted by catalyst: an interaction. Tree models discover this automatically. Linear models need an explicit term.
- **Correlation of 0.98 between two features** means they are effectively the same information. Prediction is not harmed, but linear coefficients become unstable and importance is split between them.
- **VIF:** > 5–10 is a common "investigate" flag, not a removal rule.
- **Mutual information** ranks dependence without assuming linearity. Low MI with a clear conditional effect means the effect only appears in combination with other variables.
- **Suspicious findings:** a feature almost perfectly related to the target (leakage?); relationships that reverse between time periods (drift or regime change); a category with very few rows and an extreme mean (noise, not signal).

## 8. Decision framework
| EDA finding | Decision / next action |
|---|---|
| Target strongly right-skewed and positive | Compare models on raw vs log target ([transformations.md](../02_preprocessing/transformations.md)); choose the metric deliberately |
| Target has a hard bound or mass at a value (e.g. many zeros) | Consider a two-part model (zero vs positive) or a Tweedie/Poisson loss |
| Non-linear, monotonic relationship | Transformation (log, sqrt) for linear models, or splines; trees handle it natively |
| Optimum / U shape | Quadratic or spline term (linear model), or a tree model |
| Relationship changes by category | Interaction term, or a flexible model; evaluate per segment |
| Highly correlated features | Keep for tree models; for interpretable linear models keep one, combine them, or regularise ([feature_selection.md](../06_model_improvement/feature_selection.md)) |
| Rare categorical levels | Group them as "other", or use target/frequency encoding ([categorical_features.md](../02_preprocessing/categorical_features.md)) |
| Target/feature drift over time | Chronological validation; time-aware features; consider shorter training windows |
| Many rows per entity | Grouped validation; entity-level aggregate features (computed within the training folds) |
| No visible relationship for any feature | Check the target definition, data quality and whether key drivers are missing, before trying complex models |

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| EDA on train + test | Choices tuned to the test set | Split first |
| Plotting everything, concluding nothing | Wasted time | Write EDA notes with decisions |
| Relying on Pearson correlation only | Missing non-linear and U-shaped effects | Binned plots, Spearman, MI |
| Interpreting marginal plots as effects | Missing interactions or confounding | Conditioning plots; domain reasoning |
| Ignoring sample size per bin/level | Over-interpreting noise | Show counts and CIs |
| Using normality tests to decide transformations | At large n, every variable "fails" | Use plots and the model's needs |
| Removing correlated features "because VIF > 10" | Losing information for no gain | Decide based on model type and goal |

## 10. Checklist
- [ ] EDA performed on training data only.
- [ ] Target distribution (linear and log), bounds, stability over time.
- [ ] Univariate distributions and rare levels reviewed.
- [ ] Feature–target shapes examined (binned means with CIs, box plots with counts).
- [ ] Interactions examined via conditioning plots.
- [ ] Correlation (Spearman) and, if relevant, VIF reviewed.
- [ ] Group and time structure confirmed for validation.
- [ ] EDA notes written: hypotheses → decisions → next experiments.

## 11. References
- Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
- Wickham, H., Çetinkaya-Rundel, M. & Grolemund, G. (2023). *R for Data Science*, 2nd ed., Ch. "Exploratory data analysis" (language-agnostic principles). https://r4ds.hadley.nz/eda
- Kuhn, M. & Johnson, K. (2019). *Feature Engineering and Selection*, Ch. 4 "Exploratory Visualizations". http://www.feat.engineering/
- seaborn tutorial: *Visualizing statistical relationships* / *Visualizing distributions*. https://seaborn.pydata.org/tutorial.html
- statsmodels, `variance_inflation_factor`. https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html
- scikit-learn, `mutual_info_regression`. https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html
