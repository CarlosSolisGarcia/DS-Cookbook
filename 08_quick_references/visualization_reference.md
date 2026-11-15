# Visualisation Reference: Which Plot Answers Which Question

> **Folder:** 08_quick_references · Plot code lives in the recipes. This page maps questions to plots and gives a few reusable helpers.

---

## 1. Question → plot
| Question | Plot | Library call | Read it for… |
|---|---|---|---|
| Distribution of one numeric variable | Histogram + KDE; linear **and** log scale | `sns.histplot(x, kde=True, log_scale=…)` | Skew, modes, spikes at defaults, unit errors |
| Compare distributions across groups | Box/violin plot with counts; ECDF | `sns.boxplot`, `sns.ecdfplot(hue=…)` | Shifts, spread, outliers per group |
| Category frequencies | Bar plot of counts | `sns.countplot` | Rare levels, imbalance |
| Relationship numeric–numeric | Scatter + LOWESS; binned means with CI for noisy data | `sns.regplot(lowess=True)`; binned helper below | Shape (linear, threshold, optimum) |
| Relationship conditional on a category | Faceted/coloured scatter + smoother | `sns.lmplot(hue=…, col=…)` | Interactions |
| Many pairwise correlations | Heatmap (Spearman) | `sns.heatmap(df.corr("spearman"))` | Redundancy clusters |
| Missingness pattern | Heatmap of `isna()`; bar of % missing | `sns.heatmap(df.isna())` | Co-missingness, blocks in time |
| Time series | Line at several zoom levels; rolling mean/std | `s.plot()`, `s.rolling(w).mean()` | Trend, seasonality, breaks |
| Seasonality | Seasonal subseries / box by weekday-month | `sns.boxplot(x=weekday, y=…)` | Seasonal shape and stability |
| Autocorrelation | ACF / PACF | `statsmodels.graphics.tsaplots.plot_acf/plot_pacf` | Memory, seasonal lags |
| Model fit (regression) | Predicted vs actual; residuals vs predicted | `PredictionErrorDisplay.from_predictions` | Bias, heteroscedasticity |
| Residual structure | Residuals vs feature/time (binned) | binned helper | Missing effects |
| Classifier ranking | ROC and PR curves | `RocCurveDisplay`, `PrecisionRecallDisplay` | Discrimination; compare with random |
| Classifier decisions | Confusion matrix (counts + normalised) | `ConfusionMatrixDisplay` | Error types |
| Probability quality | Reliability diagram | `CalibrationDisplay` | Over/under-confidence |
| Thresholds | Precision/recall/cost vs threshold | manual (see [threshold_optimization.md](../04_classification/threshold_optimization.md)) | Operating point |
| Learning dynamics | Learning / validation curves | `LearningCurveDisplay`, `ValidationCurveDisplay` | Bias vs variance; more data? |
| Feature effects | PDP/ICE; SHAP beeswarm/waterfall | `PartialDependenceDisplay`, `shap.plots.*` | Shapes, interactions, local reasons |
| High-dimensional structure | PCA biplot; t-SNE/UMAP embedding | see [11_dimensionality_reduction](../11_dimensionality_reduction/) | Clusters, outliers (with caveats) |
| Forecasts | History + forecast + interval band, per origin | matplotlib `fill_between` | Plausibility; interval width |

## 2. Reusable helpers

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def binned_plot(x, y, bins=15, ax=None, label=None):
    """Mean of y (± 95% CI) in quantile bins of x: a clear shape for noisy relationships and residual checks."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    b = pd.qcut(d["x"], bins, duplicates="drop")
    g = d.groupby(b, observed=True).agg(xm=("x", "median"), ym=("y", "mean"), sd=("y", "std"), n=("y", "size"))
    ax = ax or plt.gca()
    ax.errorbar(g["xm"], g["ym"], yerr=1.96 * g["sd"] / np.sqrt(g["n"]), fmt="o-", ms=3, label=label)
    return ax

def multi_scale_ts(s, windows=(None, "90D", "21D"), figsize=(13, 7)):
    """Plot a series at full scale and zoomed into the last windows."""
    fig, axes = plt.subplots(len(windows), 1, figsize=figsize)
    for ax, w in zip(axes, windows):
        part = s if w is None else s[s.index[-1] - pd.Timedelta(w):]
        part.plot(ax=ax, lw=0.7); ax.set_title("full" if w is None else f"last {w}")
    plt.tight_layout()
    return fig

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 2000); y = np.sin(x) + rng.normal(0, 0.5, 2000)
binned_plot(x, y); plt.title("binned means reveal the shape"); plt.show()
idx = pd.date_range("2023-01-01", periods=365, freq="D")
multi_scale_ts(pd.Series(np.sin(np.arange(365) / 7) + rng.normal(0, .2, 365), index=idx)); plt.show()
```

## 3. Good-practice rules
- Every plot answers a stated question. Write the question in the title.
- Show **counts or confidence intervals** with group means; small groups mislead.
- Use log scales for positive, skewed quantities spanning orders of magnitude.
- For time series, always plot against time (not just a histogram).
- For model diagnostics, use **validation (out-of-fold) predictions**.
- Use colour-blind-safe palettes (seaborn `"colorblind"`, matplotlib `"viridis"`) and label units.

## 4. References
- seaborn tutorial. https://seaborn.pydata.org/tutorial.html
- matplotlib, *Plot types* and *Tutorials*. https://matplotlib.org/stable/plot_types/index.html
- scikit-learn, *Visualizations* (Display objects). https://scikit-learn.org/stable/visualizations.html
- Wilke, C. O. (2019). *Fundamentals of Data Visualization*. O'Reilly. Free online: https://clauswilke.com/dataviz/
