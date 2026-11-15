# Classification Baselines

> **Folder:** 04_classification · **Prev:** [classification_workflow.md](classification_workflow.md) · **Next:** [model_selection.md](model_selection.md) · **Related:** [regression baselines](../03_regression/baseline_models.md), [modeling_fundamentals.md](../00_methodology/modeling_fundamentals.md)

---

## 1. Purpose
Define the zero point of skill for each metric, and an informed bar that represents current practice or a simple rule. Model value can then be stated as an improvement over something real.

## 2. When to use it
Always, before modelling. Keep the baselines in every comparison table.

## 3. Conceptual foundation
| Baseline | `DummyClassifier(strategy=…)` or rule | Score reference |
|---|---|---|
| **Majority class** | `"most_frequent"` | Accuracy = 1 − prevalence; recall(positive) = 0 |
| **Class prior** | `"prior"`: predicts the prevalence as the probability | ROC-AUC = 0.5; PR-AUC ≈ prevalence; Brier = p(1−p) |
| **Stratified random** | `"stratified"` | Expected precision = prevalence at any recall |
| **One-feature rule** | e.g. "flag if vibration > 90th percentile" | What the strongest single signal achieves |
| **Current practice** | e.g. "inspect the 20 oldest machines" | The business bar |
| **Simple logistic** | 2–3 features | How much a basic statistical model already captures |

**Expected values for random scoring.** With prevalence π, a random ranking gives precision = π at every recall, so PR-AUC ≈ π. That is why PR-AUC must always be read relative to π.

## 4. Assumptions and limitations
- Baselines must be evaluated on the same folds, with the same metrics and at the same operating point (e.g. the same k for precision@k).
- A rule chosen using validation data is mildly optimistic. Choose its threshold on training data within each fold, or keep it pre-defined by the domain.

## 5. Step-by-step procedure
1. Compute the dummy "prior" baseline for the ranking and probability metrics.
2. Compute the majority-class baseline, to expose the accuracy trap.
3. Implement 1–2 domain rules as scikit-learn-compatible estimators (they output scores).
4. Fit a small logistic regression on 2–3 interpretable features.
5. Report ROC-AUC, PR-AUC, Brier and the decision metric (e.g. precision@k) for all of them.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cookbook_data import make_equipment_failure

df = make_equipment_failure(include_leakage=False)
y, groups = df["failure_30d"].to_numpy(), df["machine_id"]

class ScoreRule(BaseEstimator, ClassifierMixin):
    """Use a single column (optionally negated) as the score; probabilities via the rank within training data."""
    def __init__(self, column: str, higher_is_riskier: bool = True):
        self.column, self.higher_is_riskier = column, higher_is_riskier
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.fill_ = X[self.column].median()
        return self
    def _score(self, X):
        s = X[self.column].fillna(self.fill_).to_numpy(dtype=float)
        return s if self.higher_is_riskier else -s
    def predict_proba(self, X):
        s = self._score(X)
        r = pd.Series(s).rank(pct=True).to_numpy()          # monotone map to (0, 1]: preserves ranking only
        return np.column_stack([1 - r, r])
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.9).astype(int)

def precision_at_k(y_true, score, k):
    top = np.argsort(-score)[:k]
    return y_true[top].mean()

baselines = {
    "majority": DummyClassifier(strategy="most_frequent"),
    "prior": DummyClassifier(strategy="prior"),
    "rule: oldest first (current practice)": ScoreRule("age_years"),
    "rule: highest vibration": ScoreRule("vibration_rms"),
    "logistic (3 features)": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                           LogisticRegression()),
}
three = ["vibration_rms", "temperature_c", "error_count_7d"]
cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
rows = []
for name, m in baselines.items():
    res = {"acc": [], "roc": [], "pr": [], "brier": [], "p@50": []}
    for tr, va in cv.split(df, y, groups):
        Xtr = df.iloc[tr][three] if "logistic" in name else df.iloc[tr]
        Xva = df.iloc[va][three] if "logistic" in name else df.iloc[va]
        m.fit(Xtr, y[tr]); p = m.predict_proba(Xva)[:, 1]
        res["acc"].append(accuracy_score(y[va], m.predict(Xva)))
        res["roc"].append(roc_auc_score(y[va], p)); res["pr"].append(average_precision_score(y[va], p))
        res["brier"].append(brier_score_loss(y[va], p)); res["p@50"].append(precision_at_k(y[va], p, 50))
    rows.append({"baseline": name, **{k: np.mean(v) for k, v in res.items()}})
print(pd.DataFrame(rows).set_index("baseline").round(3))
print("prevalence:", y.mean().round(3))
```

## 7. Interpretation
- **Majority class:** accuracy ≈ 0.95 with zero useful predictions. This is the accuracy trap, made visible.
- **Prior:** ROC-AUC 0.5, PR-AUC ≈ prevalence, Brier ≈ 0.046. These are the zero points.
- **"Oldest first" (current practice):** a modest lift. Age contributes to failure risk, but weakly.
- **Highest vibration:** stronger. A single good sensor rule can capture a large share of the achievable signal. ML must beat *this*.
- **The rule-based Brier score is meaningless** because the rank-based "probabilities" are not calibrated. Compare rules only on ranking metrics.

## 8. Decision framework
- **IF** a single-feature rule gets within the noise of your best model → consider deploying the rule (transparent, robust) or using it as a model feature.
- **IF** current practice is not quantifiable → build a heuristic with domain experts; it is still the best business baseline.
- **IF** the classes are multiclass → the baselines are `"most_frequent"`, `"prior"` and `"stratified"`; report per-class recall.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Reporting PR-AUC without the prevalence | Uninterpretable numbers | Always report the prevalence (random PR-AUC) |
| Comparing rules on Brier / log-loss | Unfair; rules don't produce probabilities | Ranking metrics for rules |
| Forgetting current practice | No business case | Implement it as a baseline |

## 10. Checklist
- [ ] Prior and majority baselines.
- [ ] Current-practice rule.
- [ ] Best single-feature rule.
- [ ] Simple logistic model.
- [ ] Same folds and metrics; prevalence reported.

## 11. References
- scikit-learn, *Dummy estimators*. https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators
- Saito, T. & Rehmsmeier, M. (2015). "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets." *PLoS ONE* 10(3): e0118432. https://doi.org/10.1371/journal.pone.0118432
