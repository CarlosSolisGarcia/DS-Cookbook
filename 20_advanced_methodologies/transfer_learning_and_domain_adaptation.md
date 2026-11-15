# Transfer Learning and Domain Adaptation (Introduction)

> **Folder:** 20_advanced_methodologies · **Prev:** [drift_and_monitoring.md](drift_and_monitoring.md) · **Next:** [ensembles_and_stacking.md](ensembles_and_stacking.md) · **Related:** [transformers_and_transfer_learning.md](../12_natural_language_processing/transformers_and_transfer_learning.md), [image_classification.md](../13_computer_vision/image_classification.md), [distribution_shift.md](../19_advanced_tabular_ml/distribution_shift.md)

---

## 1. Purpose
Reuse knowledge from a data-rich **source** domain (an existing plant, an established product, a public dataset, a pretrained network) to perform well in a data-poor **target** domain (a new site, a new instrument, a new product), and judge when transfer helps and when it hurts.

## 2. When to use it
- A model must work on a new asset/site/instrument with few labelled examples.
- Pretrained models exist for the modality (text, images), which is the standard case.
- Tabular/process data from similar but not identical systems.

## 3. Conceptual foundation
| Strategy | Idea | Typical use |
|---|---|---|
| **Pretrain → fine-tune** | Initialise from a source-trained model; continue training on the target data | NLP, CV (standard); NNs on sensor data |
| **Feature extraction** | Freeze the source representation; train a simple head on the target | Few target labels |
| **Pooling with a domain indicator** | Train on source + target with a domain feature (and interactions) | Tabular, related sites |
| **Offset / recalibration** | Keep the source model; learn a target-specific bias/scale correction | Instrument changes; a new plant with a systematic offset |
| **Importance weighting** | Re-weight the source rows to resemble the target inputs (covariate shift) | [distribution_shift.md](../19_advanced_tabular_ml/distribution_shift.md) |
| **Domain-invariant features** | Normalise per domain (z-scores per site), or align the statistics (e.g. CORAL) | Sensors with domain-specific offsets |
| **Hierarchical models** | Partial pooling across domains | Many related domains ([grouped_hierarchical_data.md](../19_advanced_tabular_ml/grouped_hierarchical_data.md)) |

**Negative transfer:** when the source and target relationships differ, source data can *hurt*. Always compare with a target-only baseline.

## 4. Assumptions and limitations
- Transfer works when the domains share structure (similar physics, similar visual features). The more different the domains, the less the source helps.
- Small target sets make the comparison noisy. Use repeated splits of the target data.
- Unsupervised domain adaptation (no target labels) is an advanced topic (adversarial alignment, self-training). It is only pointed to here.

## 5. Step-by-step procedure
1. **Characterise the domain gap:** adversarial validation (inputs); do the target labels follow the source model's predictions (an offset? a different slope?).
2. **Baselines:** source-only model on the target; target-only model (if there are enough labels).
3. **Transfer candidates:** source model + target offset/recalibration; pooled training with a domain feature; fine-tuning (NNs); per-domain normalisation.
4. **Evaluate on held-out target data** with repeated splits; plot the performance vs the number of target labels (a learning curve).
5. **Choose** the simplest strategy that beats both baselines; monitor after deployment.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from cookbook_data import load_clean_process_yield

df = load_clean_process_yield()
feats = ["temperature_c", "pressure_bar", "feed_rate_kg_h", "residence_time_min", "catalyst_age_days", "impurity_ppm"]
src = df[df["reactor"].isin(["R1", "R2", "R3"])]
tgt = df[df["reactor"] == "R4"].copy()
# The NEW reactor R4 differs systematically: 12% lower yield (different geometry) — the mechanism is otherwise shared
tgt["yield_kg"] *= 0.88
rng = np.random.default_rng(0)

src_model = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(src[feats], np.log(src["yield_kg"]))

def evaluate(n_target, reps=8):
    res = {"source only": [], "target only": [], "source + offset": [], "pooled + domain flag": []}
    for r in range(reps):
        idx = rng.permutation(len(tgt)); tr, te = tgt.iloc[idx[:n_target]], tgt.iloc[idx[n_target:]]
        y_te = te["yield_kg"]
        res["source only"].append(mean_absolute_error(y_te, np.exp(src_model.predict(te[feats]))))
        if n_target >= 20:
            t_only = HistGradientBoostingRegressor(max_iter=200, min_samples_leaf=5, random_state=0).fit(tr[feats], np.log(tr["yield_kg"]))
            res["target only"].append(mean_absolute_error(y_te, np.exp(t_only.predict(te[feats]))))
        offset = np.median(np.log(tr["yield_kg"]) - src_model.predict(tr[feats]))          # learn ONE number on the target
        res["source + offset"].append(mean_absolute_error(y_te, np.exp(src_model.predict(te[feats]) + offset)))
        pooled_X = pd.concat([src[feats].assign(is_target=0), tr[feats].assign(is_target=1)])
        pooled_y = np.log(pd.concat([src["yield_kg"], tr["yield_kg"]]))
        w = np.r_[np.ones(len(src)), np.full(len(tr), 5.0)]                                # up-weight the scarce target rows
        pooled = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(pooled_X, pooled_y, sample_weight=w)
        res["pooled + domain flag"].append(mean_absolute_error(y_te, np.exp(pooled.predict(te[feats].assign(is_target=1)))))
    return {k: (np.mean(v) if v else np.nan) for k, v in res.items()}

table = pd.DataFrame({n: evaluate(n) for n in [5, 20, 60, 150]}).T
table.index.name = "labelled target batches"
print(table.round(1).to_string())
```

```python
# not-run (requires pretrained weights; see 12_nlp and 13_cv for full examples)
# NLP: AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=K) → Trainer on target texts
# CV:  resnet18(weights=ResNet18_Weights.IMAGENET1K_V1); replace the head; train the head first, then unfreeze with a low LR
```

## 7. Interpretation
This run teaches something less tidy than the textbook story, and more realistic:
- **Source only ≈ 50 kg MAE at every target size.** It never uses the target labels. The −12% domain offset is real, but with these features (no catalyst/supplier information) the *base* error of the model is already ≈ 50 kg. The offset is a modest part of the total error.
- **Source + offset hurts with 5 target batches (≈ 57) and helps only slightly from 20+ (≈ 49–50).** An offset estimated from 5 noisy residuals is itself noisy. The correction must be larger than its own estimation error to pay off. Shrink the offset towards 0 when n is small (e.g. offset × n/(n + k)), or wait for more target data.
- **Target only** is poor with 20 labels (≈ 81) and only competitive at 150 (≈ 48). The classic small-data penalty.
- **Pooled + domain flag is the best from 60 target batches on** (≈ 47–48). It uses all the source data for the shared mechanism and learns the target difference as the target data grow, with no harm at small n (≈ 50).
- **The general lessons:** (1) quantify the domain gap relative to the base error before adapting; (2) every adaptation step has an estimation cost at small n; (3) the learning curve over the number of target labels is the decision tool. It tells you which strategy to use *now* and when to switch.

## 8. Decision framework
- **IF** the target differs by an offset/scale (calibration, unit geometry) → source model + offset/scale recalibration, shrunk towards zero while target labels are few. It pays off once the offset is large relative to its estimation noise.
- **IF** the relationships partly differ, and there are moderate target labels → pooled training with a domain feature (+ interactions), or fine-tuning.
- **IF** it is images/text → pretrained models + fine-tuning are the default ([12](../12_natural_language_processing/transformers_and_transfer_learning.md), [13](../13_computer_vision/image_classification.md)).
- **IF** there are many related domains → hierarchical/partially pooled models.
- **IF** the source-informed models are worse than target-only once enough target data exist → negative transfer. Drop the source or reduce its weight.
- **IF** there are no target labels → importance weighting and per-domain normalisation; label a few target examples as soon as possible.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| No target-only baseline | Negative transfer unnoticed | Compare both baselines |
| Evaluating on target data used for adaptation | Optimistic | Hold out target data |
| Complex adaptation for a simple offset | Variance without benefit | Try the recalibration first |
| One random target split with tiny n | Noisy conclusions | Repeated splits; learning curves |

## 10. Checklist
- [ ] Domain gap characterised (inputs and outputs).
- [ ] Source-only and target-only baselines.
- [ ] Offset/recalibration, pooled-with-flag and fine-tuning candidates as relevant.
- [ ] A learning curve over the number of target labels with repeated splits.
- [ ] The simplest strategy that beats both baselines; monitoring after deployment.

## 11. References
- Pan, S. J. & Yang, Q. (2010). "A Survey on Transfer Learning." *IEEE TKDE* 22(10), 1345–1359. https://doi.org/10.1109/TKDE.2009.191
- Ben-David, S. et al. (2010). "A theory of learning from different domains." *Machine Learning* 79, 151–175.
- Sun, B., Feng, J. & Saenko, K. (2016). "Return of Frustratingly Easy Domain Adaptation" (CORAL). *AAAI*. https://arxiv.org/abs/1511.05547
- Daumé III, H. (2007). "Frustratingly Easy Domain Adaptation." *ACL*. https://aclanthology.org/P07-1033/
- Weiss, K., Khoshgoftaar, T. M. & Wang, D. (2016). "A survey of transfer learning." *Journal of Big Data* 3, 9. https://doi.org/10.1186/s40537-016-0043-6
