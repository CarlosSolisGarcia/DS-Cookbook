# Learning with Noisy Labels and an Introduction to Weak Supervision

> **Folder:** 19_advanced_tabular_ml · **Prev:** [entity_resolution.md](entity_resolution.md) · **Next:** [20_advanced_methodologies/README.md](../20_advanced_methodologies/README.md) · **Related:** [data_quality.md](../01_data_understanding/data_quality.md) (label quality), [classical_nlp.md](../12_natural_language_processing/classical_nlp.md) (a label-noise ceiling example)

---

## 1. Purpose
Detect, quantify and mitigate label errors (miscoded work orders, inconsistent defect grading, mislabelled samples), and create training labels cheaply from heuristics when manual labels are scarce (weak supervision).

## 2. When to use it
- Labels come from busy humans (technicians, operators), from automated rules, or from proxies.
- Error analysis shows "errors" that look like mislabelled examples.
- Labelling is expensive, but domain heuristics exist ("if the text mentions 'leak' → seal_leak").

## 3. Conceptual foundation
- **Types of label noise:** random (symmetric), class-conditional (e.g. "lubrication" often logged as "bearing"), and instance-dependent (ambiguous cases mislabelled more often).
- **Effects:** label noise caps the achievable accuracy, biases the probabilities, and in flexible models leads to memorisation of the wrong labels.
- **Detection by out-of-fold disagreement (confident learning, simplified):** train with CV, obtain OOF probabilities, and flag the examples whose given label has a low OOF probability while another class has a high one. Estimate the noise rate per class pair from the "confident joint". The `cleanlab` library implements this rigorously.
- **Mitigation:** relabel the flagged examples (best, with experts); remove them (risky); down-weight them; use robust models/regularisation (simpler models, early stopping, label smoothing in NNs).
- **Weak supervision:** write several noisy **labelling functions** (LFs: rules, keyword patterns, existing models, thresholds). Each LF votes or abstains. Combine the votes (majority vote, or a label model that estimates each LF's accuracy, e.g. Snorkel), then train an end model on the resulting probabilistic labels, which generalises beyond the LFs.

## 4. Assumptions and limitations
- Detection methods find *suspected* errors. Some flagged cases are genuinely ambiguous. Expert review decides.
- Removing hard examples can remove exactly the informative cases (rare modes). Prefer relabelling over deletion.
- Weak supervision quality depends on LF coverage and accuracy. It needs a small gold-labelled set for evaluation.

## 5. Step-by-step procedure
1. **Measure the agreement** where possible (two annotators on a sample: Cohen's κ). This bounds the achievable performance.
2. **OOF probabilities** from a reasonable model (grouped CV if needed).
3. **Flag suspects:** a low probability for the given label and a high one for another label; rank by confidence.
4. **Review the top suspects** with experts; estimate the precision of the flags; relabel.
5. **Retrain** and compare on a *clean* evaluation set (a reviewed sample).
6. **Weak supervision (if labels are scarce):** write LFs; check their coverage/overlap/conflicts on unlabelled data; combine them; train the end model; evaluate on a small gold set.

## 6. Python implementation

```python
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

from cookbook_data import make_maintenance_logs

clean = make_maintenance_logs(label_noise=0.0)               # the same texts with TRUE labels (for verification only)
noisy = make_maintenance_logs(label_noise=0.10)              # 10% labels randomly reassigned (what you receive)
texts = noisy["text"].str.lower().str.replace(r"\bwo-\d+\b", " ", regex=True)
y_noisy, y_true = noisy["category"].to_numpy(), clean["category"].to_numpy()
print(f"actual label error rate in the received data: {np.mean(y_noisy != y_true):.3f}")

model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), LogisticRegression(C=5, max_iter=2000))
P = cross_val_predict(model, texts, y_noisy, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")
classes = np.sort(np.unique(y_noisy)); idx = {c: k for k, c in enumerate(classes)}
p_given = P[np.arange(len(P)), [idx[c] for c in y_noisy]]
suspect_score = P.max(1) - p_given                            # high when another class is much more likely than the given one
flagged = np.argsort(-suspect_score)[: int(0.12 * len(P))]   # review the top 12%
prec = np.mean(y_noisy[flagged] != y_true[flagged]); rec = np.sum(y_noisy[flagged] != y_true[flagged]) / np.sum(y_noisy != y_true)
print(f"flagged {len(flagged)} suspects: {prec:.0%} are true label errors; they cover {rec:.0%} of all errors")
```

```python
# Mitigation compared on a CLEAN evaluation (here: true labels; in practice an expert-reviewed sample)
eval_idx = np.random.default_rng(0).choice(len(texts), 600, replace=False); train_idx = np.setdiff1d(np.arange(len(texts)), eval_idx)
def fit_score(train_labels, mask=None):
    tr = train_idx if mask is None else train_idx[mask[train_idx]]
    m = model.fit(texts.iloc[tr], train_labels[tr])
    return f1_score(y_true[eval_idx], m.predict(texts.iloc[eval_idx]), average="macro")
relabelled = y_noisy.copy(); relabelled[flagged] = classes[P[flagged].argmax(1)]   # stand-in for expert relabelling
keep = np.ones(len(texts), bool); keep[flagged] = False
print(f"macro-F1 on clean eval — noisy labels: {fit_score(y_noisy):.3f} | flagged removed: {fit_score(y_noisy, keep):.3f} | "
      f"flagged relabelled: {fit_score(relabelled):.3f} | true labels (ceiling): {fit_score(y_true):.3f}")
```

```python
# Weak supervision: keyword labelling functions → majority vote → end model generalising beyond the keywords
ABSTAIN = None
LFS = {
    "bearing": [r"\bbearing", r"\bbrg\b", r"grinding|rumbling"],
    "seal_leak": [r"\bseal\b", r"leak|dripping|weeping|gland"],
    "electrical": [r"breaker|contactor|tripped|overcurrent"],
    "lubrication": [r"\boil\b|grease|lubric"],
    "instrumentation": [r"transmitter|sensor|recalibrat|gauge"],
}
def apply_lfs(t):
    votes = [c for c, pats in LFS.items() for p in pats if re.search(p, t)]
    return votes
votes = texts.map(apply_lfs)
coverage = np.mean(votes.str.len() > 0); conflict = np.mean(votes.map(lambda v: len(set(v)) > 1))
weak = votes.map(lambda v: pd.Series(v).mode().iloc[0] if len(v) else ABSTAIN)
labelled = weak.notna().to_numpy()
print(f"LF coverage={coverage:.2%}  conflicts={conflict:.2%}  weak-label accuracy={np.mean(weak[labelled] == y_true[labelled]):.3f}")
end = model.fit(texts[labelled & np.isin(np.arange(len(texts)), train_idx)], weak[labelled & np.isin(np.arange(len(texts)), train_idx)])
print(f"end model trained on weak labels → macro-F1 on clean eval: "
      f"{f1_score(y_true[eval_idx], end.predict(texts.iloc[eval_idx]), average='macro'):.3f}")
```

## 7. Interpretation
- **Flagging:** with 7.5% wrong labels, the top 12% of suspects contain **all** the label errors (100% recall) at 62% precision. Reviewing 360 texts instead of 3000 would find every miscoded order. In real data the separation is less clean (ambiguous texts look like errors too), so measure the flag precision on a reviewed subset.
- **Mitigation, and a surprising but important result:** on the *clean* evaluation set, the model trained on noisy labels already reaches macro-F1 ≈ 0.998. Relabelling or removing the suspects brings it to 1.0. **Random (symmetric) label noise barely hurts a regularised linear model when the signal is strong.** The wrong labels are outvoted by the many correct ones. The real damage of label noise here is to the **evaluation**: measured on noisy labels, the same model looks like ≈ 0.96 ([classical_nlp.md](../12_natural_language_processing/classical_nlp.md)). Class-conditional noise (systematic miscoding) and flexible models that memorise are much more harmful to training. Clean evaluation data matter most.
- **Weak supervision:** keyword LFs cover 95% of the texts, with 23% conflicts (texts mentioning two systems, e.g. "… oil ok"), and their majority vote is 95% accurate. The end model trained only on these weak labels reaches macro-F1 ≈ 0.98 on the clean evaluation, without a single manual label. It generalises beyond the keywords by learning from the co-occurring words. The conflicts show where to refine the LFs (priorities, negation such as "ok").

## 8. Decision framework
- **IF** the CV performance plateaus near (1 − estimated noise rate) → you are at the label ceiling. Invest in label quality, not models.
- **IF** many suspected errors are concentrated in specific class pairs → class-conditional confusion. Clarify the labelling guidelines with the data creators.
- **IF** expert time is limited → review the top-ranked suspects (confident-learning style), relabel, and repeat.
- **IF** labels are scarce but rules exist → weak supervision (LFs + label model) → an end model; evaluate on a small gold set.
- **IF** the noise is heavy and unavoidable → simpler, regularised models; avoid memorising (early stopping); consider noise-robust losses (NNs).

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Evaluating on noisy labels only | Underestimating clean performance; wrong model choice | A reviewed clean evaluation sample |
| Deleting all flagged examples | Losing hard/rare informative cases | Relabel; check the class balance of the removals |
| Flagging with in-sample predictions | The model agrees with its own memorised errors | OOF predictions |
| LFs evaluated only by coverage | Low-accuracy labels | Measure LF accuracy on a gold sample |
| Ignoring annotator disagreement | Unrealistic performance targets | Measure κ; set expectations |

## 10. Checklist
- [ ] Label agreement measured (if possible).
- [ ] OOF-based suspect ranking; expert review of the top suspects.
- [ ] Mitigation compared on a clean evaluation set.
- [ ] Weak supervision: LF coverage/conflicts/accuracy; end model evaluated on a gold set.

## 11. References
- Northcutt, C., Jiang, L. & Chuang, I. (2021). "Confident Learning: Estimating Uncertainty in Dataset Labels." *JAIR* 70, 1373–1411. https://doi.org/10.1613/jair.1.12125 · cleanlab: https://docs.cleanlab.ai/
- Frénay, B. & Verleysen, M. (2014). "Classification in the Presence of Label Noise: A Survey." *IEEE TNNLS* 25(5), 845–869.
- Ratner, A. et al. (2017). "Snorkel: Rapid Training Data Creation with Weak Supervision." *VLDB*. https://arxiv.org/abs/1711.10160
- Song, H. et al. (2022). "Learning From Noisy Labels With Deep Neural Networks: A Survey." *IEEE TNNLS*. https://arxiv.org/abs/2007.08199
