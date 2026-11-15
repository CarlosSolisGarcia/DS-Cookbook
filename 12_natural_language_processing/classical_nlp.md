# Classical NLP: Bag of Words, TF-IDF, N-grams and Text Classification (incl. Sentiment)

> **Folder:** 12_natural_language_processing · **Prev:** [nlp_workflow.md](nlp_workflow.md) · **Next:** [topic_modeling.md](topic_modeling.md) · **Related:** [classification evaluation](../04_classification/evaluation.md), [model_selection.md (classification)](../04_classification/model_selection.md)

---

## 1. Purpose
Build strong, fast, interpretable text classifiers with scikit-learn: bag-of-words and TF-IDF representations, word and character n-grams, linear models, and evaluation and error analysis suited to text. They are the baseline every transformer must beat.

## 2. When to use it
Every text classification task (topic/category, sentiment, intent, fault type), as the first model and often as the final one for short, domain-specific texts.

## 3. Conceptual foundation
- **Bag of words (BoW):** a document becomes a sparse vector of token counts. Word order is lost, except locally through n-grams.
- **TF-IDF:** term frequency × inverse document frequency. Common words are down-weighted and distinctive words up-weighted. With `sublinear_tf=True`, log(1 + tf) dampens repeated words. L2 normalisation makes documents of different lengths comparable.
- **N-grams:** word bigrams capture short phrases ("no leak", "high vibration"). **Character n-grams** (e.g. 3–5 characters, `analyzer="char_wb"`) are robust to typos, abbreviations and inflections, which suits noisy technical text.
- **Linear models on sparse text:** logistic regression, linear SVM (`LinearSVC`), SGD, and naive Bayes (a fast baseline, but uncalibrated). With a very high dimensionality and few non-zeros, linear models are efficient and strong.
- **Sentiment analysis** is text classification with polarity labels. Negation handling (bigrams), domain-specific vocabulary and sarcasm are the challenges. Classical models do reasonably well. Pretrained transformers capture nuance better.

## 4. Assumptions and limitations
- BoW ignores long-range word order and meaning similarity ("bearing" vs "brg" are unrelated tokens unless normalised, or unless char n-grams link them).
- The vocabulary is fixed at fit time. Unseen words in production are ignored (char n-grams partially mitigate this).
- Coefficients are interpretable per token, but correlated tokens share weight.

## 5. Step-by-step procedure
1. **Baselines:** majority class; keyword rules if domain experts have them.
2. **TF-IDF (words, 1–2-grams) + logistic regression** in a pipeline; stratified (grouped) CV; macro-F1.
3. **Add character n-grams** (a `FeatureUnion`) for typo/abbreviation robustness.
4. **Tune** `C`, `min_df`, the n-gram ranges, `sublinear_tf` (a small grid).
5. **Evaluate:** a per-class report, the confusion matrix, macro-F1.
6. **Interpret:** the top positive coefficients per class; check that they are sensible.
7. **Error analysis:** read the misclassified texts. Typical causes are label noise, ambiguous texts, missing normalisations, and multi-topic texts.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

from cookbook_data import make_maintenance_logs
import re

df = make_maintenance_logs()
def light_clean(t):                       # remove IDs only; keep the rest for the vectoriser
    return re.sub(r"\bwo-\d+\b", " ", t.lower())
X, y = df["text"].map(light_clean), df["category"]
groups = X.factorize()[0]                 # identical texts stay in the same fold
cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)

def evaluate(model, name):
    pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
    print(f"{name:42s} macro-F1={f1_score(y, pred, average='macro'):.3f}  accuracy={np.mean(pred == y):.3f}")
    return pred

word_tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
char_tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)
models = {
    "majority class": DummyClassifier(strategy="most_frequent"),
    "BoW counts + ComplementNB": make_pipeline(CountVectorizer(min_df=2), ComplementNB()),
    "TF-IDF words(1-2) + logistic": make_pipeline(word_tfidf, LogisticRegression(C=5, max_iter=2000)),
    "TF-IDF words + chars + logistic": make_pipeline(FeatureUnion([("w", word_tfidf), ("c", char_tfidf)]),
                                                     LogisticRegression(C=5, max_iter=2000)),
    "TF-IDF words + chars + LinearSVC": make_pipeline(FeatureUnion([("w", word_tfidf), ("c", char_tfidf)]), LinearSVC(C=0.5)),
}
preds = {n: evaluate(m, n) for n, m in models.items()}
```

```python
# Per-class evaluation and confusion matrix of the best model
best = preds["TF-IDF words + chars + logistic"]
print(classification_report(y, best, digits=3))
labels = sorted(y.unique())
print(pd.DataFrame(confusion_matrix(y, best, labels=labels), index=labels, columns=labels))
```

```python
# Interpretation: top weighted n-grams per class (word features of the fitted model)
pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), LogisticRegression(C=5, max_iter=2000)).fit(X, y)
vocab = np.array(pipe[0].get_feature_names_out()); coef = pipe[-1].coef_
for k, c in enumerate(pipe[-1].classes_):
    print(f"{c:16s}", vocab[np.argsort(-coef[k])[:8]].tolist())
```

```python
# Error analysis: read misclassified texts (and check label noise)
err = df.assign(pred=best)[best != y]
print(f"{len(err)} errors ({len(err)/len(df):.1%})")
print(err.sample(8, random_state=0)[["text", "category", "pred"]].to_string())
```

```python
# Sentiment = text classification with polarity labels. Tiny illustrative example of NEGATION handling:
# "no" is good news for "leak" but bad news for "power": only the PAIR carries the meaning.
docs = ["no leak after repair", "leak after repair", "no power after repair", "power ok after repair",
        "no vibration now", "vibration high now"] * 20
lab = ["pos", "neg", "neg", "pos", "pos", "neg"] * 20
for ngr in [(1, 1), (1, 2)]:
    m = make_pipeline(TfidfVectorizer(ngram_range=ngr), LogisticRegression(C=10)).fit(docs, lab)
    print(ngr, m.predict(["no leak", "no power", "leak", "power ok", "no vibration"]))
# Unigrams cannot represent "no leak" (good) vs "no power" (bad); bigrams can. Transformers model this contextually.
```

## 7. Interpretation
- **Baselines:** the majority class gives macro-F1 ≈ 0.09 (one class predicted). Even **BoW + ComplementNB reaches ≈ 0.96**. These short, keyword-driven texts are easy for bag-of-words models.
- **All the linear models tie at ≈ 0.963.** Character n-grams add nothing here. The reason is in the error analysis: about 4% of the labels were randomly reassigned, so ≈ 3% of the labels are wrong (a fifth of the reassignments land on the correct class by chance). **The models are at the label-noise ceiling.** More model complexity cannot help; cleaner labels can. On real data with more typos and abbreviations, char n-grams typically do help. Always test them.
- **Per-class report / confusion matrix:** the errors are spread evenly over all class pairs, which is the signature of random label noise. Systematic confusion between two specific classes would point to genuinely overlapping vocabulary instead.
- **Top n-grams per class** are domain-meaningful ("replaced bearing", "brg" for bearing; "repacked", "gland" for seal_leak; "recalibrated", "loop" for instrumentation). The model has learned the technicians' vocabulary, abbreviations included. ID-like or generic top features would indicate leakage or a cleaning gap.
- **The error sample confirms it:** "brg play 91 C … replaced bearing" labelled *electrical* and "bearing hot … replaced bearing" labelled *instrumentation* are mislabelled orders, not model errors. In real CMMS data such miscoding is common. Error analysis of text is reading, not just counting.
- **Negation demo:** with unigrams every test phrase is classified "pos": the token "no" is mostly associated with good news, so "no power" is misread. With bigrams, "no power" → neg and "no leak" → pos. Meaning lives in word combinations.

## 8. Decision framework
| Situation | Choice |
|---|---|
| Short technical texts with typos/abbreviations | Word (1–2) + char (3–5) TF-IDF + logistic/LinearSVC |
| Very large corpus, need speed | `HashingVectorizer` + `SGDClassifier` (online learning) |
| Need calibrated probabilities | Logistic regression (check calibration) |
| Strong class imbalance | `class_weight="balanced"` or threshold tuning; macro-F1 ([class_imbalance.md](../04_classification/class_imbalance.md)) |
| Multilabel (several codes per text) | `OneVsRestClassifier(LogisticRegression)` with `MultiLabelBinarizer`; micro/macro F1 |
| Classical model plateaus; texts are nuanced | Transformers ([transformers_and_transfer_learning.md](transformers_and_transfer_learning.md)) |
| Few labels | Embeddings + logistic, or active learning to label efficiently |

- **IF** the top features look like IDs/templates → clean them or group-split; re-evaluate.
- **IF** many errors are label noise → estimate the noise rate ([noisy_labels_weak_supervision.md](../19_advanced_tabular_ml/noisy_labels_weak_supervision.md)) before more modelling.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Vectoriser fitted outside CV | Leakage via vocabulary/IDF | Pipeline |
| `min_df=1` with huge vocabularies | Overfitting rare tokens; memory | `min_df` 2–5 |
| Only unigrams for sentiment/negation | Misses "not good" | Bigrams |
| Naive Bayes probabilities used as risks | Overconfident | Logistic, or calibration |
| Ignoring near-duplicates | Inflated CV | Grouped CV by text |

## 10. Checklist
- [ ] Majority + simple keyword/NB baselines.
- [ ] TF-IDF (word ± char n-grams) + linear model in a pipeline; grouped stratified CV.
- [ ] Small grid over C, n-gram ranges, min_df.
- [ ] Per-class report, confusion matrix, macro-F1.
- [ ] Top features per class sanity-checked.
- [ ] Misclassified texts read; label-noise estimate.

## 11. References
- scikit-learn User Guide, *Text feature extraction* (`CountVectorizer`, `TfidfVectorizer`, `HashingVectorizer`). https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
- scikit-learn, *Classification of text documents using sparse features*. https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html
- Manning, C. D., Raghavan, P. & Schütze, H. (2008). *Introduction to Information Retrieval*, Ch. 6 (TF-IDF), Ch. 13 (Naive Bayes). https://nlp.stanford.edu/IR-book/
- Wang, S. & Manning, C. D. (2012). "Baselines and Bigrams: Simple, Good Sentiment and Topic Classification." *ACL*. https://aclanthology.org/P12-2018/
- Rennie, J. D. M. et al. (2003). "Tackling the Poor Assumptions of Naive Bayes Text Classifiers." *ICML* (Complement NB).
