# Word and Document Embeddings, Semantic Similarity and Search

> **Folder:** 12_natural_language_processing · **Prev:** [topic_modeling.md](topic_modeling.md) · **Next:** [transformers_and_transfer_learning.md](transformers_and_transfer_learning.md) · **Related:** [visualization_embeddings.md](../11_dimensionality_reduction/visualization_embeddings.md), [entity_resolution.md](../19_advanced_tabular_ml/entity_resolution.md)

---

## 1. Purpose
Represent words and documents as dense vectors whose geometry reflects meaning. Use these vectors for semantic similarity ("find past incidents like this one"), deduplication, clustering, and as features for classifiers.

## 2. When to use it
- Search and retrieval over reports, manuals or tickets.
- Near-duplicate detection and record linkage of free text.
- Few-label classification (embeddings + logistic regression).
- Clustering/visualising documents by meaning.

## 3. Conceptual foundation
| Representation | How | Captures | Limits |
|---|---|---|---|
| **TF-IDF vectors** | Sparse weighted counts | Lexical overlap | No synonymy ("brg" ≠ "bearing" unless normalised) |
| **LSA** (TruncatedSVD of TF-IDF) | Low-rank factorisation | Co-occurrence-based synonymy | Linear; corpus-specific |
| **Static word embeddings** (word2vec, GloVe, fastText) | Predict words from contexts | Word similarity; fastText handles sub-words/typos | One vector per word (no context: "seal" as a gasket vs the animal) |
| **Contextual embeddings** (BERT-family) | A transformer's hidden states | Meaning in context | Need pooling and a model download; heavier |
| **Sentence-embedding models** (e.g. Sentence-Transformers) | Transformers trained so that similar sentences have similar vectors | Semantic similarity of whole sentences | Domain shift for technical jargon; a model download |

**Cosine similarity** cos(a, b) = a·b / (‖a‖‖b‖) is the standard similarity measure. With L2-normalised vectors it is just the dot product, so nearest-neighbour search becomes a matrix product.

**Distributional hypothesis:** words that occur in similar contexts have similar meanings. All of the embedding methods above exploit it.

## 4. Assumptions and limitations
- General-purpose embedding models may not know plant-specific jargon and abbreviations. Normalising the text, or fine-tuning on domain data, helps.
- Similarity ≠ relevance. Two reports can be similar in wording but differ in the critical detail (the equipment, a negation).
- Large embedding models require downloads (Hugging Face Hub). Plan for offline caching on secured networks.

## 5. Step-by-step procedure
1. **Define "similar"** for the use case (same failure mode? same equipment? same root cause?) and build a small **evaluation set** of query → relevant documents.
2. **Lexical baseline:** TF-IDF (word + char) cosine similarity.
3. **LSA** (TruncatedSVD, 100–300 dimensions) for co-occurrence synonymy.
4. **Sentence embeddings** from a pretrained model (if available), using the same evaluation.
5. **Evaluate retrieval:** precision@k / recall@k / MRR on the evaluation set.
6. **For classification:** embeddings → logistic regression; compare with TF-IDF.

## 6. Python implementation

```python
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import Normalizer

from cookbook_data import make_maintenance_logs

df = make_maintenance_logs()
texts = df["text"].map(lambda t: re.sub(r"\bwo-\d+\b", " ", t.lower()))

vec = FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                    ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True))])
X_tfidf = Normalizer().fit_transform(vec.fit_transform(texts))
lsa = make_pipeline(TruncatedSVD(150, random_state=0), Normalizer())
X_lsa = lsa.fit_transform(X_tfidf)

def search(query, X, k=3, fitted_vec=vec, transform=None):
    q = Normalizer().transform(fitted_vec.transform([query.lower()]))
    if transform is not None:
        q = transform.transform(q)
    sims = X @ q.T
    sims = sims.toarray().ravel() if hasattr(sims, "toarray") else np.asarray(sims).ravel()   # sparse or dense input
    top = np.argsort(-sims)[:k]
    return [(round(float(sims[i]), 3), df["category"].iloc[i], df["text"].iloc[i][:70]) for i in top]

for q in ["brg grinding noise on pump", "mechanical seal dripping water"]:
    print("\nQUERY:", q)
    for r in search(q, X_tfidf): print("  tfidf", r)
    for r in search(q, X_lsa, transform=lsa): print("  lsa  ", r)
```

```python
# Retrieval evaluation: a relevant result = same failure category (a proxy for a real relevance set)
def precision_at_k(X, labels, k=5, n_queries=300, seed=0):
    rng = np.random.default_rng(seed); idx = rng.choice(X.shape[0], n_queries, replace=False)
    S = X[idx] @ X.T
    S = S.toarray() if hasattr(S, "toarray") else np.asarray(S)
    S[np.arange(n_queries), idx] = -np.inf                     # exclude the query itself
    top = np.argsort(-S, axis=1)[:, :k]
    return np.mean(labels[top] == labels[idx][:, None])

labels = df["category"].to_numpy()
print(f"precision@5 TF-IDF = {precision_at_k(X_tfidf, labels):.3f}   LSA = {precision_at_k(X_lsa, labels):.3f}")

# Embeddings as classifier features (few-label scenario: 100 labelled examples)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
small = np.random.default_rng(1).choice(len(df), 100, replace=False)
for name, X in [("TF-IDF", X_tfidf), ("LSA-150", X_lsa)]:
    s = cross_val_score(LogisticRegression(C=10, max_iter=2000), X[small], labels[small], cv=cv, scoring="f1_macro")
    print(f"{name:8s} macro-F1 with 100 labels: {s.mean():.3f} ± {s.std():.3f}")
```

```python
# not-run (requires downloading a pretrained model from the Hugging Face Hub; CPU is fine for inference)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")      # ~90 MB, 384-dim embeddings
E = model.encode(texts.tolist(), batch_size=64, normalize_embeddings=True, show_progress_bar=True)
q = model.encode(["brg grinding noise on pump"], normalize_embeddings=True)
top = np.argsort(-(E @ q.T).ravel())[:5]
print(df.iloc[top][["category", "text"]])
print("precision@5 sentence embeddings:", precision_at_k(E, labels))
```

```python
# Word-level embeddings learned from the corpus itself: LSA term vectors (no download).
# Unigrams only (bigrams would make every word's nearest neighbours its own bigrams).
uni = TfidfVectorizer(min_df=3, token_pattern=r"[a-z]{2,}")
Xu = uni.fit_transform(texts.str.replace(r"\d+", " ", regex=True))
terms = np.array(uni.get_feature_names_out())
T = Normalizer().fit_transform(TruncatedSVD(30, random_state=0).fit(Xu).components_.T)   # one vector per word
def neighbours(word, k=5):
    i = np.where(terms == word)[0][0]
    sims = T @ T[i]
    return [(terms[j], round(float(sims[j]), 2)) for j in np.argsort(-sims)[1:k + 1]]
print("bearing →", neighbours("bearing")); print("brg →", neighbours("brg")); print("leak →", neighbours("leak"))
```

## 7. Interpretation
- **Search results:** with TF-IDF word + char n-grams, the query "brg grinding noise" retrieves bearing jobs (the char n-grams connect "brg" to similar strings, and the word overlap with "grinding noise" does the rest). LSA may also retrieve semantically related reports that use other words.
- **precision@5:** the fraction of the top-5 neighbours sharing the query's category. On real, more varied text, sentence embeddings typically outperform TF-IDF on paraphrases.
- **Word neighbours (LSA on unigrams):** "leak" → seal, dripping, spraying, weeping; "bearing" → swapped, play, rumbling and even the typos "bearigs" and "rereased". Words that occur in similar contexts end up close together: the distributional hypothesis in action. Note what does **not** happen: "brg" is close to bearing *symptoms* ("rumbling", "hot", "play") but not to the word "bearing" itself. In this corpus "brg" appears in different templates ("changed brg") than "bearing" ("replaced bearing"), and they never co-occur. Embeddings learn only what the corpus shows. Small or templated corpora need explicit normalisation (an abbreviation dictionary), or embeddings pretrained on much larger text.
- **Retrieval and few-label results:** TF-IDF and LSA tie here (precision@5 ≈ 0.93; few-label macro-F1 ≈ 0.93). This corpus is lexically very regular. On varied natural language, dense embeddings usually win when labels are scarce or when paraphrases dominate. A mislabelled order also appears among the top results for the seal query (label noise again).

## 8. Decision framework
- **IF** the queries share vocabulary with the documents (technical text, codes) → TF-IDF word + char cosine is a strong, explainable baseline.
- **IF** paraphrases/synonyms matter (natural language, different authors) → sentence embeddings; evaluate them against TF-IDF on a labelled query set.
- **IF** offline/air-gapped → LSA or locally trained fastText/word2vec; cache pretrained models in advance.
- **IF** few labels for classification → embeddings + logistic regression; compare with TF-IDF.
- **IF** large corpora (> 10⁶ docs) → approximate nearest-neighbour indexes (e.g. FAISS) on normalised embeddings.
- **IF** the critical detail is a specific entity/number → combine semantic similarity with filters (same equipment class) or a re-ranking step.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Unnormalised vectors with dot-product search | Long documents dominate | L2-normalise; cosine |
| Evaluating search by eyeballing a few queries | Over-optimism | A labelled query set; precision@k / MRR |
| General embeddings on heavy jargon without checks | Poor retrieval | Compare with TF-IDF; normalise abbreviations; domain fine-tuning |
| Fitting LSA/vocabulary on test documents for evaluation | Leakage in reported scores | Fit within the training data for evaluation |
| Ignoring negation/entities | Similar but wrong matches | Hybrid search; filters; re-ranking |

## 10. Checklist
- [ ] "Similar" defined; a labelled query/relevance set built.
- [ ] TF-IDF baseline (word + char), normalised.
- [ ] LSA and/or sentence embeddings compared on the same evaluation.
- [ ] Retrieval metrics (precision@k, MRR) reported.
- [ ] For classification: embeddings vs TF-IDF, especially with few labels.

## 11. References
- Mikolov, T., Chen, K., Corrado, G. & Dean, J. (2013). "Efficient Estimation of Word Representations in Vector Space." https://arxiv.org/abs/1301.3781
- Bojanowski, P., Grave, E., Joulin, A. & Mikolov, T. (2017). "Enriching Word Vectors with Subword Information." *TACL* 5 (fastText). https://doi.org/10.1162/tacl_a_00051
- Deerwester, S. et al. (1990). "Indexing by latent semantic analysis." *JASIS* 41(6), 391–407.
- Reimers, N. & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *EMNLP*. https://arxiv.org/abs/1908.10084 · Docs: https://www.sbert.net/
- Jurafsky, D. & Martin, J. H. *SLP3*, Ch. 6 "Vector Semantics and Embeddings". https://web.stanford.edu/~jurafsky/slp3/
