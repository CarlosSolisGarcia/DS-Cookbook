# Topic Modelling: Discovering Themes in Unlabelled Text

> **Folder:** 12_natural_language_processing · **Prev:** [classical_nlp.md](classical_nlp.md) · **Next:** [embeddings_and_similarity.md](embeddings_and_similarity.md) · **Related:** [09_unsupervised_learning](../09_unsupervised_learning/unsupervised_workflow.md) (validation logic for unsupervised results)

---

## 1. Purpose
Find recurring themes in a corpus without labels (incident reports, maintenance logs, customer feedback, abstracts), describe each theme by its words and representative documents, and use the themes for exploration, monitoring or as weak labels.

## 2. When to use it
- Large unlabelled corpora where you need to know "what people are writing about".
- To bootstrap a labelling scheme (candidate categories) or to monitor theme prevalence over time.
- **Not** as a replacement for a classifier when labels exist, and not for short texts with few words per document (topics become noisy; consider clustering sentence embeddings instead).

## 3. Conceptual foundation
| Method | Model | Input | Notes |
|---|---|---|---|
| **LDA** (Latent Dirichlet Allocation) | Probabilistic: documents are mixtures of topics, topics are distributions over words | Raw **counts** (`CountVectorizer`) | Classic; soft memberships; tune the number of topics and the priors |
| **NMF** (Non-negative Matrix Factorisation) | X ≈ W H with W, H ≥ 0 | Usually **TF-IDF** | Often more coherent topics on short texts; deterministic given the init |
| **Embedding clustering** (e.g. BERTopic-style) | Cluster sentence embeddings, then describe the clusters with class-TF-IDF | Dense embeddings | Good for short texts; needs an embedding model ([embeddings_and_similarity.md](embeddings_and_similarity.md)) |

**Evaluating topics** (no ground truth): human interpretability (are the top words coherent?), **coherence** measures (co-occurrence of the top words in the corpus), stability across seeds and subsamples, and usefulness (does the theme prevalence track known events?). Perplexity (for LDA) often correlates poorly with human judgement.

## 4. Assumptions and limitations
- BoW assumption: word order is ignored, and synonyms/abbreviations split topics unless normalised.
- The number of topics k is a modelling choice. Several values can be "valid" at different granularities.
- Topics are descriptive. Naming them is an interpretation that requires domain review.

## 5. Step-by-step procedure
1. **Clean and normalise** ([nlp_workflow.md](nlp_workflow.md)); remove corpus-specific boilerplate words ("checked", "as per").
2. **Vectorise:** counts for LDA, TF-IDF for NMF; `min_df`/`max_df` to drop rare and ubiquitous words.
3. **Fit for several k;** compute a coherence proxy and check the stability.
4. **Inspect the top words and representative documents** per topic; name the topics with experts.
5. **Assign** the dominant topic per document; analyse its prevalence over time and segments.
6. **If labels exist for a subset,** compare topics and labels (ARI/NMI, cross-tab) as external validation.

## 6. Python implementation

```python
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score

from cookbook_data import make_maintenance_logs

df = make_maintenance_logs()
BOILER = {"checked", "inspected", "site", "operator", "reported", "night", "shift", "pm", "previous", "wo", "urgent",
          "safety", "ok", "isolated", "equipment", "returned", "service", "pump", "compressor", "fan", "agitator", "conveyor"}
def clean(t):
    t = re.sub(r"\bwo-\d+\b|\b[a-z]-\d+\b|\d+", " ", t.lower())
    return " ".join(w for w in re.findall(r"[a-z]+", t) if w not in BOILER)
docs = df["text"].map(clean)

tfidf = TfidfVectorizer(min_df=5, max_df=0.5, stop_words="english")
Xt = tfidf.fit_transform(docs); vocab = np.array(tfidf.get_feature_names_out())
counts = CountVectorizer(min_df=5, max_df=0.5, stop_words="english")
Xc = counts.fit_transform(docs); vocab_c = np.array(counts.get_feature_names_out())

def top_words(H, vocab, n=7):
    return [", ".join(vocab[np.argsort(-h)[:n]]) for h in H]

def umass_coherence(H, X_bin, n=7):
    """UMass coherence per topic: mean log P(w_i, w_j)/P(w_j) over top-word pairs (higher = more coherent)."""
    D = X_bin.shape[0]; df_ = np.asarray(X_bin.sum(0)).ravel(); scores = []
    for h in H:
        top = np.argsort(-h)[:n]; s = []
        for i in range(1, n):
            for j in range(i):
                co = X_bin[:, top[i]].multiply(X_bin[:, top[j]]).sum()
                s.append(np.log((co + 1) / df_[top[j]]))
        scores.append(np.mean(s))
    return np.mean(scores)

Xbin = (Xc > 0).astype(int)
for k in [3, 5, 8]:
    nmf = NMF(n_components=k, init="nndsvda", random_state=0, max_iter=500).fit(Xt)
    H_c = np.zeros((k, len(vocab_c)))                       # map NMF topics onto the count vocabulary for coherence
    idx = {w: i for i, w in enumerate(vocab_c)}
    for t, h in enumerate(nmf.components_):
        for w, val in zip(vocab, h):
            if w in idx: H_c[t, idx[w]] = val
    print(f"NMF k={k}: UMass coherence = {umass_coherence(H_c, Xbin):.2f}")
```

```python
k = 5
nmf = NMF(n_components=k, init="nndsvda", random_state=0, max_iter=500).fit(Xt)
lda = LatentDirichletAllocation(n_components=k, learning_method="batch", random_state=0).fit(Xc)
print("NMF topics:"); [print(f"  T{i}: {w}") for i, w in enumerate(top_words(nmf.components_, vocab))]
print("LDA topics:"); [print(f"  T{i}: {w}") for i, w in enumerate(top_words(lda.components_, vocab_c))]

# Representative documents and external validation against the (here known) categories
W = nmf.transform(Xt); dominant = W.argmax(1)
for t in range(k):
    ex = df["text"].iloc[np.argsort(-W[:, t])[:2]].tolist()
    print(f"T{t} examples: {ex}")
print("ARI(NMF dominant topic, category) =", round(adjusted_rand_score(df["category"], dominant), 3))
print(pd.crosstab(dominant, df["category"]))
```

```python
# Topic prevalence over time (monitoring themes)
prev = pd.DataFrame(W / W.sum(1, keepdims=True), index=df["created"]).resample("QE").mean()
prev.columns = [f"T{i}" for i in range(k)]
print(prev.round(2))
```

## 7. Interpretation
- **Coherence vs k:** UMass coherence is highest at k = 3 and similar for k = 5 and k = 8. Like silhouette in clustering, coherence often prefers coarse solutions. Use it as a hint and inspect the topics at neighbouring k.
- **NMF topics with k = 5 are instructive:** four topics map onto categories (bearing, seal, electrical, lubrication). The **fifth topic is "brg, changed, noise, grinding…"**, which is the *abbreviation* "brg" forming its own bearing topic. Instrumentation has no topic: its documents are spread over the electrical and lubrication topics (see the cross-tab). The model found a real pattern in the *vocabulary* (two ways of writing bearing jobs), not in the *failure mechanisms*. **Fix:** normalise the abbreviations (`brg → bearing`, [nlp_workflow.md](nlp_workflow.md)) and refit. Instrumentation will then likely get its own topic. Topic models faithfully expose vocabulary artefacts, which is useful in itself.
- **LDA** finds a similar structure, with more mixing (e.g. "vibration" words appearing in an electrical topic), which is typical on short texts.
- **The ARI with the categories (≈ 0.64)** (known here, unknown in a real unlabelled setting) quantifies the agreement. The cross-tab shows exactly which topics split or merge categories.
- **Prevalence over time** is roughly stable here (the synthetic data have no trend). In real data, a rising theme (e.g. seal leaks after a supplier change) is a lead worth investigating.

## 8. Decision framework
- **IF** the documents are short and keyword-driven → NMF on TF-IDF first, then LDA for comparison. Remove boilerplate words.
- **IF** the documents are short and semantically varied (no shared keywords) → cluster sentence embeddings and describe the clusters with TF-IDF terms.
- **IF** the topics are incoherent → improve the cleaning (boilerplate, abbreviations), adjust min_df/max_df, try other k.
- **IF** the topics will become labels → validate them on a labelled sample; treat them as weak labels.
- **IF** stakeholders need stable categories over time → fix the vocabulary and model, and assign new documents with `transform` rather than refitting.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| LDA on TF-IDF | Violates its count-based model | Counts for LDA |
| Not removing boilerplate | Topics full of generic words | Corpus-specific stop words |
| Choosing k by perplexity alone | Uninterpretable topics | Coherence + human review |
| Refitting on every update | Topic identities change | Fixed model + transform; scheduled refits |
| Treating topics as ground truth | Over-interpretation | Expert naming; external validation |

## 10. Checklist
- [ ] Cleaning with corpus-specific stop words.
- [ ] Counts (LDA) / TF-IDF (NMF) with min_df/max_df.
- [ ] Several k compared (coherence, stability, inspection).
- [ ] Top words + representative documents reviewed with experts.
- [ ] External validation where labels exist.
- [ ] Assignment/monitoring plan.

## 11. References
- Blei, D. M., Ng, A. Y. & Jordan, M. I. (2003). "Latent Dirichlet Allocation." *JMLR* 3, 993–1022. https://jmlr.org/papers/v3/blei03a.html
- Lee, D. D. & Seung, H. S. (1999). "Learning the parts of objects by non-negative matrix factorization." *Nature* 401, 788–791.
- Mimno, D., Wallach, H., Talley, E., Leenders, M. & McCallum, A. (2011). "Optimizing Semantic Coherence in Topic Models." *EMNLP* (UMass coherence). https://aclanthology.org/D11-1024/
- scikit-learn, *Topic extraction with NMF and LDA*. https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
- Grootendorst, M. (2022). "BERTopic: Neural topic modeling with a class-based TF-IDF procedure." https://arxiv.org/abs/2203.05794
