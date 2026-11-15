# Recommender Baselines and Content-Based Filtering

> **Folder:** 14_recommender_systems · **Prev:** [recsys_workflow.md](recsys_workflow.md) · **Next:** [collaborative_filtering.md](collaborative_filtering.md) · **Related:** [classical_nlp.md](../12_natural_language_processing/classical_nlp.md) (TF-IDF), [embeddings_and_similarity.md](../12_natural_language_processing/embeddings_and_similarity.md)

---

## 1. Purpose
Establish strong non-personalised baselines (popularity, recent popularity, segment popularity) and content-based recommenders (item-attribute and text similarity), which also solve the item cold-start problem.

## 2. When to use it
Always first. Popularity is the bar any personalised model must clear. Content-based methods are used when item metadata is rich, items are new, or interaction data are sparse.

## 3. Conceptual foundation
- **Global popularity:** the top-k items by interaction count. Surprisingly hard to beat on accuracy metrics, and zero personalisation.
- **Recency-weighted popularity:** counts weighted by exp(−age/τ). It adapts to trends and to new items gaining traction.
- **Segment popularity:** the popularity within the user's segment (site, role, equipment type). Cheap personalisation.
- **Content-based filtering:** represent each item by features (category, text description → TF-IDF/embeddings, technical attributes). Build a **user profile** as the (weighted) average of the vectors of the items they interacted with. Recommend the unseen items most similar to the profile.
  - Strength: works for new items with no interactions. It is explainable ("because you used X").
  - Weakness: over-specialisation (recommends more of the same), and it needs good item features.

## 4. Assumptions and limitations
- Popularity baselines inherit exposure bias (popular because promoted).
- Content quality bounds content-based performance. Poor or generic descriptions give poor recommendations.
- User profiles from very few interactions are noisy.

## 5. Step-by-step procedure
1. Use the temporal split and the evaluation harness from [recsys_workflow.md](recsys_workflow.md).
2. Global popularity (excluding the seen items); recency-weighted popularity (tune τ on a validation window).
3. Segment popularity if user segments exist.
4. Content-based: item vectors (one-hot category + TF-IDF of the description); user profiles; cosine ranking.
5. Evaluate all of them; examine coverage and popularity bias; evaluate the cold-start items separately.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, normalize

from cookbook_data import make_interactions

inter, items = make_interactions()
T = inter["timestamp"].max() - pd.Timedelta(days=90)
train, test = inter[inter["timestamp"] < T], inter[inter["timestamp"] >= T]
test = test[test["user_id"].isin(train["user_id"])]
truth = test.groupby("user_id")["item_id"].apply(set)
seen = train.groupby("user_id")["item_id"].apply(set)
K = 10

def evaluate(recs, truth=truth, k=K):                                     # compact version of the harness in recsys_workflow.md
    P, R, N = [], [], []
    for u, rel in truth.items():
        rec = recs.get(u, [])[:k]; hits = np.array([r in rel for r in rec], float)
        P.append(hits.sum() / k); R.append(hits.sum() / len(rel))
        N.append((hits / np.log2(np.arange(2, len(hits) + 2))).sum() / (1 / np.log2(np.arange(2, min(len(rel), k) + 2))).sum())
    rec_items = {i for u in truth.index for i in recs.get(u, [])[:k]}
    return {"precision@10": np.mean(P), "recall@10": np.mean(R), "nDCG@10": np.mean(N), "coverage": len(rec_items) / len(items)}

def top_k_excluding(scores: pd.Series, user, k=K):
    s = scores.drop(list(seen.get(user, set())), errors="ignore")
    return list(s.nlargest(k).index)

# 1. Global popularity
pop = train["item_id"].value_counts().astype(float)
recs_pop = {u: top_k_excluding(pop, u) for u in truth.index}

# 2. Recency-weighted popularity (τ = 60 days)
age = (T - train["timestamp"]).dt.days
recent_pop = train.assign(w=np.exp(-age / 60)).groupby("item_id")["w"].sum()
recs_recent = {u: top_k_excluding(recent_pop, u) for u in truth.index}
```

```python
# 3. Content-based: item vectors from category (one-hot) + TF-IDF of the description
item_ids = items["item_id"].to_numpy(); pos = {i: k for k, i in enumerate(item_ids)}
V_cat = OneHotEncoder().fit_transform(items[["category"]])
V_txt = TfidfVectorizer().fit_transform(items["description"])
V = normalize(sparse.hstack([V_cat, 0.5 * V_txt]).tocsr())                # weight text vs category

def content_recs(user, k=K):
    hist = [pos[i] for i in train.loc[train["user_id"] == user, "item_id"]]
    profile = normalize(np.asarray(V[hist].mean(axis=0)))                  # user profile = mean item vector
    scores = pd.Series(np.asarray(V @ profile.T).ravel(), index=item_ids)
    return top_k_excluding(scores, user, k)

recs_content = {u: content_recs(u) for u in truth.index}

# 4. Hybrid: content similarity × popularity prior (a simple, robust blend)
def hybrid_recs(user, alpha=0.5, k=K):
    hist = [pos[i] for i in train.loc[train["user_id"] == user, "item_id"]]
    sim = np.asarray(V @ normalize(np.asarray(V[hist].mean(axis=0))).T).ravel()
    p = np.log1p(pop.reindex(item_ids).fillna(0).to_numpy()); p = p / p.max()
    return top_k_excluding(pd.Series(alpha * sim + (1 - alpha) * p, index=item_ids), user, k)
recs_hybrid = {u: hybrid_recs(u) for u in truth.index}

res = pd.DataFrame({"popularity": evaluate(recs_pop), "recent popularity": evaluate(recs_recent),
                    "content-based": evaluate(recs_content), "hybrid content+pop": evaluate(recs_hybrid)}).T
print(res.round(4))
```

```python
# Cold-start items: how often are NEW items (unseen in training) recommended and hit?
new_items = set(items["item_id"]) - set(train["item_id"])
for name, recs in [("popularity", recs_pop), ("content-based", recs_content)]:
    share = np.mean([i in new_items for u in truth.index for i in recs[u]])
    print(f"{name:14s}: share of recommendations that are new items = {share:.2%}")
```

## 7. Interpretation
- **Popularity:** precision@10 ≈ 0.023 and recall@10 ≈ 0.11, about 3–4× random (≈ 0.007 precision), with **7% coverage** (essentially the same items for everyone). This is the bar to beat.
- **Recent popularity is the best baseline here** (recall@10 ≈ 0.12). The catalogue evolves (new items are released over time), and recency weighting tracks what is currently being used.
- **Content-based is weak on accuracy** (precision ≈ 0.009). Its item vectors only encode the category and a very short, templated description, so "similar items" means "same category", which is too coarse to capture individual preferences. Its **coverage is very high (≈ 77%)**, and it is the only method that recommends new items.
- **The hybrid content + popularity blend** matches popularity's accuracy with **4× the coverage**. It personalises within popular regions of the catalogue, which is often the better product, even when the offline accuracy is equal.
- **Cold-start share:** popularity never recommends new items; content-based does (≈ 1.3% of its recommendations). With richer item descriptions (manuals, specifications), content-based accuracy improves considerably. Here the content signal is the bottleneck, not the method.

## 8. Decision framework
- **IF** there is little interaction data or many new items → content-based + popularity hybrid.
- **IF** the user segments are meaningful (site, role) → segment popularity as a cheap baseline.
- **IF** trends change → recency weighting (tune τ on a validation window).
- **IF** the item text is rich → TF-IDF or sentence embeddings for the item vectors ([embeddings_and_similarity.md](../12_natural_language_processing/embeddings_and_similarity.md)).
- **IF** content-based recommendations are too narrow → diversify (MMR re-ranking), or blend with collaborative signals ([collaborative_filtering.md](collaborative_filtering.md)).

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Not excluding seen items | Inflated metrics for re-consumption | Exclude per use case |
| Skipping the popularity baseline | Overestimating the value of personalisation | Always report it |
| Profiles averaged without weighting the interaction types | Noisy profiles | Weight orders > views; recency weights |
| Item features fitted with future items included in the evaluation | Mild leakage (vocabulary) | Fit on catalogue snapshots as of the split date where it matters |

## 10. Checklist
- [ ] Global, recent and segment popularity baselines.
- [ ] Content vectors (attributes + text), user profiles, cosine ranking.
- [ ] Hybrid with a popularity prior.
- [ ] Accuracy + coverage + cold-start item share.

## 11. References
- Lops, P., de Gemmis, M. & Semeraro, G. (2011). "Content-based Recommender Systems: State of the Art and Trends." In *Recommender Systems Handbook*, Springer.
- Aggarwal, C. C. (2016). *Recommender Systems: The Textbook*, Ch. 4 (content-based). Springer.
- Carbonell, J. & Goldstein, J. (1998). "The use of MMR, diversity-based reranking for reordering documents and producing summaries." *SIGIR*.
