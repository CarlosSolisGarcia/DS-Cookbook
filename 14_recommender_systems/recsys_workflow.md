# Recommender Systems Workflow: Formulation, Feedback Types, Splitting and Offline Evaluation

> **Folder:** 14_recommender_systems · **Start here.** · **Next:** [baselines_and_content_based.md](baselines_and_content_based.md) · **Related:** [collaborative_filtering.md](collaborative_filtering.md), [metrics_reference.md](../08_quick_references/metrics_reference.md) §5

---

## 1. Purpose
Frame recommendation problems: what to recommend, to whom, when, and to optimise what. Choose the right kind of feedback data, split interactions in a way that mimics deployment, and evaluate rankings offline with appropriate metrics, before building any model.

## 2. When to use it
- Suggesting items (spare parts, documents, products, materials, procedures, articles) to users (engineers, customers, operators).
- Any "rank N candidates for this context" problem (e.g. likely root causes for an alarm, similar past cases).

## 3. Conceptual foundation

### 3.1 Feedback types
| Feedback | Examples | Properties | Typical models |
|---|---|---|---|
| **Explicit** | Star ratings, thumbs up/down | Sparse; biased (people rate what they like or hate); has negatives | Rating prediction (MF with RMSE), but ranking is usually the real goal |
| **Implicit** | Views, clicks, downloads, orders, time spent | Abundant; **no explicit negatives** (unobserved ≠ disliked); noisy | Ranking with sampled negatives; implicit MF (ALS), item-kNN, popularity |

### 3.2 The interaction matrix
Users × items, very sparse (often > 99% empty). Long tails on both sides: a few heavy users and blockbuster items dominate. **Cold start** is the problem of new users or items that have no interactions.

### 3.3 Families of methods
| Family | Idea | Handles cold start? | Recipe |
|---|---|---|---|
| Popularity (global, recent, per segment) | Recommend what is popular | Users: yes (non-personal) | [baselines_and_content_based.md](baselines_and_content_based.md) |
| Content-based | Recommend items similar to the ones the user liked (by attributes/text) | **Items: yes** | [baselines_and_content_based.md](baselines_and_content_based.md) |
| Collaborative filtering (item-kNN, user-kNN) | Users who used X also used Y | No | [collaborative_filtering.md](collaborative_filtering.md) |
| Matrix factorisation (SVD, ALS, BPR) | Latent user/item factors | No (unless features are added) | [collaborative_filtering.md](collaborative_filtering.md) |
| Hybrid / learning-to-rank | Combine signals and features in a ranking model (e.g. GBM ranker) | Yes, with features | Extension |

### 3.4 Offline evaluation
- **Ranking metrics at k** (top-k list per user, averaged over users):
  - **Precision@k:** the share of recommended items that are relevant. **Recall@k:** the share of the user's relevant held-out items that appear in the top k. **Hit rate@k:** at least one relevant item in the top k.
  - **MAP@k:** the average precision over the ranks of the relevant items (rewards ranking them early).
  - **nDCG@k:** discounted cumulative gain (log-position discount), normalised by the ideal ordering.
  - **MRR:** the mean reciprocal rank of the first relevant item.
- **Beyond accuracy:** catalogue **coverage** (the share of items ever recommended), **novelty/popularity bias** (the mean popularity of the recommended items), diversity.
- **Exclude already-seen items** from the recommendations when the use case is discovery (or include them when re-ordering is the use case).

### 3.5 Splitting interactions
| Split | How | Mimics | Leakage risk |
|---|---|---|---|
| **Global temporal** | Train on interactions before time T, test after T | Real deployment | Lowest; recommended |
| **Leave-last-out per user** | Each user's last (or last n) interactions as test | Next-item prediction | Mild (global time is not respected across users) |
| Random interaction split | Random 80/20 of the interactions | Nothing real | Future interactions leak into training; optimistic |

## 4. Assumptions and limitations
- Offline metrics measure agreement with past behaviour, which was itself shaped by the old recommender (exposure bias). They approximate online value but do not equal it. A/B tests are the gold standard.
- Implicit feedback treats unobserved items as negatives during evaluation. Some of them are relevant but simply not seen, so offline precision underestimates the true value.
- Popularity bias: models that recommend popular items score well offline yet add little value. Report popularity/coverage alongside accuracy.

## 5. Step-by-step procedure
1. **Frame:** the users, the items, the context (when is a recommendation shown?), the objective (discovery, re-ordering, cross-sell), the constraints (business rules, availability) and k (the list length shown).
2. **Data:** the interaction log (user, item, timestamp, type, value); item metadata; user metadata. Choose what counts as "relevant" (e.g. ordered, not just viewed).
3. **Explore:** sparsity, interactions per user/item (long tails), the share of new users/items over time, repeat interactions.
4. **Split** temporally; decide how to handle users/items unseen in training (cold start), and report them separately.
5. **Baselines:** popularity (global and recent), then content-based ([baselines_and_content_based.md](baselines_and_content_based.md)).
6. **Collaborative models:** item-kNN, matrix factorisation ([collaborative_filtering.md](collaborative_filtering.md)).
7. **Evaluate:** precision/recall/MAP/nDCG@k, hit rate, coverage, popularity of the recommendations; segment by user activity (light vs heavy users) and cold-start status.

## 6. Python implementation: exploration, temporal split and the evaluation harness

```python
import numpy as np
import pandas as pd
from cookbook_data import make_interactions

inter, items = make_interactions()
print(f"{inter['user_id'].nunique()} users, {inter['item_id'].nunique()} items, {len(inter)} interactions")
density = len(inter.drop_duplicates(["user_id", "item_id"])) / (inter["user_id"].nunique() * inter["item_id"].nunique())
print(f"matrix density: {density:.2%}")
print("interactions per user:", inter.groupby("user_id").size().describe()[["50%", "max"]].to_dict())
pop = inter["item_id"].value_counts()
print(f"top 10% items account for {pop.iloc[:int(0.1 * len(pop))].sum() / pop.sum():.0%} of interactions")

# Global temporal split: last 3 months = test
T = inter["timestamp"].max() - pd.Timedelta(days=90)
train, test = inter[inter["timestamp"] < T], inter[inter["timestamp"] >= T]
test = test[test["user_id"].isin(train["user_id"])]                     # warm users (cold users evaluated separately)
new_items_share = (~test["item_id"].isin(train["item_id"])).mean()
print(f"test interactions with items unseen in training (item cold start): {new_items_share:.1%}")
truth = test.groupby("user_id")["item_id"].apply(set)                   # relevant items per test user
```

```python
# Evaluation harness: ranking metrics @k for a dict user -> ranked list of item ids
def evaluate_rankings(recs: dict, truth: pd.Series, k=10, item_pop=None, n_items=None):
    P, R, AP, NDCG, HIT = [], [], [], [], []
    for u, rel in truth.items():
        rec = recs.get(u, [])[:k]
        hits = np.array([r in rel for r in rec], dtype=float)
        P.append(hits.sum() / k); R.append(hits.sum() / len(rel)); HIT.append(float(hits.any()))
        prec_at_i = np.cumsum(hits) / (np.arange(len(hits)) + 1)
        AP.append((prec_at_i * hits).sum() / min(len(rel), k))
        dcg = (hits / np.log2(np.arange(2, len(hits) + 2))).sum()
        idcg = (1 / np.log2(np.arange(2, min(len(rel), k) + 2))).sum()
        NDCG.append(dcg / idcg)
    out = {f"precision@{k}": np.mean(P), f"recall@{k}": np.mean(R), f"MAP@{k}": np.mean(AP),
           f"nDCG@{k}": np.mean(NDCG), f"hit_rate@{k}": np.mean(HIT)}
    rec_items = [i for u in truth.index for i in recs.get(u, [])[:k]]
    if n_items:
        out["coverage"] = len(set(rec_items)) / n_items
    if item_pop is not None:
        out["mean_popularity_rank"] = float(np.mean([item_pop.get(i, len(item_pop)) for i in rec_items]))
    return out

pop_rank = {item: r for r, item in enumerate(train["item_id"].value_counts().index)}
seen = train.groupby("user_id")["item_id"].apply(set)
```

```python
# Sanity checks of the harness: a random recommender (lower bound) and an "oracle" (upper bound)
rng = np.random.default_rng(0)
all_items = items["item_id"].to_numpy()
random_recs = {u: list(rng.choice(all_items, 10, replace=False)) for u in truth.index}
oracle_recs = {u: list(truth[u]) + [i for i in rng.permutation(all_items) if i not in truth[u]][:10] for u in truth.index}
print("random :", {k: round(v, 4) for k, v in evaluate_rankings(random_recs, truth, item_pop=pop_rank, n_items=len(all_items)).items()})
print("oracle :", {k: round(v, 4) for k, v in evaluate_rankings(oracle_recs, truth).items()})
```

## 7. Interpretation
- **Density ≈ 3%** and heavy-tailed activity (median 11 interactions per user, max 201) are typical (real catalogues are often < 0.1% dense). Most users have few interactions, which limits personalisation for them.
- **Popularity concentration:** here the top 10% of the items account for ≈ 38% of the interactions. The stronger this concentration, the stronger a popularity baseline is on accuracy metrics. That is why it must always be reported.
- **Item cold start in test:** the share of test interactions with unseen items, which collaborative methods cannot recommend. Content-based methods must cover them.
- **Random vs oracle:** these bracket the metric range (oracle recall/MAP/nDCG ≈ 1; its precision@10 < 1 only because many users have fewer than 10 relevant items). Random precision@10 ≈ (number of relevant items) / (number of items). The oracle gives the maximum. Real models fall between them. If a model is near random, check the ID alignment and the split logic first.

## 8. Decision framework
- **IF** there is implicit feedback only → treat it as ranking (precision/recall/nDCG@k). Evaluate with a temporal split; never use RMSE on 0/1 matrices.
- **IF** explicit ratings exist but the product shows ranked lists → evaluate rankings anyway (RMSE optimisation ≠ good top-k).
- **IF** many new items → content-based or hybrid models (item features). **IF** many new users → popularity/segment-based lists or onboarding preferences.
- **IF** popular items dominate → report coverage and popularity bias; consider re-ranking for novelty when discovery is the goal.
- **IF** the offline gains are small but the business case depends on them → an online A/B test before rollout.
- **IF** the "users" are engineers and the "items" are documents/parts → domain constraints (equipment compatibility, safety relevance) often matter more than CF signals. Encode them as filters/features.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Random interaction split | Leakage of future behaviour; optimistic | Temporal split |
| Recommending already-consumed items when evaluating discovery | Inflated hit rates | Exclude the seen items (as the use case dictates) |
| Only accuracy metrics | Popularity-biased systems look best | Coverage, novelty, segment results |
| Ignoring cold-start users/items in the evaluation | Overestimated real-world performance | Report them separately |
| RMSE as the main metric for ranking use | Misaligned optimisation | Ranking metrics @k |
| Mixing interaction types (view = order) | Noisy relevance | Define relevance; weight the types |

## 10. Checklist
- [ ] Users, items, context, objective, k defined.
- [ ] Relevance defined (which interactions count).
- [ ] Sparsity, long tails and cold-start shares explored.
- [ ] Temporal split; cold-start segments separated.
- [ ] Evaluation harness validated with random/oracle bounds.
- [ ] Accuracy + coverage + popularity metrics; segment breakdown.

## 11. References
- Aggarwal, C. C. (2016). *Recommender Systems: The Textbook*. Springer.
- Ricci, F., Rokach, L. & Shapira, B. (eds.) (2022). *Recommender Systems Handbook*, 3rd ed. Springer.
- Hu, Y., Koren, Y. & Volinsky, C. (2008). "Collaborative Filtering for Implicit Feedback Datasets." *ICDM*. https://doi.org/10.1109/ICDM.2008.22
- Ferrari Dacrema, M., Cremonesi, P. & Jannach, D. (2019). "Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches." *RecSys*. https://arxiv.org/abs/1907.06902
- Järvelin, K. & Kekäläinen, J. (2002). "Cumulated gain-based evaluation of IR techniques." *ACM TOIS* 20(4) (nDCG).
