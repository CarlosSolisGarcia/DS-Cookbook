# Collaborative Filtering: Item-kNN and Matrix Factorisation (Implicit and Explicit Feedback)

> **Folder:** 14_recommender_systems · **Prev:** [baselines_and_content_based.md](baselines_and_content_based.md) · **Next:** [graph_ml_workflow.md](../15_graph_machine_learning/graph_ml_workflow.md) · **Related:** [pca.md](../11_dimensionality_reduction/pca.md) (low-rank factorisation), [recsys_workflow.md](recsys_workflow.md) (splits, metrics)

---

## 1. Purpose
Personalise recommendations from interaction patterns alone: "engineers who consulted X also consulted Y". Use neighbourhood methods (item-kNN) and latent-factor models (SVD, implicit ALS), and compare them honestly with the popularity and content baselines.

## 2. When to use it
When there are enough interactions per user and per item (warm users and items), and the behaviour shows co-usage structure. Combine it with content/popularity models for the cold start.

## 3. Conceptual foundation
- **Item-kNN:** item–item similarity from co-occurrence, e.g. the cosine between the item columns of the user–item matrix. Score(u, i) = Σ over the items j the user interacted with of sim(i, j). It is simple, strong, explainable ("because you used j"), and robust in many benchmarks (Ferrari Dacrema et al., 2019).
- **Matrix factorisation (MF):** approximate the interaction matrix R ≈ U Vᵀ with low-rank user and item factors (k ≈ 10–200).
  - **Explicit ratings:** minimise the squared error on the *observed* ratings, plus biases and an L2 penalty (Funk-SVD-style, via SGD/ALS).
  - **Implicit feedback (ALS, Hu–Koren–Volinsky 2008):** preference p_ui = 1 if any interaction, else 0, with confidence c_ui = 1 + α·count. Minimise Σ c_ui (p_ui − u_uᵀv_i)² + λ(‖U‖² + ‖V‖²) over **all** cells. Alternating least squares has closed-form updates.
  - **Truncated SVD** of the (normalised) interaction matrix is a quick unweighted approximation.
- **Popularity bias:** CF tends to recommend popular items, and it improves recall for heavy users. Check the gains per user-activity segment.

## 4. Assumptions and limitations
- No recommendations for new items or users without interactions (cold start) → hybrids.
- Sparse users (≤ 3 interactions) get noisy factors. Fall back to popularity/content.
- Hyperparameters (k neighbours, factors, regularisation, α) need validation on a *temporal* validation window.

## 5. Step-by-step procedure
1. **Build the sparse user–item matrix** from the training period (binary or counts; weight the interaction types).
2. **Item-kNN:** cosine similarities (optionally keep the top-N neighbours per item); score the unseen items.
3. **MF:** truncated SVD (quick) and implicit ALS (weighted); tune the factors/λ/α on a validation window carved from the end of training.
4. **Evaluate** against popularity/content on the test window: precision/recall/nDCG@k, coverage, popularity bias; per user-activity segment.
5. **Hybridise:** blend the CF scores with popularity/content for cold or light users.
6. **For explicit ratings:** a biased MF with RMSE on held-out ratings, but evaluate the ranking quality as well.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from cookbook_data import make_interactions

inter, items = make_interactions()
T = inter["timestamp"].max() - pd.Timedelta(days=90)
train, test = inter[inter["timestamp"] < T], inter[inter["timestamp"] >= T]
test = test[test["user_id"].isin(train["user_id"])]
truth = test.groupby("user_id")["item_id"].apply(set)
users = np.sort(train["user_id"].unique()); item_ids = items["item_id"].to_numpy()
u_pos = {u: k for k, u in enumerate(users)}; i_pos = {i: k for k, i in enumerate(item_ids)}
counts = train.groupby(["user_id", "item_id"]).size()
R = sparse.csr_matrix((counts.to_numpy(dtype=float),
                       ([u_pos[u] for u, _ in counts.index], [i_pos[i] for _, i in counts.index])),
                      shape=(len(users), len(item_ids)))
B = (R > 0).astype(float)                                         # binary interactions
K = 10

def recommend_from_scores(S):
    """S: dense users × items score matrix → dict user -> top-K unseen item ids."""
    S = np.asarray(S, dtype=float).copy()
    S[B.toarray() > 0] = -np.inf                                  # exclude items already seen
    top = np.argsort(-S, axis=1)[:, :K]
    return {u: list(item_ids[top[u_pos[u]]]) for u in truth.index}

def evaluate(recs, k=K):
    P, R_, N = [], [], []
    for u, rel in truth.items():
        rec = recs[u][:k]; hits = np.array([r in rel for r in rec], float)
        P.append(hits.sum() / k); R_.append(hits.sum() / len(rel))
        N.append((hits / np.log2(np.arange(2, k + 2))).sum() / (1 / np.log2(np.arange(2, min(len(rel), k) + 2))).sum())
    cov = len({i for u in truth.index for i in recs[u]}) / len(item_ids)
    return {"precision@10": np.mean(P), "recall@10": np.mean(R_), "nDCG@10": np.mean(N), "coverage": cov}

pop = np.asarray(B.sum(0)).ravel()
results = {"popularity": evaluate(recommend_from_scores(np.tile(pop, (len(users), 1))))}

# Item-kNN: cosine similarity between item columns
Bn = normalize(B, axis=0)                                         # unit-norm item columns
S_items = (Bn.T @ Bn).toarray(); np.fill_diagonal(S_items, 0)
results["item-kNN (cosine)"] = evaluate(recommend_from_scores(B @ S_items))

# Truncated SVD of the row-normalised binary matrix
svd = TruncatedSVD(n_components=20, random_state=0).fit(normalize(B))
results["TruncatedSVD (k=20)"] = evaluate(recommend_from_scores(svd.transform(normalize(B)) @ svd.components_))
```

```python
# Implicit ALS (Hu, Koren & Volinsky 2008), dense implementation for small matrices
def implicit_als(R_counts, factors=20, reg=0.1, alpha=10.0, iters=10, seed=0):
    rng = np.random.default_rng(seed)
    C = 1 + alpha * R_counts.toarray()                            # confidence
    P = (R_counts.toarray() > 0).astype(float)                    # preference
    n_u, n_i = P.shape
    U = rng.normal(0, 0.1, (n_u, factors)); V = rng.normal(0, 0.1, (n_i, factors))
    I = reg * np.eye(factors)
    for _ in range(iters):
        for u in range(n_u):                                      # solve per user: (Vᵀ C_u V + λI) x = Vᵀ C_u p_u
            Cu = C[u]; U[u] = np.linalg.solve(V.T @ (V * Cu[:, None]) + I, V.T @ (Cu * P[u]))
        for i in range(n_i):
            Ci = C[:, i]; V[i] = np.linalg.solve(U.T @ (U * Ci[:, None]) + I, U.T @ (Ci * P[:, i]))
    return U, V

# Tune on a VALIDATION window carved from the end of the training period (never on the test window)
U, V = implicit_als(R, factors=20, reg=0.1, alpha=10)
results["implicit ALS (k=20)"] = evaluate(recommend_from_scores(U @ V.T))
res = pd.DataFrame(results).T
print(res.round(4))
```

```python
# Segment analysis: who benefits from personalisation? (light vs heavy users)
n_train = train.groupby("user_id").size()
seg = pd.cut(n_train.reindex(truth.index), [0, 5, 15, 1000], labels=["light (≤5)", "medium (6–15)", "heavy (>15)"])
recs_pop = recommend_from_scores(np.tile(pop, (len(users), 1))); recs_knn = recommend_from_scores(B @ S_items)
rows = []
for s in seg.cat.categories:
    us = seg.index[seg == s]
    r = lambda recs: np.mean([len(set(recs[u]) & truth[u]) / len(truth[u]) for u in us])
    rows.append({"segment": s, "users": len(us), "recall@10 popularity": r(recs_pop), "recall@10 item-kNN": r(recs_knn)})
print(pd.DataFrame(rows).round(3))
```

```python
# Explicit ratings: biased matrix factorisation (SGD) vs bias-only baseline, RMSE on held-out ratings
rated = inter.dropna(subset=["rating"])
r_tr, r_te = rated[rated["timestamp"] < T], rated[rated["timestamp"] >= T]
r_te = r_te[r_te["user_id"].isin(r_tr["user_id"]) & r_te["item_id"].isin(r_tr["item_id"])]
mu = r_tr["rating"].mean()
def fit_biased_mf(df, k=8, lr=0.01, reg=0.05, epochs=30, seed=0):
    rng = np.random.default_rng(seed)
    uix = {u: n for n, u in enumerate(df["user_id"].unique())}; iix = {i: n for n, i in enumerate(df["item_id"].unique())}
    bu, bi = np.zeros(len(uix)), np.zeros(len(iix)); Pu = rng.normal(0, 0.1, (len(uix), k)); Qi = rng.normal(0, 0.1, (len(iix), k))
    data = df[["user_id", "item_id", "rating"]].to_numpy()
    for _ in range(epochs):
        for u, i, r in data[rng.permutation(len(data))]:
            a, b = uix[u], iix[i]; e = r - (mu + bu[a] + bi[b] + Pu[a] @ Qi[b])
            bu[a] += lr * (e - reg * bu[a]); bi[b] += lr * (e - reg * bi[b])
            Pu[a], Qi[b] = Pu[a] + lr * (e * Qi[b] - reg * Pu[a]), Qi[b] + lr * (e * Pu[a] - reg * Qi[b])
    return lambda u, i, factors=True: mu + bu[uix[u]] + bi[iix[i]] + (Pu[uix[u]] @ Qi[iix[i]] if factors else 0)
pred = fit_biased_mf(r_tr)
rmse = lambda f: np.sqrt(np.mean([(r - f(u, i)) ** 2 for u, i, r in r_te[["user_id", "item_id", "rating"]].to_numpy()]))
print(f"RMSE global mean={np.sqrt(np.mean((r_te['rating'] - mu) ** 2)):.3f}  "
      f"biases only={rmse(lambda u, i: pred(u, i, factors=False)):.3f}  biased MF={rmse(pred):.3f}")
```

## 7. Interpretation
- **Item-kNN is the best model** (recall@10 ≈ 0.125 vs 0.105 for popularity, nDCG +23%), with 8× the coverage (≈ 60% vs 7.5%). Simple neighbourhood methods are strong and explainable. Here they are the right default.
- **Untuned matrix factorisation is worse than popularity.** Truncated SVD is a weak, unweighted approximation. Implicit ALS with arbitrary hyperparameters (k = 20, α = 10, λ = 0.1) even loses to popularity, although it has the highest coverage. MF is sensitive to its hyperparameters and to the data size (here ~16 interactions per user). Tune the factors, α and λ on a validation window carved from the training period, and compare them with item-kNN. Do not assume "latent factors > neighbours". The benchmarking literature (Ferrari Dacrema et al., 2019) found well-tuned simple baselines beating many complex models.
- **Segments:** item-kNN beats popularity in all segments, here including light users (≤ 5 interactions). Heavy users show *lower* recall for both, because they have more relevant test items, so recall@10 is capped by k. Compare segments with precision or nDCG as well, and always check the light/new users before deploying CF.
- **Explicit ratings:** the biases alone reduce the RMSE from 1.20 to 1.15. The latent factors add almost nothing (1.147), because only ~30% of the events carry a rating, so each user has very few ratings. With more ratings per user, factors usually help. A better RMSE still does not guarantee better top-k lists.

## 8. Decision framework
| Situation | Approach |
|---|---|
| Implicit feedback, moderate data | Item-kNN baseline → implicit ALS/BPR; tune on a temporal validation window |
| Explicit ratings, rating prediction needed | Biased MF; but evaluate rankings if lists are shown |
| Very sparse users / many cold users | Popularity/content for them; CF for warm users (hybrid routing) |
| Need explanations | Item-kNN ("because you used j") |
| Large scale (10⁶+ users/items) | Sparse ALS libraries (e.g. `implicit`), approximate nearest neighbours for retrieval |
| Rich context (time, site, equipment) | Learning-to-rank (GBM ranker) with CF scores as features |

- **IF** CF improves recall but collapses the coverage onto popular items → add popularity-penalised re-ranking, or diversity constraints.
- **IF** the offline gains are small → check the per-segment gains; consider whether personalisation is worth the complexity.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Tuning on the test window | Optimistic results | A validation window from the training period |
| Dense matrices at scale | Memory blow-up | Sparse ops; specialised libraries |
| Treating zeros as negatives with full confidence | Poor implicit models | Confidence weighting (ALS) or negative sampling (BPR) |
| No segment analysis | Hidden failure for light/new users | Per-segment metrics |
| Comparing CF only with random | Overstated value | Popularity + content baselines |

## 10. Checklist
- [ ] Sparse user–item matrix from the training period; the interaction weighting defined.
- [ ] Item-kNN, SVD and implicit ALS compared with popularity/content.
- [ ] Hyperparameters tuned on a temporal validation window.
- [ ] Accuracy, coverage, popularity bias; per user-activity segment.
- [ ] Hybrid routing for cold/light users.
- [ ] Explicit ratings: biased MF RMSE + ranking evaluation.

## 11. References
- Koren, Y., Bell, R. & Volinsky, C. (2009). "Matrix Factorization Techniques for Recommender Systems." *IEEE Computer* 42(8), 30–37. https://doi.org/10.1109/MC.2009.263
- Hu, Y., Koren, Y. & Volinsky, C. (2008). "Collaborative Filtering for Implicit Feedback Datasets." *ICDM*. https://doi.org/10.1109/ICDM.2008.22
- Sarwar, B., Karypis, G., Konstan, J. & Riedl, J. (2001). "Item-based collaborative filtering recommendation algorithms." *WWW*. https://doi.org/10.1145/371920.372071
- Rendle, S., Freudenthaler, C., Gantner, Z. & Schmidt-Thieme, L. (2009). "BPR: Bayesian Personalized Ranking from Implicit Feedback." *UAI*. https://arxiv.org/abs/1205.2618
- Ferrari Dacrema, M., Cremonesi, P. & Jannach, D. (2019). "Are We Really Making Much Progress?" *RecSys*. https://arxiv.org/abs/1907.06902
- `implicit` library (fast ALS/BPR). https://benfred.github.io/implicit/
