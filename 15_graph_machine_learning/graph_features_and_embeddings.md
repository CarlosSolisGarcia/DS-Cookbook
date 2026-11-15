# Graph Features, Embeddings and Classical Baselines (Node Classification and Link Prediction)

> **Folder:** 15_graph_machine_learning · **Prev:** [graph_ml_workflow.md](graph_ml_workflow.md) · **Next:** [gnn_pytorch_geometric.md](gnn_pytorch_geometric.md) · **Related:** [dimensionality_reduction_workflow.md](../11_dimensionality_reduction/dimensionality_reduction_workflow.md) (spectral embeddings)

---

## 1. Purpose
Build strong non-GNN baselines for graph tasks. Structural features + tabular models, label propagation, spectral node embeddings, and link-prediction heuristics. They tell you how much a GNN really adds, and they are often good enough on their own.

## 2. When to use it
Before (and as a benchmark for) any GNN. They are also the main approach when compute is limited, or when the graph changes too often to retrain GNNs.

## 3. Conceptual foundation
- **Structural node features:** degree, clustering coefficient, centralities (PageRank, betweenness), k-core number, the counts of neighbours with given attributes, and **neighbour feature aggregates** (the mean/max of the neighbours' features, a hand-made 1-hop "message passing").
- **Label propagation:** iteratively spread the known labels to neighbours (it assumes homophily). A very strong baseline on homophilous graphs.
- **Spectral embeddings:** eigenvectors of the (normalised) graph Laplacian give coordinates in which connected nodes are close (`sklearn.manifold.SpectralEmbedding`). Random-walk embeddings (DeepWalk/node2vec) are similar in spirit, via random walks + word2vec.
- **Link-prediction heuristics:** common neighbours |N(u) ∩ N(v)|, Jaccard, Adamic–Adar Σ 1/log(deg(w)) over the common neighbours w, preferential attachment deg(u)·deg(v). Evaluate them with ROC-AUC/AP against sampled non-edges.

## 4. Assumptions and limitations
- Structural features fixed on the full graph are transductive. For new nodes they must be recomputed (as-of the time of prediction).
- Spectral embeddings do not transfer between graphs, and they change when the graph changes.
- For link prediction, compute the features on the graph **with the test edges removed**.

## 5. Step-by-step procedure
1. **Node classification:** compute the structural features + neighbour aggregates of the node features (training labels are never used as features of the same node).
2. **Train a GBM/logistic model** with stratified node splits; compare it with a feature-only model.
3. **Label propagation** with the training labels only.
4. **Spectral embedding** features + logistic regression.
5. **Link prediction:** remove the test edges; sample negatives; score with the heuristics; ROC-AUC/AP.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
sizes = [60, 50, 40, 50]; P = np.full((4, 4), 0.004); np.fill_diagonal(P, 0.08)
G = nx.stochastic_block_model(sizes, P.tolist(), seed=0)
y = np.concatenate([[k] * s for k, s in enumerate(sizes)]); n = len(y)
# Node features: weakly informative noisy sensors (the class shifts the means slightly)
X_feat = rng.normal(0, 1, (n, 8)) + 0.4 * np.eye(4)[y] @ rng.normal(0, 1, (4, 8))
idx_tr, idx_te = train_test_split(np.arange(n), test_size=0.5, stratify=y, random_state=0)

A = nx.to_scipy_sparse_array(G, nodelist=range(n), format="csr")
deg = np.asarray(A.sum(1)).ravel()
neigh_mean = (A @ X_feat) / np.maximum(deg, 1)[:, None]               # 1-hop aggregation of FEATURES (no labels)
struct = np.column_stack([deg, list(nx.clustering(G).values()), list(nx.pagerank(G).values())])

def fit_eval(X, name):
    m = LogisticRegression(max_iter=3000).fit(X[idx_tr], y[idx_tr])
    print(f"{name:38s} accuracy = {accuracy_score(y[idx_te], m.predict(X[idx_te])):.3f}")

fit_eval(X_feat, "node features only")
fit_eval(struct, "structural features only")
fit_eval(np.hstack([X_feat, neigh_mean]), "features + neighbour-mean features")
emb = SpectralEmbedding(n_components=8, affinity="precomputed", random_state=0).fit_transform(A.toarray())
fit_eval(emb, "spectral embedding (8 dims)")
```

```python
# Label propagation: uses the graph + training labels (test labels hidden as -1)
labels_partial = np.full(n, -1); labels_partial[idx_tr] = y[idx_tr]
# scikit-learn's LabelSpreading builds its graph from FEATURE similarity (rbf/knn kernels).
# For a GIVEN graph, propagate over its adjacency directly:
def label_propagation(A, labels, n_classes, alpha=0.9, iters=50):
    d = np.asarray(A.sum(1)).ravel(); Dm = 1 / np.sqrt(np.maximum(d, 1))
    S = A.multiply(Dm[:, None]).multiply(Dm[None, :]).tocsr()        # symmetric normalisation D^-1/2 A D^-1/2
    Y0 = np.zeros((A.shape[0], n_classes)); Y0[labels >= 0, labels[labels >= 0]] = 1
    F = Y0.copy()
    for _ in range(iters):
        F = alpha * (S @ F) + (1 - alpha) * Y0
    return F.argmax(1)
pred_lp = label_propagation(A, labels_partial, 4)
print(f"{'label propagation (graph only)':38s} accuracy = {accuracy_score(y[idx_te], pred_lp[idx_te]):.3f}")
```

```python
# Link prediction with heuristics: hide 10% of edges, sample equally many non-edges
edges = np.array(G.edges()); rng.shuffle(edges)
n_test = len(edges) // 10
test_pos = edges[:n_test]; G_train = G.copy(); G_train.remove_edges_from(map(tuple, test_pos))   # features from G_train ONLY
neg = set()
while len(neg) < n_test:
    u, v = rng.integers(0, n, 2)
    if u != v and not G.has_edge(u, v):
        neg.add((min(u, v), max(u, v)))
test_neg = np.array(list(neg)); pairs = np.vstack([test_pos, test_neg]); lab = np.r_[np.ones(n_test), np.zeros(n_test)]

pair_list = [(int(u), int(v)) for u, v in pairs]          # plain Python ints: networkx silently skips NumPy-int tuples here
heuristics = {
    "common neighbours": lambda g, p: [len(list(nx.common_neighbors(g, u, v))) for u, v in p],
    "Jaccard": lambda g, p: [s for _, _, s in nx.jaccard_coefficient(g, p)],
    "Adamic-Adar": lambda g, p: [s for _, _, s in nx.adamic_adar_index(g, p)],
    "preferential attachment": lambda g, p: [s for _, _, s in nx.preferential_attachment(g, p)],
}
for name, f in heuristics.items():
    s = np.array(f(G_train, pair_list), dtype=float)
    print(f"{name:24s} ROC-AUC={roc_auc_score(lab, s):.3f}  AP={average_precision_score(lab, s):.3f}")
```

## 7. Interpretation
- **Features only (0.56) vs + neighbour-mean (0.78):** aggregating the neighbours' features gives a big jump. The graph carries the label information (homophily 0.88). That is exactly what a GNN layer learns to do, with learned weights.
- **Structural features only (0.34):** degree/clustering/PageRank barely identify the *community*, because all the SBM blocks look alike structurally. They are useful for *role* tasks (hubs, bridges), not here.
- **Spectral embedding (0.30) fails here:** the graph has **three components** (two isolated nodes). The first Laplacian eigenvectors then encode the components, not the communities. This is exactly the pitfall listed in §9. Use the largest component (or more dimensions) and re-check. A useful reminder that every method has preconditions.
- **Label propagation (0.95) is the best method by far**, and it uses no features at all. With strong homophily and half the nodes labelled, the graph alone almost determines the labels. **A GNN must beat 0.95 to justify itself** ([gnn_pytorch_geometric.md](gnn_pytorch_geometric.md)).
- **Link heuristics are weak here (AUC ≈ 0.57):** common-neighbour methods need triangles, and this sparse SBM has a low clustering coefficient (≈ 0.06), so most true edges have no common neighbours. Preferential attachment is even below random, because the degrees are homogeneous. On real networks with higher clustering (social, citation, molecular co-occurrence), these heuristics are strong baselines. Always check the clustering coefficient first.

## 8. Decision framework
- **IF** the homophily is high and labels are partially known → label propagation as a baseline (often hard to beat).
- **IF** the node features are informative and the structure adds context → features + neighbour aggregates + GBM (a strong, cheap "manual GNN").
- **IF** the task concerns roles (bridges, hubs, critical nodes) → centralities and k-core features.
- **IF** link prediction → Adamic–Adar/common-neighbour baselines; GNN encoders if node features matter.
- **IF** the graph changes often and fast inference is needed → structural features + tabular models, recomputed incrementally.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Using neighbours' *labels* (including test labels) as features | Leakage | Aggregate features, or only training labels with care |
| Link features computed on the full graph | Leakage of the test edges | Remove the test edges first |
| Evaluating link prediction without negatives | Meaningless metrics | Sample non-edges (or rank against all candidates) |
| Spectral embedding on disconnected graphs without care | Degenerate eigenvectors | Use the largest component, or per-component handling |
| Passing NumPy-integer node pairs to networkx link functions | Silently empty results (observed with networkx 3.7) | Convert to Python `int` tuples |

## 10. Checklist
- [ ] Feature-only, structure-only, neighbour-aggregate and label-propagation baselines.
- [ ] Spectral (or random-walk) embeddings compared.
- [ ] Link prediction: test edges removed, negatives sampled, heuristics scored.
- [ ] Results table to compare against GNNs.

## 11. References
- Zhu, X. & Ghahramani, Z. (2002). "Learning from labeled and unlabeled data with label propagation." CMU Tech. Report CMU-CALD-02-107.
- Zhou, D. et al. (2004). "Learning with Local and Global Consistency." *NeurIPS* (normalised propagation).
- Liben-Nowell, D. & Kleinberg, J. (2007). "The link-prediction problem for social networks." *JASIST* 58(7), 1019–1031.
- Adamic, L. A. & Adar, E. (2003). "Friends and neighbors on the Web." *Social Networks* 25(3), 211–230.
- Grover, A. & Leskovec, J. (2016). "node2vec: Scalable Feature Learning for Networks." *KDD*. https://arxiv.org/abs/1607.00653
- Huang, Q. et al. (2021). "Combining Label Propagation and Simple Models Out-performs Graph Neural Networks." *ICLR*. https://arxiv.org/abs/2010.13993
