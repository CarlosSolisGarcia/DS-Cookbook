# Graph Neural Networks with PyTorch Geometric: Message Passing, GCN, GraphSAGE and GIN

> **Folder:** 15_graph_machine_learning · **Prev:** [graph_features_and_embeddings.md](graph_features_and_embeddings.md) · **Next:** [sensor_data_workflow.md](../16_industrial_sensor_data/sensor_data_workflow.md) · **Related:** [18_scientific_machine_learning](../18_scientific_machine_learning/scientific_ml_workflow.md) (molecules/materials), [efficient_development.md](../20_advanced_methodologies/efficient_development.md)

---

## 1. Purpose
Implement and evaluate GNNs for node classification, link prediction and graph classification in PyTorch Geometric (PyG). Understand message passing well enough to choose an architecture, and measure whether the GNN beats the classical baselines.

## 2. When to use it
After the graph baselines ([graph_features_and_embeddings.md](graph_features_and_embeddings.md)), when node/edge features and structure *jointly* determine the target, or when learned representations should transfer to new graphs (molecules, materials, flowsheets).

## 3. Conceptual foundation
**Message passing (one layer):** each node aggregates "messages" from its neighbours and updates its own representation:

  h_v^(l+1) = UPDATE( h_v^(l), AGGREGATE_{u ∈ N(v)} MESSAGE(h_u^(l), h_v^(l), e_uv) )

Stacking L layers gives each node information from its L-hop neighbourhood.

| Architecture | Aggregation | Notes |
|---|---|---|
| **GCN** (Kipf & Welling) | Degree-normalised mean, including a self-loop: Σ h_u / √(d_u d_v), then a linear layer | Simple, transductive by default; assumes homophily |
| **GraphSAGE** | Mean/max/LSTM of the neighbours, **concatenated with the node's own** representation | Inductive; neighbour sampling for large graphs; better under heterophily than GCN |
| **GAT** | Attention-weighted neighbours | Learns which neighbours matter |
| **GIN** | Sum aggregation + MLP | As expressive as the Weisfeiler–Lehman test; a strong default for **graph classification** (molecules) |
| **MPNN / edge-conditioned** | Messages depend on edge features (bond types) | Chemistry, where edge features matter |

**Graph-level readout:** pool the node embeddings (mean/sum/max, `global_mean_pool`) → an MLP head.

**Depth:** 2–3 layers are typical. Deeper GNNs suffer from **over-smoothing** (all node embeddings become similar) and over-squashing. Use residual connections and normalisation if more depth is needed.

## 4. Assumptions and limitations
- GNNs need informative structure *and* enough labels. With strong homophily and few features, label propagation can match them. With heterophily, a naive GCN can lose to an MLP.
- Small graph datasets (≲ 1000 graphs) give high variance: use repeated splits and seeds, and compare with descriptor + GBM baselines.
- Full-batch training on large graphs exhausts memory. Use `NeighborLoader` (GraphSAGE-style sampling).

## 5. Step-by-step procedure
1. **Build the PyG `Data`/`Dataset` objects:** `x`, `edge_index` (undirected edges in both directions), `y`, the masks or graph lists.
2. **Baselines:** an MLP on the node features (same pipeline, no edges); classical baselines from the previous recipe.
3. **Node classification:** GCN and GraphSAGE (2 layers, hidden 32–64, dropout, weight decay); early stopping on the validation mask.
4. **Link prediction:** `RandomLinkSplit` (separate message-passing vs supervision edges) → a GNN encoder + dot-product decoder; BCE with negative sampling; ROC-AUC/AP.
5. **Graph classification:** GIN + a global pooling readout; a graph-level split (scaffold/cluster-like for molecules); compare with descriptor baselines.
6. **Repeat with several seeds/splits;** report the mean ± std.

## 6. Python implementation

```python
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import from_networkx
from sklearn.metrics import roc_auc_score, average_precision_score

torch.manual_seed(0); rng = np.random.default_rng(0)
# Same plant-like graph as in graph_features_and_embeddings.md (4 homophilous areas, weak node features)
sizes = [60, 50, 40, 50]; P = np.full((4, 4), 0.004); np.fill_diagonal(P, 0.08)
G = nx.stochastic_block_model(sizes, P.tolist(), seed=0)
y = np.concatenate([[k] * s for k, s in enumerate(sizes)]); n = len(y)
X = rng.normal(0, 1, (n, 8)) + 0.4 * np.eye(4)[y] @ rng.normal(0, 1, (4, 8))
edge_index = from_networkx(G).edge_index                                  # both directions included
data = Data(x=torch.tensor(X, dtype=torch.float32), edge_index=edge_index, y=torch.tensor(y))
perm = torch.randperm(n); data.train_mask = torch.zeros(n, dtype=torch.bool); data.val_mask = data.train_mask.clone(); data.test_mask = data.train_mask.clone()
data.train_mask[perm[:80]] = True; data.val_mask[perm[80:100]] = True; data.test_mask[perm[100:]] = True
```

```python
# Node classification: MLP vs GCN vs GraphSAGE (same training loop)
class MLP(nn.Module):
    def __init__(self, d_in, d_h, d_out): super().__init__(); self.l1, self.l2 = nn.Linear(d_in, d_h), nn.Linear(d_h, d_out)
    def forward(self, x, edge_index): return self.l2(F.dropout(F.relu(self.l1(x)), 0.5, self.training))
class GCN(nn.Module):
    def __init__(self, d_in, d_h, d_out): super().__init__(); self.c1, self.c2 = GCNConv(d_in, d_h), GCNConv(d_h, d_out)
    def forward(self, x, edge_index): return self.c2(F.dropout(F.relu(self.c1(x, edge_index)), 0.5, self.training), edge_index)
class SAGE(nn.Module):
    def __init__(self, d_in, d_h, d_out): super().__init__(); self.c1, self.c2 = SAGEConv(d_in, d_h), SAGEConv(d_h, d_out)
    def forward(self, x, edge_index): return self.c2(F.dropout(F.relu(self.c1(x, edge_index)), 0.5, self.training), edge_index)

def train_node_model(cls, data, epochs=200, seed=0):
    torch.manual_seed(seed); model = cls(data.num_features, 32, 4)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4); best = (0, None)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(1)
        val = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
        if val >= best[0]:
            best = (val, (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item())
    return best[1]                                                          # test accuracy at the best validation epoch

for name, cls in [("MLP (features only)", MLP), ("GCN", GCN), ("GraphSAGE", SAGE)]:
    accs = [train_node_model(cls, data, seed=s) for s in range(3)]
    print(f"{name:20s} test accuracy = {np.mean(accs):.3f} ± {np.std(accs):.3f}")
```

```python
# Link prediction: RandomLinkSplit keeps supervision edges OUT of the message-passing graph
split = RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True, add_negative_train_samples=False)
train_d, val_d, test_d = split(Data(x=data.x, edge_index=data.edge_index, num_nodes=n))

class Encoder(nn.Module):
    def __init__(self): super().__init__(); self.c1, self.c2 = GCNConv(8, 32), GCNConv(32, 32)
    def forward(self, x, ei): return self.c2(F.relu(self.c1(x, ei)), ei)
enc = Encoder(); opt = torch.optim.Adam(enc.parameters(), lr=0.01)
def decode(z, pairs): return (z[pairs[0]] * z[pairs[1]]).sum(-1)       # dot-product decoder
for epoch in range(200):
    enc.train(); opt.zero_grad()
    z = enc(train_d.x, train_d.edge_index)
    pos = train_d.edge_label_index
    neg = torch.randint(0, n, pos.shape)                                  # random negatives each epoch
    logits = torch.cat([decode(z, pos), decode(z, neg)])
    labels = torch.cat([torch.ones(pos.shape[1]), torch.zeros(neg.shape[1])])
    F.binary_cross_entropy_with_logits(logits, labels).backward(); opt.step()
enc.eval()
with torch.no_grad():
    z = enc(test_d.x, test_d.edge_index)                                  # message passing on train(+val) edges only
    s = decode(z, test_d.edge_label_index).numpy()
print(f"GCN link prediction: ROC-AUC={roc_auc_score(test_d.edge_label.numpy(), s):.3f}  AP={average_precision_score(test_d.edge_label.numpy(), s):.3f}")
```

```python
# Graph classification: synthetic "molecules" — does the graph contain a 6-ring with a heteroatom?
def make_molecule(rng):
    n_atoms = rng.integers(6, 16)
    g = nx.random_labeled_tree(n_atoms, seed=int(rng.integers(1e9))) if hasattr(nx, "random_labeled_tree") else nx.random_tree(n_atoms, seed=int(rng.integers(1e9)))
    has_ring = rng.random() < 0.5
    if has_ring:                                                          # attach a 6-membered ring
        ring = list(range(n_atoms, n_atoms + 6)); nx.add_cycle(g, ring); g.add_edge(0, ring[0])
    atom = {v: rng.choice(["C", "N", "O"], p=[0.8, 0.1, 0.1]) for v in g.nodes}
    hetero_in_ring = has_ring and any(atom[v] != "C" for v in range(n_atoms, n_atoms + 6))
    x = torch.tensor([[a == "C", a == "N", a == "O"] for a in (atom[v] for v in sorted(g.nodes))], dtype=torch.float32)
    d = from_networkx(nx.convert_node_labels_to_integers(g, ordering="sorted"))
    return Data(x=x, edge_index=d.edge_index, y=torch.tensor([int(hetero_in_ring)]))

graphs = [make_molecule(rng) for _ in range(800)]
print("positive share:", np.mean([g.y.item() for g in graphs]).round(3))
tr, te = graphs[:600], graphs[600:]

class GIN(nn.Module):
    def __init__(self, d_in=3, d_h=32):
        super().__init__()
        mlp = lambda a, b: nn.Sequential(nn.Linear(a, b), nn.ReLU(), nn.Linear(b, b))
        self.c1, self.c2, self.c3 = GINConv(mlp(d_in, d_h)), GINConv(mlp(d_h, d_h)), GINConv(mlp(d_h, d_h))
        self.head = nn.Linear(3 * d_h, 1)
    def forward(self, x, ei, batch):
        h1 = F.relu(self.c1(x, ei)); h2 = F.relu(self.c2(h1, ei)); h3 = F.relu(self.c3(h2, ei))
        return self.head(torch.cat([global_add_pool(h, batch) for h in (h1, h2, h3)], 1)).squeeze(-1)  # jumping-knowledge readout

torch.manual_seed(0); gin = GIN(); opt = torch.optim.Adam(gin.parameters(), lr=0.005)
for epoch in range(40):
    gin.train()
    for b in DataLoader(tr, batch_size=32, shuffle=True):
        opt.zero_grad(); F.binary_cross_entropy_with_logits(gin(b.x, b.edge_index, b.batch), b.y.float()).backward(); opt.step()
gin.eval()
with torch.no_grad():
    s = torch.cat([gin(b.x, b.edge_index, b.batch) for b in DataLoader(te, batch_size=64)]).numpy()
yte = np.array([g.y.item() for g in te])
print(f"GIN graph classification ROC-AUC = {roc_auc_score(yte, s):.3f}")

# Descriptor baseline: counts of atom types + cycle count (graph-level features) + logistic regression
from sklearn.linear_model import LogisticRegression
def descriptors(g):
    nxg = nx.Graph(); nxg.add_edges_from(g.edge_index.t().tolist())
    return [g.x[:, 0].sum().item(), g.x[:, 1].sum().item(), g.x[:, 2].sum().item(), len(nx.cycle_basis(nxg)), g.num_nodes]
Dtr, Dte = np.array([descriptors(g) for g in tr]), np.array([descriptors(g) for g in te])
ytr = np.array([g.y.item() for g in tr])
print(f"descriptor baseline ROC-AUC = {roc_auc_score(yte, LogisticRegression(max_iter=2000).fit(Dtr, ytr).predict_proba(Dte)[:, 1]):.3f}")
```

## 7. Interpretation
- **Node classification:** MLP 0.52 (weak features only), GCN 0.94, GraphSAGE 0.93. Message passing adds ~40 points of accuracy by pooling the information of homophilous neighbours. Label propagation reached 0.95 in the previous recipe, with *more* labelled nodes (100 vs 80 here) and no features. The GNN essentially **matches** it. On a strongly homophilous graph with weak features, the simpler method is competitive. Whether to deploy the GNN then depends on inductive needs (new nodes) and on the feature richness.
- **Link prediction:** the GCN encoder reaches ROC-AUC ≈ 0.85 vs ≈ 0.57 for the common-neighbour heuristics, which failed on this low-clustering graph. Nodes in the same area get similar embeddings even without shared neighbours. Here the learned representations add real value.
- **Graph classification:** GIN reaches ROC-AUC = 1.0, and the descriptor baseline (atom counts + number of cycles) 0.96. The baseline is strong because the "ring present" and "heteroatom count" descriptors already correlate with the target. Only message passing can locate the heteroatom *inside* the ring, which closes the remaining gap. Synthetic structure-defined targets are easy for GINs. With real molecular properties, compare GIN with fingerprints (e.g. RDKit Morgan) + GBM, which are often hard to beat on small datasets, and use scaffold splits.
- **Variability:** the ± std over seeds (here small) shows how noisy the results are. On small real graphs it is often large. Never compare architectures on a single seed.

## 8. Decision framework
| Situation | Recommended |
|---|---|
| Homophilous graph, few/weak features | Label propagation baseline; GCN; choose the simpler if tied |
| Rich node features + structure | GraphSAGE/GAT; compare with features + neighbour aggregates + GBM |
| Heterophily | GraphSAGE (separate self/neighbour weights) or heterophily-aware models; MLP baseline essential |
| New nodes/graphs at prediction time (inductive) | GraphSAGE / GIN; no transductive embeddings |
| Molecules / materials, property prediction | GIN or edge-aware MPNN; scaffold/cluster splits; fingerprint + GBM baseline |
| Large graphs | `NeighborLoader` mini-batching; or structural features + GBM |
| Link prediction | GNN encoder + dot/MLP decoder; `RandomLinkSplit`; heuristics as baselines |

- **IF** a deeper GNN performs worse → over-smoothing. Stay at 2–3 layers, add residuals/jumping knowledge.
- **IF** the results vary a lot across seeds → report the distributions; increase the data or regularise.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Supervision edges left in the message-passing graph (link prediction) | Leakage | `RandomLinkSplit` (`edge_index` vs `edge_label_index`) |
| Missing reverse edges for undirected graphs | Asymmetric propagation | `to_undirected` / `from_networkx` |
| Selecting the epoch on the test mask | Optimistic accuracy | Validation-mask early stopping |
| Too many layers | Over-smoothing | 2–3 layers + residuals |
| Random graph-level splits for molecules | Over-optimism | Scaffold/cluster splits |
| No MLP/descriptor baseline | Unknown value of the structure | Always include it |

## 10. Checklist
- [ ] PyG data objects built correctly (undirected edges, masks/graph lists).
- [ ] MLP and classical baselines on the same splits.
- [ ] GCN/GraphSAGE (nodes), GNN encoder + decoder (links), GIN + readout (graphs).
- [ ] Early stopping on validation; several seeds.
- [ ] Results compared with the label-propagation/heuristic/descriptor baselines.

## 11. References
- Kipf, T. N. & Welling, M. (2017). "Semi-Supervised Classification with Graph Convolutional Networks." *ICLR*. https://arxiv.org/abs/1609.02907
- Hamilton, W., Ying, R. & Leskovec, J. (2017). "Inductive Representation Learning on Large Graphs" (GraphSAGE). *NeurIPS*. https://arxiv.org/abs/1706.02216
- Xu, K., Hu, W., Leskovec, J. & Jegelka, S. (2019). "How Powerful are Graph Neural Networks?" (GIN). *ICLR*. https://arxiv.org/abs/1810.00826
- Gilmer, J. et al. (2017). "Neural Message Passing for Quantum Chemistry." *ICML*. https://arxiv.org/abs/1704.01212
- Fey, M. & Lenssen, J. E. (2019). "Fast Graph Representation Learning with PyTorch Geometric." https://arxiv.org/abs/1903.02428 · Docs and tutorials: https://pytorch-geometric.readthedocs.io/
- Hamilton, W. L. (2020). *Graph Representation Learning*. https://www.cs.mcgill.ca/~wlh/grl_book/
