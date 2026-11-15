# Graph Machine Learning Workflow: Representation, Exploration, Tasks and Pitfalls

> **Folder:** 15_graph_machine_learning · **Start here for graph-structured data.** · **Next:** [graph_features_and_embeddings.md](graph_features_and_embeddings.md) · **Related:** [gnn_pytorch_geometric.md](gnn_pytorch_geometric.md), [data_leakage.md](../00_methodology/data_leakage.md)

---

## 1. Purpose
Recognise when data are naturally a graph, represent them correctly (nodes, edges, features, direction, weights), explore the structure with network statistics, choose the task (node, edge or graph level), and validate without the leakage that is specific to graphs. The aim is to reconnect the graph-neural-network knowledge you already have to a practical workflow.

## 2. When to use it
- Relations matter as much as attributes: molecules (atoms/bonds), materials (crystal graphs), process flowsheets and P&IDs (units/streams), pipe and power networks, sensor networks, supply chains, citation/knowledge graphs, user–item graphs.
- **Not** when the "graph" is an artificial kNN graph over tabular rows with no relational meaning. Tabular models are usually better then.

## 3. Conceptual foundation

### 3.1 Representation
- **G = (V, E)**: nodes with feature vectors x_v; edges (directed or undirected, weighted or typed) with optional features e_uv; graph-level attributes.
- **Data structures:** adjacency matrix A (dense, n²), edge list / COO `edge_index` (2 × |E|, as in PyTorch Geometric), adjacency lists (networkx).
- **Heterogeneous graphs:** several node/edge types (e.g. equipment–sensor–alarm). PyG offers `HeteroData`.

### 3.2 Tasks
| Level | Task | Example | Typical setting |
|---|---|---|---|
| **Node** | Node classification/regression | Equipment criticality in a plant network; atom-level properties; user type | Usually *transductive*: one graph, some labelled nodes |
| **Edge** | Link prediction / edge classification | Missing interactions, future connections, bond type | Hide edges, predict them |
| **Graph** | Graph classification/regression | Molecular property (solubility, toxicity), material property | *Inductive*: many graphs, new graphs at test time |

### 3.3 The relationship between structure, features and predictions
- **Homophily:** connected nodes tend to share labels (communities). Message passing and label propagation exploit it.
- **Heterophily:** connected nodes tend to differ (e.g. bipartite-like interactions). Naive GCNs can then perform worse than an MLP on the node features.
- **Structural roles:** hubs, bridges, periphery. Captured by centrality measures, not by homophily.
- **The key diagnostic question:** *how much of the target is explained by the node features alone, by the structure alone, and by both?* Always compare: an MLP on features, structure-only features/embeddings, and a GNN using both.

### 3.4 Network statistics for exploration
Degree distribution (hubs? power law?), connected components, density, clustering coefficient (triangles), shortest-path lengths/diameter, centralities (degree, betweenness, closeness, PageRank, eigenvector), assortativity (do high-degree nodes connect to each other?), communities (modularity-based detection), and label homophily (the fraction of edges joining same-label nodes).

## 4. Assumptions and limitations
- The edges must be meaningful and measured consistently. Missing edges are a form of missing data.
- Graph-level datasets in chemistry/materials are often small (hundreds to thousands of graphs). Classical descriptors + GBM are strong baselines (e.g. RDKit fingerprints/descriptors).
- Large graphs (10⁶+ nodes) require neighbour sampling (GraphSAGE-style mini-batching).

## 5. Step-by-step procedure
1. **Define the nodes, the edges (direction, weights, types) and the features;** check them against the domain meaning.
2. **Explore:** components, the degree distribution, clustering, centralities, communities; the label distribution and **homophily**.
3. **Choose the task level** and the *inductive vs transductive* setting from the deployment (will new nodes/graphs appear?).
4. **Design the split** (§8): random node masks (transductive), edge splits with negative sampling (link prediction), or graph-level splits by scaffold/cluster (molecules).
5. **Baselines:** feature-only (MLP/GBM), structure-only (centralities, embeddings), label propagation; classical graph descriptors for graph-level tasks ([graph_features_and_embeddings.md](graph_features_and_embeddings.md)).
6. **GNNs** (GCN, GraphSAGE, GIN) in PyTorch Geometric ([gnn_pytorch_geometric.md](gnn_pytorch_geometric.md)).
7. **Evaluate** with the appropriate metrics and compare with the baselines.

## 6. Python implementation: exploration with networkx

```python
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities, modularity

# A plant-like network: 4 process areas (communities) with sparse links between areas
sizes = [60, 50, 40, 50]
P = np.full((4, 4), 0.004); np.fill_diagonal(P, 0.08)
G = nx.stochastic_block_model(sizes, P.tolist(), seed=0)
area = np.concatenate([[k] * s for k, s in enumerate(sizes)])
nx.set_node_attributes(G, {n: int(area[n]) for n in G.nodes}, "area")

print(f"nodes={G.number_of_nodes()} edges={G.number_of_edges()} density={nx.density(G):.4f}")
comps = list(nx.connected_components(G)); print("connected components:", len(comps), "largest:", max(map(len, comps)))
deg = pd.Series(dict(G.degree())); print("degree:", deg.describe()[["mean", "50%", "max"]].round(2).to_dict())
print("average clustering:", round(nx.average_clustering(G), 3))
Gc = G.subgraph(max(comps, key=len))
print("diameter (largest component):", nx.diameter(Gc), "| avg shortest path:", round(nx.average_shortest_path_length(Gc), 2))
```

```python
# Centralities: who are the hubs and bridges?
cent = pd.DataFrame({"degree": nx.degree_centrality(G), "betweenness": nx.betweenness_centrality(G, seed=0),
                     "pagerank": nx.pagerank(G), "area": pd.Series(area)})
print(cent.sort_values("betweenness", ascending=False).head(5).round(4))      # bridges between areas

# Communities and homophily of the 'area' label
comms = greedy_modularity_communities(G)
print(f"detected communities: {len(comms)}, modularity={modularity(G, comms):.3f}")
same = np.mean([area[u] == area[v] for u, v in G.edges()])
print(f"edge homophily w.r.t. area: {same:.2f}  (random expectation ≈ {np.sum((np.array(sizes)/sum(sizes))**2):.2f})")
print("degree assortativity:", round(nx.degree_assortativity_coefficient(G), 3))
```

## 7. Interpretation
- **Density and components:** sparse (density ≈ 2%). One large component (198 of 200 nodes) plus two isolated nodes, which have no neighbours to learn from (GNNs treat them like an MLP would). Message passing can reach most nodes in a few hops (the diameter is small).
- **Degree distribution:** fairly homogeneous here (SBM). Real networks often have hubs, whose influence dominates the aggregation. Consider degree normalisation (GCN) or attention.
- **Betweenness top nodes:** the nodes on the inter-area links. Bridges between process areas are critical for propagation (failures, disturbances) and are often what an operator calls "critical equipment".
- **Communities:** the modularity-based detection finds 6 communities (modularity ≈ 0.61) for 4 true areas. It splits some areas into sub-communities, because the graph is sparse (mean degree ≈ 4.7) and the detection optimises modularity, not "the true number of areas". As with clustering ([cluster_validation.md](../09_unsupervised_learning/cluster_validation.md)), the community count is a modelling outcome to interpret, not a fact.
- **Homophily ≫ random:** the area label is strongly homophilous, so neighbours' labels and features are informative. GNNs/label propagation will help. With low homophily, start from feature-only models and heterophily-aware architectures.

## 8. Decision framework
**IF the data are graph-structured:**
- Will predictions be needed for **new nodes/graphs** not seen in training? → an **inductive** setup (GraphSAGE/GIN; graph-level splits). **ELSE** transductive node classification on a fixed graph (GCN, label propagation).
- **IF** the homophily is high → label propagation / GCN baselines are strong. **IF** it is low → feature MLP/GBM baselines first; heterophily-aware GNNs (e.g. separate self and neighbour embeddings, as GraphSAGE does).
- **IF** graph-level property prediction with small datasets (molecules/materials) → descriptors/fingerprints + GBM baseline, then GNNs (GIN, MPNN); scaffold/cluster splits.
- **IF** link prediction → edge splits with negative sampling; remove the test edges from the message-passing graph (§9); heuristics (common neighbours, Adamic–Adar) as baselines.
- **IF** the graph is huge → neighbour sampling, mini-batches, or precomputed structural features + GBM.

**Validation designs:** node-level transductive: random train/val/test node masks (stratified). Inductive: split by subgraph/community/time. Graph-level: random → scaffold split (chemistry) → cluster split (by structural similarity) for an honest estimate of generalisation to new chemotypes.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Test edges still present in the message-passing graph (link prediction) | Severe leakage | `RandomLinkSplit`: separate message-passing vs supervision edges |
| Random splits for molecules | Optimistic: near-duplicate scaffolds in train and test | Scaffold/cluster splits |
| Skipping the feature-only baseline | Can't tell if the structure helps | MLP/GBM on node features |
| Ignoring heterophily | A GNN worse than an MLP | Measure homophily first |
| Treating directed relations as undirected without thought | Wrong propagation semantics | Model the direction (e.g. flow direction in plants) |
| Normalising/scaling features with all nodes in inductive settings | Leakage | Fit on training nodes/graphs |

## 10. Checklist
- [ ] Nodes/edges/features defined with domain meaning; direction/weights decided.
- [ ] Exploration: components, degrees, clustering, centralities, communities.
- [ ] Label homophily measured.
- [ ] Task level and inductive/transductive setting chosen.
- [ ] Split designed for the setting (node masks, edge split, scaffold/cluster split).
- [ ] Feature-only and structure-only baselines before GNNs.

## 11. References
- Hamilton, W. L. (2020). *Graph Representation Learning*. Morgan & Claypool. Free pre-print: https://www.cs.mcgill.ca/~wlh/grl_book/
- Newman, M. (2018). *Networks*, 2nd ed. Oxford University Press.
- NetworkX documentation (algorithms: centrality, community, shortest paths). https://networkx.org/documentation/stable/
- Zhu, J. et al. (2020). "Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs." *NeurIPS*. https://arxiv.org/abs/2006.11468
- Wu, Z. et al. (2018). "MoleculeNet: a benchmark for molecular machine learning." *Chemical Science* 9, 513–530 (scaffold splits). https://doi.org/10.1039/C7SC02664A
- PyTorch Geometric documentation. https://pytorch-geometric.readthedocs.io/
