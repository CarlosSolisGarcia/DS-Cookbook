# Visualising High-Dimensional Data: PCA, t-SNE and UMAP (and Their Limits)

> **Folder:** 11_dimensionality_reduction · **Prev:** [high_dimensional_data.md](high_dimensional_data.md) · **Next:** [nlp_workflow.md](../12_natural_language_processing/nlp_workflow.md) · **Related:** [clustering_algorithms.md](../09_unsupervised_learning/clustering_algorithms.md), [visualization_reference.md](../08_quick_references/visualization_reference.md)

---

## 1. Purpose
Produce 2-D maps of high-dimensional data (sensor snapshots, spectra, image or text embeddings) that reveal neighbourhoods, clusters and outliers, without being misled by the artefacts of non-linear embeddings.

## 2. When to use it
- Exploratory analysis of high-dimensional data and of learned embeddings.
- Communicating structure (regimes, classes, batches) visually.
- **Not** as a source of quantitative conclusions (cluster sizes, distances between clusters), and usually not as model features.

## 3. Conceptual foundation
| Method | Preserves | Good for | Pitfalls |
|---|---|---|---|
| **PCA** | Global linear structure; variances; distances approximately (in the retained subspace) | First look; trends; outliers; batch effects | Misses non-linear structure; dominated by high-variance directions |
| **t-SNE** | **Local neighbourhoods** (who is near whom) | Revealing clusters in complex data | Cluster sizes and between-cluster distances are not meaningful; depends on perplexity and seed; slow for large n; no `transform` for new points in sklearn |
| **UMAP** (`umap-learn`) | Local neighbourhoods, with somewhat more global structure than t-SNE | Faster on large n; supports `transform` of new points | Still distorts densities/distances; depends on `n_neighbors`/`min_dist`; seeds |

**Key parameters:**
- t-SNE `perplexity` (≈ the effective number of neighbours; typical 5–50). Small → many fragmented clusters; large → more global structure.
- UMAP `n_neighbors` (local vs global balance) and `min_dist` (how tightly points pack).

**The essential caution:** t-SNE and UMAP can show clusters in structureless data (at low perplexity) and can make clusters of very different spread look equal. Always cross-check with PCA, with the clustering metrics on the original features, and with several parameter settings.

## 4. Assumptions and limitations
- Run PCA (e.g. to 30–50 dimensions) before t-SNE/UMAP on very high-dimensional data. It denoises and speeds things up.
- The embeddings are stochastic. Fix the seed for reproducibility, and check stability across seeds.
- Colouring the map by a label you then "discover" is circular. Use colours to *test* hypotheses (batch, regime, target), not to invent them.

## 5. Step-by-step procedure
1. **Scale/preprocess** as for modelling.
2. **PCA map** first (PC1 vs PC2, with the explained variance in the axis labels).
3. **t-SNE/UMAP** on the PCA-reduced data, with 2–3 parameter settings and 2 seeds.
4. **Colour by metadata** (batch, regime, class, time) to test hypotheses.
5. **Confirm** any apparent cluster in the original space (clustering metrics, nearest-neighbour checks).

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

digits = load_digits()                                   # 1797 images of 8×8 handwritten digits (bundled with sklearn)
X, labels = StandardScaler().fit_transform(digits.data), digits.target

Z_pca = PCA(2, random_state=0).fit_transform(X)
Z_tsne = {p: TSNE(n_components=2, perplexity=p, init="pca", random_state=0).fit_transform(PCA(30, random_state=0).fit_transform(X))
          for p in (5, 30, 100)}

fig, ax = plt.subplots(1, 4, figsize=(18, 4))
ax[0].scatter(*Z_pca.T, c=labels, cmap="tab10", s=4); ax[0].set_title("PCA")
for a, (p, Z) in zip(ax[1:], Z_tsne.items()):
    a.scatter(*Z.T, c=labels, cmap="tab10", s=4); a.set_title(f"t-SNE perplexity={p}")
plt.tight_layout(); plt.show()
```

```python
# UMAP (umap-learn), and its transform() for new points
import umap
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0)
Z_umap = reducer.fit_transform(X[:1500])
Z_new = reducer.transform(X[1500:])                      # place unseen points on the existing map
plt.figure(figsize=(5, 4)); plt.scatter(*Z_umap.T, c=labels[:1500], cmap="tab10", s=4)
plt.scatter(*Z_new.T, c="k", s=4, marker="x"); plt.title("UMAP (x = new points via transform)"); plt.show()
```

```python
# The caution: t-SNE on STRUCTURELESS data (Gaussian noise) and on clusters of very different spread
rng = np.random.default_rng(0)
noise = rng.normal(size=(800, 50))
two = np.vstack([rng.normal(0, 0.2, (400, 10)), rng.normal(5, 3.0, (400, 10))])   # tight vs very wide cluster
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].scatter(*TSNE(perplexity=5, random_state=0).fit_transform(noise).T, s=3); ax[0].set_title("t-SNE of pure noise (perplexity 5)")
Zt = TSNE(perplexity=30, random_state=0).fit_transform(two)
ax[1].scatter(*Zt.T, c=[0] * 400 + [1] * 400, cmap="coolwarm", s=3); ax[1].set_title("t-SNE: tight vs wide cluster")
ax[2].scatter(*PCA(2).fit_transform(two).T, c=[0] * 400 + [1] * 400, cmap="coolwarm", s=3); ax[2].set_title("PCA of the same data")
plt.tight_layout(); plt.show()
spread = lambda Z, g: np.mean(np.std(Z[g], axis=0))
g0, g1 = np.arange(400), np.arange(400, 800)
print(f"true spread ratio (wide/tight) ≈ {3.0/0.2:.0f}x; in t-SNE map ≈ {spread(Zt, g1)/spread(Zt, g0):.1f}x")
```

## 7. Interpretation
- **Digits:** PCA shows overlapping classes (linear projection). t-SNE separates most digits clearly. At perplexity 5 the classes fragment into sub-clusters. At 100 the global arrangement is more compact.
- **UMAP transform:** new points land in the regions of their class, which is useful for monitoring where new data fall relative to the reference data. The positions are still approximate.
- **Pure noise:** at low perplexity t-SNE produces blob-like "structure" from structureless data. **Never conclude that clusters exist from a t-SNE map alone.**
- **Tight vs wide cluster:** the true spreads differ by ~15×, but the t-SNE map shows them with similar sizes. t-SNE equalises densities. PCA preserves the difference.

## 8. Decision framework
- **IF** the goal is a first look or a communication-safe plot → PCA.
- **IF** the goal is to see whether local neighbourhood structure exists (classes, regimes in complex data) → t-SNE/UMAP on PCA-reduced data, with several settings and seeds.
- **IF** new data must be placed on an existing map → UMAP (`transform`) or PCA, not sklearn t-SNE.
- **IF** a cluster appears only in t-SNE/UMAP → confirm it in the original space before reporting it.
- **IF** you are tempted to use t-SNE coordinates as features → don't; use PCA/PLS, or the original features.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Reading cluster sizes or distances from t-SNE | Wrong conclusions | Use only for neighbourhood structure |
| A single perplexity/seed | Artefacts mistaken for structure | Several settings and seeds |
| t-SNE directly on raw very-high-dimensional data | Slow, noisy | PCA to ~30–50 dimensions first |
| Circular colouring | "Discovering" what you put in | Colour to test independent metadata |
| Using embeddings as model inputs without CV | Leakage / instability | Fit inside the pipeline, or avoid |

## 10. Checklist
- [ ] PCA map produced first.
- [ ] t-SNE/UMAP with ≥ 2 parameter settings and seeds, on PCA-reduced data.
- [ ] Colouring by independent metadata.
- [ ] Apparent clusters confirmed in the original space.
- [ ] Caveats stated with every non-linear map shown.

## 11. References
- van der Maaten, L. & Hinton, G. (2008). "Visualizing Data using t-SNE." *JMLR* 9, 2579–2605. https://jmlr.org/papers/v9/vandermaaten08a.html
- Wattenberg, M., Viégas, F. & Johnson, I. (2016). "How to Use t-SNE Effectively." *Distill*. https://distill.pub/2016/misread-tsne/
- McInnes, L., Healy, J. & Melville, J. (2018). "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." https://arxiv.org/abs/1802.03426 · Docs: https://umap-learn.readthedocs.io/
- Kobak, D. & Berens, P. (2019). "The art of using t-SNE for single-cell transcriptomics." *Nature Communications* 10, 5416. https://doi.org/10.1038/s41467-019-13056-x
- scikit-learn, *Manifold learning*. https://scikit-learn.org/stable/modules/manifold.html
