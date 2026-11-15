# Entity Resolution and Record Linkage

> **Folder:** 19_advanced_tabular_ml · **Prev:** [imbalance_beyond_binary.md](imbalance_beyond_binary.md) · **Next:** [noisy_labels_weak_supervision.md](noisy_labels_weak_supervision.md) · **Related:** [embeddings_and_similarity.md](../12_natural_language_processing/embeddings_and_similarity.md), [data_quality.md](../01_data_understanding/data_quality.md) (duplicates)

---

## 1. Purpose
Identify records that refer to the same real-world entity (the same supplier, part, equipment tag, customer, chemical compound) across or within datasets, when no reliable common key exists. The output is a clean master table and correct joins.

## 2. When to use it
- Merging an ERP supplier list with procurement invoices, a CMMS equipment list with historian tags, lab sample registries with LIMS exports.
- Deduplicating master data before modelling (duplicates leak across splits and distort counts).

## 3. Conceptual foundation
The standard pipeline:
1. **Standardisation:** case, whitespace, punctuation, legal suffixes ("S.A.", "GmbH", "Ltd"), abbreviations, unit formats, transliteration.
2. **Blocking (candidate generation):** comparing all pairs is O(n²). Restrict the comparisons to pairs sharing a cheap key (the same postcode, the first 3 letters, a sorted token, or nearest neighbours in TF-IDF space).
3. **Pairwise comparison features:** string similarities (Jaro–Winkler, Levenshtein ratio, token-set Jaccard, TF-IDF cosine of character n-grams), numeric differences, field equality flags.
4. **Classification:** rules/thresholds (simple), a probabilistic linkage model (Fellegi–Sunter), or a supervised classifier trained on labelled pairs.
5. **Clustering:** take the transitive closure (connected components) of the matched pairs, with checks against chaining errors.
6. **Evaluation:** precision/recall of pairs on a labelled sample; blocking recall (the share of true matches surviving blocking).

## 4. Assumptions and limitations
- True matches are rare among all pairs (heavy imbalance). Report precision/recall, not accuracy.
- Labelled pairs are expensive. Active labelling of uncertain pairs is efficient.
- Transitive closure can merge distinct entities through a chain of weak matches. Inspect the large clusters.

## 5. Step-by-step procedure
1. **Profile the fields;** choose the identifying fields; standardise them.
2. **Blocking strategy;** measure the blocking recall on a labelled sample.
3. **Similarity features per candidate pair.**
4. **Label a sample of pairs** (stratified by similarity score) → train a classifier (or set a threshold).
5. **Evaluate** pair precision/recall; choose the threshold by the cost of false merges vs misses.
6. **Cluster;** review the large/odd clusters; create the master IDs.

## 6. Python implementation

```python
import re
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix

# Synthetic supplier master data: 300 true suppliers, recorded 1–4 times with variations
rng = np.random.default_rng(0)
stems = ["Acme", "Iberica", "Norte", "Quimica", "Valves", "Bombas", "Flux", "Tecno", "Hidro", "Metal", "Sur", "Delta"]
suffixes = [" S.A.", " SL", " S.L.", " GmbH", " Ltd", ""]
true = [f"{rng.choice(stems)} {rng.choice(stems)} {rng.choice(['Industrial', 'Supplies', 'Systems', 'Group'])}" for _ in range(300)]
cities = rng.choice(["Madrid", "Bilbao", "Valencia", "Sevilla", "Tarragona"], 300)
def variant(name):
    v = name + rng.choice(suffixes)
    if rng.random() < 0.3: v = v.upper()
    if rng.random() < 0.3: v = v.replace("Industrial", "Ind.").replace("Supplies", "Supp.")
    if rng.random() < 0.2: i = rng.integers(1, len(v) - 1); v = v[:i] + v[i + 1:]          # typo
    return v
rows = [(e, variant(true[e]), cities[e]) for e in range(300) for _ in range(rng.integers(1, 5))]
rec = pd.DataFrame(rows, columns=["entity", "name", "city"]).sample(frac=1, random_state=0).reset_index(drop=True)
print(len(rec), "records for", rec["entity"].nunique(), "true suppliers"); print(rec.head())

LEGAL = r"\b(s\.?a\.?|s\.?l\.?|gmbh|ltd)\b"
def standardise(s):
    s = s.lower().replace("ind.", "industrial").replace("supp.", "supplies")
    s = re.sub(LEGAL, " ", s); s = re.sub(r"[^a-z ]", " ", s)
    return " ".join(s.split())
rec["std"] = rec["name"].map(standardise)
```

```python
# Blocking: nearest neighbours in character-3-gram TF-IDF space, within the same city
tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3)).fit(rec["std"])
pairs = set()
for city, g in rec.groupby("city"):
    V = tf.transform(g["std"]); nn = NearestNeighbors(n_neighbors=min(6, len(g)), metric="cosine").fit(V)
    _, idx = nn.kneighbors(V)
    for i, row in enumerate(idx):
        for j in row:
            a, b = g.index[i], g.index[j]
            if a != b:                                    # identical strings can return another record before "self"
                pairs.add((min(a, b), max(a, b)))
pairs = pd.DataFrame(sorted(pairs), columns=["a", "b"])
pairs["match"] = (rec.loc[pairs["a"], "entity"].to_numpy() == rec.loc[pairs["b"], "entity"].to_numpy()).astype(int)
n_true = int(rec.groupby("entity").size().apply(lambda k: k * (k - 1) / 2).sum())
print(f"candidate pairs: {len(pairs)} (vs {len(rec) * (len(rec) - 1) // 2} all pairs); blocking recall = {pairs['match'].sum() / n_true:.3f}")
```

```python
# Pair features and a classifier trained on labelled pairs (here the synthetic truth; in practice a labelled sample)
A, B = rec.loc[pairs["a"], "std"].to_numpy(), rec.loc[pairs["b"], "std"].to_numpy()
def jaccard(x, y):
    s, t = set(x.split()), set(y.split()); return len(s & t) / len(s | t)
pairs["seq_ratio"] = [SequenceMatcher(None, x, y).ratio() for x, y in zip(A, B)]
pairs["token_jaccard"] = [jaccard(x, y) for x, y in zip(A, B)]
pairs["tfidf_cos"] = np.asarray(tf.transform(A).multiply(tf.transform(B)).sum(1)).ravel()
pairs["first_token_eq"] = [x.split()[0] == y.split()[0] for x, y in zip(A, B)]
F = pairs[["seq_ratio", "token_jaccard", "tfidf_cos", "first_token_eq"]].astype(float)
p = cross_val_predict(LogisticRegression(max_iter=2000), F, pairs["match"], cv=5, method="predict_proba")[:, 1]
for thr in (0.3, 0.5, 0.8):
    pred = (p >= thr).astype(int)
    print(f"threshold {thr}: pair precision={precision_score(pairs['match'], pred):.3f}  recall={recall_score(pairs['match'], pred):.3f}")
print("simple rule tfidf_cos > 0.8: precision =", round(precision_score(pairs["match"], F["tfidf_cos"] > 0.8), 3),
      "recall =", round(recall_score(pairs["match"], F["tfidf_cos"] > 0.8), 3))
```

```python
# Clustering by connected components of the matched pairs → master IDs; check purity and over-merging
m = pairs[p >= 0.5]
G = coo_matrix((np.ones(len(m)), (m["a"], m["b"])), shape=(len(rec), len(rec)))
n_clusters, labels = connected_components(G, directed=False)
rec["master_id"] = labels
purity = rec.groupby("master_id")["entity"].agg(lambda s: s.value_counts().iloc[0] / len(s)).mean()
print(f"{n_clusters} master records for {rec['entity'].nunique()} true suppliers; mean cluster purity = {purity:.3f}")
print("largest clusters (inspect for chaining):", rec["master_id"].value_counts().head(3).to_dict())
```

## 7. Interpretation
- **Blocking:** 2621 candidate pairs instead of ~298 000 (a 99% reduction), with **blocking recall ≈ 0.98**. Almost every true match survives, and this number is the upper bound for the whole pipeline's recall. If it is low, widen the blocking (more neighbours, several blocking keys).
- **Classifier vs rule:** at threshold 0.5 the learned combination of similarities reaches precision ≈ 0.91 / recall ≈ 0.89, against ≈ 0.76 / 0.71 for the single rule "TF-IDF cosine > 0.8". Shared stems ("Acme Delta Group" vs "Acme Delta Systems") are the hard negatives. Combining character-, token- and sequence-level similarities separates them better.
- **Threshold choice:** raising the threshold to 0.8 barely raises precision but cuts recall sharply. Use the precision–recall trade-off at the decision's cost. A false merge (two suppliers combined) is often worse than a miss in master data.
- **Clusters vs true entities:** 332 clusters for 300 suppliers, with purity ≈ 0.99. Mostly correct, with some suppliers split into several clusters (missed links). The largest clusters have 7–8 records, although no true supplier has more than 4. That is **chaining**: transitive closure merged distinct suppliers through intermediate weak matches. Inspect clusters above the expected size, and split them with stricter cluster-level rules.

## 8. Decision framework
- **IF** the records are few (< ~5000) → full pairwise comparison is feasible. **ELSE** blocking (TF-IDF nearest neighbours, sorted neighbourhood, phonetic keys).
- **IF** labelled pairs are unavailable → start with rules on standardised fields; label a sample (stratified by similarity) to evaluate them.
- **IF** false merges are costly → high-precision thresholds; manual review of the borderline pairs.
- **IF** several fields exist (name, address, tax ID) → field-wise similarities as features; exact matches on strong identifiers first.
- **IF** the scale or complexity grows → dedicated libraries (`recordlinkage`, Splink, dedupe).

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| No standardisation before matching | Misses obvious matches | Normalise case, punctuation, legal forms, abbreviations |
| Reporting accuracy over all pairs | Meaningless (almost all pairs are non-matches) | Pair precision/recall; blocking recall |
| Blindly taking the transitive closure | Over-merged clusters | Inspect the large clusters; stricter thresholds |
| Evaluating only on the easy pairs | Optimistic | Stratified labelled samples incl. hard negatives |
| Self-pairs from nearest-neighbour blocking (identical strings) | Inflated "matches" (blocking recall > 1 was the symptom here) | Exclude a == b explicitly |

## 10. Checklist
- [ ] Fields standardised.
- [ ] Blocking with measured recall.
- [ ] Pair similarity features; a labelled sample; a classifier/threshold by cost.
- [ ] Clusters built; the large clusters reviewed; master IDs created.

## 11. References
- Christen, P. (2012). *Data Matching: Concepts and Techniques for Record Linkage, Entity Resolution, and Duplicate Detection*. Springer.
- Fellegi, I. P. & Sunter, A. B. (1969). "A Theory for Record Linkage." *JASA* 64(328), 1183–1210.
- Papadakis, G., Skoutas, D., Thanos, E. & Palpanas, T. (2020). "Blocking and Filtering Techniques for Entity Resolution: A Survey." *ACM Computing Surveys* 53(2). https://doi.org/10.1145/3377455
- Python Record Linkage Toolkit. https://recordlinkage.readthedocs.io/ · Splink: https://moj-analytical-services.github.io/splink/
