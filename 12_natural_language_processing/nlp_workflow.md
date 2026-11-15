# NLP Workflow: Framing, Text Exploration, Cleaning, Tokenisation and Evaluation

> **Folder:** 12_natural_language_processing · **Start here for text data.** · **Next:** [classical_nlp.md](classical_nlp.md) · **Related:** [classification_workflow.md](../04_classification/classification_workflow.md), [data_leakage.md](../00_methodology/data_leakage.md)

---

## 1. Purpose
Frame text problems (classification, extraction, similarity, topics), explore and clean text data, understand tokenisation and normalisation choices, and set up an evaluation that reflects how the system will be used. This page is the entry point for a progression from classical scikit-learn NLP to pretrained transformers.

## 2. When to use it
Whenever an input or target is free text: maintenance work orders, incident reports, customer feedback, lab notebooks, scientific abstracts, alarm messages.

## 3. Conceptual foundation

### 3.1 Common task types
| Task | Example | First approach | Section |
|---|---|---|---|
| **Text classification** (incl. sentiment) | Work order → failure category; feedback → positive/negative | TF-IDF + logistic regression | [classical_nlp.md](classical_nlp.md) |
| **Multilabel classification** | Report → several fault codes | One-vs-rest linear models | [classical_nlp.md](classical_nlp.md) |
| **Topic discovery** | What themes appear in 10 000 incident reports? | NMF/LDA on TF-IDF/counts | [topic_modeling.md](topic_modeling.md) |
| **Semantic similarity / search / deduplication** | Find similar past incidents | TF-IDF/LSA cosine → sentence embeddings | [embeddings_and_similarity.md](embeddings_and_similarity.md) |
| **Information extraction** (entities, values) | Extract equipment tags and temperatures | Regex/rules → NER models (spaCy / transformers) | Rules here; NER: future extension |
| **Generation / summarisation** | Summarise long reports | Pretrained seq2seq / LLMs | Out of scope (not the focus) |

### 3.2 The progression
1. **Rules/regex:** for highly structured fragments (tags like `P-101`, numbers and units). A strong baseline for extraction.
2. **Bag-of-words / TF-IDF + linear models:** fast, strong, interpretable baselines for classification. Often hard to beat on short, domain-specific texts.
3. **Static embeddings and LSA:** dense representations; similarity.
4. **Pretrained transformers** (BERT-family encoders): contextual representations; fine-tuning for classification; sentence-embedding models for similarity. Best for nuanced language, but they need a GPU for comfortable fine-tuning ([transformers_and_transfer_learning.md](transformers_and_transfer_learning.md)).

### 3.3 Text preprocessing choices
| Step | What it does | Use with classical models | Use with transformers |
|---|---|---|---|
| Lowercasing | "Pump" = "pump" | Usually yes | **No**: the model has its own tokenizer (cased/uncased models) |
| Removing codes/IDs/numbers | Removes noise like `WO-12345` | Yes (or normalise to a token like `<num>`) | Often keep, or mask IDs |
| Tokenisation | Split into tokens | Regex word tokens; character n-grams for typos | **Subword** tokenizer of the model (WordPiece/BPE) |
| Stopword removal | Drops frequent function words | Optional (TF-IDF already down-weights them); careful with negations ("no leak") | No |
| Stemming / lemmatisation | "leaking" → "leak" | Can help on small data; may hurt precision | No |
| Spelling/abbreviation normalisation | "brg" → "bearing" | Very valuable in technical text | Helps too |

## 4. Assumptions and limitations
- Text labels are often noisy (inconsistent coding by technicians). Inspect the disagreements before blaming the model.
- Duplicates and templates (copy-pasted reports) inflate scores if they span train and test. Deduplicate, or group them.
- Language, domain jargon and abbreviations differ from general-purpose pretrained models. Domain adaptation or simple normalisation may matter more than model size.

## 5. Step-by-step procedure
1. **Frame:** the task type, the label definition, the unit (document, sentence), the decision supported, the metric (macro-F1 for imbalanced categories; precision@k for search).
2. **Explore:** document lengths, vocabulary size, frequent tokens and n-grams per class, label distribution, duplicates, languages, the share of codes/numbers.
3. **Clean** to the extent the chosen model family needs (§3.3). Keep the raw text.
4. **Split:** stratified; grouped by template/author/equipment if near-duplicates exist; temporal if vocabulary or practices drift.
5. **Baseline:** majority class + TF-IDF + logistic regression ([classical_nlp.md](classical_nlp.md)).
6. **Iterate:** error analysis on the misclassified texts (read them), then normalisation, n-grams, character features, embeddings, and transformers if needed.

## 6. Python implementation: text exploration and cleaning

```python
import re
from collections import Counter
import numpy as np
import pandas as pd
from cookbook_data import make_maintenance_logs

df = make_maintenance_logs()
print(df["category"].value_counts(normalize=True).round(3))
df["n_chars"] = df["text"].str.len(); df["n_tokens"] = df["text"].str.split().str.len()
print(df[["n_chars", "n_tokens"]].describe().round(1).T)
print(df.sample(5, random_state=3)["text"].tolist())
```

```python
# Tokenisation + normalisation for classical models
ABBREV = {"brg": "bearing", "mtr": "motor", "temp": "temperature", "wo": "work_order"}
def normalise(text: str) -> str:
    t = text.lower()
    t = re.sub(r"\bwo-\d+\b", " ", t)                      # remove work-order IDs (no information about the category)
    t = re.sub(r"\b[a-z]{1,3}-\d+\b", " <equip> ", t)      # equipment tags → placeholder (P-101, K-12)
    t = re.sub(r"\d+(\.\d+)?\s*(c|bar)\b", r" <value_\2> ", t)  # numeric values with units → typed placeholder
    t = re.sub(r"\d+", " <num> ", t)
    t = " ".join(ABBREV.get(w, w) for w in re.findall(r"<\w+>|[a-z]+", t))
    return t

df["clean"] = df["text"].map(normalise)
print(df[["text", "clean"]].head(4).to_string())

# Frequent tokens per class (after cleaning): sanity check of the signal
def top_tokens(texts, k=8):
    return [w for w, _ in Counter(" ".join(texts).split()).most_common(k)]
for c, g in df.groupby("category"):
    print(f"{c:16s}", top_tokens(g["clean"]))
```

```python
# Stemming (no external data needed) and stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
stem = SnowballStemmer("english").stem
example = "leaking seals were replaced and bearings regreased; no leak after"
tokens = re.findall(r"[a-z]+", example.lower())
print("stemmed:", [stem(w) for w in tokens])
print("without stopwords:", [w for w in tokens if w not in ENGLISH_STOP_WORDS])   # note: 'no' is removed → negation lost!
# Lemmatisation (dictionary-based, e.g. spaCy: nlp = spacy.load("en_core_web_sm")) requires a model download.
```

```python
# Duplicates / near-duplicates (template texts) that could leak across splits
dup_rate = df["clean"].duplicated().mean()
print(f"exact duplicate cleaned texts: {dup_rate:.1%}")
# Use the cleaned text as a GROUP key so identical texts never sit in both train and test
groups = df["clean"].factorize()[0]
```

## 7. Interpretation
- **Class distribution:** imbalanced (bearing ≈ 30%, instrumentation ≈ 12%). Report macro-F1 and per-class recall, not accuracy alone.
- **Lengths:** short texts (≈ 10–20 tokens). Bag-of-words models do well on short, keyword-driven texts. Long-range context matters little.
- **Cleaning output:** IDs removed; equipment tags and values replaced by placeholders (`<equip>`, `<value_c>`, `<value_bar>`). The *presence* of a temperature value may be informative (bearing/electrical), while the specific ID is not.
- **Top tokens per class:** they should be the expected keywords (bearing → "bearing", "noise", "vibration"…). Generic tokens shared by all classes ("pump", "checked") carry little information. TF-IDF down-weights them.
- **Stopword caveat:** removing "no" turns "no leak" into "leak". In technical and medical text, negation matters. Keep negations, or use n-grams.
- **Duplicates:** a non-trivial share of identical cleaned texts. Use grouped splitting by text, or deduplicate, to avoid inflated scores.

## 8. Decision framework
**IF the input is text, which approach first?**
- Short, jargon-heavy, keyword-driven texts (work orders, alarms) → TF-IDF (word + char n-grams) + linear model. Often near the best.
- Nuanced language (sentiment with negation and irony, long reports) → a strong TF-IDF baseline, then fine-tune a pretrained transformer (GPU or Colab).
- Few labelled examples (< a few hundred) → a sentence-embedding model + logistic regression, or zero-shot classification (a pretrained NLI model), as a baseline.
- No labels; want themes → topic modelling, or clustering of embeddings.
- Need to find similar documents → TF-IDF cosine baseline, then sentence embeddings.
- Need to extract values/tags → regex rules first; NER models if the rules are insufficient.

**IF texts are near-duplicates/templates** → grouped split. **IF vocabulary drifts over time** → temporal split.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Fitting the vectoriser on all data | Mild leakage (IDF, vocabulary) | Vectoriser inside the pipeline |
| Aggressive cleaning before transformers | Destroys information the model uses | Raw(ish) text for transformers |
| Removing negations as stopwords | Meaning inverted | Custom stopword lists; n-grams |
| Accuracy on imbalanced classes | Misleading | Macro-F1, per-class report |
| Ignoring duplicates/templates | Inflated scores | Dedup / group split |
| Not reading misclassified texts | Missing label noise and normalisation gaps | Manual error review |

## 10. Checklist
- [ ] Task type, label definition, metric decided.
- [ ] Lengths, vocabulary, class balance, duplicates explored.
- [ ] Cleaning matched to the model family; raw text kept.
- [ ] Split stratified, grouped/temporal as needed.
- [ ] TF-IDF + linear baseline before transformers.
- [ ] Error analysis by reading the texts.

## 11. References
- Jurafsky, D. & Martin, J. H. *Speech and Language Processing*, 3rd ed. draft (Ch. 2 tokenisation/normalisation, Ch. 4–5 classification). https://web.stanford.edu/~jurafsky/slp3/
- scikit-learn User Guide, *Text feature extraction*. https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
- Bird, S., Klein, E. & Loper, E. *Natural Language Processing with Python* (NLTK book). https://www.nltk.org/book/
- Hugging Face, *LLM/NLP Course* (tokenizers, transformers). https://huggingface.co/learn/nlp-course
- Brundage, M. P., Sexton, T., Hodkiewicz, M., Dima, A. & Lukens, S. (2021). "Technical language processing: Unlocking maintenance knowledge." *Manufacturing Letters* 27, 42–46. https://doi.org/10.1016/j.mfglet.2020.11.001
