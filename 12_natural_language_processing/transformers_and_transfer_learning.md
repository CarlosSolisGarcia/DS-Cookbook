# Transformers, Pretrained Models and Transfer Learning with Hugging Face

> **Folder:** 12_natural_language_processing · **Prev:** [embeddings_and_similarity.md](embeddings_and_similarity.md) · **Next:** [cv_workflow.md](../13_computer_vision/cv_workflow.md) · **Related:** [transfer_learning_and_domain_adaptation.md](../20_advanced_methodologies/transfer_learning_and_domain_adaptation.md), [efficient_development.md](../20_advanced_methodologies/efficient_development.md)

---

## 1. Purpose
Understand the transformer architecture well enough to use it sensibly. Apply pretrained models from the Hugging Face ecosystem: zero-shot use, feature extraction, and fine-tuning for classification. Evaluate them fairly against the classical baselines, on modest hardware (a laptop GPU such as an RTX 3060 with 6 GB, or Colab).

## 2. When to use it
- A TF-IDF + linear baseline exists, and the errors involve meaning that keywords miss (negation, paraphrase, context, long-range dependencies).
- Few labels: pretrained knowledge transfers (fine-tuning with hundreds of examples often works).
- Multilingual text, or text in a domain covered by an available pretrained model (scientific, biomedical, legal, code).

## 3. Conceptual foundation

### 3.1 The transformer in a few lines
- **Tokenisation** into sub-words (WordPiece/BPE): unknown words are split into known pieces ("rewired" → "re", "##wired"). Each model has its *own* tokenizer, and you must use it.
- **Embeddings + positional information** → a stack of **self-attention** layers. Each token's representation is updated as a weighted combination of all the tokens, with the weights computed from token similarity (query–key dot products). Attention lets "no" modify "leak" regardless of distance.
- **Encoder models** (BERT, RoBERTa, DeBERTa, MiniLM): bidirectional; good for classification, NER and embeddings. **Decoder models** (GPT-style): generation. **Encoder–decoder** (T5, BART): translation, summarisation.
- **Pretraining** (masked-language modelling on huge corpora) learns general language. **Fine-tuning** adapts all (or some) of the weights to your task with a small labelled set. A classification head on the [CLS]/pooled representation predicts the labels.

### 3.2 Ways to use a pretrained model
| Approach | Labels needed | Compute | When |
|---|---|---|---|
| **Zero-shot** (NLI-based `zero-shot-classification`) | None | Inference only | Quick baseline; label names are meaningful words |
| **Feature extraction** (frozen embeddings + logistic regression) | Few–moderate | Inference only (CPU ok) | Limited compute; quick strong baseline |
| **Fine-tuning** (all weights) | Hundreds+ | GPU recommended | Best accuracy for the task |
| **Parameter-efficient fine-tuning** (LoRA/adapters, `peft`) | Hundreds+ | Less memory | Large models on small GPUs |

**Model size vs hardware.** On a 6 GB GPU, base-size encoders (~110M parameters) fine-tune comfortably with a batch of 16–32 at sequence length 128, with mixed precision (`fp16=True`). DistilBERT/MiniLM are faster. Large models need gradient accumulation, PEFT or Colab.

## 4. Assumptions and limitations
- Pretrained models encode the biases and vocabulary of their training corpora. Technical abbreviations may tokenise poorly. Check the tokenisation of domain text.
- Fine-tuning on small datasets is high-variance. Use several seeds and report the mean ± std.
- Transformers are not automatically better on short keyword-driven texts. Compare them with the TF-IDF baseline on the **same splits**.
- Models must be downloaded (the Hugging Face Hub) and are tied to library versions. Pin them and cache them.

## 5. Step-by-step procedure
1. **Baseline first:** TF-IDF + logistic regression with grouped stratified CV ([classical_nlp.md](classical_nlp.md)).
2. **Choose a checkpoint:** a small encoder (e.g. `distilbert-base-uncased`, `microsoft/MiniLM-L12-H384-uncased`), or a domain model (e.g. a science- or biomedical-pretrained BERT) if relevant.
3. **Inspect the tokenisation** of typical texts; set `max_length` from the token-length distribution.
4. **Quick wins:** zero-shot and frozen embeddings + logistic regression.
5. **Fine-tune** with `Trainer`: learning rate 2e-5–5e-5, 2–5 epochs, weight decay 0.01, warm-up, evaluation each epoch, early stopping on validation macro-F1.
6. **Evaluate on the same test split** as the baseline; several seeds; per-class report.
7. **Error analysis:** where does the transformer win or lose compared with TF-IDF? Is the gain worth the cost (latency, maintenance)?

## 6. Python implementation

**(a) Mechanics, runnable offline:** train a WordPiece tokenizer on the corpus, and run a *tiny, randomly initialised* BERT, to see the shapes and the API. This is not a useful model. Pretrained weights are what make transformers powerful.

```python
import re
import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

from cookbook_data import make_maintenance_logs

df = make_maintenance_logs()
texts = df["text"].str.lower().str.replace(r"\bwo-\d+\b", " ", regex=True).tolist()
labels = sorted(df["category"].unique()); y = df["category"].map({l: i for i, l in enumerate(labels)}).to_numpy()

tok_model = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tok_model.pre_tokenizer = pre_tokenizers.Whitespace()
tok_model.train_from_iterator(texts, trainers.WordPieceTrainer(vocab_size=800, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok_model, unk_token="[UNK]", pad_token="[PAD]",
                                    cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
print(tokenizer.tokenize("pump p-101: brg grinding noise - rewired contactor"))

enc = tokenizer(texts[:4], padding=True, truncation=True, max_length=32, return_tensors="pt")
print({k: v.shape for k, v in enc.items()})          # input_ids, attention_mask, (token_type_ids)
tiny = BertForSequenceClassification(BertConfig(vocab_size=tokenizer.vocab_size, hidden_size=64, num_hidden_layers=2,
                                                num_attention_heads=2, intermediate_size=128, num_labels=len(labels),
                                                max_position_embeddings=64))
with torch.no_grad():
    out = tiny(**{k: v for k, v in enc.items() if k != "token_type_ids"})
print("logits shape:", out.logits.shape, "| parameters:", sum(p.numel() for p in tiny.parameters()))
```

**(b) Zero-shot and pipelines.** Requires downloading pretrained models (internet access to the Hugging Face Hub):

```python
# not-run (downloads models)
from transformers import pipeline
zsc = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")       # ~1.6 GB; CPU ok but slow
print(zsc("motor tripped on overcurrent, replaced contactor",
          candidate_labels=["bearing", "seal leak", "electrical", "lubrication", "instrumentation"]))
sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
print(sentiment(["The repair fixed the leak, great job", "Still vibrating after the second visit"]))
```

**(c) Frozen embeddings + logistic regression** (CPU-friendly):

```python
# not-run (downloads a model)
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
name = "sentence-transformers/all-MiniLM-L6-v2"
tok, enc_model = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval()
def embed(batch_texts):
    b = tok(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        h = enc_model(**b).last_hidden_state                      # (batch, tokens, hidden)
    mask = b["attention_mask"].unsqueeze(-1)
    return ((h * mask).sum(1) / mask.sum(1)).numpy()              # mean pooling over real tokens
E = np.vstack([embed(texts[i:i + 64]) for i in range(0, len(texts), 64)])
print(cross_val_score(LogisticRegression(max_iter=2000), E, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                      scoring="f1_macro").mean())
```

**(d) Fine-tuning with `Trainer`** (GPU recommended: ~2–5 minutes on an RTX 3060 for this dataset size):

```python
# not-run (downloads a model; GPU recommended)
from datasets import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

ckpt = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(ckpt)
tr_idx, te_idx = train_test_split(np.arange(len(texts)), test_size=0.2, stratify=y, random_state=0)  # SAME split as the baseline
tr_idx, va_idx = train_test_split(tr_idx, test_size=0.15, stratify=y[tr_idx], random_state=0)
def to_ds(idx):
    return Dataset.from_dict({"text": [texts[i] for i in idx], "label": y[idx].tolist()}).map(
        lambda b: tok(b["text"], truncation=True, max_length=64), batched=True)
ds_tr, ds_va, ds_te = to_ds(tr_idx), to_ds(va_idx), to_ds(te_idx)

model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=len(labels),
                                                           id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)})
def compute_metrics(p):
    return {"macro_f1": f1_score(p.label_ids, p.predictions.argmax(-1), average="macro")}
args = TrainingArguments(output_dir="out", learning_rate=3e-5, per_device_train_batch_size=32, per_device_eval_batch_size=64,
                         num_train_epochs=5, weight_decay=0.01, warmup_ratio=0.1, eval_strategy="epoch", save_strategy="epoch",
                         load_best_model_at_end=True, metric_for_best_model="macro_f1", fp16=torch.cuda.is_available(),
                         seed=0, report_to="none")
trainer = Trainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va, processing_class=tok,
                  data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
                  callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
trainer.train()
print(trainer.evaluate(ds_te))            # compare with the TF-IDF baseline on the same test indices
```

> **API notes (transformers ≥ 4.46 / 5.x):** `TrainingArguments(eval_strategy=...)` replaced `evaluation_strategy`, and `Trainer(processing_class=tok)` replaced `tokenizer=tok`. Check the docs of your installed version. On Colab, select a GPU runtime and `pip install transformers datasets accelerate`.

## 7. Interpretation
- **Tokenisation output (a):** domain words split into sub-words. Abbreviations like "brg" become single tokens if frequent in the training corpus of the tokenizer. With a general pretrained tokenizer they may fragment, which is a hint that normalisation helps.
- **Tiny random BERT (a):** shows the input/output shapes and the parameter count (~10⁵ here vs ~6.6 × 10⁷ for DistilBERT). Without pretraining, a transformer on 3000 short texts would not beat TF-IDF. Pretraining is the source of the advantage.
- **Zero-shot (b):** a useful sanity baseline when no labels exist. Expect it to be clearly below supervised models on domain jargon.
- **Frozen embeddings (c) and fine-tuning (d):** on these synthetic, keyword-driven texts, expect *no* large gain over TF-IDF (which is already at the label-noise ceiling). On real texts with nuanced language, fine-tuning typically gives the best macro-F1. Report the mean ± std over seeds, and the cost (training time, inference latency).

## 8. Decision framework
- **IF** TF-IDF already reaches the label-noise ceiling → stop. Transformers add cost without benefit.
- **IF** no labels → zero-shot as a baseline; label a few hundred examples (active learning) for real gains.
- **IF** few labels (100–1000) → frozen sentence embeddings + logistic regression, then fine-tuning with several seeds.
- **IF** you have a GPU with ≥ 6 GB → fine-tune base-size encoders with fp16, max_length from the data (often 64–256). **ELSE** → DistilBERT/MiniLM on CPU for inference/embeddings, or Colab for fine-tuning.
- **IF** the texts are long (> 512 tokens) → truncation strategy (head+tail), chunking with aggregation, or long-context models.
- **IF** the domain is specialised (chemistry, materials, biomedical) → try a domain-pretrained checkpoint; compare it with a general one.
- **IF** generative LLMs are considered → evaluate them like any classifier on a held-out set; consider cost, privacy and reproducibility (not the focus of this cookbook).

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Comparing a transformer and TF-IDF on different splits | Unfair conclusions | Same indices, same metric |
| A single fine-tuning run | High-variance results mistaken for real gains | ≥ 3 seeds |
| Heavy text cleaning before a pretrained tokenizer | Information loss | Minimal cleaning (remove IDs only) |
| Learning rate too high (1e-3) | Catastrophic forgetting / divergence | 2e-5–5e-5; warm-up |
| `max_length` 512 by default for short texts | Wasted compute | Set from the token-length distribution |
| Test data used for early stopping | Optimistic results | A separate validation split |
| Unpinned model versions | Irreproducible results | Record the checkpoint name + revision |

## 10. Checklist
- [ ] TF-IDF baseline on fixed splits.
- [ ] Checkpoint chosen for the domain and hardware; tokenisation inspected.
- [ ] Zero-shot / frozen-embedding quick baselines.
- [ ] Fine-tuning with validation-based early stopping; several seeds.
- [ ] Test comparison on the same split; per-class report; cost noted.
- [ ] Model name/revision and library versions recorded.

## 11. References
- Vaswani, A. et al. (2017). "Attention Is All You Need." *NeurIPS 30*. https://arxiv.org/abs/1706.03762
- Devlin, J., Chang, M.-W., Lee, K. & Toutanova, K. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL*. https://arxiv.org/abs/1810.04805
- Sanh, V., Debut, L., Chaumond, J. & Wolf, T. (2019). "DistilBERT, a distilled version of BERT." https://arxiv.org/abs/1910.01108
- Hugging Face, *LLM/NLP Course* (Ch. 2–3: using and fine-tuning transformers). https://huggingface.co/learn/nlp-course
- Hugging Face Transformers documentation (`Trainer`, `pipeline`, tokenizers). https://huggingface.co/docs/transformers
- Tunstall, L., von Werra, L. & Wolf, T. (2022). *Natural Language Processing with Transformers*, revised ed. O'Reilly.
- Hu, E. J. et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." https://arxiv.org/abs/2106.09685
