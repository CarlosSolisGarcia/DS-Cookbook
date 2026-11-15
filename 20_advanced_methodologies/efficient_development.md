# Efficient Model Development with Limited Compute (Laptop GPU, 16 GB RAM, Colab)

> **Folder:** 20_advanced_methodologies · **Prev:** [ensembles_and_stacking.md](ensembles_and_stacking.md) · **Next:** [versioning_and_pipelines.md](versioning_and_pipelines.md) · **Related:** [hyperparameter_optimization.md](../06_model_improvement/hyperparameter_optimization.md), [image_classification.md](../13_computer_vision/image_classification.md), [transformers_and_transfer_learning.md](../12_natural_language_processing/transformers_and_transfer_learning.md)

---

## 1. Purpose
Get the most out of a typical personal workstation, for example an **Intel i7-12700H (14 cores), 16 GB RAM and an NVIDIA RTX 3060 laptop GPU (6 GB VRAM)**. Iterate quickly, avoid out-of-memory errors, use the GPU only where it pays off, and move heavy jobs to Google Colab when needed.

## 2. When to use it
Whenever experiments are slow, memory is tight, or you are deciding between CPU, local GPU and Colab.

## 3. Conceptual foundation
| Workload | Best hardware | Why |
|---|---|---|
| pandas / scikit-learn / statsmodels | CPU (multi-core via `n_jobs=-1`) | Most of these libraries do not use GPUs |
| LightGBM / XGBoost on tabular data | CPU is usually fine; the GPU helps for large data (XGBoost `device="cuda"`) | Tree training is memory-bound; the gains appear above ~10⁶ rows |
| CNNs / transformers training | GPU | Orders of magnitude faster than CPU |
| Transformer **inference** for embeddings (small models) | CPU acceptable, GPU faster | MiniLM-size models run on CPU |
| Models > ~300M parameters, long sequences, large images | Colab (T4/L4/A100), or PEFT/quantisation locally | 6 GB VRAM limit |

**Memory rules of thumb:**
- A pandas DataFrame takes roughly n_rows × n_cols × 8 bytes (float64), plus extra for object/string columns. Downcasting to float32 halves it, and `category` dtype saves a lot for repeated strings.
- GPU memory for training ≈ weights + gradients + optimiser state (Adam ≈ 2× the weights) + activations (∝ batch × sequence length/resolution). Mixed precision (fp16/bf16) roughly halves the activation memory.

**Speed levers (by typical impact):** iterate on a **subsample** first → reduce the data size (dtypes, columns) → early stopping → smaller models/resolutions → fewer CV folds during exploration (full CV for the final comparisons) → parallelism (`n_jobs`) → GPU.

## 4. Assumptions and limitations
- Subsample results are indicative. Confirm the important decisions on the full data.
- The GPU speed-up depends on the batch size and the data-loading pipeline. A starved GPU (slow DataLoader) gains little.
- Colab sessions are time-limited and ephemeral. Save checkpoints to Drive, and pin package versions.

## 5. Step-by-step procedure
1. **Profile first:** time each stage (load, features, fit, predict); measure the memory (`df.memory_usage(deep=True)`, `torch.cuda.max_memory_allocated`).
2. **Shrink the data:** select columns, downcast dtypes, use `category` for strings, use Parquet for I/O.
3. **Develop on a subsample** (e.g. 10–20%, stratified/grouped/time-consistent); scale up for the final runs.
4. **Bound the training:** early stopping (GBMs, NNs); reasonable iteration caps.
5. **Parallelise the CPU work** (`n_jobs=-1` in CV/RF; LightGBM threads), avoiding nested over-subscription (n_jobs in both CV *and* the model).
6. **GPU training:** mixed precision, a batch size tuned to fit, gradient accumulation for effectively larger batches, `pin_memory` + `num_workers` in the DataLoader.
7. **Offload to Colab** for heavy fine-tuning, with checkpointing.

## 6. Python implementation

```python
import time
import numpy as np
import pandas as pd

# 1–2. Memory: dtype optimisation
rng = np.random.default_rng(0)
n = 1_000_000
df = pd.DataFrame({"sensor_a": rng.normal(size=n), "sensor_b": rng.normal(size=n), "count": rng.integers(0, 100, n),
                   "site": rng.choice(["North", "South", "East", "West"], n), "machine": rng.choice([f"M{i:04d}" for i in range(500)], n)})
def mb(d): return d.memory_usage(deep=True).sum() / 1e6
before = mb(df)
opt = df.assign(sensor_a=df["sensor_a"].astype("float32"), sensor_b=df["sensor_b"].astype("float32"),
                count=pd.to_numeric(df["count"], downcast="unsigned"),
                site=df["site"].astype("category"), machine=df["machine"].astype("category"))
print(f"memory: {before:.0f} MB → {mb(opt):.0f} MB")
# Parquet keeps the dtypes and is much faster than CSV:  opt.to_parquet("data.parquet"); pd.read_parquet("data.parquet", columns=[...])
```

```python
# 3–4. Subsample for iteration + early stopping: same conclusion, a fraction of the time
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
X = opt[["sensor_a", "sensor_b", "count", "site"]]
y = 2 * opt["sensor_a"] - opt["sensor_b"] ** 2 + (opt["site"] == "North") * 0.5 + rng.normal(0, 0.5, n)
for frac in (0.02, 0.2):
    idx = rng.choice(n, int(frac * n), replace=False)
    t0 = time.time()
    s = cross_val_score(HistGradientBoostingRegressor(categorical_features="from_dtype", early_stopping=True, random_state=0),
                        X.iloc[idx], y.iloc[idx], cv=KFold(3, shuffle=True, random_state=0), scoring="r2")
    print(f"{frac:>4.0%} of rows: R2={s.mean():.4f}  time={time.time() - t0:.1f}s")
```

```python
# 6. GPU training pattern: device selection, mixed precision, gradient accumulation, efficient DataLoader
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Sequential(nn.Linear(32, 256), nn.ReLU(), nn.Linear(256, 1)).to(device)
data = TensorDataset(torch.randn(20000, 32), torch.randn(20000, 1))
loader = DataLoader(data, batch_size=256, shuffle=True, num_workers=0,            # use 2–6 workers for disk/augmentation-heavy data
                    pin_memory=(device == "cuda"))
opt_ = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
accum = 4                                                                          # effective batch = 256 × 4
for step, (xb, yb) in enumerate(loader):
    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
    with torch.autocast(device_type=device, dtype=torch.float16 if device == "cuda" else torch.bfloat16, enabled=(device == "cuda")):
        loss = nn.functional.mse_loss(model(xb), yb) / accum
    scaler.scale(loss).backward()
    if (step + 1) % accum == 0:
        scaler.step(opt_); scaler.update(); opt_.zero_grad(set_to_none=True)
print("device:", device, "| last loss:", round(loss.item() * accum, 4))
if device == "cuda":
    print("peak GPU memory (MB):", torch.cuda.max_memory_allocated() / 1e6)
```

**Colab checklist (when local resources are insufficient):**
```text
1. Runtime → Change runtime type → GPU (T4 is free-tier; L4/A100 with paid plans).
2. !pip install -q <exact versions from requirements.txt>      # reproducibility
3. from google.colab import drive; drive.mount('/content/drive')  # checkpoints + data on Drive
4. Save checkpoints every epoch (trainer save_strategy="epoch"); sessions can disconnect.
5. Download artefacts (model, metrics, logs) at the end; record the GPU type and versions.
```

## 7. Interpretation
- **Memory:** downcasting and categories typically cut a mixed DataFrame's memory several-fold (string columns dominate). This is often the difference between "fits in 16 GB" and "swap/crash".
- **Subsample vs full:** if the 2% and 20% runs reach similar R², use small subsamples for feature/model iteration, and reserve the full data for the final comparisons. If the scores differ a lot (learning-curve regime), the model is data-hungry. Then iterate on larger samples, or prioritise data volume.
- **The GPU pattern:** mixed precision and gradient accumulation let models that would not fit in 6 GB train with acceptable batch sizes. Peak-memory logging guides the batch-size choice.

## 8. Decision framework
- **IF** the workload is classical ML on < 10⁶ rows → CPU, `n_jobs=-1`, early stopping; the GPU is unnecessary.
- **IF** it is CNN/transformer training → the local GPU for small/medium models (fp16, batch ≤ what fits, accumulation); Colab for larger ones.
- **IF** the data exceed RAM → column selection, dtype optimisation, Parquet with column pruning, chunked processing, Polars/DuckDB.
- **IF** experiments take hours → a subsample for iteration; fewer folds while exploring; Optuna with pruning/halving ([hyperparameter_optimization.md](../06_model_improvement/hyperparameter_optimization.md)).
- **IF** CPU over-subscription (everything slow despite many cores) → set `n_jobs` in one place only (CV *or* the model).

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Developing on the full dataset from day 1 | Slow iteration | Subsample, then scale up |
| float64 and object columns everywhere | Memory blow-up | Downcast; categories; Parquet |
| GPU for scikit-learn models | No speed-up | CPU parallelism |
| Batch size too large → CUDA OOM | Crashes | Mixed precision; accumulation; smaller batches |
| Nested `n_jobs` | Thread contention | Parallelise one level only |
| Colab without checkpoints | Lost work | Save to Drive regularly |

## 10. Checklist
- [ ] Stage timings and memory profiled.
- [ ] dtypes optimised; Parquet; only the needed columns.
- [ ] A subsample for iteration; full data for the final decisions.
- [ ] Early stopping / iteration caps.
- [ ] Correct hardware per workload; mixed precision on the GPU.
- [ ] A Colab plan with versions and checkpoints for heavy jobs.

## 11. References
- pandas, *Scaling to large datasets*. https://pandas.pydata.org/docs/user_guide/scale.html
- PyTorch, *Automatic Mixed Precision* and *Performance Tuning Guide*. https://pytorch.org/docs/stable/amp.html · https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- Hugging Face, *Methods and tools for efficient training on a single GPU*. https://huggingface.co/docs/transformers/perf_train_gpu_one
- XGBoost, *GPU support*; LightGBM, *GPU tutorial*. https://xgboost.readthedocs.io/en/stable/gpu/index.html · https://lightgbm.readthedocs.io/en/stable/GPU-Tutorial.html
- Google Colab FAQ (resource limits). https://research.google.com/colaboratory/faq.html
