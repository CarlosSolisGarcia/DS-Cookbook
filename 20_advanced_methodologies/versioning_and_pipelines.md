# Dataset and Model Versioning, Experiment Management and Reproducible Pipelines

> **Folder:** 20_advanced_methodologies · **Prev:** [efficient_development.md](efficient_development.md) · **Next:** [kaggle_workflow.md](../21_practical_projects/kaggle_workflow.md) · **Related:** [reproducibility.md](../07_experimentation/reproducibility.md) (seeds, environments, persisted pipelines: the basics), [experiment_tracking.md](../07_experimentation/experiment_tracking.md)

---

## 1. Purpose
Scale the reproducibility basics to projects whose data change, whose models are deployed and whose experiments are numerous. Version the data and models together with the code, organise the work as a reproducible pipeline (stages with explicit inputs and outputs), and keep a registry of which model version runs where.

## 2. When to use it
- Data are refreshed (new extracts, new labels); you must reproduce "the model from March".
- Several people or many runs; models are deployed and must be traceable.
- Multi-step workflows (extract → clean → features → train → evaluate) that you rerun often.

## 3. Conceptual foundation
| Concern | Lightweight solution | Tool-based solution |
|---|---|---|
| Code version | git (commit hash in every run) | git |
| Data version | Immutable, dated snapshots + content hashes in a manifest | **DVC** (data tracked by hash, stored in remote storage, linked to git commits) |
| Pipeline | A `Makefile` or a Python CLI with stages; each stage reads/writes declared files | **DVC pipelines** (`dvc.yaml`: stages, deps, outs, params) |
| Parameters | YAML/JSON config committed to git | `params.yaml` + DVC/MLflow param tracking |
| Experiments | JSONL log ([experiment_tracking.md](../07_experimentation/experiment_tracking.md)) | MLflow tracking, DVC experiments |
| Models | `models/<name>/<version>/` with model + metadata (data hash, git hash, metrics) | **MLflow Model Registry** (versions, stages/aliases) |
| Environment | Pinned `requirements.txt` / lock files | conda-lock, uv/pip-tools; Docker (only if needed) |

**Lineage principle:** every model artefact should answer *which code (commit), which data (hash), which parameters, which environment, and which evaluation*.

## 4. Assumptions and limitations
- Tooling adds overhead. For a solo project with static data, a manifest + git + JSONL logs are enough.
- DVC/MLflow are local-first tools (no cloud needed): DVC remotes can be a local folder or a network drive, and MLflow can use a local `mlruns` directory.

## 5. Step-by-step procedure
1. **Freeze the raw data snapshots;** write a data manifest (file, hash, date, source).
2. **Structure the pipeline in stages** with explicit inputs and outputs (`src/` scripts reading `params.yaml`).
3. **Track the parameters** in one config file; never hard-code them in notebooks.
4. **Every training run writes:** the model, the metrics, the config, the data hash, the git hash and the library versions.
5. **Register model versions** with their evaluation report; mark which version is "production".
6. **Reproduce:** check out a commit + its data version → rerun the pipeline → identical artefacts/metrics.

## 6. Python implementation: a lightweight, tool-free version

```python
import hashlib, json, platform, subprocess, tempfile, time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from cookbook_data import load_clean_process_yield

root = Path(tempfile.mkdtemp())                                       # stands in for the project directory
(root / "data/raw").mkdir(parents=True); (root / "models").mkdir()

def file_hash(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()[:16]

# 1. Data snapshot + manifest
raw = root / "data/raw/batches_2024-06-01.parquet"
load_clean_process_yield().to_parquet(raw)
manifest = {"files": [{"path": str(raw.relative_to(root)), "sha256_16": file_hash(raw), "rows": len(pd.read_parquet(raw)),
                       "extracted": "2024-06-01", "source": "historian export v3"}]}
(root / "data/manifest.json").write_text(json.dumps(manifest, indent=2))

# 2–4. A pipeline stage driven by params, writing a fully described model version
params = {"features": ["temperature_c", "pressure_bar", "feed_rate_kg_h", "residence_time_min"], "max_iter": 300, "seed": 42}
def train_stage(params, data_path, out_dir):
    from sklearn.ensemble import HistGradientBoostingRegressor
    import joblib
    d = pd.read_parquet(data_path)
    m = HistGradientBoostingRegressor(max_iter=params["max_iter"], random_state=params["seed"]).fit(d[params["features"]], np.log(d["yield_kg"]))
    version = time.strftime("%Y%m%d-%H%M%S")
    vdir = out_dir / "yield_model" / version; vdir.mkdir(parents=True)
    joblib.dump(m, vdir / "model.joblib")
    try:
        git = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        git = "no-git"
    meta = {"version": version, "git": git, "data": manifest["files"][0], "params": params,
            "env": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__},
            "metrics": {"train_rows": len(d)}, "status": "candidate"}
    (vdir / "model_card.json").write_text(json.dumps(meta, indent=2))
    return vdir

vdir = train_stage(params, raw, root / "models")
print(json.loads((vdir / "model_card.json").read_text())["data"])
print("model versions:", [p.name for p in (root / "models/yield_model").iterdir()])
```

DVC equivalent: declarative stages that rerun only when their dependencies change.

```text
# dvc.yaml
stages:
  clean:
    cmd: python src/clean.py
    deps: [src/clean.py, data/raw/batches.parquet]
    outs: [data/interim/clean.parquet]
  train:
    cmd: python src/train.py
    deps: [src/train.py, data/interim/clean.parquet]
    params: [features, max_iter, seed]          # read from params.yaml
    outs: [models/model.joblib]
    metrics: [reports/metrics.json]
# Usage: `dvc repro` (rerun changed stages), `dvc exp run -S max_iter=500` (experiment), `dvc push` (store data)
```

MLflow model registry (local):

```python
# not-run (requires `pip install mlflow`)
import mlflow
mlflow.set_tracking_uri("file:./mlruns")
with mlflow.start_run():
    mlflow.log_params(params); mlflow.log_metric("cv_mae", 0.12)
    info = mlflow.sklearn.log_model(sk_model=model, name="model", registered_model_name="yield_model")
# client = mlflow.MlflowClient(); client.set_registered_model_alias("yield_model", "production", version=info.registered_model_version)
```

## 7. Interpretation
- **The manifest** ties the model to an exact data file (its hash). If someone overwrites the file, the hash mismatch reveals it.
- **The model card** answers the lineage questions. Its `status` (candidate/production/retired) is the poor man's registry.
- **DVC/MLflow** automate exactly these records, plus caching, remote storage and UIs, once the project grows.

## 8. Decision framework
- **IF** solo, static data → git + manifest + JSONL experiment log + model cards.
- **IF** the data change regularly or are large → DVC (data versions linked to commits; pipelines with caching).
- **IF** a team, or many runs and deployed models → MLflow tracking + registry (local or a shared server).
- **IF** deployment to other machines → pinned environments; Docker only if environment drift becomes a real problem.
- **IF** a result must be reproduced for an audit → commit + data hash + params + environment, all recorded *at training time*.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Overwriting data files in place | Irreproducible models | Immutable dated snapshots + hashes |
| Parameters scattered in notebooks | Can't rerun the best model | One params file |
| Models saved without metadata | "Which data/code made this?" | Model cards / registry |
| Heavy tooling for a tiny project | Overhead > benefit | Start lightweight, adopt tools when needed |

## 10. Checklist
- [ ] Raw snapshots immutable; manifest with hashes.
- [ ] Pipeline stages with explicit inputs/outputs; params in one file.
- [ ] Each model version with git hash, data hash, params, env, metrics, status.
- [ ] A reproduction test from a clean checkout.
- [ ] Tools (DVC/MLflow) adopted when the scale justifies them.

## 11. References
- DVC documentation (data versioning, pipelines, experiments). https://dvc.org/doc
- MLflow documentation (tracking, model registry). https://mlflow.org/docs/latest/
- Mitchell, M. et al. (2019). "Model Cards for Model Reporting." *FAT\**. https://arxiv.org/abs/1810.03993
- Gebru, T. et al. (2021). "Datasheets for Datasets." *CACM* 64(12). https://arxiv.org/abs/1803.09010
- Sculley, D. et al. (2015). "Hidden Technical Debt in Machine Learning Systems." *NeurIPS*.
