# Reproducibility and Project Organisation

> **Folder:** 07_experimentation · **Prev:** [experiment_tracking.md](experiment_tracking.md) · **Next:** [metrics_reference.md](../08_quick_references/metrics_reference.md) · **Related:** [project_lifecycle.md](../00_methodology/project_lifecycle.md) §6, [20_advanced_methodologies](../20_advanced_methodologies/) (dataset & model versioning, reproducible pipelines)

---

## 1. Purpose
Make results **re-creatable**, by you in six months or by someone else. Control the randomness, the data versions, the code, the environment, the configuration and the preprocessing consistency between training and inference.

## 2. When to use it
From day 1 of every project. Retrofitting reproducibility is much harder than building it in.

## 3. Conceptual foundation
Sources of irreproducibility and their controls:

| Source | Control |
|---|---|
| **Randomness** (splits, model init, subsampling, resampling) | Explicit `random_state` everywhere; one seed in the config; `np.random.default_rng(seed)` instead of the global state |
| **Data changes** (new extracts, edited files) | Immutable `data/raw`; version files by hash/snapshot; record the query + date |
| **Code changes** | git; commit before experiments; record the commit hash |
| **Environment** (library versions change behaviour and defaults) | Pinned `requirements.txt`/lock file; record the versions in the run |
| **Configuration drift** (notebook cells run out of order) | Config files + script entry points; restart-and-run-all before trusting notebook results |
| **Preprocessing inconsistency** (training vs inference) | One fitted `Pipeline` object persisted and loaded for inference |
| **Non-determinism in parallel/GPU code** | Accept tiny differences, or set deterministic flags (PyTorch: `torch.use_deterministic_algorithms(True)`) |

**Levels of reproducibility:** (1) same code + data + environment → identical numbers; (2) different seeds → results within the reported variability (robustness). Report (2): a result that only holds for one seed is not a result.

## 4. Assumptions and limitations
- Bit-identical results across machines and OSs are not always achievable (BLAS threading, GPU kernels). Aim for statistical reproducibility, and document the platform.
- Pickled models (`joblib`) are tied to library versions. Store the versions with the model, and retrain rather than loading a model across major version changes.

## 5. Step-by-step procedure
1. **Project layout** as in [project_lifecycle.md](../00_methodology/project_lifecycle.md) §6; `data/raw` read-only.
2. **Environment:** a virtual environment; pinned requirements; record `sklearn.__version__` etc. in the logs.
3. **Seeds:** a single `SEED` in the config, passed to every splitter, model and sampler.
4. **Data versioning:** a hash of the input files/DataFrames logged with each run; DVC or dated snapshots for large data.
5. **Pipelines:** all fitted preprocessing inside one sklearn `Pipeline`; persist it with `joblib` together with metadata.
6. **Configuration:** YAML/JSON configs; scripts read the config; notebooks only for exploration.
7. **Seed-robustness check:** rerun the final configuration with 3–5 seeds; report the mean ± std.
8. **Final reproduction test:** clone into a fresh environment, run `train.py` and compare the metrics.

## 6. Python implementation

```python
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from cookbook_data import load_clean_process_yield

CONFIG = {                       # in a project: configs/hgb_v3.json, loaded by train.py
    "seed": 42,
    "features": {"num": ["temperature_c", "pressure_bar", "feed_rate_kg_h", "residence_time_min",
                         "catalyst_age_days", "impurity_ppm"],
                 "cat": ["reactor", "catalyst", "shift", "supplier"]},
    "model": {"learning_rate": 0.05, "max_iter": 500, "max_leaf_nodes": 15},
    "cv": {"n_splits": 5},
}

def build_pipeline(cfg):
    num, cat = cfg["features"]["num"], cfg["features"]["cat"]
    pre = ColumnTransformer([("num", "passthrough", num),
                             ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), cat)])
    model = HistGradientBoostingRegressor(categorical_features=list(range(len(num), len(num) + len(cat))),
                                          random_state=cfg["seed"], **cfg["model"])
    return Pipeline([("pre", pre), ("model", model)])

df = load_clean_process_yield()
X = df[CONFIG["features"]["num"] + CONFIG["features"]["cat"]]
y = np.log(df["yield_kg"])

# Determinism check: same seed → identical scores
cv = KFold(CONFIG["cv"]["n_splits"], shuffle=True, random_state=CONFIG["seed"])
s1 = cross_val_score(build_pipeline(CONFIG), X, y, cv=cv, scoring="neg_mean_absolute_error")
s2 = cross_val_score(build_pipeline(CONFIG), X, y, cv=cv, scoring="neg_mean_absolute_error")
print("identical with same seed:", np.allclose(s1, s2))

# Seed robustness: vary the seed of BOTH the splitter and the model
seed_scores = []
for seed in range(5):
    cfg = dict(CONFIG, seed=seed)
    cv_s = KFold(5, shuffle=True, random_state=seed)
    seed_scores.append(-cross_val_score(build_pipeline(cfg), X, y, cv=cv_s, scoring="neg_mean_absolute_error").mean())
print(f"MAE(log) across seeds: {np.mean(seed_scores):.4f} ± {np.std(seed_scores):.4f}")
```

```python
# Persist the FULL fitted pipeline + metadata (preprocessing consistency between training and inference)
out = Path("artifacts"); out.mkdir(exist_ok=True)
pipe = build_pipeline(CONFIG).fit(X, y)
joblib.dump(pipe, out / "model.joblib")
meta = {"config": CONFIG, "python": sys.version.split()[0], "platform": platform.platform(),
        "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "data_hash": pd.util.hash_pandas_object(pd.concat([X, y], axis=1)).sum().item() % 10**12,
        "feature_columns": list(X.columns), "target_transform": "log"}
(out / "model_meta.json").write_text(json.dumps(meta, indent=2))

# Inference: load and predict with the SAME preprocessing
loaded = joblib.load(out / "model.joblib")
new_batch = X.iloc[:3].copy()
print(np.exp(loaded.predict(new_batch[meta["feature_columns"]])).round(1))
import shutil; shutil.rmtree(out)           # clean up the demo
```

```text
# requirements.txt (pin exact versions you validated with; see the repository's requirements.txt)
numpy==2.5.3
pandas==3.0.5
scikit-learn==1.9.1
```

## 7. Interpretation
- **"identical with same seed: True"** confirms the pipeline is deterministic given the seed. If it is False, some component lacks a seed, or parallel execution introduces non-determinism.
- **The std across seeds** is the robustness band. Differences between configurations smaller than this band are not meaningful.
- **Metadata next to the model** lets you detect version mismatches at load time and reproduce the training run.

## 8. Decision framework
- **IF** data are small and static → file hashes + git are enough. **IF** large or frequently updated → DVC, or a data catalogue with snapshot IDs.
- **IF** using notebooks → restart-and-run-all before reporting; move stable code to `src/`.
- **IF** a result changes with the seed beyond the reported std → it is not robust. Increase the data/folds, simplify, or report the variability.
- **IF** deploying → persist one pipeline object; validate the input schema at inference (same columns and dtypes).
- **IF** a library's major version changes → retrain and re-validate; do not trust old pickles.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Missing `random_state` in splitters/models | Different results per run | One `SEED`, passed everywhere |
| Preprocessing re-implemented at inference | Training–serving skew | Persist the full pipeline |
| Editing raw data | Irreproducible, lost provenance | Immutable raw + cleaning code |
| Unpinned environment | Silent behaviour changes | Pinned requirements; record the versions |
| Out-of-order notebook state | Unrepeatable results | Scripts + configs; restart-and-run-all |
| Reporting one lucky seed | Overclaiming | Seed-robustness check |

## 10. Checklist
- [ ] Project layout; immutable raw data.
- [ ] Pinned environment; versions recorded.
- [ ] Single seed passed to all random components; determinism verified.
- [ ] Data version/hash logged per run.
- [ ] Config files; scripts as entry points.
- [ ] Full pipeline persisted with metadata; inference uses it.
- [ ] Seed-robustness reported.
- [ ] Fresh-environment reproduction tested before final reporting.

## 11. References
- scikit-learn, *Controlling randomness* (common pitfalls). https://scikit-learn.org/stable/common_pitfalls.html#controlling-randomness
- scikit-learn, *Model persistence*. https://scikit-learn.org/stable/model_persistence.html
- Wilson, G. et al. (2017). "Good enough practices in scientific computing." *PLoS Comput Biol* 13(6): e1005510. https://doi.org/10.1371/journal.pcbi.1005510
- DVC documentation. https://dvc.org/doc
- PyTorch, *Reproducibility*. https://pytorch.org/docs/stable/notes/randomness.html
- Cookiecutter Data Science (project structure). https://cookiecutter-data-science.drivendata.org/
