# Experiment Tracking and Comparison

> **Folder:** 07_experimentation · **Prev:** [iterative_improvement.md](../06_model_improvement/iterative_improvement.md) · **Next:** [reproducibility.md](reproducibility.md) · **Related:** [21_practical_projects/systematic_experimentation.md](../21_practical_projects/systematic_experimentation.md), [20_advanced_methodologies](../20_advanced_methodologies/) (dataset/model versioning)

---

## 1. Purpose
Record every experiment (what was changed, on which data and code, with which result and conclusion) so that results are comparable, decisions are traceable and work is not repeated.

## 2. When to use it
From the first baseline onwards. A spreadsheet-level log is enough for solo projects. A tool such as MLflow pays off once there are tens of runs, several people or artefacts (models, plots) to keep.

## 3. Conceptual foundation
A useful experiment record contains:

| Field | Why |
|---|---|
| ID, timestamp, author | Traceability |
| **Hypothesis** and **change** relative to the parent experiment | Makes the result interpretable |
| Code version (git commit hash) | Reproducibility |
| Data version (file hash / snapshot ID / query + date) | Reproducibility; detects silent data changes |
| Configuration (features, model, hyperparameters, seeds) | Rerun and compare |
| Validation scheme ID | Only results under the same scheme are comparable |
| Metrics (mean ± std per fold; per segment if relevant) | Decisions |
| Artefacts (plots, predictions, model file) | Inspection |
| **Conclusion / decision** (accepted / rejected / inconclusive and why) | The most often missing, most valuable field |

**Comparability rule:** compare experiments only when they share the data version, the validation scheme and the metric definition. When any of these change, re-run the reference (baseline and current best) under the new conditions.

## 4. Assumptions and limitations
- Tools record what you log. They cannot make an ill-designed comparison valid.
- Logging every tuning trial in a tracker can drown out the important runs. Log tuning *studies* as one parent run with child runs, or log only the summary.

## 5. Step-by-step procedure
1. **Decide the log format** (CSV/JSON lines for solo work; MLflow for more).
2. **Define a validation-scheme ID** (e.g. `tss5_dev2023` = TimeSeriesSplit 5 folds on dev data v2023-06).
3. **Write a `run_experiment(config)` function** that trains, evaluates and logs everything automatically, so that no manual copying is needed.
4. **Before each run,** write the hypothesis. **After it,** write the conclusion.
5. **Review the log regularly:** the leaderboard under one scheme, the trend of the best score, rejected ideas.

## 6. Python implementation

```python
import hashlib
import json
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_validate

from cookbook_data import load_clean_process_yield

LOG = Path("experiment_log.jsonl")          # one JSON object per line: append-only, diff-friendly

def data_fingerprint(df: pd.DataFrame) -> str:
    """Stable hash of the data content (detects silent changes in the data)."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:12]

def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "no-git"

def run_experiment(name, hypothesis, model, X, y, cv, cv_id, config, parent=None):
    t0 = time.time()
    r = cross_validate(model, X, y, cv=cv, scoring={"MAE": "neg_mean_absolute_error", "R2": "r2"},
                       return_train_score=True)
    rec = {"id": uuid.uuid4().hex[:8], "time": time.strftime("%Y-%m-%d %H:%M:%S"), "name": name,
           "parent": parent, "hypothesis": hypothesis, "git": git_commit(),
           "data_hash": data_fingerprint(pd.concat([X, y], axis=1)), "cv_id": cv_id, "config": config,
           "val_MAE": float(-r["test_MAE"].mean()), "val_MAE_std": float(r["test_MAE"].std()),
           "train_MAE": float(-r["train_MAE"].mean()), "val_R2": float(r["test_R2"].mean()),
           "fold_MAE": (-r["test_MAE"]).round(4).tolist(), "seconds": round(time.time() - t0, 1),
           "conclusion": None}
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec

def set_conclusion(run_id, text):
    rows = [json.loads(l) for l in LOG.read_text().splitlines()]
    for r in rows:
        if r["id"] == run_id:
            r["conclusion"] = text
    LOG.write_text("".join(json.dumps(r) + "\n" for r in rows))

df = load_clean_process_yield()
train = df.iloc[: int(0.8 * len(df))]
num = ["temperature_c", "pressure_bar", "feed_rate_kg_h", "residence_time_min", "catalyst_age_days", "impurity_ppm"]
cat = ["reactor", "catalyst", "shift", "supplier"]
X = train[num + cat].astype({c: "category" for c in cat}); y = np.log(train["yield_kg"])
cv, cv_id = TimeSeriesSplit(5), "tss5_dev_v1"

a = run_experiment("hgb default", "reference", HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=0),
                   X, y, cv, cv_id, {"model": "HGB", "params": "default"})
b = run_experiment("hgb small trees", "additive structure → shallow trees generalise better",
                   HistGradientBoostingRegressor(categorical_features="from_dtype", max_leaf_nodes=4,
                                                 learning_rate=0.1, max_iter=400, random_state=0),
                   X, y, cv, cv_id, {"model": "HGB", "max_leaf_nodes": 4, "lr": 0.1, "max_iter": 400}, parent=a["id"])
gain = np.array(a["fold_MAE"]) - np.array(b["fold_MAE"])
set_conclusion(b["id"], f"{'accepted' if (gain > 0).mean() >= 0.8 else 'inconclusive'}: paired gain {gain.mean():.4f}")

board = pd.read_json(LOG, lines=True)
print(board[board.cv_id == cv_id][["id", "name", "val_MAE", "val_MAE_std", "train_MAE", "conclusion"]].round(4))
LOG.unlink()                                # clean up the demo log
```

MLflow equivalent. It is a local tool: no server is needed, and runs are stored in `./mlruns`, browsable with `mlflow ui`:

```python
# not-run (requires `pip install mlflow`)
import mlflow
mlflow.set_experiment("process-yield")
with mlflow.start_run(run_name="hgb small trees"):
    mlflow.set_tags({"hypothesis": "shallow trees generalise better", "cv_id": cv_id, "git": git_commit()})
    mlflow.log_params({"max_leaf_nodes": 4, "learning_rate": 0.1, "max_iter": 400})
    mlflow.log_metrics({"val_MAE": b["val_MAE"], "val_MAE_std": b["val_MAE_std"]})
    # mlflow.sklearn.log_model(fitted_model, name="model")   # store the artefact
    # mlflow.log_artifact("reports/residuals.png")
```

## 7. Interpretation
- **The leaderboard (one `cv_id` only):** sort by the primary metric and read it with `val_MAE_std`. Look at `train_MAE` to spot overfitting configurations.
- **Parent links** make the lineage visible: which change produced which gain.
- **A changed `data_hash`** between runs with the same name means the data changed underneath you. Treat the comparisons as invalid until you re-run the references.

## 8. Decision framework
- **IF** solo, fewer than ~50 runs, no large artefacts → a JSONL/CSV log + git is sufficient.
- **IF** a team, many runs, model artefacts, or a need for UI comparison → MLflow (local file store), or a similar tool.
- **IF** the validation scheme or the data version changes → start a new `cv_id`; re-run the baseline and the current best.
- **IF** tuning with many trials → one parent run per study; log the best params and a summary, and keep the full trial table as an artefact.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Logging metrics without the configuration | Can't reproduce the best run | Log the config + git + data hash automatically |
| Comparing runs across validation schemes | Invalid conclusions | `cv_id` field; filter by it |
| No hypothesis/conclusion | Log becomes noise | Mandatory fields |
| Manual copy of numbers into a spreadsheet | Errors, gaps | Log from code |
| Uncommitted code changes during runs | Runs not reproducible | Commit before runs (or log the diff) |

## 10. Checklist
- [ ] Log format chosen; `run_experiment` logs automatically.
- [ ] Each run: hypothesis, change, git hash, data hash, cv_id, config, metrics ± std, conclusion.
- [ ] Leaderboard reviewed within one cv_id.
- [ ] References re-run when the data/scheme change.

## 11. References
- MLflow documentation, *Tracking*. https://mlflow.org/docs/latest/tracking.html
- DVC documentation, *Experiment management*. https://dvc.org/doc/user-guide/experiment-management
- Pineau, J. et al. (2021). "Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)." *JMLR* 22(164). https://jmlr.org/papers/v22/20-303.html
- pandas, `hash_pandas_object`. https://pandas.pydata.org/docs/reference/api/pandas.util.hash_pandas_object.html
