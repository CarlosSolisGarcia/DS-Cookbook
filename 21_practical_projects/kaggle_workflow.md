# Practical Project & Competition Playbook: Overview

> **Folder:** 21_practical_projects · **Start here for any Kaggle competition or independent project.** · **Next:** [project_reconnaissance.md](project_reconnaissance.md) · **Related:** [project_lifecycle.md](../00_methodology/project_lifecycle.md) (the general lifecycle; this playbook is its competition/portfolio version)

---

## 1. Purpose
Give you a systematic, problem-family-agnostic routine for unfamiliar competitions and portfolio projects. It takes you from a fresh problem page to a robust final solution and a written retrospective, so that every project leaves you with skill, not just a score.

## 2. When to use it
- Kaggle (or similar) competitions, hackathons, take-home assignments, personal portfolio projects.
- Any project with a fixed evaluation metric and a hidden or future test set.

## 3. Conceptual foundation
**Competitions vs real projects:**
| Aspect | Competition | Real project |
|---|---|---|
| Metric | Given, fixed | Must be derived from costs ([problem_framing.md](../00_methodology/problem_framing.md)) |
| Test set | Hidden, fixed; public/private leaderboard (LB) split | Future data; drifts |
| Leakage | Sometimes exploitable (and often banned) | Always a bug |
| Goal | Rank on the private LB | Value, robustness, maintainability |
| Complexity | Large ensembles are acceptable | Usually penalised (maintenance cost) |

**The two biggest risks in competitions:**
1. **A local validation that does not mimic the hidden test split.** The CV then disagrees with the leaderboard, and you optimise noise.
2. **Overfitting the public leaderboard:** choosing models by public-LB score, which is computed on a small sample. The private LB (the final ranking) then "shakes up".

**The seven-stage playbook:**
| Stage | Recipe | Output |
|---|---|---|
| 1. Reconnaissance | [project_reconnaissance.md](project_reconnaissance.md) | Notes: target, metric, data structure, test construction, leakage risks |
| 2. Baseline | [baseline_development.md](baseline_development.md) | A valid end-to-end submission + local CV score |
| 3. Validation design | [competition_validation.md](competition_validation.md) | A CV scheme that correlates with the LB |
| 4. Systematic experimentation | [systematic_experimentation.md](systematic_experimentation.md) | An experiment log; a steady improvement trajectory |
| 5. Advanced improvement | [advanced_improvement.md](advanced_improvement.md) | Features, models, ensembles, tuning |
| 6. Final solution | [final_solution.md](final_solution.md) | Selected submissions; refit; verified files; write-up |
| 7. Retrospective | [project_retrospective.md](project_retrospective.md) | Lessons learned; cookbook updates |

## 4. Assumptions and limitations
- Competition rules (external data, team size, code requirements, leakage exploitation) override everything here. Read them.
- Time budgets differ (a weekend vs three months). Scale the depth, not the order.

## 5. Step-by-step procedure (time allocation for a ~4-week effort)
| Week | Focus |
|---|---|
| 1 | Reconnaissance, EDA, baseline submission, validation scheme checked against the LB |
| 2 | Feature engineering and model families, logged experiments |
| 3 | Deeper features, tuning, diverse models for ensembling; error analysis |
| 4 | Ensembling/stacking, robustness checks, final selection, write-up |

**Daily habits:** one hypothesis per experiment; log everything; check the CV–LB correlation weekly; read the discussion forum and the public notebooks for *ideas*, not for blind copying.

## 6. Python implementation: project skeleton

```text
competition-name/
├── README.md               # recon notes, validation scheme, current best, submission log
├── data/raw/               # untouched competition files
├── src/
│   ├── features.py         # feature functions (train and test through the SAME code path)
│   ├── cv.py               # the frozen fold assignment + run_cv() returning OOF and test predictions
│   └── models/             # one file per model family
├── oof/                    # saved out-of-fold predictions per experiment (for ensembling)
├── subs/                   # submission files, named exp_id__cvscore.csv
└── experiments.csv         # the log (systematic_experimentation.md)
```

**Adaptations by problem family** (what changes in each stage):
| Family | Reconnaissance focus | Validation | Typical strong baseline | Main improvement levers |
|---|---|---|---|---|
| Tabular regression/classification | Feature meaning, IDs, groups, time columns, train/test drift | (Stratified/Group) K-fold matching the test split | GBM (LightGBM/HistGB) with minimal preprocessing | Feature engineering, target/aggregate encodings, ensembles |
| Time series | Forecast horizon, test period, available exogenous data | Rolling origin ending where the test starts | Seasonal naive, then GBM on lags/calendar | Lag/rolling features, multi-step strategy, ensembles |
| NLP | Text length, language, label noise, duplicates between train and test | Stratified K-fold; group by author/thread if relevant | TF-IDF + logistic regression | Pretrained transformers, fine-tuning, ensembles ([12_nlp](../12_natural_language_processing/nlp_workflow.md)) |
| Computer vision | Image sizes, classes, annotation format, near-duplicates | Stratified/grouped by source (patient, camera, site) | Pretrained CNN fine-tuned at low resolution | Augmentation, resolution, architectures, TTA ([13_cv](../13_computer_vision/cv_workflow.md)) |
| Recommender / ranking | Interaction types, user/item overlap with test, timing | Temporal split per user; ranking metrics | Popularity / recent popularity | Collaborative + content features, re-ranking ([14_recsys](../14_recommender_systems/recsys_workflow.md)) |
| Graph | Transductive vs inductive test, edge leakage | Edge/node splits consistent with the task | Node features + simple GNN or GBM on graph features | Graph features, GNN depth/architecture ([15_graph](../15_graph_machine_learning/graph_ml_workflow.md)) |
| Signals / sensors | Sampling rate, segment structure, unit IDs | Group by unit/recording | Window statistics + GBM | Spectral features, domain normalisation ([16](../16_industrial_sensor_data/sensor_data_workflow.md)) |

## 7. Interpretation
Use the table as a lookup. It tells you where to look first in reconnaissance and which baseline to build on day one. The playbook is iterative: stage 4 regularly sends you back to stage 3 (the validation is not trustworthy) or to stage 1 (the data are not understood).

## 8. Decision framework
- **IF** your CV and the public LB disagree in ranking → stop modelling and fix the validation ([competition_validation.md](competition_validation.md)).
- **IF** the public LB is computed on a small sample (a few hundred rows or few positives) → trust CV over the public LB.
- **IF** improvements stall for several days → do error analysis; read the forums for data insights; try a different model family for ensemble diversity.
- **IF** the last week is approaching → freeze feature work; focus on robust ensembling and selecting two final submissions (one "best CV", one "robust").
- **IF** it is a portfolio project → apply the same stages and add the [problem_framing.md](../00_methodology/problem_framing.md) step: why does this problem matter, and what decision would the model support?

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Jumping into complex models on day 1 | No reference; confusing results | Baseline submission first |
| Validation not mimicking the test split | CV–LB mismatch; leaderboard shake-up | Reconstruct the test construction |
| Selecting by the public LB | Overfitting a small sample | Select by CV; use the LB as a sanity check |
| Unlogged experiments | Can't reproduce the best submission | Log + saved OOF/test predictions |
| Copying public notebooks blindly | Hidden leakage/bugs; no learning | Understand, re-validate, cite |
| No retrospective | Same mistakes next time | Write the post-project notes |

## 10. Checklist
- [ ] Rules read (data, external resources, team, compute).
- [ ] Repository skeleton created.
- [ ] Stages 1–7 planned with a time budget.
- [ ] Weekly CV–LB correlation check.
- [ ] Two final submissions planned (best CV + robust).
- [ ] Retrospective scheduled.

## 11. References
- Kaggle, *How to Use Kaggle* / competition documentation (leaderboards, submissions). https://www.kaggle.com/docs/competitions
- Bojer, C. S. & Meldgaard, J. P. (2021). "Kaggle forecasting competitions: An overlooked learning opportunity." *IJF* 37(2), 587–603. https://doi.org/10.1016/j.ijforecast.2020.07.007
- Roelofs, R. et al. (2019). "A Meta-Analysis of Overfitting in Machine Learning." *NeurIPS 32*. On Kaggle public/private leaderboards. https://papers.nips.cc/paper/2019/hash/ee39e503b6bedf0c98c388b7e8589aca-Abstract.html
- Blum, A. & Hardt, M. (2015). "The Ladder: A Reliable Leaderboard for Machine Learning Competitions." *ICML*. https://arxiv.org/abs/1502.04585
