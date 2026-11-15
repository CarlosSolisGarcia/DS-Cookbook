# Metrics Reference

> **Folder:** 08_quick_references · A one-page lookup. The reasoning lives in [regression evaluation](../03_regression/evaluation.md), [classification evaluation](../04_classification/evaluation.md), [ts evaluation](../05_time_series/evaluation.md), and in the evaluation recipes of the extension sections (09–15).
>
> *Quick references use a compact format instead of the 11-section recipe structure.*

---

## 1. How to choose a metric (decision summary)
1. **What decision does the output support?** (threshold, ranking under capacity, quantity planning, risk communication)
2. **What does each error type cost?** (symmetric? are large errors much worse? FP vs FN?)
3. **Is there a reference to compare against?** (baseline → skill scores, MASE, PR-AUC vs prevalence)
4. Choose **one primary** metric for decisions and **2–3 secondary** metrics for context. Report them with uncertainty.

## 2. Regression
| Metric | sklearn | Optimal point forecast | Use | Avoid / caveat |
|---|---|---|---|---|
| MAE | `mean_absolute_error`, `"neg_mean_absolute_error"` | Median | Costs ∝ \|error\|; robust | Hides rare large errors |
| RMSE | `root_mean_squared_error`, `"neg_root_mean_squared_error"` | Mean | Large errors costly | Dominated by outliers |
| MSE | `mean_squared_error` | Mean | Optimisation | Squared units |
| R² | `r2_score`, `"r2"` | Mean | Relative to the mean baseline | Not comparable across datasets; can be < 0 |
| MAPE | `mean_absolute_percentage_error` | Biased low | All y ≫ 0, relative errors | Near-zero y; favours under-prediction |
| WAPE | Σ\|e\|/Σ\|y\| (manual) | — | Volume-weighted relative error | Per-item errors hidden |
| MedAE | `median_absolute_error` | — | Typical error | Ignores the tail |
| Max error / P90 AE | `max_error` / quantile | — | Worst case / tail | Unstable (max) |
| Pinball (α) | `mean_pinball_loss` | α-quantile | Quantiles, asymmetric costs | — |
| Poisson/Gamma/Tweedie deviance | `mean_poisson_deviance`, … | Mean (GLM family) | Counts / positive skewed | Requires y ≥ 0 / > 0 |
| Interval coverage & width | manual | — | Prediction intervals | Check per segment |

## 3. Classification
| Metric | sklearn | Threshold-free? | Use | Caveat |
|---|---|---|---|---|
| Accuracy | `accuracy_score` | No | Balanced classes, equal costs | Meaningless under imbalance |
| Balanced accuracy | `balanced_accuracy_score` | No | Imbalance-robust summary | Not cost-aware |
| Precision / Recall / F1 / F-β | `precision_score`, `recall_score`, `f1_score`, `fbeta_score` | No | Operating-point quality | F1 assumes equal P/R importance |
| Specificity | `recall_score(pos_label=0)` | No | False-alarm control | Tiny changes are meaningful when negatives are numerous |
| MCC | `matthews_corrcoef` | No | All four cells summarised | Not cost-aware |
| ROC-AUC | `roc_auc_score` | Yes | Ranking, balanced-ish classes | Optimistic under heavy imbalance |
| PR-AUC (average precision) | `average_precision_score` | Yes | Ranking, rare positives | Compare with the prevalence |
| Precision@k / Recall@k | manual | — | Capacity-constrained actions | Depends on k and period |
| Log-loss | `log_loss` | Yes (probabilities) | Probability quality | Punishes confident errors heavily |
| Brier | `brier_score_loss` | Yes (probabilities) | Probability quality | Compare with π(1−π) |
| Expected cost | manual | No | Business decision | Needs a cost matrix |
| Multiclass averages | `average="macro" / "weighted" / "micro"` | — | Macro: all classes equal; weighted: by support | State the choice |

## 4. Time series / forecasting
| Metric | Definition | Use | Caveat |
|---|---|---|---|
| MAE, RMSE | as for regression, **per horizon** | Single series | Scale-dependent |
| MASE | MAE / in-sample seasonal-naive MAE (`cookbook_utils.mase`) | Across series; vs naive | State the seasonal period m |
| sMAPE | 2\|e\|/(\|y\|+\|ŷ\|) (`cookbook_utils.smape`) | Competitions (M4) | Several definitions; unstable near 0 |
| Bias | mean(ŷ − y) | Systematic over/under-forecast | Can be 0 with large errors |
| Coverage / width / pinball | per horizon | Intervals, quantiles | Drift degrades coverage |

## 5. Unsupervised and other families (see the extension sections)
| Family | Metrics | Recipe |
|---|---|---|
| Clustering | Silhouette, Davies–Bouldin, Calinski–Harabasz; ARI/NMI when labels exist; stability (ARI across bootstraps) | [09 cluster validation](../09_unsupervised_learning/cluster_validation.md) |
| Anomaly detection | Precision@k, PR-AUC on scarce labels; alarm rate; time to detection; event-level recall | [10 evaluation](../10_anomaly_detection/evaluation_and_thresholds.md) |
| Dimensionality reduction | Explained variance; reconstruction error; trustworthiness; downstream CV score | [11](../11_dimensionality_reduction/) |
| NLP classification | Macro-F1, per-class report; for ranking/retrieval: MRR, nDCG | [12](../12_natural_language_processing/) |
| Computer vision | Accuracy/F1 per class; IoU; Dice; mAP@[.5:.95] | [13](../13_computer_vision/) |
| Recommenders | Precision@k, Recall@k, MAP@k, nDCG@k, hit rate, coverage, novelty | [14](../14_recommender_systems/) |
| Graph ML | Node/graph classification metrics; link prediction: ROC-AUC, AP, Hits@k, MRR | [15](../15_graph_machine_learning/) |

## 6. Red flags
- A metric improvement without a matching improvement in the *decision* metric.
- R² or accuracy reported alone.
- Metrics from a single split with no uncertainty.
- MAPE on data with near-zero values.
- PR-AUC reported without the prevalence.
- Different validation schemes behind compared numbers.

## 7. References
- scikit-learn User Guide, *Metrics and scoring*. https://scikit-learn.org/stable/modules/model_evaluation.html
- Hyndman, R. J. & Koehler, A. B. (2006). "Another look at measures of forecast accuracy." *IJF* 22(4). https://doi.org/10.1016/j.ijforecast.2006.03.001
- Saito, T. & Rehmsmeier, M. (2015). *PLoS ONE* 10(3): e0118432. https://doi.org/10.1371/journal.pone.0118432
- Gneiting, T. (2011). "Making and Evaluating Point Forecasts." *JASA* 106(494). https://doi.org/10.1198/jasa.2011.r10138
