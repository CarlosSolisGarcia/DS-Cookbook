# 20 — Advanced Cross-Cutting Methodologies: Map and Relationship to the Core Cookbook

> **Folder:** 20_advanced_methodologies · These methods apply across problem families. Several of them already have core recipes in sections 03–07. This page tells you **where each topic lives**, so nothing is duplicated.

| Methodology | Core recipe (start here) | Advanced / cross-family recipe | When you need the advanced part |
|---|---|---|---|
| Model interpretability: SHAP, permutation importance, PDP/ICE | [06 model_interpretation.md](../06_model_improvement/model_interpretation.md) | [robustness_and_sensitivity.md](robustness_and_sensitivity.md) (sensitivity analysis); [10 root_cause_analysis.md](../10_anomaly_detection/root_cause_analysis.md) (explaining anomalies) | Explaining unsupervised scores; global sensitivity of simulators/surrogates |
| Probability calibration | [04 calibration.md](../04_classification/calibration.md) | [uncertainty_estimation.md](uncertainty_estimation.md) (prediction sets) | Calibrated *sets* of classes; calibration under shift |
| Uncertainty estimation | [03 uncertainty.md](../03_regression/uncertainty.md) (regression intervals), [05 evaluation.md](../05_time_series/evaluation.md) (forecast intervals) | [uncertainty_estimation.md](uncertainty_estimation.md) | Classification sets, ensembles, deep-learning uncertainty, epistemic vs aleatoric |
| Robustness and sensitivity analysis | [06 model_interpretation.md](../06_model_improvement/model_interpretation.md) | [robustness_and_sensitivity.md](robustness_and_sensitivity.md) | Stress tests, perturbations, Sobol indices |
| Data drift and concept drift; model monitoring | [19 distribution_shift.md](../19_advanced_tabular_ml/distribution_shift.md) (train vs target) | [drift_and_monitoring.md](drift_and_monitoring.md) | Models in production |
| Domain adaptation; transfer learning | [12 transformers](../12_natural_language_processing/transformers_and_transfer_learning.md), [13 image_classification](../13_computer_vision/image_classification.md) | [transfer_learning_and_domain_adaptation.md](transfer_learning_and_domain_adaptation.md) | New site/instrument/product with little data |
| Ensemble learning and stacking | [21 advanced_improvement.md](../21_practical_projects/advanced_improvement.md) (OOF-based blending in competitions) | [ensembles_and_stacking.md](ensembles_and_stacking.md) | Bagging/boosting/stacking with scikit-learn APIs |
| Hyperparameter optimisation strategies | [06 hyperparameter_optimization.md](../06_model_improvement/hyperparameter_optimization.md) | [efficient_development.md](efficient_development.md) (budget-aware tuning) | Limited compute; GPUs |
| Experiment management | [07 experiment_tracking.md](../07_experimentation/experiment_tracking.md), [21 systematic_experimentation.md](../21_practical_projects/systematic_experimentation.md) | [versioning_and_pipelines.md](versioning_and_pipelines.md) | Teams; many runs; artefacts |
| Dataset and model versioning; reproducible pipelines | [07 reproducibility.md](../07_experimentation/reproducibility.md) | [versioning_and_pipelines.md](versioning_and_pipelines.md) | Data that change; models that are deployed |
| Efficient development with limited compute | — | [efficient_development.md](efficient_development.md) | A laptop GPU (e.g. RTX 3060 6 GB), 16 GB RAM, Colab |

**How these fit the lifecycle** ([project_lifecycle.md](../00_methodology/project_lifecycle.md)): uncertainty, calibration and robustness extend *evaluation* (stage 10). Drift/monitoring and versioning extend the lifecycle *beyond* the final test into operation. Transfer learning and ensembles are *modelling* levers (stages 8–9). Efficient development cuts across all of the stages.
