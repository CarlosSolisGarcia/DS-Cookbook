# Data Science Practical Cookbook

A **field manual** for solving real Data Science and Machine Learning problems. It is a set of methodological playbooks, decision frameworks, checklists and runnable Python recipes. The examples come from industry, science and engineering: reactors, pumps, sensors, energy, maintenance logs, surface inspection, experiments.

It is not a textbook. Each recipe tells you **what to do, why, when (and when not), how to implement it, how to read the outputs, and what to do next**.

---

## How to use this cookbook

1. **Find your situation** in the [problem and domain index](#problem-and-domain-index) below.
2. **Open the workflow recipe** of that family (the file named `*_workflow.md`). It sequences the other recipes.
3. **Follow the steps.** Each recipe has the same 11 sections: *Purpose · When to use it · Conceptual foundation · Assumptions and limitations · Step-by-step procedure · Python implementation · Interpretation · Decision framework · Common mistakes · Checklist · References*.
4. **Run the code.** Every recipe uses shared synthetic datasets with known structure (see [Running the examples](#running-the-examples)), so you can check whether a method recovers the truth.
5. **Read §7 Interpretation and §8 Decision framework** before deciding the next step. They report the *actual* results of the code, including the cases where a popular technique did **not** help.
6. When stuck, go to [00_methodology/decision_framework.md](00_methodology/decision_framework.md) (the master index of decisions) or to [project_lifecycle.md](00_methodology/project_lifecycle.md) §8 ("when to go back, and where").

Quick references (metrics, validation schemes, plots, code patterns, glossary) are in [08_quick_references](08_quick_references/). They use a compact format.

---

## Problem and domain index

| I have… / I need to… | Start here | Then |
|---|---|---|
| **Any new project** (frame it before touching data) | [problem_framing.md](00_methodology/problem_framing.md) | [project_lifecycle.md](00_methodology/project_lifecycle.md), [data_leakage.md](00_methodology/data_leakage.md) |
| A **continuous target** (yield, energy, strength, cost) | [regression_workflow.md](03_regression/regression_workflow.md) | [02_preprocessing](02_preprocessing/), [06_model_improvement](06_model_improvement/) |
| A **categorical target** (fail/no fail, defect type) | [classification_workflow.md](04_classification/classification_workflow.md) | [class_imbalance.md](04_classification/class_imbalance.md), [threshold_optimization.md](04_classification/threshold_optimization.md) |
| **Future values** of a time series | [time_series_workflow.md](05_time_series/time_series_workflow.md) | [validation.md](05_time_series/validation.md), [forecasting_models.md](05_time_series/forecasting_models.md) |
| **No labels**: find groups/regimes | [unsupervised_workflow.md](09_unsupervised_learning/unsupervised_workflow.md) | [clustering_algorithms.md](09_unsupervised_learning/clustering_algorithms.md) |
| **Unusual behaviour**: faults, glitches, novelty | [anomaly_detection_workflow.md](10_anomaly_detection/anomaly_detection_workflow.md) | [statistical_methods.md](10_anomaly_detection/statistical_methods.md), [evaluation_and_thresholds.md](10_anomaly_detection/evaluation_and_thresholds.md) |
| **Many features** / spectra / need to visualise high-dimensional data | [dimensionality_reduction_workflow.md](11_dimensionality_reduction/dimensionality_reduction_workflow.md) | [pca.md](11_dimensionality_reduction/pca.md), [high_dimensional_data.md](11_dimensionality_reduction/high_dimensional_data.md) |
| **Multiple sensor streams**, equipment, predictive maintenance, RUL | [sensor_data_workflow.md](16_industrial_sensor_data/sensor_data_workflow.md) | [predictive_maintenance_rul.md](16_industrial_sensor_data/predictive_maintenance_rul.md), [process_monitoring.md](16_industrial_sensor_data/process_monitoring.md) |
| **Text** (reports, work orders, feedback) | [nlp_workflow.md](12_natural_language_processing/nlp_workflow.md) | [classical_nlp.md](12_natural_language_processing/classical_nlp.md) → [transformers](12_natural_language_processing/transformers_and_transfer_learning.md) |
| **Images** (inspection, microscopy, thermal) | [cv_workflow.md](13_computer_vision/cv_workflow.md) | [image_classification.md](13_computer_vision/image_classification.md), [detection_and_segmentation.md](13_computer_vision/detection_and_segmentation.md) |
| **User–item interactions** (recommend parts/documents) | [recsys_workflow.md](14_recommender_systems/recsys_workflow.md) | [collaborative_filtering.md](14_recommender_systems/collaborative_filtering.md) |
| **Graph-structured data** (molecules, networks, flowsheets) | [graph_ml_workflow.md](15_graph_machine_learning/graph_ml_workflow.md) | [gnn_pytorch_geometric.md](15_graph_machine_learning/gnn_pytorch_geometric.md) |
| **Locations / maps / satellite images** | [geospatial_workflow.md](17_geospatial_data/geospatial_workflow.md) | [remote_sensing_intro.md](17_geospatial_data/remote_sensing_intro.md) |
| **Experiments**, small data, replicates, simulators, physics | [scientific_ml_workflow.md](18_scientific_machine_learning/scientific_ml_workflow.md) | [small_data_and_uncertainty.md](18_scientific_machine_learning/small_data_and_uncertainty.md), [surrogate_and_physics_informed.md](18_scientific_machine_learning/surrogate_and_physics_informed.md) |
| Hard tabular cases: high cardinality, shift, groups, noisy labels, record linkage | [19_advanced_tabular_ml](19_advanced_tabular_ml/) | See the per-topic files below |
| Uncertainty, robustness, drift, transfer, ensembles, limited compute, versioning | [20_advanced_methodologies/README.md](20_advanced_methodologies/README.md) | The relationship map to the core recipes |
| A **Kaggle competition** or portfolio project | [kaggle_workflow.md](21_practical_projects/kaggle_workflow.md) | The seven-stage playbook in [21_practical_projects](21_practical_projects/) |

---

## Contents

### Part I — Core cookbook (methodology, tabular ML, time series)

**[00 Methodology](00_methodology/)**: [project_lifecycle](00_methodology/project_lifecycle.md) · [problem_framing](00_methodology/problem_framing.md) · [decision_framework](00_methodology/decision_framework.md) · [data_leakage](00_methodology/data_leakage.md) · [modeling_fundamentals](00_methodology/modeling_fundamentals.md) (baselines, model selection, training, evaluation, over/underfitting)

**[01 Data understanding](01_data_understanding/)**: [initial_inspection](01_data_understanding/initial_inspection.md) · [data_quality](01_data_understanding/data_quality.md) · [missing_values](01_data_understanding/missing_values.md) · [outliers](01_data_understanding/outliers.md) · [exploratory_data_analysis](01_data_understanding/exploratory_data_analysis.md)

**[02 Preprocessing](02_preprocessing/)**: [numerical_features](02_preprocessing/numerical_features.md) · [categorical_features](02_preprocessing/categorical_features.md) · [transformations](02_preprocessing/transformations.md) · [scaling](02_preprocessing/scaling.md) · [feature_engineering](02_preprocessing/feature_engineering.md)

**[03 Regression](03_regression/)**: [regression_workflow](03_regression/regression_workflow.md) · [baseline_models](03_regression/baseline_models.md) · [model_selection](03_regression/model_selection.md) · [evaluation](03_regression/evaluation.md) · [uncertainty](03_regression/uncertainty.md) (prediction intervals, conformal) · [error_analysis](03_regression/error_analysis.md)

**[04 Classification](04_classification/)**: [classification_workflow](04_classification/classification_workflow.md) · [baseline_models](04_classification/baseline_models.md) · [model_selection](04_classification/model_selection.md) · [class_imbalance](04_classification/class_imbalance.md) · [evaluation](04_classification/evaluation.md) · [threshold_optimization](04_classification/threshold_optimization.md) · [calibration](04_classification/calibration.md) · [error_analysis](04_classification/error_analysis.md)

**[05 Time series](05_time_series/)**: [time_series_workflow](05_time_series/time_series_workflow.md) · [time_series_understanding](05_time_series/time_series_understanding.md) · [preprocessing](05_time_series/preprocessing.md) · [exploratory_analysis](05_time_series/exploratory_analysis.md) · [feature_engineering](05_time_series/feature_engineering.md) · [validation](05_time_series/validation.md) · [baseline_models](05_time_series/baseline_models.md) · [forecasting_models](05_time_series/forecasting_models.md) · [multi_step_forecasting](05_time_series/multi_step_forecasting.md) · [evaluation](05_time_series/evaluation.md)

**[06 Model improvement](06_model_improvement/)**: [cross_validation](06_model_improvement/cross_validation.md) · [hyperparameter_optimization](06_model_improvement/hyperparameter_optimization.md) · [feature_selection](06_model_improvement/feature_selection.md) · [model_interpretation](06_model_improvement/model_interpretation.md) · [iterative_improvement](06_model_improvement/iterative_improvement.md)

**[07 Experimentation](07_experimentation/)**: [experiment_tracking](07_experimentation/experiment_tracking.md) · [reproducibility](07_experimentation/reproducibility.md)

**[08 Quick references](08_quick_references/)**: [metrics_reference](08_quick_references/metrics_reference.md) · [validation_reference](08_quick_references/validation_reference.md) · [visualization_reference](08_quick_references/visualization_reference.md) · [python_code_patterns](08_quick_references/python_code_patterns.md) · [glossary](08_quick_references/glossary.md)

### Part II — Extended domains and problem families

| Tier | Section | Recipes |
|---|---|---|
| 1 | **[09 Unsupervised learning](09_unsupervised_learning/)** | [unsupervised_workflow](09_unsupervised_learning/unsupervised_workflow.md) · [clustering_algorithms](09_unsupervised_learning/clustering_algorithms.md) · [cluster_validation](09_unsupervised_learning/cluster_validation.md) · [cluster_interpretation](09_unsupervised_learning/cluster_interpretation.md) |
| 1 | **[10 Anomaly detection](10_anomaly_detection/)** | [anomaly_detection_workflow](10_anomaly_detection/anomaly_detection_workflow.md) · [statistical_methods](10_anomaly_detection/statistical_methods.md) (robust z, Mahalanobis, PCA T²/SPE, EWMA/CUSUM) · [ml_methods](10_anomaly_detection/ml_methods.md) (IF, LOF, OCSVM, GMM, autoencoder) · [time_series_anomalies](10_anomaly_detection/time_series_anomalies.md) · [evaluation_and_thresholds](10_anomaly_detection/evaluation_and_thresholds.md) · [root_cause_analysis](10_anomaly_detection/root_cause_analysis.md) |
| 1 | **[11 Dimensionality reduction](11_dimensionality_reduction/)** | [dimensionality_reduction_workflow](11_dimensionality_reduction/dimensionality_reduction_workflow.md) · [pca](11_dimensionality_reduction/pca.md) (+PLS, spectra) · [high_dimensional_data](11_dimensionality_reduction/high_dimensional_data.md) · [visualization_embeddings](11_dimensionality_reduction/visualization_embeddings.md) (t-SNE, UMAP) |
| 1 | **[16 Industrial & sensor data](16_industrial_sensor_data/)** | [sensor_data_workflow](16_industrial_sensor_data/sensor_data_workflow.md) · [sensor_preprocessing](16_industrial_sensor_data/sensor_preprocessing.md) · [predictive_maintenance_rul](16_industrial_sensor_data/predictive_maintenance_rul.md) · [process_monitoring](16_industrial_sensor_data/process_monitoring.md) |
| 1 | **[21 Practical Project & Competition Playbook](21_practical_projects/)** | [kaggle_workflow](21_practical_projects/kaggle_workflow.md) · [project_reconnaissance](21_practical_projects/project_reconnaissance.md) · [baseline_development](21_practical_projects/baseline_development.md) · [competition_validation](21_practical_projects/competition_validation.md) · [systematic_experimentation](21_practical_projects/systematic_experimentation.md) · [advanced_improvement](21_practical_projects/advanced_improvement.md) · [final_solution](21_practical_projects/final_solution.md) · [project_retrospective](21_practical_projects/project_retrospective.md) |
| 2 | **[12 NLP](12_natural_language_processing/)** | [nlp_workflow](12_natural_language_processing/nlp_workflow.md) · [classical_nlp](12_natural_language_processing/classical_nlp.md) · [topic_modeling](12_natural_language_processing/topic_modeling.md) · [embeddings_and_similarity](12_natural_language_processing/embeddings_and_similarity.md) · [transformers_and_transfer_learning](12_natural_language_processing/transformers_and_transfer_learning.md) |
| 2 | **[13 Computer vision](13_computer_vision/)** | [cv_workflow](13_computer_vision/cv_workflow.md) · [image_preprocessing_augmentation](13_computer_vision/image_preprocessing_augmentation.md) · [image_classification](13_computer_vision/image_classification.md) · [detection_and_segmentation](13_computer_vision/detection_and_segmentation.md) |
| 2 | **[14 Recommender systems](14_recommender_systems/)** | [recsys_workflow](14_recommender_systems/recsys_workflow.md) · [baselines_and_content_based](14_recommender_systems/baselines_and_content_based.md) · [collaborative_filtering](14_recommender_systems/collaborative_filtering.md) |
| 3 | **[15 Graph ML](15_graph_machine_learning/)** | [graph_ml_workflow](15_graph_machine_learning/graph_ml_workflow.md) · [graph_features_and_embeddings](15_graph_machine_learning/graph_features_and_embeddings.md) · [gnn_pytorch_geometric](15_graph_machine_learning/gnn_pytorch_geometric.md) |
| 3 | **[17 Geospatial data](17_geospatial_data/)** | [geospatial_workflow](17_geospatial_data/geospatial_workflow.md) · [remote_sensing_intro](17_geospatial_data/remote_sensing_intro.md) |
| 3 | **[18 Scientific ML](18_scientific_machine_learning/)** | [scientific_ml_workflow](18_scientific_machine_learning/scientific_ml_workflow.md) · [small_data_and_uncertainty](18_scientific_machine_learning/small_data_and_uncertainty.md) · [surrogate_and_physics_informed](18_scientific_machine_learning/surrogate_and_physics_informed.md) |
| 3 | **[19 Advanced tabular ML](19_advanced_tabular_ml/)** | [high_cardinality_and_wide_data](19_advanced_tabular_ml/high_cardinality_and_wide_data.md) · [distribution_shift](19_advanced_tabular_ml/distribution_shift.md) · [grouped_hierarchical_data](19_advanced_tabular_ml/grouped_hierarchical_data.md) · [imbalance_beyond_binary](19_advanced_tabular_ml/imbalance_beyond_binary.md) · [entity_resolution](19_advanced_tabular_ml/entity_resolution.md) · [noisy_labels_weak_supervision](19_advanced_tabular_ml/noisy_labels_weak_supervision.md) |
| × | **[20 Advanced methodologies](20_advanced_methodologies/README.md)** | [uncertainty_estimation](20_advanced_methodologies/uncertainty_estimation.md) · [robustness_and_sensitivity](20_advanced_methodologies/robustness_and_sensitivity.md) · [drift_and_monitoring](20_advanced_methodologies/drift_and_monitoring.md) · [transfer_learning_and_domain_adaptation](20_advanced_methodologies/transfer_learning_and_domain_adaptation.md) · [ensembles_and_stacking](20_advanced_methodologies/ensembles_and_stacking.md) · [efficient_development](20_advanced_methodologies/efficient_development.md) · [versioning_and_pipelines](20_advanced_methodologies/versioning_and_pipelines.md) |

**How the parts connect:** every extension section reuses the core methodology. That includes framing ([00](00_methodology/)), data quality ([01](01_data_understanding/)), leakage-safe pipelines ([02](02_preprocessing/)), validation that mimics deployment ([06](06_model_improvement/cross_validation.md), [08](08_quick_references/validation_reference.md)), baselines first, metrics from costs ([08](08_quick_references/metrics_reference.md)), error analysis and tracked iteration ([06](06_model_improvement/iterative_improvement.md), [07](07_experimentation/)). The extension recipes link back instead of repeating these. [20_advanced_methodologies/README.md](20_advanced_methodologies/README.md) maps each cross-cutting method to its core recipe.

---

## Suggested learning progression

Learn through implementation: each step = read the recipe, run the code, change something, explain the result.

| Stage | Goal | Recipes | Practice |
|---|---|---|---|
| 1. Foundations (1–2 weeks) | Framing, leakage, validation, baselines | 00 (all), 01, 02, [cross_validation](06_model_improvement/cross_validation.md), [08 quick refs](08_quick_references/) | Reproduce the leakage demos; write a problem statement for a real project |
| 2. Tabular core (2–3 weeks) | Full regression and classification workflows | 03, 04, 06, 07 | Run both workflows end to end on your own dataset |
| 3. Time series (2–3 weeks) | Forecasting discipline | 05 (in workflow order) | Forecast a real energy/production series with rolling origin vs seasonal naive |
| 4. Tier 1 extensions (3–4 weeks) | Unsupervised, anomalies, DR, sensors | 09 → 11 → 10 → 16 | Build a regime-aware monitoring prototype on plant/sensor data |
| 5. Competition practice (ongoing) | Systematic project execution | 21 (all) | One tabular Kaggle playground competition with the full playbook + retrospective |
| 6. Tier 2 domains (4–6 weeks) | Text, images, recommenders | 12 → 13 → 14 | Classify maintenance texts; train a defect classifier with transfer learning (GPU/Colab) |
| 7. Tier 3 specialisations (as needed) | Graphs, geospatial, scientific, advanced tabular | 15, 17, 18, 19 | Revisit GNNs on molecular data (RDKit + PyG); a GP surrogate + BO for a lab problem |
| 8. Production thinking | Uncertainty, drift, versioning | 20 | Add conformal intervals, drift monitoring and model cards to one earlier project |

---

## Running the examples

**Prerequisites:** Python ≥ 3.10 (validated with 3.13), and the packages in [requirements.txt](requirements.txt). The core recipes need only the scientific Python stack. NLP/CV/graph/geospatial need torch, transformers, torchvision, torch-geometric, opencv, geopandas and rasterio. **All runnable examples execute on CPU**, and none needs paid APIs or cloud services. Code blocks that download pretrained models (Hugging Face Hub, torchvision weights) are marked `# not-run` and need internet access. A GPU (e.g. an RTX 3060 laptop, 6 GB) or Google Colab is recommended for those ([efficient_development.md](20_advanced_methodologies/efficient_development.md)).

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Run notebooks/scripts **from the repository root** (or add it to `PYTHONPATH`), so that the shared modules import:

- [`cookbook_data.py`](cookbook_data.py): synthetic datasets with known structure, used throughout. They include a batch-reactor yield dataset (regression), a machine-fleet failure dataset (classification), site energy demand (time series), process sensors with injected faults, a pump fleet run-to-failure, multi-rate streams, maintenance work-order texts, surface-defect images with masks, user–item interactions, air-quality stations, catalytic-reactor experiments with replicates, and NIR spectra. The data contain realistic problems on purpose: missing-data mechanisms, sentinels, unit errors, duplicates, leakage columns, drift, label noise.
- [`cookbook_utils.py`](cookbook_utils.py): shared helpers (forecast metrics, rolling-origin evaluation, direct multi-horizon features), explained in the time-series recipes.

**Quality tools:**

```bash
python tools/validate_code_blocks.py          # executes every ```python block of every recipe (≈ 15–20 min on CPU)
```

```bash
python tools/check_links.py                   # verifies all relative links between recipes
```

---

## Conventions

- **Terminology:** see the [glossary](08_quick_references/glossary.md) (training/validation/test, OOF, origin/horizon, bias sign conventions…).
- **Seeds:** `random_state=42` (or 0) everywhere. `np.random.default_rng(seed)`, never the global seed.
- **Pipelines:** every fitted preprocessing step lives inside a scikit-learn `Pipeline` / `ColumnTransformer`.
- **Honest results:** the *Interpretation* sections describe what the code actually produced, including negative and surprising results (e.g. SMOTE not helping, GBMs failing to extrapolate, label noise hurting the evaluation more than the training). Those are lessons, not errors.
- **References:** official documentation, textbooks and peer-reviewed papers. The links were not re-checked live during writing. Report any broken URL.

---

## Scope, limitations and future extensions

The cookbook covers methodology, tabular ML, time series, unsupervised learning, anomaly detection, dimensionality reduction, introductory NLP/CV/recommenders/graphs/geospatial/scientific ML, and the cross-cutting methods listed above. It intentionally **defers**:
- **Reinforcement learning** (a future section `22_reinforcement_learning/`: bandits for experimentation, process control with RL, simulation environments).
- **Deep time-series models** (global models across many series, N-BEATS/TFT-style architectures) and hierarchical forecast reconciliation (in `05_time_series/`).
- **Survival analysis** in depth (lifelines/scikit-survival; started in [predictive_maintenance_rul.md](16_industrial_sensor_data/predictive_maintenance_rul.md)).
- **Causal inference** (DoE analysis, uplift, causal graphs) beyond the warnings in [problem_framing.md](00_methodology/problem_framing.md).
- **Generative AI / LLM agents**, object tracking/video, speech, and full MLOps (serving, CI/CD for models).
- **Chemistry-specific tooling** (RDKit descriptors, fingerprints, scaffold splits in code); referenced in [18](18_scientific_machine_learning/) and [15](15_graph_machine_learning/).

New recipes should follow the 11-section structure and the update rule in [project_retrospective.md](21_practical_projects/project_retrospective.md): general, validated, structured.
