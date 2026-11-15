# Glossary: Consistent Terminology

> **Folder:** 08_quick_references · *Added to the original architecture.* The terms below are used with these meanings throughout the cookbook.

| Term | Meaning in this cookbook |
|---|---|
| **Training set** | Data used to fit model parameters and preprocessing statistics. |
| **Validation set / validation folds** | Data used to make choices (features, models, hyperparameters, thresholds). In CV, the held-out fold of each split. |
| **Test set (hold-out)** | Data used **once** for the final estimate, never for decisions. |
| **Development data** | Training + validation data (everything except the test set). |
| **Cross-validation (CV)** | Repeated train/validation splitting of the development data; reported as mean ± std over folds. |
| **Nested CV** | CV with an inner loop for tuning and an outer loop for estimation. |
| **Out-of-fold (OOF) predictions** | Predictions for each row made by a model that did not see that row during training. |
| **Baseline** | A trivial or simple reference model that any proposed model must beat (mean, majority class, seasonal naive, domain rule). |
| **Leakage** | Information unavailable at prediction time entering training/evaluation ([data_leakage.md](../00_methodology/data_leakage.md)). |
| **Prediction time (t₀) / forecast origin** | The moment a prediction is made; only information available then may be used. |
| **Horizon (h, H)** | Steps ahead of the origin being forecast; H = maximum horizon. |
| **Rolling origin** | Time-series CV: repeated forecasts from successive origins. |
| **Group** | An entity whose rows are correlated (machine, patient, formulation, site, user); grouped splitting keeps it in one fold. |
| **Prevalence (π)** | Fraction of positives in a binary problem. |
| **Operating point** | A chosen threshold (or k) at which predictions become decisions. |
| **Calibration** | Agreement between predicted probabilities and observed frequencies. |
| **Prediction interval** | A range expected to contain a *new observation* with a given probability (vs. a confidence interval for a parameter or mean). |
| **Coverage** | Fraction of true values inside the prediction intervals. |
| **Residual** | y − ŷ (actual minus predicted); positive means under-prediction. |
| **Bias (of a forecast/prediction)** | mean(ŷ − y): positive means over-prediction. Note the opposite sign to the residual convention. |
| **MASE** | MAE scaled by the in-sample MAE of the seasonal-naive forecast. |
| **Drift** | Change over time in the input distribution (data drift) or in the input–target relationship (concept drift). |
| **Regime** | A distinct operating mode (start-up, steady state, maintenance) with different statistical behaviour. |
| **Feature (column) contract** | The declared list of inputs available at t₀, with types; checked in code. |
| **Pipeline** | A scikit-learn `Pipeline` containing every fitted preprocessing step and the model. |
| **Experiment** | One logged run: hypothesis, change, configuration, data/code version, results, conclusion. |
| **Noise floor** | The irreducible error (measurement/label noise); also used loosely for the CV std below which differences are not meaningful. |
| **Anomaly / outlier** | *Outlier*: an unusual value in data-quality or EDA work ([outliers.md](../01_data_understanding/outliers.md)). *Anomaly*: an unusual observation or behaviour that is itself the target of detection ([10_anomaly_detection](../10_anomaly_detection/)). |
| **Embedding** | A learned dense vector representation of an object (word, document, image, node). |
| **Transfer learning / fine-tuning** | Re-using a model pretrained on large data; fine-tuning updates its weights on the target task. |
