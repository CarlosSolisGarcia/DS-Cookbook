# Validation Reference

> **Folder:** 08_quick_references · A one-page lookup. The reasoning lives in [cross_validation.md](../06_model_improvement/cross_validation.md), [ts validation](../05_time_series/validation.md) and [data_leakage.md](../00_methodology/data_leakage.md). The competition-specific version is in [competition_validation.md](../21_practical_projects/competition_validation.md).

---

## 1. Choose the scheme from the data structure

```
Will the model be used on FUTURE data, and does the process drift or trend?
├── Yes → time-based hold-out (last period) + TimeSeriesSplit / rolling origin
│        Also repeated entities? → time-based folds; check performance on new AND existing entities
└── No  → Do rows share an entity (machine, patient, formulation, site, user)?
         ├── Yes, and production sees NEW entities → GroupKFold / StratifiedGroupKFold (+ group hold-out)
         ├── Spatial data, production in new areas → spatial blocks as groups (17_geospatial_data)
         └── No → Classification? → StratifiedKFold : KFold (shuffle=True)
                  Small n (< ~500)? → Repeated(Stratified)KFold
Tuning AND reporting from the same data? → nested CV, or keep an untouched test set
```

## 2. scikit-learn splitters
| Splitter | Use | Key arguments |
|---|---|---|
| `KFold` | i.i.d. regression | `n_splits`, `shuffle=True`, `random_state` |
| `StratifiedKFold` | i.i.d. classification | same |
| `RepeatedKFold` / `RepeatedStratifiedKFold` | Small n | `n_repeats` |
| `GroupKFold` | Repeated entities | pass `groups=` (sklearn ≥ 1.6 also supports `shuffle`) |
| `StratifiedGroupKFold` | Repeated entities + imbalance | `groups=`, `shuffle`, `random_state` |
| `LeaveOneGroupOut` | Few groups (sites, campaigns) | `groups=` |
| `GroupShuffleSplit` | One grouped hold-out | `test_size`, `groups=` |
| `TimeSeriesSplit` | Temporal | `n_splits`, `test_size`, `gap`, `max_train_size` |
| `PredefinedSplit` | Externally defined folds | `test_fold` |
| Iterable of (train, test) index pairs | Anything custom (rolling origin, spatial blocks) | pass as `cv=` |

## 3. Hold-out (test set) rules
- Create the test set **first**, before EDA, mimicking deployment (time / groups / space).
- Size: large enough for a stable metric (hundreds of rows; ≥ ~50–100 positives for classification metrics).
- Use it **once**. If it has been used for decisions, report that the estimate is optimistic.

## 4. Leakage quick audit
- [ ] Duplicates removed before splitting; near-duplicates/replicates grouped.
- [ ] All fitted preprocessing inside `Pipeline`.
- [ ] Resampling only inside training folds (`imblearn.pipeline.Pipeline`).
- [ ] Target encodings and target aggregates cross-fitted or past-only.
- [ ] Time features past-only (`shift` before `rolling`); exogenous inputs as forecasts.
- [ ] No column created after the prediction time.
- [ ] Feature selection and threshold choice inside CV.

## 5. Comparing models
- Same splitter object, same seed, same metric for all.
- Look at **paired** fold differences; repeat the CV for small data.
- Treat differences below about one fold std as ties; prefer the simpler model.

## 6. References
- scikit-learn, *Cross-validation* and *Visualizing cross-validation behavior*. https://scikit-learn.org/stable/modules/cross_validation.html · https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html
- Roberts, D. R. et al. (2017). "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure." *Ecography* 40(8), 913–929. https://doi.org/10.1111/ecog.02881
