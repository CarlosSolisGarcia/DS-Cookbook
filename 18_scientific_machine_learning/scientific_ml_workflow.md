# Scientific and Engineering ML Workflow: Small Data, Replicates, Physics and Scientific Validity

> **Folder:** 18_scientific_machine_learning · **Start here for experimental/scientific datasets.** · **Next:** [small_data_and_uncertainty.md](small_data_and_uncertainty.md) · **Related:** [problem_framing.md](../00_methodology/problem_framing.md) (prediction vs causation), [feature_engineering.md](../02_preprocessing/feature_engineering.md), [high_dimensional_data.md](../11_dimensionality_reduction/high_dimensional_data.md)

---

## 1. Purpose
Adapt the data-science workflow to the particular nature of scientific and engineering data. These datasets are small and expensive, come with replicates and measurement uncertainty, obey physical laws and constraints, carry rich domain knowledge, and are judged by *scientific validity*, not just by predictive scores.

## 2. When to use it
Laboratory and pilot-plant experiments (reaction yields, conversions, selectivities), materials properties (strength, conductivity, band gaps), formulations, characterisation data (spectra, chromatograms), simulation outputs (CFD/FEM/process simulators), environmental and energy measurements.

## 3. Conceptual foundation

### 3.1 What is different
| Feature | Consequence | Practice |
|---|---|---|
| **Small n** (10–500) | High-variance estimates; flexible models overfit | Simple/regularised models, GPs; repeated/nested CV; uncertainty everywhere ([small_data_and_uncertainty.md](small_data_and_uncertainty.md)) |
| **Replicates** | Rows are not independent; the replicate spread gives the noise floor | Group by condition in CV; compare the model error with the replicate error |
| **Measurement uncertainty** (known σ per measurement) | Some points are more reliable than others | Weighted fitting (1/σ²); errors-in-variables awareness; intervals |
| **Designed experiments (DoE)** | Factors are varied systematically; often orthogonal | Response-surface models; interpretable effects; sequential design |
| **Physical constraints** (bounds, monotonicity, conservation, units) | Implausible predictions are unacceptable | Constrained models, physics-based features, hybrid models ([surrogate_and_physics_informed.md](surrogate_and_physics_informed.md)) |
| **Domain knowledge** (mechanisms, dimensionless groups) | Strong priors | Mechanistic features (Arrhenius 1/T, ln P, Reynolds…) |
| **Extrapolation needs** (new materials/conditions) | Interpolation-focused CV is optimistic | Cluster/leave-region-out CV; applicability domain |
| **Scientific claims** | Explanations must be valid and reproducible | Separate prediction from mechanism; validate claims with new experiments |

### 3.2 Representations in chemistry and materials (for orientation)
- **Molecules:** descriptors (MW, logP, TPSA…), fingerprints (Morgan/ECFP) via RDKit, graphs for GNNs ([15_graph_machine_learning](../15_graph_machine_learning/gnn_pytorch_geometric.md)).
- **Materials:** compositional features (element property statistics, e.g. matminer), structural descriptors, crystal graphs.
- **Process/reaction conditions:** physically transformed variables (1/T for rates, ln of concentrations/pressures, residence time, dimensionless numbers).

## 4. Assumptions and limitations
- With small n, model comparisons are noisy. Prefer few, well-motivated candidate models over broad searches.
- ML on observational lab data does not establish mechanisms. Mechanistic claims need designed confirmation experiments.
- Simulation data are noise-free but carry model-form error. Validate the surrogates against experiments.

## 5. Step-by-step procedure
1. **Define the scientific question:** prediction (screening, optimisation), understanding (effects, mechanisms), or a surrogate of an expensive simulation/experiment.
2. **Data audit:** the experimental design, replicates, measurement uncertainty, failed/aborted runs, batch/day effects, units.
3. **Noise floor from the replicates:** the replicate standard deviation sets the best achievable error.
4. **Domain features:** the transformations implied by theory (e.g. ln k = ln A − Ea/RT → 1/T), and dimensionless groups.
5. **Models, by data size:** linear/polynomial response surface → Gaussian process → GBM (if n is large enough) → hybrid physics + ML.
6. **Validation:** grouped by condition (replicates), repeated; extrapolation tests (leave-region-out).
7. **Uncertainty and the applicability domain:** intervals; flag predictions outside the data domain.
8. **Scientific validity checks:** the signs and magnitudes of the effects against theory; monotonicity; plausibility of the extrapolations.
9. **Close the loop:** propose new experiments (where the uncertainty is high, or the optimum is likely) → sequential design.

## 6. Python implementation: auditing an experimental dataset

```python
import numpy as np
import pandas as pd
from cookbook_data import make_catalyst_experiments

df = make_catalyst_experiments()                     # 24 conditions × 3 replicates (LHS design)
factors = ["temperature_K", "pressure_bar", "catalyst_loading_pct", "h2_co_ratio"]
print(df[factors].describe().loc[["min", "max"]])
print("design correlation between factors (low = good design):\n", df.drop_duplicates("condition_id")[factors].corr().round(2))

# Failed runs: flagged anomalies; inspect before deciding (here: exclude, with the decision documented)
print("failed runs:", int(df["failed_run"].sum()))
clean = df[df["failed_run"] == 0]

# Replicate-based noise floor
rep = clean.groupby("condition_id")["conversion_pct"].agg(["mean", "std", "count"])
pooled_sd = np.sqrt(np.nanmean(rep["std"] ** 2))
print(f"pooled replicate SD = {pooled_sd:.2f} points (instrument SD = {df['conversion_sd_instrument'].iloc[0]})")
print("→ no model can be expected to predict single runs better than ≈", round(pooled_sd, 2), "points RMSE")
```

```python
# Domain features from kinetics: Arrhenius → 1/T; power-law orders → logs
clean = clean.assign(inv_T=1 / clean["temperature_K"], ln_P=np.log(clean["pressure_bar"]),
                     ln_load=np.log(clean["catalyst_loading_pct"]),
                     # for first-order kinetics, X = 1 - exp(-k tau) ⇒ ln(-ln(1 - X)) is linear in ln k
                     lnk_obs=np.log(-np.log(1 - clean["conversion_pct"].clip(upper=99.5) / 100)))
from sklearn.linear_model import LinearRegression
lin = LinearRegression().fit(clean[["inv_T", "ln_P", "ln_load"]], clean["lnk_obs"])
Ea_est = -lin.coef_[0] * 8.314
print(f"estimated apparent activation energy Ea ≈ {Ea_est / 1000:.0f} kJ/mol; orders: P {lin.coef_[1]:.2f}, loading {lin.coef_[2]:.2f}")
```

## 7. Interpretation
- **The design table:** the factors span their ranges with low mutual correlation, a well-conditioned DoE (Latin hypercube). Effects can therefore be estimated separately. Observational datasets often have strongly correlated factors, which makes the effects unidentifiable.
- **Failed runs:** one run has anomalously low conversion (a plugged reactor). Deciding to exclude them is a scientific judgement. Document it and, ideally, confirm it from lab notes.
- **Pooled replicate SD (≈ 2 points):** this is the noise floor. A model with CV RMSE close to it is as good as the data allow. A model far below it is suspicious (leakage across replicates).
- **Mechanistic linearisation:** fitting ln k against 1/T recovers an apparent activation energy close to the true 85 kJ/mol, with reaction orders ≈ 0.5 (pressure) and ≈ 0.8 (loading). This is the strongest form of "feature engineering": the theory tells you the functional form, and the parameters have physical meaning and can be checked against the literature.

## 8. Decision framework
| Situation | Approach |
|---|---|
| n < ~50, smooth response | Linear/quadratic response surface or GP; LOO/grouped CV; mechanistic features |
| n ~ 50–500 | GP, regularised linear models with features, small GBMs; repeated grouped CV |
| n > ~500 (or simulations) | GBM/NN surrogates; still physics features + constraints |
| Known mechanism with unknown parameters | Fit the mechanistic model (nonlinear least squares); ML on its residuals (hybrid) |
| Replicates present | GroupKFold by condition; compare the RMSE with the replicate SD |
| Known measurement σ | Weighted regression (`sample_weight = 1/σ²`); GP with heteroscedastic noise |
| Need to extrapolate | Physics-based/hybrid models; leave-region-out validation; applicability-domain checks |
| Goal = optimisation with costly experiments | Bayesian optimisation / sequential DoE ([surrogate_and_physics_informed.md](surrogate_and_physics_informed.md)) |

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Random CV with replicates split across folds | Optimistic: the model "predicts" a replicate from its siblings | Group by condition ([data_leakage.md](../00_methodology/data_leakage.md)) |
| Ignoring the noise floor | Chasing impossible accuracy; overfitting | Replicate-based noise estimate |
| Black-box models without physical checks | Implausible predictions in extrapolation | Monotonicity/sign checks; hybrids |
| Deleting "outliers" without lab evidence | Biased science | Document; flag; sensitivity analysis |
| Over-interpreting feature importances as mechanisms | Wrong scientific claims | Designed confirmation experiments |
| Mixing units/scales across sources | Silent errors | A units audit; dimensionless groups |

## 10. Checklist
- [ ] Scientific question (predict / understand / surrogate) stated.
- [ ] Design, replicates, uncertainty, failed runs and batch effects audited.
- [ ] Noise floor from the replicates computed.
- [ ] Mechanistic/dimensionless features derived.
- [ ] Model complexity matched to n; grouped and repeated validation.
- [ ] Uncertainty and the applicability domain reported.
- [ ] Physical plausibility checks passed; next experiments proposed.

## 11. References
- Montgomery, D. C. (2017). *Design and Analysis of Experiments*, 9th ed. Wiley.
- Box, G. E. P., Hunter, J. S. & Hunter, W. G. (2005). *Statistics for Experimenters*, 2nd ed. Wiley.
- Butler, K. T., Davies, D. W., Cartwright, H., Isayev, O. & Walsh, A. (2018). "Machine learning for molecular and materials science." *Nature* 559, 547–555. https://doi.org/10.1038/s41586-018-0337-2
- Wang, A. Y.-T. et al. (2020). "Machine Learning for Materials Scientists: An Introductory Guide toward Best Practices." *Chemistry of Materials* 32(12), 4954–4965. https://doi.org/10.1021/acs.chemmater.0c01907
- Artrith, N. et al. (2021). "Best practices in machine learning for chemistry." *Nature Chemistry* 13, 505–508. https://doi.org/10.1038/s41557-021-00716-z
- RDKit documentation (descriptors, fingerprints). https://www.rdkit.org/docs/
