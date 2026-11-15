# Robustness and Sensitivity Analysis

> **Folder:** 20_advanced_methodologies · **Prev:** [uncertainty_estimation.md](uncertainty_estimation.md) · **Next:** [drift_and_monitoring.md](drift_and_monitoring.md) · **Related:** [model_interpretation.md](../06_model_improvement/model_interpretation.md), [surrogate_and_physics_informed.md](../18_scientific_machine_learning/surrogate_and_physics_informed.md)

---

## 1. Purpose
Test how predictions respond to realistic perturbations (sensor noise, missing inputs, calibration offsets, shifted operating conditions), and quantify which inputs drive the output variance (global sensitivity analysis). Robust models degrade gracefully, and sensitivity results guide measurement and control priorities.

## 2. When to use it
Before deploying models fed by sensors or manual inputs. Use it when a surrogate replaces a simulator (which inputs matter?), and when stakeholders ask "how sure can we be if X is measured badly?".

## 3. Conceptual foundation
- **Perturbation / stress tests:** add noise of a realistic magnitude to each input, apply offsets (miscalibration), set inputs to missing, or shift their distributions. Measure the change in predictions and in the metrics.
- **Local sensitivity:** partial derivatives or finite differences around a point (one-at-a-time, OAT). Cheap, but blind to interactions.
- **Global sensitivity (variance-based, Sobol):** the first-order index S_i is the share of the output variance explained by input i alone. The total index S_Ti includes all its interactions. Estimated by Monte Carlo (the Saltelli/Jansen estimators) over the input distribution. A large S_Ti − S_i means strong interactions.
- **Robustness vs importance:** a model can rely heavily on a feature (high importance) that is also fragile (noisy sensor). That combination is a deployment risk.

## 4. Assumptions and limitations
- The results depend on the assumed input distributions and perturbation sizes. Use measurement-system data (sensor accuracy, drift rates).
- Sobol indices assume independent inputs. With correlated inputs, interpret them carefully or sample from the realistic joint distribution.
- Monte Carlo sensitivity needs many model evaluations: fine for ML models and surrogates, expensive for raw simulators.

## 5. Step-by-step procedure
1. **List the inputs** with their realistic uncertainty (sensor accuracy, calibration drift, missingness rates).
2. **Stress tests:** noise, offsets and missingness per input; record the metric degradation.
3. **Global sensitivity** (Sobol) on the model/surrogate over the operating envelope.
4. **Combine:** inputs with high sensitivity *and* high measurement uncertainty are the priorities (better sensors, redundancy, monitoring).
5. **Mitigate fragility:** regularisation, robust features (ratios, smoothing), training with noise augmentation, fallbacks when inputs are missing.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from cookbook_data import load_clean_process_yield

df = load_clean_process_yield()
feats = ["temperature_c", "pressure_bar", "feed_rate_kg_h", "residence_time_min", "catalyst_age_days", "impurity_ppm"]
cut = int(0.8 * len(df)); tr, te = df.iloc[:cut], df.iloc[cut:]
model = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(tr[feats], np.log(tr["yield_kg"]))
base_mae = mean_absolute_error(te["yield_kg"], np.exp(model.predict(te[feats])))

# Stress tests per input: realistic sensor noise (in units), a calibration offset, and missingness
sensor = {"temperature_c": (1.0, 3.0), "pressure_bar": (0.1, 0.5), "feed_rate_kg_h": (1.0, 5.0),
          "residence_time_min": (1.0, 3.0), "catalyst_age_days": (0.0, 5.0), "impurity_ppm": (10, 40)}   # (noise SD, offset)
rng = np.random.default_rng(0); rows = []
for f, (sd, off) in sensor.items():
    def mae_with(Xp): return mean_absolute_error(te["yield_kg"], np.exp(model.predict(Xp)))
    noisy = te[feats].copy(); noisy[f] = noisy[f] + rng.normal(0, sd, len(noisy)) if sd > 0 else noisy[f]
    shifted = te[feats].copy(); shifted[f] = shifted[f] + off
    missing = te[feats].copy(); missing[f] = np.nan
    rows.append({"input": f, "noise": mae_with(noisy) - base_mae, "offset": mae_with(shifted) - base_mae,
                 "missing": mae_with(missing) - base_mae})
print(f"baseline test MAE = {base_mae:.1f} kg; increase in MAE under each perturbation:")
print(pd.DataFrame(rows).set_index("input").round(1))
```

```python
# Global sensitivity (first-order and total Sobol indices) with the Saltelli/Jansen Monte Carlo estimators
def sobol_indices(f, sample_inputs, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    A, B = sample_inputs(n, rng), sample_inputs(n, rng)
    fA, fB = f(A), f(B); var = np.var(np.r_[fA, fB])
    out = {}
    for c in A.columns:
        ABc = A.copy(); ABc[c] = B[c].to_numpy()                       # A with column c taken from B
        fABc = f(ABc)
        S1 = np.mean(fB * (fABc - fA)) / var                          # Saltelli (2010)
        ST = 0.5 * np.mean((fA - fABc) ** 2) / var                    # Jansen (1999)
        out[c] = (S1, ST)
    return pd.DataFrame(out, index=["S1 (first order)", "ST (total)"]).T

def sample_operating_envelope(n, rng):
    """Independent sampling from the empirical marginals of the training data (the operating envelope)."""
    return pd.DataFrame({c: rng.choice(tr[c].dropna().to_numpy(), n) for c in feats})

S = sobol_indices(lambda X: model.predict(X[feats]), sample_operating_envelope)
print(S.round(3).sort_values("ST (total)", ascending=False))
```

## 7. Interpretation
- **Stress tests (baseline MAE ≈ 52 kg):** realistic random noise changes almost nothing, but a **+3 °C temperature calibration offset adds ≈ 9 kg MAE**. The model has learned the temperature optimum, so a systematic bias moves every prediction in the same direction. A **missing temperature or feed-rate reading adds ≈ 34–38 kg**. HGB routes NaN to a learned branch, so the predictions continue but become much worse. Conclusion for deployment: the temperature sensor needs calibration checks and redundancy, and both critical sensors need a fallback when they are missing.
- **Sobol indices (on log-yield, over the operating envelope):** temperature dominates (S1 ≈ 0.54), because sampling across the whole envelope includes temperatures far from the optimum, where the yield falls steeply. Feed rate follows (S1 ≈ 0.26, ST ≈ 0.43). The gap ST − S1 ≈ 0.16 signals **interactions**: the effect of feed depends on the other inputs within the model. The remaining inputs have ST ≈ 0.03–0.04. They could be fixed at nominal values in a surrogate with little loss. Small negative S1 values are Monte Carlo noise; increase n for precise estimates.
- **Combined view:** temperature has high sensitivity *and* a calibration risk, so it is the top measurement priority.

## 8. Decision framework
- **IF** a high-importance input is noisy or drifts → robust features (smoothed/aggregated), redundancy, monitoring of that sensor, noise augmentation in training.
- **IF** predictions break when an input is missing → explicit fallback models without that input, or imputation with uncertainty flags.
- **IF** strong interactions (ST ≫ S1) → do not interpret the effects one at a time; use 2-D analyses (PDP/ICE).
- **IF** a surrogate has inputs with ST ≈ 0 → fix them; simplify the model and the experiments.
- **IF** the inputs are correlated → sample from the realistic joint distribution (e.g. resampling historical rows) for sensitivity statements.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Arbitrary perturbation sizes | Meaningless robustness scores | Use instrument specifications/historical drift |
| OAT sensitivity only | Missed interactions | Global (Sobol) indices |
| Sobol with unrealistic independent sampling of correlated inputs | Evaluating impossible operating points | Sample realistic joint inputs; state the assumption |
| Treating importance as robustness | Deploying fragile models | Separate stress tests |

## 10. Checklist
- [ ] Input uncertainties from the measurement-system data.
- [ ] Noise/offset/missingness stress tests with metric deltas.
- [ ] Global sensitivity indices (S1, ST).
- [ ] Priorities: sensitivity × uncertainty; mitigation plan.

## 11. References
- Saltelli, A. et al. (2008). *Global Sensitivity Analysis: The Primer*. Wiley.
- Saltelli, A. et al. (2010). "Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index." *Computer Physics Communications* 181(2), 259–270. https://doi.org/10.1016/j.cpc.2009.09.018
- Jansen, M. J. W. (1999). "Analysis of variance designs for model output." *Computer Physics Communications* 117, 35–43.
- SALib (Python sensitivity analysis library). https://salib.readthedocs.io/
- Hendrycks, D. & Dietterich, T. (2019). "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations." *ICLR*. https://arxiv.org/abs/1903.12261
