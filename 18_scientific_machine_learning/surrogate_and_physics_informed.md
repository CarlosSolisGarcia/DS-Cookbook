# Surrogate Modelling, Bayesian Optimisation and Physics-Informed Approaches

> **Folder:** 18_scientific_machine_learning · **Prev:** [small_data_and_uncertainty.md](small_data_and_uncertainty.md) · **Next:** [high_cardinality_and_wide_data.md](../19_advanced_tabular_ml/high_cardinality_and_wide_data.md) · **Related:** [model_selection.md (regression)](../03_regression/model_selection.md) (monotonic constraints), [process_monitoring.md](../16_industrial_sensor_data/process_monitoring.md)

---

## 1. Purpose
Replace expensive simulations or experiments with fast, accurate **surrogates**. Use them to choose the next experiments efficiently (**Bayesian optimisation**). Inject physics into ML models (constraints, hybrid physics + ML, physics-informed neural networks) so that predictions stay plausible and extrapolate better.

## 2. When to use it
- A process/CFD/FEM simulator takes minutes to hours per run, and you need thousands of evaluations (optimisation, uncertainty propagation, real-time use).
- Experiments are costly, and you want to reach an optimum (yield, selectivity, strength) in few runs.
- A partial physical model exists (balances, kinetics), but it misses effects the data show.

## 3. Conceptual foundation
- **Design of computer experiments:** space-filling designs (Latin hypercube, Sobol sequences via `scipy.stats.qmc`) for the initial samples.
- **Surrogate models:** GPs (small n, smooth responses, uncertainty), polynomial response surfaces, GBMs/NNs (large n). Validate them on held-out simulator runs, and check the error over the whole input space.
- **Bayesian optimisation (BO):** fit a GP to the evaluated points. Choose the next point by maximising an **acquisition function** that balances exploitation (a high predicted value) and exploration (high uncertainty), e.g. **Expected Improvement (EI)**. Repeat. Very sample-efficient for ≲ 10–20 dimensions.
- **Physics-informed ML (a spectrum):**
  1. *Physics-based features* (dimensionless groups, Arrhenius terms).
  2. *Constraints:* monotonicity (`monotonic_cst` in HistGB), bounds via output transforms, non-negativity.
  3. *Hybrid / grey-box models:* y = f_physics(x; θ) + g_ML(x), where the ML model learns the residual of the physics.
  4. *Physics-informed neural networks (PINNs):* a NN whose loss includes the residual of a differential equation (and the boundary/initial conditions), in addition to the data misfit.

## 4. Assumptions and limitations
- A surrogate is only valid within its training domain. Out-of-domain queries during optimisation must be prevented (bounds) or flagged.
- BO assumes a reasonably smooth objective. Noisy objectives need noise-aware GPs and replicate evaluations.
- PINNs are delicate (loss balancing, optimisation difficulties). For well-posed ODE/PDE problems, classical solvers are usually more reliable. PINNs shine in inverse problems and in combining sparse data with physics.

## 5. Step-by-step procedure
1. **Define the inputs, bounds and outputs** of the expensive function; the evaluation budget.
2. **Initial design:** LHS/Sobol with ~10× the number of dimensions as points (when affordable).
3. **Fit a surrogate** (a GP for small n); validate it by LOO/holdout; inspect the uncertainty.
4. **Optimisation:** a BO loop (EI) within the bounds; stop at the budget or when the improvement stalls; confirm the optimum with a real run.
5. **Physics integration:** start from a mechanistic model; fit an ML residual model (hybrid); enforce the constraints; compare the extrapolation with pure ML.
6. **(Optional) PINN** for ODE/PDE-governed systems with sparse data.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

# "Expensive" simulator: CSTR yield of an intermediate B in A → B → C, as a function of temperature and residence time
def simulator(T, tau):
    R = 8.314
    k1 = 4e6 * np.exp(-60000 / (R * T)); k2 = 8e8 * np.exp(-90000 / (R * T))       # 1/min
    cA = 1 / (1 + k1 * tau); cB = k1 * tau * cA / (1 + k2 * tau)                   # steady-state CSTR, cA0 = 1
    return 100 * cB                                                                   # yield of B (%)

lo, hi = np.array([300.0, 1.0]), np.array([420.0, 120.0])                           # T (K), tau (min)
X0 = qmc.scale(qmc.LatinHypercube(d=2, seed=0).random(8), lo, hi)                   # initial space-filling design
y0 = simulator(*X0.T)

def fit_gp(X, y):
    k = ConstantKernel(1.0) * Matern(length_scale=[30, 30], nu=2.5) + WhiteKernel(1e-4, noise_level_bounds=(1e-8, 1e-1))
    return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=5, random_state=0).fit((X - lo) / (hi - lo), y)

def expected_improvement(gp, Xc, best, xi=0.01):
    mu, sd = gp.predict((Xc - lo) / (hi - lo), return_std=True)
    z = (mu - best - xi) / np.maximum(sd, 1e-9)
    return (mu - best - xi) * norm.cdf(z) + sd * norm.pdf(z)

X, y = X0.copy(), y0.copy()
cand = qmc.scale(qmc.Sobol(d=2, seed=1).random(2048), lo, hi)                       # candidate pool for the acquisition
for it in range(12):                                                                # BO loop: 12 further "expensive" runs
    gp = fit_gp(X, y)
    x_next = cand[np.argmax(expected_improvement(gp, cand, y.max()))]
    X = np.vstack([X, x_next]); y = np.append(y, simulator(*x_next))
grid = qmc.scale(qmc.Sobol(d=2, seed=2).random(4096), lo, hi); true_best = simulator(*grid.T).max()
print(f"BO best after {len(y)} runs: {y.max():.2f}% at T={X[y.argmax(), 0]:.0f} K, tau={X[y.argmax(), 1]:.0f} min "
      f"(dense-grid optimum ≈ {true_best:.2f}%)")
rand = qmc.scale(qmc.LatinHypercube(d=2, seed=5).random(len(y)), lo, hi)
print(f"pure space-filling design with the same budget: best = {simulator(*rand.T).max():.2f}%")
```

```python
# Surrogate validation on held-out simulator runs (accuracy over the domain, not just near the optimum)
X_test = qmc.scale(qmc.LatinHypercube(d=2, seed=9).random(200), lo, hi); y_test = simulator(*X_test.T)
gp = fit_gp(X, y)
mu, sd = gp.predict((X_test - lo) / (hi - lo), return_std=True)
print(f"surrogate RMSE over the domain: {np.sqrt(np.mean((mu - y_test) ** 2)):.2f}  | 95% coverage: {np.mean(np.abs(mu - y_test) <= 1.96 * sd):.2f}")
```

```python
# Hybrid (grey-box) model: an imperfect physics model + ML on the residual, vs pure ML, in EXTRAPOLATION
from sklearn.ensemble import HistGradientBoostingRegressor
from cookbook_data import make_catalyst_experiments
df = make_catalyst_experiments(); df = df[df["failed_run"] == 0]
T, P, L, r = df["temperature_K"].to_numpy(), df["pressure_bar"].to_numpy(), df["catalyst_loading_pct"].to_numpy(), df["h2_co_ratio"].to_numpy()
y_obs = df["conversion_pct"].to_numpy()
# A simplified physics model the engineer knows (Arrhenius + loading order), WITHOUT the pressure and ratio effects:
def physics(T, L, A=1.2e7, Ea=85000):
    return 100 * (1 - np.exp(-A * np.exp(-Ea / (8.314 * T)) * L ** 0.8 * 2.0))
Xf = np.column_stack([T, P, L, r])
train = T < np.quantile(T, 0.75); test = ~train                                     # extrapolate to the hottest conditions
pure = HistGradientBoostingRegressor(max_iter=200, max_depth=2, min_samples_leaf=5, random_state=0).fit(Xf[train], y_obs[train])
resid = HistGradientBoostingRegressor(max_iter=200, max_depth=2, min_samples_leaf=5, random_state=0).fit(
    Xf[train], y_obs[train] - physics(T[train], L[train]))
rmse = lambda p: np.sqrt(np.mean((y_obs[test] - p) ** 2))
print(f"extrapolation RMSE — physics only: {rmse(physics(T[test], L[test])):.2f} | pure ML: {rmse(pure.predict(Xf[test])):.2f} | "
      f"hybrid physics+ML: {rmse(np.clip(physics(T[test], L[test]) + resid.predict(Xf[test]), 0, 100)):.2f}")

# Monotonic constraint: conversion must not decrease with temperature or loading (all else equal)
mono = HistGradientBoostingRegressor(max_iter=200, max_depth=2, min_samples_leaf=5, monotonic_cst=[1, 0, 1, 0], random_state=0)
print(f"pure ML with monotonic constraints, extrapolation RMSE: {rmse(mono.fit(Xf[train], y_obs[train]).predict(Xf[test])):.2f}")
```

```python
# Physics-informed neural network (PINN) intro: first-order decay dc/dt = −k c, k unknown, from 6 noisy points
import torch
torch.manual_seed(0)
k_true, t_obs = 0.35, torch.tensor([[0.0], [1.0], [2.0], [3.5], [5.0], [7.0]])
c_obs = torch.exp(-k_true * t_obs) + 0.02 * torch.randn_like(t_obs)
net = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Tanh(), torch.nn.Linear(32, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
log_k = torch.nn.Parameter(torch.tensor(0.0))                               # learn k jointly (inverse problem)
opt = torch.optim.Adam(list(net.parameters()) + [log_k], lr=1e-2)
t_col = torch.linspace(0, 12, 100).reshape(-1, 1).requires_grad_(True)    # collocation points (incl. extrapolation range)
for step in range(3000):
    opt.zero_grad()
    loss_data = ((net(t_obs) - c_obs) ** 2).mean()
    c = net(t_col); dc_dt = torch.autograd.grad(c, t_col, torch.ones_like(c), create_graph=True)[0]
    loss_phys = ((dc_dt + torch.exp(log_k) * c) ** 2).mean()               # ODE residual
    (loss_data + loss_phys).backward(); opt.step()
t_new = torch.tensor([[10.0], [12.0]])
print(f"PINN: estimated k = {torch.exp(log_k).item():.3f} (true {k_true}); c(10), c(12) predicted "
      f"{net(t_new).detach().ravel().numpy().round(3)} vs true {np.exp(-k_true * np.array([10, 12])).round(3)}")
```

## 7. Interpretation
- **Bayesian optimisation works:** after 8 initial + 12 guided runs, BO found 75.7% at T ≈ 394 K, τ ≈ 119 min, essentially the dense-grid optimum (75.7%). A pure space-filling design with the same 20-run budget reached 74.0%. Each "expensive" run was spent where improvement was likely. Always confirm the optimum with a real run.
- **But the BO-trained surrogate is poor over the whole domain** (RMSE ≈ 33 points, with very wide intervals: 99% "coverage"). BO concentrates its samples near the optimum, so the GP knows that region well and the rest badly, and it says so through its large uncertainty. **A surrogate meant for global use (sensitivity, uncertainty propagation, real-time prediction) needs a space-filling design**, not BO samples. Same model, different purpose, different data.
- **Hybrid vs physics vs pure ML in extrapolation:** pure ML fails beyond the training temperatures (RMSE ≈ 26: the trees predict flat values), and monotonic constraints do not rescue it (a monotone flat line is still flat). The simplified physics model alone extrapolates best here (≈ 4.0). **The hybrid (≈ 5.0) is slightly worse than the physics alone**: its tree-based residual correction, learned at lower temperatures, is carried flat into the hot region, where the true residual is different. Lessons: (1) the physics carries the extrapolation; (2) the residual model must itself behave sensibly outside the data. Prefer smooth or regularised residual models (linear, GP with a prior mean of zero), or shrink the correction outside the training domain. Always test the hybrid's added value on an extrapolation split.
- **PINN:** from six noisy points it estimates k ≈ 0.38 (true 0.35) and extrapolates the decay to t = 10–12 with the right order of magnitude (0.019/0.009 vs 0.030/0.015), using no data at all in that range, because the ODE residual is enforced there. The ~10% error in k comes from the noise with only six points. For this simple ODE, classical nonlinear least squares is simpler and more reliable. PINNs pay off for complex, partially known PDE systems and inverse problems.

## 8. Decision framework
| Situation | Approach |
|---|---|
| Expensive simulator, ≤ ~20 inputs, need an optimum | LHS/Sobol initial design + GP + BO (EI) |
| Need a fast surrogate over the whole domain | Space-filling design (not BO samples) + GP/GBM/NN; validate on held-out runs |
| Partial physics known | Hybrid: physics + ML residual; constraints |
| Monotone relationships known | Monotonic constraints (HistGB/LightGBM/XGBoost) |
| ODE/PDE-governed with sparse data / inverse problem | PINN (after trying classical parameter estimation) |
| Noisy experiments | Noise-aware GP (WhiteKernel), replicates at promising points |
| Many inputs (> 20–30) | Screening designs / sensitivity analysis first to reduce dimension |

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Validating a surrogate only near the optimum | Unreliable elsewhere | A held-out space-filling test set |
| Optimising outside the surrogate's domain | Fake optima | Bounds; AD checks |
| BO without confirming the optimum | Reporting surrogate artefacts | A real confirmation run |
| PINNs where classical solvers suffice | Fragile, slow results | Start with standard methods |
| Ignoring units/scaling in GP inputs | Poor length-scales | Normalise the inputs to [0, 1] |
| Pure black-box ML for extrapolation | Implausible predictions | Physics features, hybrids, constraints |

## 10. Checklist
- [ ] Inputs, bounds, outputs and budget defined.
- [ ] Space-filling initial design.
- [ ] GP surrogate validated on held-out runs (accuracy + coverage).
- [ ] BO loop with the optimum confirmed by a real evaluation.
- [ ] Physics integrated (features/constraints/hybrid); extrapolation compared with pure ML.
- [ ] PINN only if justified; compared with classical estimation.

## 11. References
- Forrester, A., Sóbester, A. & Keane, A. (2008). *Engineering Design via Surrogate Modelling: A Practical Guide*. Wiley.
- Jones, D. R., Schonlau, M. & Welch, W. J. (1998). "Efficient Global Optimization of Expensive Black-Box Functions." *J. Global Optimization* 13, 455–492 (EI).
- Shahriari, B. et al. (2016). "Taking the Human Out of the Loop: A Review of Bayesian Optimization." *Proc. IEEE* 104(1), 148–175. https://doi.org/10.1109/JPROC.2015.2494218
- Raissi, M., Perdikaris, P. & Karniadakis, G. E. (2019). "Physics-informed neural networks." *J. Computational Physics* 378, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045
- Karniadakis, G. E. et al. (2021). "Physics-informed machine learning." *Nature Reviews Physics* 3, 422–440. https://doi.org/10.1038/s42254-021-00314-5
- von Stosch, M., Oliveira, R., Peres, J. & Feyo de Azevedo, S. (2014). "Hybrid semi-parametric modeling in process systems engineering: Past, present and future." *Computers & Chemical Engineering* 60, 86–101. https://doi.org/10.1016/j.compchemeng.2013.08.008
- SciPy, `scipy.stats.qmc` (Latin hypercube, Sobol). https://docs.scipy.org/doc/scipy/reference/stats.qmc.html
