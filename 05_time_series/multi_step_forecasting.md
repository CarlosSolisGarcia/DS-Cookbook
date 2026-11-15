# Multi-Step Forecasting: Recursive, Direct and Pooled Strategies

> **Folder:** 05_time_series · **Prev:** [forecasting_models.md](forecasting_models.md) · **Next:** [evaluation.md](evaluation.md) · **Related:** [feature_engineering.md](feature_engineering.md), [validation.md](validation.md)
>
> *Added to the original architecture: the choice of multi-step strategy is a distinct, recurring decision for ML forecasters.*

---

## 1. Purpose
Produce forecasts for h = 1…H with ML/regression models, and choose the strategy deliberately. The strategy controls error accumulation, leakage risk and computational cost.

## 2. When to use it
Whenever an ML/regression model must forecast more than one step ahead. Statistical models (ETS, ARIMA) produce multi-step forecasts natively, via the recursive logic built into their state equations.

## 3. Conceptual foundation
| Strategy | How it works | Pros | Cons |
|---|---|---|---|
| **Recursive (iterated)** | One model for t+1 using lags y_{t}, y_{t−1}…; for t+2, feed in ŷ_{t+1} as the lag | One model; uses the most recent information; consistent dynamics | **Errors accumulate** with h; exogenous features needed for every step; bias compounds |
| **Direct** | A separate model per horizon h, trained on targets y_{t+h} with features known at t | No error accumulation; each model specialises | H models; less data per model (the last h rows are lost); forecasts across h may be inconsistent |
| **Pooled direct (multi-horizon)** | One model; each row is (origin, h); `h` is a feature | One model; more data; shares strength across horizons | Bigger table; the model must learn the horizon dependence |
| **Multi-output (MIMO)** | One model outputs the H-vector at once (e.g. a neural net, or `MultiOutputRegressor` = direct) | Coherent vector forecasts | Model type restrictions |
| **DirRec / hybrids** | Direct models that also take the earlier predictions as inputs | Middle ground | Complexity |

The empirical evidence is mixed. No strategy dominates (Taieb et al., 2012). Direct strategies tend to win when the model is misspecified, because they do not compound bias. Recursive strategies win when the one-step model is very good and the data are scarce. **Validate both.**

**Leakage note for direct training.** When training a direct model at origin O, the training rows must have **targets** at or before O. A row with origin O − 3 and h = 7 has its target at O + 4, which is in the future. `cookbook_utils.evaluate_direct_model` filters on `target_time <= O` for this reason.

## 4. Assumptions and limitations
- Recursive forecasts need the future values of every exogenous feature for each step. Calendar features are fine; weather needs forecasts.
- Recursive strategies with tree models can drift quickly when the predictions leave the training range.
- Direct models need H × (fits per origin). The pooled layout keeps this to one fit per origin.

## 5. Step-by-step procedure
1. Fix H and the feature set ([feature_engineering.md](feature_engineering.md)).
2. Implement the **recursive** forecaster with the one-step model.
3. Implement the **pooled direct** forecaster (or per-horizon direct models if H is small).
4. Evaluate both on identical rolling origins; compare the MAE **by horizon** and the bias.
5. Choose the strategy, or a horizon-dependent mix (recursive for small h, direct for large h), if the evidence supports it.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cookbook_data import load_regular_energy_demand
from cookbook_utils import (evaluate_forecaster, summarize_forecasts, make_direct_features,
                            evaluate_direct_model)

df = load_regular_energy_demand()
TEST, H, INITIAL, STEP = 182, 14, 730, 28
dev = df.iloc[:-TEST]
y_dev = dev["demand_mwh"]
LAGS = [1, 2, 3, 7, 14]

def calendar(idx):
    """Known-in-advance features for any set of dates."""
    doy = idx.dayofyear.to_numpy()
    out = pd.DataFrame({f"dow_{d}": (idx.dayofweek == d).astype(int) for d in range(6)}, index=idx)
    out["sin_y"], out["cos_y"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    out["holiday"] = idx.strftime("%m-%d").isin(
        ["01-01", "01-06", "05-01", "08-15", "10-12", "11-01", "12-06", "12-08", "12-25", "12-26"]).astype(int)
    out["post_new_line"] = (idx >= "2022-06-01").astype(int)
    return out

def one_step_table(y):
    X = calendar(y.index)
    for k in LAGS:
        X[f"lag_{k}"] = y.shift(k)
    return X.assign(target=y).dropna()

def recursive_forecaster(y_tr, h, X_tr=None, X_fu=None):
    """Fit a one-step model, then iterate: each prediction becomes a lag for the next step."""
    tab = one_step_table(y_tr)
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))
    feats = [c for c in tab.columns if c != "target"]
    model.fit(tab[feats], tab["target"])
    history = list(y_tr.values)
    future_idx = pd.date_range(y_tr.index[-1] + pd.Timedelta("1D"), periods=h, freq="D")
    cal = calendar(future_idx)
    preds = []
    for i, t in enumerate(future_idx):
        row = cal.iloc[[i]].copy()
        for k in LAGS:
            row[f"lag_{k}"] = history[-k]           # uses PREDICTIONS once i >= k
        p = model.predict(row[feats])[0]
        preds.append(p); history.append(p)
    return np.array(preds)

res_rec = evaluate_forecaster(y_dev, recursive_forecaster, INITIAL, H, STEP, name="recursive ridge")
```

```python
# Pooled direct: same model family, same calendar information (no weather, for a fair comparison)
direct = make_direct_features(df, H=H, temp_col=None, events={"post_new_line": "2022-06-01"})
direct_dev = direct[direct["target_time"] <= y_dev.index[-1]]
cols = [c for c in direct.columns if c not in ("origin", "target_time", "y_target")]
origins = y_dev.index[INITIAL - 1: len(y_dev) - H: STEP]
dow_dummies = pd.get_dummies(direct_dev["dow"], prefix="dow", drop_first=True).astype(int)
direct_dev = pd.concat([direct_dev, dow_dummies], axis=1)
cols_d = [c for c in cols if c != "dow"] + list(dow_dummies.columns)

ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))
res_dir = evaluate_direct_model(direct_dev, ridge, origins, cols_d, H, y_dev, name="pooled direct ridge")

res = pd.concat([res_rec, res_dir])
print(summarize_forecasts(res).round(3))
print(summarize_forecasts(res, by=["model", "h"])[["MAE", "bias"]].unstack(0).iloc[[0, 3, 6, 9, 13]].round(2))
```

```python
# Per-horizon direct models (classic "direct" strategy) with scikit-learn's MultiOutputRegressor:
# one regressor per horizon column, trained on features at the origin.
from sklearn.multioutput import MultiOutputRegressor
wide = direct_dev.pivot_table(index="origin", columns="h", values="y_target")
X_orig = direct_dev[direct_dev["h"] == 1].set_index("origin")[[f"y_lag{k}" for k in range(14)] + ["roll7_mean", "roll28_mean"]]
data = X_orig.join(wide, how="inner").dropna()
O = data.index[-60]                                   # one illustrative origin
train = data[data.index <= O - pd.Timedelta(days=H)]  # targets of training rows must end before O
mo = MultiOutputRegressor(make_pipeline(StandardScaler(), RidgeCV())).fit(train[X_orig.columns], train[list(range(1, H + 1))])
print("direct (14 models) forecast from", O.date(), ":", mo.predict(data.loc[[O], X_orig.columns]).round(1))
```

## 7. Interpretation
- **MAE by horizon:** both strategies have similar errors at h = 1. The recursive errors typically grow faster with h, because the lag features increasingly consist of predictions. A direct model with `y_same_weekday` and calendar features degrades more slowly.
- **Bias by horizon:** a growing bias in the recursive model means systematic errors compound (e.g. it under-predicts after the level shift and then feeds those under-predictions back in).
- **If recursive wins at short h and direct at long h,** a horizon-dependent combination is legitimate, provided it is validated on the same origins.
- **MultiOutputRegressor training rows** must have *all* H targets before the origin. That is why the training set ends H days before O. Forgetting this is a subtle leak.

## 8. Decision framework
- **IF** H is small (1–3) → recursive is simple and adequate.
- **IF** H is large, or the model is imperfect (most real cases) → pooled direct as the default; compare with recursive.
- **IF** exogenous inputs are only available at the origin (not as per-step forecasts) → direct.
- **IF** forecasts must be coherent across horizons (e.g. a smooth weekly profile) → multi-output, or post-smoothing; check the cost.
- **IF** compute is limited → pooled direct (one model) rather than H separate models.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Using true lags (not predictions) inside a recursive evaluation | Optimistic: this evaluates one-step accuracy, not H-step | Feed predictions back in |
| Direct training rows whose targets extend past the origin | Look-ahead leakage | Filter `target_time <= origin` |
| Evaluating multi-step models only at h = 1 | Hides the horizon decay | Report MAE by h |
| Recursive trees on trending data | Runaway drift | Level-relative targets, linear models, or direct |

## 10. Checklist
- [ ] H fixed; strategy candidates implemented (recursive and pooled direct at least).
- [ ] Same origins; MAE and bias by horizon compared.
- [ ] Direct training targets filtered to ≤ origin.
- [ ] Strategy (or horizon mix) justified by validation.

## 11. References
- Taieb, S. B., Bontempi, G., Atiya, A. F. & Sorjamaa, A. (2012). "A review and comparison of strategies for multi-step ahead time series forecasting based on the NN5 forecasting competition." *Expert Systems with Applications* 39(8), 7067–7083. https://doi.org/10.1016/j.eswa.2012.01.039
- Chevillon, G. (2007). "Direct multi-step estimation and forecasting." *J. Economic Surveys* 21(4), 746–785. https://doi.org/10.1111/j.1467-6419.2007.00518.x
- Hyndman, R. J. & Athanasopoulos, G. (2021). *FPP3*, §5.8 (multi-step forecast evaluation). https://otexts.com/fpp3/
- scikit-learn, `MultiOutputRegressor`. https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html
