# Model Interpretation: Coefficients, Permutation Importance, PDP/ICE and SHAP

> **Folder:** 06_model_improvement · **Prev:** [feature_selection.md](feature_selection.md) · **Next:** [iterative_improvement.md](../06_model_improvement/iterative_improvement.md) · **Related:** [problem_framing.md](../00_methodology/problem_framing.md) (prediction vs causation), [20_advanced_methodologies](../20_advanced_methodologies/) (robustness and sensitivity analysis)

---

## 1. Purpose
Understand **what a model has learned**. Use that to (a) check it against domain knowledge, which catches leakage, artefacts and implausible behaviour; (b) explain predictions to stakeholders; and (c) generate hypotheses for improvement and for experiments. Every claim must stay within what these tools can actually support.

## 2. When to use it
- After evaluation, before deployment: a sanity check of the learned relationships.
- When stakeholders need explanations (global or for individual predictions).
- During error analysis, to see why particular cases fail.

## 3. Conceptual foundation
| Tool | Question | Scope | Pitfalls |
|---|---|---|---|
| **Coefficients** (linear/logistic) | Effect of one unit/std change, holding the others fixed | Global | Depend on scaling and collinearity; conditional on the other features |
| **Impurity importance** (`feature_importances_` of trees) | How much the splits on a feature reduce the training loss | Global | Computed on **training** data; biased towards high-cardinality/continuous features; no direction |
| **Permutation importance** | How much the validation score drops when a feature's values are shuffled | Global | Correlated features share importance; shuffling creates unrealistic combinations |
| **Partial dependence (PDP)** | Average prediction as a function of one/two features | Global, shape | Averages hide heterogeneity; extrapolates into unrealistic regions when features are correlated |
| **ICE curves** | PDP per individual row | Local + heterogeneity | Busy plots; same correlation caveat |
| **SHAP values** | Additive attribution of a prediction to the features (relative to a baseline) | Local, aggregatable to global | Expensive for non-tree models; depends on the background data; **not causal** |
| **ALE plots** (not covered in code) | Effect shape using conditional distributions | Global | Better than PDP with correlated features |

**The crucial limit: these tools explain the *model*, not the *world*.** "The model relies on vibration" ≠ "vibration causes failure". Under correlation, the model can rely on a proxy. For causal claims you need designed experiments or causal methods ([problem_framing.md](../00_methodology/problem_framing.md) §3.1).

## 4. Assumptions and limitations
- Interpretation of a poorly performing model is meaningless. Interpret models that generalise.
- Compute permutation importance on **held-out** data; on training data it reflects overfitting.
- PDP/ICE/SHAP with strongly correlated features can be misleading (evaluating unrealistic inputs). Use grouped importance, ALE or conditional explanations, and state the caveat.
- SHAP for tree models (`TreeExplainer`) is fast and exact for the model. KernelSHAP is slow and approximate.

## 5. Step-by-step procedure
1. **Global importance** on validation data (permutation; grouped for correlated clusters).
2. **Effect shapes** for the top features: PDP + ICE (check against physics and domain expectations: direction, monotonicity, optimum location).
3. **Interactions:** a 2-D PDP or SHAP interaction values for suspected pairs (e.g. temperature × catalyst).
4. **Local explanations** (SHAP) for representative and for surprising predictions (large errors, top-risk cases).
5. **Sanity checks:** any implausible effect is a lead. Investigate leakage, data artefacts, extrapolation or confounding.
6. **Communicate:** the global ranking with uncertainty (repeats); shapes with the data density (rug plots); local explanations with clear caveats.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

from cookbook_data import load_clean_process_yield

df = load_clean_process_yield()
num = ["temperature_c", "jacket_temperature_c", "pressure_bar", "feed_rate_kg_h", "residence_time_min",
       "catalyst_age_days", "impurity_ppm", "ambient_humidity_pct"]
cat = ["reactor", "catalyst", "shift", "supplier"]
cut = int(0.8 * len(df))
# Encode categories as float codes (keeps PDP and SHAP happy) and remember the mapping for display.
X = df[num + cat].astype({"catalyst_age_days": float})          # PDP requires float numerics
codes = {}
for c in cat:
    X[c] = X[c].astype("category")
    codes[c] = dict(enumerate(X[c].cat.categories))
    X[c] = X[c].cat.codes.astype(float)
y = np.log(df["yield_kg"])
X_tr, X_va, y_tr, y_va = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]

model = HistGradientBoostingRegressor(categorical_features=cat, learning_rate=0.05, max_iter=500,
                                      random_state=0).fit(X_tr, y_tr)
print("validation MAE(log):", round(np.mean(np.abs(y_va - model.predict(X_va))), 4))

# 1. Permutation importance on VALIDATION data, repeated for uncertainty
pi = permutation_importance(model, X_va, y_va, scoring="neg_mean_absolute_error", n_repeats=10, random_state=0, n_jobs=-1)
imp = pd.DataFrame({"mean": pi.importances_mean, "std": pi.importances_std}, index=X.columns).sort_values("mean")
imp["mean"].plot.barh(xerr=imp["std"], figsize=(6, 4), title="permutation importance (ΔMAE, validation)")
plt.tight_layout(); plt.show()
print(imp.sort_values("mean", ascending=False).round(4))
```

```python
# 2–3. PDP + ICE for the top features; 2-D PDP for a suspected interaction
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
PartialDependenceDisplay.from_estimator(model, X_va, ["temperature_c"], kind="both", subsample=200,
                                        ice_lines_kw={"alpha": 0.1}, random_state=0, ax=ax[0])
PartialDependenceDisplay.from_estimator(model, X_va, ["catalyst_age_days"], kind="average", ax=ax[1])
PartialDependenceDisplay.from_estimator(model, X_va, ["catalyst"], categorical_features=["catalyst"], ax=ax[2])
ax[2].set_xlabel(f"catalyst {codes['catalyst']}")
plt.tight_layout(); plt.show()

# Interaction check: temperature effect per catalyst (conditional PDP by subsetting)
fig, ax = plt.subplots(figsize=(6, 4))
inv = {v: k for k, v in codes["catalyst"].items()}
for c in ["Pt", "Pd", "Ni"]:
    sub = X_va[X_va["catalyst"] == inv[c]]
    grid = np.linspace(170, 200, 31)
    preds = [model.predict(sub.assign(temperature_c=t)).mean() for t in grid]
    ax.plot(grid, preds, label=c)
ax.set(xlabel="temperature_c", ylabel="mean predicted log(yield)", title="temperature effect by catalyst"); ax.legend()
plt.show()
```

```python
# 4. SHAP: global summary and local explanation (TreeExplainer: exact and fast for tree ensembles)
import shap
explainer = shap.TreeExplainer(model)
X_sample = X_va.sample(300, random_state=0)
sv = explainer(X_sample)                               # Explanation object (values, base_values, data)
shap.plots.beeswarm(sv, max_display=10, show=False); plt.tight_layout(); plt.show()

i = int(np.argmax(np.abs(y_va.loc[X_sample.index] - model.predict(X_sample))))  # the worst-predicted sample
print("actual:", round(float(y_va.loc[X_sample.index[i]]), 3), " predicted:", round(float(model.predict(X_sample.iloc[[i]])[0]), 3))
shap.plots.waterfall(sv[i], max_display=10, show=False); plt.tight_layout(); plt.show()
```

> **Note on categorical inputs.** The model above receives categories as float codes, with `categorical_features=cat`, so HGB still treats them as categories. Interpretation tools (PDP's fast recursion method, SHAP's `TreeExplainer`) need numeric arrays, and string/pandas-`category` columns often fail with them, especially with the pandas 3 string dtype. Keep the `codes` mapping to label the plots.

## 7. Interpretation
- **Permutation importance:** `feed_rate_kg_h` and `temperature_c` lead, followed by `catalyst`. Then come `jacket_temperature_c`, `residence_time_min`, `pressure_bar`, `catalyst_age_days`, `supplier` and `impurity_ppm`. `ambient_humidity_pct` and `shift` are ≈ 0, which matches the data-generating process (humidity has no effect; the shift effect is tiny). Note the **non-zero importance of `jacket_temperature_c`**: it has no causal role, but it is an almost perfect proxy for reactor temperature and, unlike `temperature_c`, it is never missing. The model uses it when the temperature sensor drops out. This is a typical example of the model relying on a proxy. It is fine for prediction and wrong as a causal conclusion. Correlated features also share importance ([feature_selection.md](feature_selection.md): grouped importance). Importance close to 0 with a std overlapping 0 means "not used by the model".
- **PDP/ICE of temperature:** an inverted U. ICE lines with different shapes or peak locations indicate an interaction. **Conditional curves by catalyst** show optima at roughly 182 (Pd), 190 (Pt) and 195 °C (Ni), recovering the true mechanism. This is a *hypothesis-generating* result; confirm it with lab or DoE data before acting.
- **PDP of catalyst age:** decreasing yield with age. Consistent with deactivation, and a plausible basis for a replacement schedule discussion (again, causal confirmation needed).
- **SHAP beeswarm:** the global ranking plus direction (high feed → high SHAP). Points coloured by feature value show monotonic and non-monotonic effects.
- **Waterfall for the worst prediction:** which features pushed the prediction away from the truth. Typical findings are an unusual supplier, an extreme temperature, or missing impurity. These are leads for error analysis.
- **Red flags:** an effect with an implausible direction (e.g. yield increasing with catalyst age); a dominant feature you would not expect (leakage?); steps at strange values (data artefacts, e.g. a sentinel value treated as real).

## 8. Decision framework
| Need | Tool |
|---|---|
| Which features does the model use? | Permutation importance (validation, repeated); grouped for correlated clusters |
| What shape does an effect have? | PDP + ICE (ALE if features are strongly correlated) |
| Do effects interact? | ICE heterogeneity, 2-D PDP, conditional PDPs, SHAP interaction values |
| Why this prediction? | SHAP waterfall/force plot; compare with similar cases |
| Interpretability is a hard requirement | Use an inherently interpretable model (linear + domain features, GAM-like splines, shallow trees), and verify that it performs comparably |
| Stakeholder asks "what should we change?" | Explain the difference between association and causation; propose experiments |

- **IF** the explanations contradict domain knowledge → investigate the data/model before presenting them.
- **IF** importance rankings change a lot between folds → the model's reliance is unstable. Report the ranges, and avoid strong claims.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Impurity importance as "the importance" | Biased rankings | Permutation importance on validation data |
| Interpreting importance as causal effect | Wrong interventions | State the predictive scope; experiments for causation |
| Ignoring correlation in importance/PDP | Misleading low importance / unrealistic PDP regions | Grouped importance; ALE; caveats |
| Explaining a poor model | Explaining noise | Interpret only validated models |
| SHAP on training data only | Explains fitting, not generalisation | Use validation/representative data |
| Cherry-picking local explanations | Biased narrative | Show representative + extreme cases |

## 10. Checklist
- [ ] The model performs well on validation data before interpretation.
- [ ] Permutation importance on validation data, with repeats; grouped for correlated features.
- [ ] PDP/ICE for the top features; shapes checked against domain knowledge.
- [ ] Interactions investigated where suspected.
- [ ] SHAP local explanations for representative and surprising cases.
- [ ] Red flags investigated (leakage, artefacts, implausible directions).
- [ ] Causal caveat communicated.

## 11. References
- Molnar, C. (2022). *Interpretable Machine Learning*, 2nd ed. https://christophm.github.io/interpretable-ml-book/ (chapters on PDP, ICE, ALE, permutation importance, SHAP)
- scikit-learn User Guide, *Permutation feature importance*; *Partial Dependence and ICE plots*. https://scikit-learn.org/stable/inspection.html
- Lundberg, S. M. & Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS 30*. https://arxiv.org/abs/1705.07874
- Lundberg, S. M. et al. (2020). "From local explanations to global understanding with explainable AI for trees." *Nature Machine Intelligence* 2, 56–67. https://doi.org/10.1038/s42256-019-0138-9
- Strobl, C., Boulesteix, A.-L., Zeileis, A. & Hothorn, T. (2007). "Bias in random forest variable importance measures." *BMC Bioinformatics* 8, 25. https://doi.org/10.1186/1471-2105-8-25
- Apley, D. W. & Zhu, J. (2020). "Visualizing the effects of predictor variables in black box supervised learning models." *JRSS B* 82(4), 1059–1086 (ALE).
- SHAP documentation. https://shap.readthedocs.io/
