"""
cookbook_data.py — synthetic but realistic datasets used throughout the cookbook.

Every recipe uses one of three running examples so that you can follow a
single problem from framing to error analysis:

1. make_process_yield()      -> tabular REGRESSION
   Batch chemical reactor: predict product yield (kg) per batch.
2. make_equipment_failure()  -> tabular CLASSIFICATION (binary + multiclass)
   Predictive maintenance: will a machine fail in the next 30 days?
3. make_energy_demand()      -> TIME SERIES forecasting
   Daily (or hourly) electricity demand of an industrial site with
   temperature as an exogenous driver.

The generators deliberately include the problems you will meet in real data:
missing values with different mechanisms (MCAR / MAR / MNAR), sentinel codes,
unit errors, duplicates, skewed and heteroscedastic targets, correlated
features, grouped observations, class imbalance, missing timestamps, level
shifts, and *leakage columns* that look predictive but would not be available
at prediction time.

Because the data-generating process is known, you can check whether the
methods in the recipes recover the truth — a luxury you never have in
real projects, and a good way to build intuition.

Usage
-----
Put this file on your Python path (e.g. run notebooks from the repository
root, or `sys.path.append("/path/to/DS-Cookbook")`) and then:

    from cookbook_data import make_process_yield
    df = make_process_yield()
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "make_process_yield",
    "load_clean_process_yield",
    "make_equipment_failure",
    "make_energy_demand",
    "load_regular_energy_demand",
    "LEAKAGE_COLUMNS",
]

#: Columns that are generated *after* the outcome and must never be used as features.
LEAKAGE_COLUMNS = {
    "process_yield": ["qc_purity_pct"],
    "equipment_failure": ["downtime_hours_30d", "failure_mode"],
}


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# 1. Regression: batch reactor yield
# ---------------------------------------------------------------------------
def make_process_yield(n: int = 2000, seed: int = 42, include_leakage: bool = True) -> pd.DataFrame:
    """Batch-reactor dataset. Target: ``yield_kg`` (continuous, right-skewed,
    multiplicative noise -> heteroscedastic errors).

    Known structure (useful for checking your analysis):
    * Temperature has a catalyst-dependent optimum (non-linear + interaction).
    * ``jacket_temperature_c`` is ~collinear with ``temperature_c``;
      ``feed_purity_pct`` is ~collinear with ``impurity_ppm``.
    * ``catalyst_age_days`` follows a saw-tooth in time (catalyst replaced
      every 120 days) and degrades yield -> drift over time.
    * ``temperature_c``: 2% MCAR missing + 0.5% sentinel value -999.
    * ``impurity_ppm``: MNAR — high values are often missing (sensor saturates).
    * ``ambient_humidity_pct``: MAR — mostly missing for reactor R4.
    * 5 rows have ``yield_kg`` recorded in tonnes instead of kg (unit error).
    * 10 exact duplicate rows (export bug).
    * ``qc_purity_pct`` is measured *after* the batch -> leakage.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2022-01-03 06:00")
    ts = start + pd.to_timedelta(np.cumsum(rng.uniform(6, 10, n)), unit="h")

    reactor = rng.choice(["R1", "R2", "R3", "R4"], n, p=[0.35, 0.30, 0.20, 0.15])
    suppliers = np.array([f"S{i:02d}" for i in range(1, 16)])
    supplier = rng.choice(suppliers, n, p=rng.dirichlet(np.full(15, 0.8)))
    catalyst = rng.choice(["Pt", "Pd", "Ni"], n, p=[0.5, 0.3, 0.2])
    hour = ts.hour.to_numpy()
    shift = np.where((hour >= 6) & (hour < 18), "Day", "Night")

    temperature = rng.normal(186, 7, n)
    jacket_temperature = temperature + 6 + rng.normal(0, 1.2, n)
    pressure = rng.normal(12, 1.5, n)
    feed_rate = rng.gamma(6, 50 / 6, n)  # kg/h, right-skewed
    residence_time = rng.lognormal(np.log(45), 0.25, n)  # min
    reactor_offset = {"R1": 0, "R2": 30, "R3": 60, "R4": 90}
    days = (ts - start).days.to_numpy()
    catalyst_age = (days + np.vectorize(reactor_offset.get)(reactor)) % 120
    doy = ts.dayofyear.to_numpy()
    humidity = 55 + 18 * np.sin(2 * np.pi * (doy - 100) / 365.25) + rng.normal(0, 6, n)
    sup_impurity = dict(zip(suppliers, rng.lognormal(np.log(180), 0.35, 15)))
    impurity = np.vectorize(sup_impurity.get)(supplier) * rng.lognormal(0, 0.45, n)
    feed_purity = 99.9 - impurity / 120 + rng.normal(0, 0.15, n)

    t_opt = np.vectorize({"Pt": 190.0, "Pd": 182.0, "Ni": 195.0}.get)(catalyst)
    sup_eff = dict(zip(suppliers, rng.normal(0, 0.15, 15)))
    reac_eff = np.vectorize({"R1": 0.0, "R2": 0.10, "R3": -0.20, "R4": 0.05}.get)(reactor)
    logit = (
        1.1
        - 0.010 * (temperature - t_opt) ** 2
        + 0.12 * (pressure - 12)
        + 0.8 * np.log(residence_time / 45)
        - 0.005 * catalyst_age
        - 0.0015 * (impurity - 200)
        + np.vectorize(sup_eff.get)(supplier)
        + reac_eff
        - 0.08 * (shift == "Night")
    )
    conversion = _sigmoid(logit)
    yield_kg = feed_rate * 8.0 * conversion * rng.lognormal(0, 0.07, n)

    df = pd.DataFrame(
        {
            "batch_id": [f"B{i:06d}" for i in range(1, n + 1)],
            "start_time": ts,
            "reactor": reactor,
            "supplier": supplier,
            "catalyst": catalyst,
            "shift": shift,
            "temperature_c": temperature.round(2),
            "jacket_temperature_c": jacket_temperature.round(2),
            "pressure_bar": pressure.round(3),
            "feed_rate_kg_h": feed_rate.round(2),
            "residence_time_min": residence_time.round(2),
            "catalyst_age_days": catalyst_age,
            "ambient_humidity_pct": humidity.round(1),
            "impurity_ppm": impurity.round(1),
            "feed_purity_pct": feed_purity.round(3),
            "yield_kg": yield_kg.round(2),
        }
    )
    if include_leakage:
        df["qc_purity_pct"] = (90 + 9 * conversion + rng.normal(0, 0.3, n)).round(2)

    # --- data-quality problems -------------------------------------------
    t = df["temperature_c"].to_numpy(copy=True)
    t[rng.random(n) < 0.02] = np.nan
    t[rng.random(n) < 0.005] = -999.0
    df["temperature_c"] = t

    imp = df["impurity_ppm"].to_numpy(copy=True)
    high = imp > np.quantile(imp, 0.92)
    imp[(high & (rng.random(n) < 0.6)) | (rng.random(n) < 0.01)] = np.nan
    df["impurity_ppm"] = imp

    hum = df["ambient_humidity_pct"].to_numpy(copy=True)
    hum[(reactor == "R4") & (rng.random(n) < 0.7)] = np.nan
    hum[rng.random(n) < 0.02] = np.nan
    df["ambient_humidity_pct"] = hum

    bad = rng.choice(n, 5, replace=False)
    df.loc[bad, "yield_kg"] = (df.loc[bad, "yield_kg"] / 1000).round(4)

    dups = df.sample(10, random_state=seed)
    df = pd.concat([df, dups]).sort_values("start_time").reset_index(drop=True)
    return df


def load_clean_process_yield(seed: int = 42) -> pd.DataFrame:
    """``make_process_yield`` with the data-quality decisions of
    01_data_understanding/data_quality.md applied (dedupe, sentinel -> NaN,
    drop unit-error rows, drop the leakage column), sorted by time.
    Used by the regression recipes so they don't repeat the cleaning code."""
    df = make_process_yield(seed=seed).drop_duplicates()
    df["temperature_c"] = df["temperature_c"].mask(df["temperature_c"] == -999)
    df = df[df["yield_kg"] > 1].drop(columns="qc_purity_pct")
    return df.sort_values("start_time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Classification: predictive maintenance
# ---------------------------------------------------------------------------
def make_equipment_failure(
    n_machines: int = 500, n_months: int = 12, seed: int = 42,
    prevalence: float = 0.05, include_leakage: bool = True,
) -> pd.DataFrame:
    """Monthly snapshots of a machine fleet. Target: ``failure_30d`` (binary,
    ~5% positives). Multiclass target: ``failure_mode``
    ({"none", "bearing", "electrical", "overheating"}).

    Known structure:
    * Repeated snapshots per ``machine_id`` share a latent frailty -> rows are
      NOT independent; use grouped splits.
    * ``vibration_rms`` is MAR-missing for old machines (no sensor fitted).
    * ``firmware`` changes over time (v3 rolled out mid-year) -> temporal drift.
    * ``downtime_hours_30d`` and ``failure_mode`` are outcomes -> leakage.
    """
    rng = np.random.default_rng(seed)
    machines = pd.DataFrame(
        {
            "machine_id": [f"M{i:04d}" for i in range(n_machines)],
            "machine_type": rng.choice(["L", "M", "H"], n_machines, p=[0.5, 0.3, 0.2]),
            "site": rng.choice(["North", "South", "East", "West", "Central"], n_machines),
            "install_age_years": rng.uniform(0.5, 18, n_machines).round(1),
            "frailty": rng.normal(0, 0.6, n_machines),
        }
    )
    dates = pd.date_range("2023-01-31", periods=n_months, freq="ME")
    df = machines.merge(pd.DataFrame({"snapshot_date": dates}), how="cross")
    m = len(df)
    month_idx = df["snapshot_date"].dt.month.to_numpy()

    df["age_years"] = (df["install_age_years"] + (month_idx - 1) / 12).round(2)
    type_load = df["machine_type"].map({"L": 0.55, "M": 0.65, "H": 0.75}).to_numpy()
    df["load_pct"] = (100 * np.clip(rng.beta(8, 4, m) * type_load / 0.66, 0.05, 1.0)).round(1)
    df["operating_hours"] = np.clip(rng.normal(480, 90, m), 50, 720).round(0)
    df["days_since_maintenance"] = rng.integers(0, 365, m)
    health = df["frailty"].to_numpy() + 0.06 * df["age_years"].to_numpy()
    df["vibration_rms"] = (rng.lognormal(np.log(2.0), 0.3, m) * np.exp(0.35 * health)).round(3)
    df["temperature_c"] = (55 + 0.25 * df["load_pct"] + 3 * health + rng.normal(0, 4, m)).round(1)
    df["error_count_7d"] = rng.poisson(np.exp(-0.5 + 0.6 * health))
    df["firmware"] = np.where(
        month_idx >= 7,
        rng.choice(["v2", "v3"], m, p=[0.3, 0.7]),
        rng.choice(["v1", "v2"], m, p=[0.4, 0.6]),
    )

    c_bearing = 1.4 * np.log(df["vibration_rms"].to_numpy() / 2.0) + 0.004 * df["days_since_maintenance"].to_numpy()
    c_heat = 0.08 * (df["temperature_c"].to_numpy() - 72) + 0.6 * (df["machine_type"].to_numpy() == "H")
    c_elec = 0.35 * df["error_count_7d"].to_numpy() - 0.5 * (df["firmware"].to_numpy() == "v3")
    lin = c_bearing + c_heat + c_elec + 1.5 * df["frailty"].to_numpy()

    lo, hi = -15.0, 5.0  # bisection for intercept giving the requested prevalence
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _sigmoid(mid + lin).mean() < prevalence else (lo, mid)
    p_fail = _sigmoid((lo + hi) / 2 + lin)
    y = (rng.random(m) < p_fail).astype(int)
    df["failure_30d"] = y

    contrib = np.column_stack([c_bearing, c_elec, c_heat])
    probs = np.exp(contrib - contrib.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    mode = np.array(["bearing", "electrical", "overheating"])[
        (probs.cumsum(axis=1) > rng.random((m, 1))).argmax(axis=1)
    ]
    df["failure_mode"] = np.where(y == 1, mode, "none")
    if include_leakage:
        df["downtime_hours_30d"] = np.where(
            y == 1, rng.gamma(3, 12, m), np.where(rng.random(m) < 0.03, rng.gamma(1, 3, m), 0.0)
        ).round(1)
    else:
        df = df.drop(columns="failure_mode")

    vib = df["vibration_rms"].to_numpy(copy=True)
    vib[(df["install_age_years"].to_numpy() > 12) & (rng.random(m) < 0.6)] = np.nan
    df["vibration_rms"] = vib
    tmp = df["temperature_c"].to_numpy(copy=True)
    tmp[rng.random(m) < 0.01] = np.nan
    df["temperature_c"] = tmp

    return df.drop(columns=["frailty", "install_age_years"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Time series: industrial site energy demand
# ---------------------------------------------------------------------------
HOLIDAYS_MMDD = ["01-01", "01-06", "05-01", "08-15", "10-12", "11-01", "12-06", "12-08", "12-25", "12-26"]


def make_energy_demand(
    start: str = "2020-01-01", end: str = "2023-12-31", freq: str = "D", seed: int = 42,
    drop_timestamps: bool = True,
) -> pd.DataFrame:
    """Electricity demand (MWh per period) of an industrial site.

    ``freq="D"`` (daily, default) or ``freq="h"`` (hourly; adds a daily profile).

    Known structure:
    * Linear upward trend + level shift (+15%) from 2022-06-01 (new line).
    * Weekly seasonality (weekends lower) and yearly seasonality driven by
      temperature (heating below 15 °C, cooling above 22 °C).
    * Holidays reduce demand (``is_holiday`` is known in advance; temperature
      is NOT known in advance — it must itself be forecast).
    * AR(1) noise -> residual autocorrelation if the model misses dynamics.
    * Missing timestamps: ~0.5% random plus a 6-day meter outage (rows
      absent, not NaN), a few NaN values and one spike (meter glitch).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq=freq)
    n = len(idx)
    t_days = (idx - idx[0]) / pd.Timedelta("1D")
    doy = idx.dayofyear.to_numpy()

    temp_noise = np.zeros(n)
    eps = rng.normal(0, 2.0 if freq == "D" else 0.6, n)
    phi_t = 0.7 if freq == "D" else 0.97
    for i in range(1, n):
        temp_noise[i] = phi_t * temp_noise[i - 1] + eps[i]
    temperature = 13 + 10 * np.sin(2 * np.pi * (doy - 110) / 365.25) + temp_noise
    if freq != "D":
        temperature = temperature + 4 * np.sin(2 * np.pi * (idx.hour.to_numpy() - 9) / 24)

    is_holiday = idx.strftime("%m-%d").isin(HOLIDAYS_MMDD).astype(int)
    dow_factor = np.select([idx.dayofweek == 5, idx.dayofweek == 6], [0.82, 0.70], 1.0)
    level = 100 + 0.012 * t_days.to_numpy()
    level = level * np.where(idx >= pd.Timestamp("2022-06-01"), 1.15, 1.0)
    weather = 1.6 * np.clip(15 - temperature, 0, None) + 2.2 * np.clip(temperature - 22, 0, None)
    mean = (level + weather) * dow_factor * np.where(is_holiday == 1, 0.65, 1.0)

    ar = np.zeros(n)
    e = rng.normal(0, 3.0, n)
    for i in range(1, n):
        ar[i] = 0.6 * ar[i - 1] + e[i]
    demand = mean + ar
    if freq != "D":
        h = idx.hour.to_numpy()
        profile = 0.6 + 0.5 * np.exp(-((h - 11) ** 2) / 18) + 0.3 * np.exp(-((h - 16) ** 2) / 8)
        demand = demand * profile / 24

    df = pd.DataFrame(
        {"timestamp": idx, "demand_mwh": demand.round(3), "temperature_c": temperature.round(2),
         "is_holiday": is_holiday}
    )
    spike = rng.integers(n // 4, n // 2)
    df.loc[spike, "demand_mwh"] *= 3.5
    df.loc[rng.choice(n, 4, replace=False), "demand_mwh"] = np.nan
    if drop_timestamps:
        keep = rng.random(n) > 0.005
        o_start = pd.Timestamp("2021-03-10")
        keep &= ~((df["timestamp"] >= o_start) & (df["timestamp"] < o_start + pd.Timedelta("6D")))
        df = df[keep]
    return df.reset_index(drop=True)


def load_regular_energy_demand(seed: int = 42) -> pd.DataFrame:
    """Daily demand on a complete, regular DatetimeIndex with the decisions of
    05_time_series/preprocessing.md applied: missing days re-inserted, the
    meter-glitch spike replaced by NaN, gaps filled by time interpolation
    (``demand_was_missing`` flags them). Index name: ``timestamp``; freq "D"."""
    df = make_energy_demand(seed=seed).set_index("timestamp").asfreq("D")
    d = df["demand_mwh"]
    # reference = rolling median of the same weekday; known holidays are never flagged
    ref = d.groupby(d.index.dayofweek).transform(lambda g: g.rolling(5, center=True, min_periods=2).median())
    resid = d - ref
    mad = resid.abs().rolling(61, center=True, min_periods=10).median()
    holiday = pd.Series(d.index.strftime("%m-%d").isin(HOLIDAYS_MMDD), index=d.index)
    d = d.mask((resid.abs() > 10 * 1.4826 * mad) & ~holiday)
    df["demand_was_missing"] = d.isna().astype(int)
    df["demand_mwh"] = d.interpolate(method="time", limit_direction="both")
    df["temperature_c"] = df["temperature_c"].interpolate(method="time", limit_direction="both")
    idx = df.index
    df["is_holiday"] = idx.strftime("%m-%d").isin(HOLIDAYS_MMDD).astype(int)
    return df


if __name__ == "__main__":  # quick smoke test
    for f in (make_process_yield, make_equipment_failure, make_energy_demand):
        d = f()
        print(f.__name__, d.shape)
        print(d.head(3).to_string(), "\n")
