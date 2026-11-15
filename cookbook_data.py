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
    # extension datasets (sections 09-21)
    "make_process_sensors",
    "make_pump_fleet",
    "make_multirate_streams",
    "make_maintenance_logs",
    "make_surface_defect_images",
    "make_interactions",
    "make_air_quality_stations",
    "make_catalyst_experiments",
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


# ===========================================================================
# EXTENSION DATASETS (sections 09–21)
# ===========================================================================

# ---------------------------------------------------------------------------
# 4. Multivariate process sensors with injected faults (anomaly detection, monitoring)
# ---------------------------------------------------------------------------
def make_process_sensors(days: int = 45, freq: str = "5min", seed: int = 42) -> pd.DataFrame:
    """Heat-exchanger + pump loop sampled every 5 minutes (45 days by default).

    Columns: load_pct (operating regime driver), flow_m3h, inlet_temp_c, coolant_in_c,
    outlet_temp_c, pressure_drop_bar, pump_current_a, vibration_mm_s, regime, fault, is_anomaly.

    Normal physics: flow ∝ load; outlet temperature follows a heat-exchanger
    effectiveness that depends on flow; pressure drop ∝ flow²; current and
    vibration grow with flow. Days 0–9 contain normal operation only (plus isolated
    sensor spikes), suitable as a "training on normal data" period.

    Injected faults (label in ``fault``):
    * ``spike``        — isolated single-sample sensor glitches (point anomalies)
    * ``sensor_drift`` — outlet temperature sensor drifts +3 °C over days 10–12
    * ``process_upset``— inlet temperature step +6 °C on day 20 for 6 hours
    * ``bearing``      — vibration rises over days 28–29.5 while flow is unchanged
                         (contextual: values still within the range seen at high load)
    * ``fouling``      — from day 38 effectiveness decays and pressure drop rises (slow, collective);
                         labelled from day 40, when the efficiency loss becomes operationally relevant
    * ``shutdown`` periods (load 0) are normal planned events, not faults (regime = "shutdown").
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-03-01", periods=int(days * 24 * 60 / pd.Timedelta(freq).total_seconds() * 60), freq=freq)
    n = len(idx)
    t_days = ((idx - idx[0]) / pd.Timedelta("1D")).to_numpy()
    # operating regimes: piecewise-constant load with random durations
    levels, load = np.array([60.0, 80.0, 100.0]), np.empty(n)
    i = 0
    while i < n:
        dur = int(rng.integers(6, 48) * 12)                      # 6–48 h in 5-min steps
        load[i:i + dur] = rng.choice(levels, p=[0.3, 0.45, 0.25])
        i += dur
    for start_day in rng.choice(np.arange(2, days - 2), 4, replace=False):   # planned shutdowns (8 h)
        m = (t_days >= start_day) & (t_days < start_day + 1 / 3)
        load[m] = 0.0
    load = pd.Series(load).rolling(6, min_periods=1).mean().to_numpy()    # ramps (30 min)
    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60
    flow = 0.5 * load + rng.normal(0, 0.6, n)
    inlet = 80 + 1.5 * np.sin(2 * np.pi * (hour - 14) / 24) + rng.normal(0, 0.3, n)
    coolant = 20 + 3 * np.sin(2 * np.pi * (hour - 15) / 24) + rng.normal(0, 0.2, n)
    fouling = 0.02 * np.clip(t_days - 38, 0, None) ** 1.5
    eff = (0.72 - 0.002 * (flow - 40)) * (1 - fouling)
    outlet = inlet - np.where(flow > 1, eff, 0.0) * (inlet - coolant) + rng.normal(0, 0.25, n)
    dp = 0.0009 * flow ** 2 * (1 + 1.5 * fouling) + rng.normal(0, 0.03, n)
    current = 5 + 0.9 * flow + rng.normal(0, 0.5, n)
    bearing = np.where((t_days >= 28) & (t_days < 29.5), 1.2 * np.clip((t_days - 28) / 0.75, 0, 1), 0.0)
    vib = 0.8 + 0.035 * flow + bearing + np.abs(rng.normal(0, 0.12, n))
    upset = (t_days >= 20) & (t_days < 20.25)
    inlet = inlet + 6 * upset
    outlet = outlet + 6 * upset * (1 - 0.7)
    drift = np.clip((t_days - 10) / 2, 0, 1) * ((t_days >= 10) & (t_days < 12)) * 3.0
    outlet_measured = outlet + drift

    df = pd.DataFrame({"load_pct": load.round(1), "flow_m3h": flow.round(2), "inlet_temp_c": inlet.round(2),
                       "coolant_in_c": coolant.round(2), "outlet_temp_c": outlet_measured.round(2),
                       "pressure_drop_bar": dp.round(3), "pump_current_a": current.round(2),
                       "vibration_mm_s": vib.round(3)}, index=idx)
    df.index.name = "timestamp"
    fault = np.full(n, "none", dtype=object)
    fault[(t_days >= 10) & (t_days < 12)] = "sensor_drift"
    fault[upset] = "process_upset"
    fault[(t_days >= 28) & (t_days < 29.5)] = "bearing"
    fault[t_days >= 40] = "fouling"                    # labelled from when it becomes operationally relevant
    sensors = ["flow_m3h", "inlet_temp_c", "outlet_temp_c", "pressure_drop_bar", "pump_current_a", "vibration_mm_s"]
    spikes = rng.choice(n, 25, replace=False)
    for k in spikes:
        col = sensors[rng.integers(len(sensors))]
        df.iloc[k, df.columns.get_loc(col)] *= rng.choice([1.6, 0.4])
        if fault[k] == "none":
            fault[k] = "spike"
    df["regime"] = pd.cut(df["load_pct"], [-1, 5, 70, 90, 101], labels=["shutdown", "low", "nominal", "high"]).astype(str)
    df["fault"] = fault
    df["is_anomaly"] = (df["fault"] != "none").astype(int)
    return df


# ---------------------------------------------------------------------------
# 5. Run-to-failure pump fleet (predictive maintenance, RUL)
# ---------------------------------------------------------------------------
def make_pump_fleet(n_units: int = 80, seed: int = 42) -> pd.DataFrame:
    """Daily aggregated sensor data of a pump fleet, each unit observed until failure.

    Columns: unit_id, day, setting (operating condition A/B/C), vibration_rms,
    bearing_temp_c, discharge_bar, flow_m3h, motor_current_a, rul_days (target),
    plus site. Degradation is exponential in normalised age with unit-specific
    lifetimes (120–360 days); operating settings shift all sensor levels
    (normalise per setting!); ~15% of units have a calibration offset on the
    bearing temperature sensor (drift between sites).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        life = int(rng.integers(120, 361))
        site = rng.choice(["P1", "P2", "P3"])
        offset = rng.normal(4, 1) if rng.random() < 0.15 else 0.0
        b = rng.uniform(2.5, 4.5)
        day = np.arange(1, life + 1)
        health = np.exp(b * (day / life) ** 2) - 1                      # 0 → e^b − 1 at failure
        health = health / (np.exp(b) - 1)                               # normalised 0..1
        setting = rng.choice(["A", "B", "C"], life, p=[0.5, 0.3, 0.2])
        lvl = pd.Series(setting).map({"A": 1.0, "B": 1.25, "C": 0.8}).to_numpy()
        rows.append(pd.DataFrame({
            "unit_id": f"U{u:03d}", "site": site, "day": day, "setting": setting,
            "vibration_rms": (2.0 * lvl + 4.0 * health + rng.normal(0, 0.25, life)).round(3),
            "bearing_temp_c": (55 + 8 * (lvl - 1) + 15 * health ** 1.5 + offset + rng.normal(0, 1.0, life)).round(2),
            "discharge_bar": (6.0 * lvl - 0.8 * health + rng.normal(0, 0.08, life)).round(3),
            "flow_m3h": (100 * lvl - 6 * health + rng.normal(0, 1.5, life)).round(2),
            "motor_current_a": (40 * lvl + 5 * health + rng.normal(0, 0.8, life)).round(2),
            "rul_days": life - day,
        }))
    return pd.concat(rows, ignore_index=True)


def make_multirate_streams(minutes: int = 120, seed: int = 42) -> dict:
    """Three raw sensor streams from one machine with realistic timing problems:
    * ``vibration``  : 1 Hz, regular, with a 3-minute logger gap
    * ``temperature``: irregular, ~every 30 s (event-driven, jitter)
    * ``pressure``   : every 10 s, but its clock is 7 s ahead (offset to correct)
    The true underlying load profile (``load``) is shared, so aligned signals correlate.
    """
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2024-05-01 08:00:00")
    secs = np.arange(minutes * 60)
    load = 0.6 + 0.3 * (secs > 40 * 60) - 0.4 * ((secs > 80 * 60) & (secs < 95 * 60))
    load = pd.Series(load).rolling(60, min_periods=1).mean().to_numpy()
    vib_t = t0 + pd.to_timedelta(secs, unit="s")
    vib = 1.0 + 2.0 * load + rng.normal(0, 0.15, len(secs))
    keep = ~((secs >= 55 * 60) & (secs < 58 * 60))
    vibration = pd.DataFrame({"timestamp": vib_t[keep], "vibration_mm_s": vib[keep].round(3)})
    tt = np.cumsum(rng.uniform(15, 45, int(minutes * 60 / 30) + 10)); tt = tt[tt < minutes * 60]
    temperature = pd.DataFrame({"timestamp": t0 + pd.to_timedelta(tt, unit="s"),
                                "temp_c": (60 + 15 * np.interp(tt, secs, load) + rng.normal(0, 0.3, len(tt))).round(2)})
    ps = np.arange(0, minutes * 60, 10)
    pressure = pd.DataFrame({"timestamp": t0 + pd.to_timedelta(ps + 7, unit="s"),      # clock 7 s ahead
                             "pressure_bar": (3 + 2.5 * np.interp(ps, secs, load) + rng.normal(0, 0.05, len(ps))).round(3)})
    return {"vibration": vibration, "temperature": temperature, "pressure": pressure}


# ---------------------------------------------------------------------------
# 6. Maintenance work-order texts (NLP)
# ---------------------------------------------------------------------------
def make_maintenance_logs(n: int = 3000, seed: int = 42, label_noise: float = 0.04) -> pd.DataFrame:
    """Free-text maintenance work orders written by technicians, with a failure category.

    Categories (imbalanced): bearing, seal_leak, electrical, lubrication, instrumentation.
    The text contains realistic noise: abbreviations (brg, temp, mtr), casing,
    typos, work-order codes, numbers and units, and shared generic vocabulary.
    ``label_noise`` fraction of labels is randomly reassigned (as in real CMMS data).
    """
    rng = np.random.default_rng(seed)
    vocab = {
        "bearing": (["bearing", "brg", "bearings"], ["noise", "grinding noise", "high vibration", "rumbling", "hot", "play"],
                    ["replaced bearing", "changed brg", "bearing swapped", "regreased and replaced bearing"]),
        "seal_leak": (["mechanical seal", "seal", "gland", "packing"], ["leak", "leaking", "dripping", "weeping", "spraying"],
                      ["replaced seal", "seal kit installed", "tightened gland", "repacked"]),
        "electrical": (["motor", "mtr", "cable", "contactor", "breaker", "VFD"], ["tripped", "burnt smell", "no power", "overcurrent", "fault code"],
                       ["reset breaker", "replaced contactor", "rewired", "megger test ok, replaced cable"]),
        "lubrication": (["oil", "lube oil", "grease", "lubricant"], ["low level", "contaminated", "dark", "milky", "dry"],
                        ["topped up oil", "oil change", "regreased", "flushed and refilled"]),
        "instrumentation": (["transmitter", "sensor", "PT", "TT", "flow meter", "gauge"], ["reading wrong", "frozen value", "erratic", "no signal", "out of range"],
                            ["recalibrated", "replaced transmitter", "loop checked", "cleaned sensor tip"]),
    }
    cats = list(vocab)
    p = np.array([0.30, 0.22, 0.20, 0.16, 0.12])
    equipment = ["pump P-101", "pump P-204", "compressor K-12", "fan F-3", "agitator M-7", "conveyor C-2", "pump P-330"]
    generic = ["checked", "inspected on site", "operator reported", "during night shift", "as per PM", "see previous WO",
               "urgent", "safety ok", "isolated equipment", "returned to service", "", "", ""]
    rows = []
    for i in range(n):
        c = rng.choice(cats, p=p)
        obj, sym, act = vocab[c]
        parts = [f"{rng.choice(equipment)}:", rng.choice(obj), rng.choice(sym), "-", rng.choice(generic),
                 rng.choice(act), rng.choice(generic)]
        if rng.random() < 0.3:
            parts.insert(3, f"{rng.integers(40, 95)} C" if c in ("bearing", "electrical") else f"{rng.integers(1, 9)} bar")
        if rng.random() < 0.25:                                        # cross-category confusion words
            other = rng.choice([k for k in cats if k != c])
            parts.append(rng.choice(vocab[other][0]) + " ok")
        text = " ".join(w for w in parts if w)
        if rng.random() < 0.5:
            text = text.upper() if rng.random() < 0.3 else text.lower()
        if rng.random() < 0.1:                                          # typo
            j = rng.integers(1, max(2, len(text) - 1)); text = text[:j] + text[j + 1:]
        text = f"WO-{rng.integers(10000, 99999)} " + text
        rows.append((text, c))
    df = pd.DataFrame(rows, columns=["text", "category"])
    flip = rng.random(n) < label_noise
    df.loc[flip, "category"] = rng.choice(cats, flip.sum())
    df["created"] = pd.Timestamp("2022-01-01") + pd.to_timedelta(np.sort(rng.uniform(0, 730, n)), unit="D")
    return df


# ---------------------------------------------------------------------------
# 7. Surface-defect images (computer vision)
# ---------------------------------------------------------------------------
def make_surface_defect_images(n: int = 1200, size: int = 64, seed: int = 42):
    """Grey-scale images of a machined metal surface with one of four classes:
    'ok', 'scratch' (thin bright line), 'spot' (dark blob), 'crack' (dark jagged line).
    Returns (images uint8 [n, size, size], labels np.array[str], masks bool [n, size, size]).
    Varying texture, illumination gradients and noise make the task non-trivial;
    masks give pixel-level ground truth for segmentation, and bounding boxes can be derived from them.
    """
    import cv2
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    classes = np.array(["ok", "scratch", "spot", "crack"])
    labels = rng.choice(classes, n, p=[0.4, 0.2, 0.2, 0.2])
    imgs = np.empty((n, size, size), np.uint8)
    masks = np.zeros((n, size, size), bool)
    yy, xx = np.mgrid[0:size, 0:size]
    for i, lab in enumerate(labels):
        base = gaussian_filter(rng.normal(0, 1, (size, size)), sigma=rng.uniform(0.8, 2.5))
        base = 128 + 18 * base / base.std()
        ang = rng.uniform(0, 2 * np.pi)
        base += rng.uniform(0, 25) * ((xx * np.cos(ang) + yy * np.sin(ang)) / size - 0.5)   # illumination gradient
        m = np.zeros((size, size), np.uint8)
        if lab == "scratch":
            p1, p2 = rng.integers(0, size, 2), rng.integers(0, size, 2)
            cv2.line(m, tuple(int(v) for v in p1), tuple(int(v) for v in p2), 1, int(rng.integers(1, 3)))
            base = np.where(m > 0, base + rng.uniform(35, 60), base)
        elif lab == "spot":
            c = tuple(int(v) for v in rng.integers(10, size - 10, 2))
            cv2.ellipse(m, c, (int(rng.integers(3, 8)), int(rng.integers(3, 8))), float(rng.uniform(0, 180)), 0, 360, 1, -1)
            base = np.where(m > 0, base - rng.uniform(35, 60), base)
        elif lab == "crack":
            pts = [rng.integers(5, size - 5, 2)]
            for _ in range(int(rng.integers(4, 8))):
                pts.append(np.clip(pts[-1] + rng.integers(-9, 10, 2), 0, size - 1))
            cv2.polylines(m, [np.array(pts, np.int32).reshape(-1, 1, 2)], False, 1, 1)
            base = np.where(m > 0, base - rng.uniform(30, 50), base)
        base += rng.normal(0, rng.uniform(3, 9), (size, size))
        imgs[i] = np.clip(base, 0, 255).astype(np.uint8)
        masks[i] = m > 0
    return imgs, labels, masks


# ---------------------------------------------------------------------------
# 8. User–item interactions (recommender systems)
# ---------------------------------------------------------------------------
def make_interactions(n_users: int = 600, n_items: int = 400, n_factors: int = 6, seed: int = 42):
    """Engineers (users) consulting technical documents / ordering spare parts (items).

    Returns (interactions, items):
    * interactions: user_id, item_id, timestamp, rating (explicit 1–5, available for ~30% of events)
    * items: item_id, category, description (short text), popularity (true latent)
    Preferences follow latent factors plus a power-law item popularity; activity per
    user is heavy-tailed; new items keep appearing over time (cold start).
    """
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 1, (n_users, n_factors))
    V = rng.normal(0, 1, (n_items, n_factors))
    cats = np.array(["pumps", "valves", "motors", "instrumentation", "safety", "lubrication"])
    item_cat = rng.choice(cats, n_items)
    cat_vec = {c: rng.normal(0, 1.0, n_factors) for c in cats}
    V += np.vstack([cat_vec[c] for c in item_cat])                      # items in a category are similar
    pop = rng.pareto(1.2, n_items) + 1
    item_release = rng.uniform(0, 600, n_items) * (rng.random(n_items) < 0.3)   # 30% of items released later
    n_events = np.clip(rng.lognormal(2.5, 0.9, n_users).astype(int), 3, 300)
    rows = []
    for u in range(n_users):
        t = np.sort(rng.uniform(0, 730, n_events[u]))
        for ti in t:
            avail = np.where(item_release <= ti)[0]
            score = U[u] @ V[avail].T / np.sqrt(n_factors) + 0.8 * np.log(pop[avail])
            p = np.exp(score - score.max()); p /= p.sum()
            it = avail[rng.choice(len(avail), p=p)]
            rating = np.clip(np.round(3 + 1.2 * (U[u] @ V[it]) / np.sqrt(n_factors) + rng.normal(0, 0.7)), 1, 5)
            rows.append((f"u{u:04d}", f"i{it:04d}", ti, rating if rng.random() < 0.3 else np.nan))
    inter = pd.DataFrame(rows, columns=["user_id", "item_id", "t_days", "rating"])
    inter["timestamp"] = pd.Timestamp("2023-01-01") + pd.to_timedelta(inter.pop("t_days"), unit="D")
    words = {"pumps": "centrifugal pump impeller casing", "valves": "gate valve actuator stem",
             "motors": "induction motor winding rotor", "instrumentation": "pressure transmitter calibration loop",
             "safety": "relief valve lockout procedure", "lubrication": "grease oil bearing lubrication"}
    items = pd.DataFrame({"item_id": [f"i{i:04d}" for i in range(n_items)], "category": item_cat,
                          "description": [words[c] + " " + rng.choice(["manual", "spare kit", "datasheet", "guide"])
                                          for c in item_cat],
                          "release_day": item_release.round(0), "popularity": pop.round(2)})
    return inter.drop_duplicates(["user_id", "item_id", "timestamp"]).reset_index(drop=True), items


# ---------------------------------------------------------------------------
# 9. Air-quality monitoring stations (geospatial)
# ---------------------------------------------------------------------------
def make_air_quality_stations(n: int = 160, seed: int = 42):
    """NO2 annual mean (µg/m³) at monitoring stations in a 30 km × 30 km area,
    with coordinates in a projected CRS (ETRS89 / UTM zone 30N, EPSG:25830, metres).

    Returns (stations, roads, industry):
    * stations: station_id, x, y, no2, elevation_m (+ station_type)
    * roads: list of ((x1, y1), (x2, y2)) major road segments
    * industry: DataFrame of point sources (x, y, emission)
    NO2 = background + traffic (decays with distance to roads) + industry plumes
          + a smooth spatially correlated field + noise; stations cluster in the city centre.
    """
    rng = np.random.default_rng(seed)
    x0, y0, L = 425000.0, 4460000.0, 30000.0
    centre = np.array([x0 + L / 2, y0 + L / 2])
    n_c = int(0.6 * n)
    pts = np.vstack([centre + rng.normal(0, 3500, (n_c, 2)), np.column_stack([rng.uniform(x0, x0 + L, n - n_c),
                                                                              rng.uniform(y0, y0 + L, n - n_c)])])
    pts = np.clip(pts, [x0, y0], [x0 + L, y0 + L])
    roads = []
    for ang in np.deg2rad([10, 55, 100, 145]):
        d = np.array([np.cos(ang), np.sin(ang)]) * L * 0.7
        roads.append((tuple(centre - d), tuple(centre + d)))
    roads.append(((x0, y0 + 0.2 * L), (x0 + L, y0 + 0.25 * L)))

    def dist_to_segment(p, a, b):
        a, b = np.array(a), np.array(b)
        t = np.clip(((p - a) @ (b - a)) / ((b - a) @ (b - a)), 0, 1)
        return np.linalg.norm(p - (a + t[:, None] * (b - a)), axis=1)

    d_road = np.min([dist_to_segment(pts, a, b) for a, b in roads], axis=0)
    industry = pd.DataFrame({"x": rng.uniform(x0, x0 + L, 4), "y": rng.uniform(y0, y0 + L, 4),
                             "emission": rng.uniform(5, 15, 4)})
    ind = sum(r.emission * np.exp(-np.hypot(pts[:, 0] - r.x, pts[:, 1] - r.y) / 2500) for r in industry.itertuples())
    k = 25
    cx, cy, w = rng.uniform(x0, x0 + L, k), rng.uniform(y0, y0 + L, k), rng.normal(0, 3, k)
    field = sum(w[j] * np.exp(-((pts[:, 0] - cx[j]) ** 2 + (pts[:, 1] - cy[j]) ** 2) / (2 * 4000 ** 2)) for j in range(k))
    elev = 650 + 0.004 * (pts[:, 1] - y0) + rng.normal(0, 10, n)
    no2 = 15 + 28 * np.exp(-d_road / 600) + ind + field - 0.01 * (elev - 650) + rng.normal(0, 2.5, n)
    stations = pd.DataFrame({"station_id": [f"S{i:03d}" for i in range(n)], "x": pts[:, 0].round(1),
                             "y": pts[:, 1].round(1), "elevation_m": elev.round(1), "no2": no2.round(2),
                             "station_type": np.where(d_road < 300, "traffic", "background")})
    return stations, roads, industry


# ---------------------------------------------------------------------------
# 10. Small experimental dataset with replicates (scientific ML)
# ---------------------------------------------------------------------------
def make_catalyst_experiments(n_conditions: int = 24, n_rep: int = 3, seed: int = 42) -> pd.DataFrame:
    """Lab-scale catalytic reactor experiments (a designed experiment with replicates).

    Factors: temperature_K (520–600), pressure_bar (10–40), catalyst_loading_pct (1–5),
    h2_co_ratio (1–3). Response: conversion_pct, from Arrhenius kinetics with first-order
    residence-time conversion X = 1 − exp(−k·τ), plus a run-to-run (replicate) random effect
    and instrument noise (sd ≈ 1.5 points; reported as ``conversion_sd_instrument``).
    Physical constraints: 0 ≤ X ≤ 100, X increases with T and loading.
    Columns also include ``condition_id`` (replicate group) and ``failed_run`` flags
    (~3% of runs with a plugged reactor → anomalously low conversion).
    """
    from scipy.stats import qmc
    rng = np.random.default_rng(seed)
    lhs = qmc.LatinHypercube(d=4, seed=seed).random(n_conditions)
    lo, hi = np.array([520, 10, 1, 1]), np.array([600, 40, 5, 3])
    X = qmc.scale(lhs, lo, hi)
    R, Ea, A = 8.314, 85000.0, 1.2e7
    rows = []
    for c, (T, P, load, ratio) in enumerate(X):
        k = A * np.exp(-Ea / (R * T)) * load ** 0.8 * (P / 20) ** 0.5 * (1 - 0.15 * (ratio - 2) ** 2)
        tau = 2.0
        for r in range(n_rep):
            k_run = k * np.exp(rng.normal(0, 0.06))                    # run-to-run variability
            conv = 100 * (1 - np.exp(-k_run * tau)) + rng.normal(0, 1.5)
            failed = rng.random() < 0.03
            if failed:
                conv *= rng.uniform(0.3, 0.6)
            rows.append({"condition_id": f"C{c:02d}", "replicate": r + 1, "temperature_K": round(T, 1),
                         "pressure_bar": round(P, 2), "catalyst_loading_pct": round(load, 3),
                         "h2_co_ratio": round(ratio, 3), "conversion_pct": float(np.clip(conv, 0, 100).round(2)),
                         "conversion_sd_instrument": 1.5, "failed_run": int(failed)})
    return pd.DataFrame(rows)


if __name__ == "__main__":  # quick smoke test
    for f in (make_process_yield, make_equipment_failure, make_energy_demand, make_process_sensors,
              make_pump_fleet, make_maintenance_logs, make_catalyst_experiments):
        d = f()
        print(f.__name__, d.shape)
        print(d.head(3).to_string(), "\n")
