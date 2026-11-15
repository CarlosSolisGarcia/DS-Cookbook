"""
cookbook_utils.py — small, reusable helpers shared by several recipes.

These functions are developed and explained step by step in the recipes
named in each docstring. They are collected here so that later recipes can
reuse them without copying code. Read the explaining recipe before using one.
"""
from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
import pandas as pd

__all__ = ["mase", "smape", "rolling_origin_splits", "evaluate_forecaster", "summarize_forecasts",
           "make_direct_features", "evaluate_direct_model"]


# ---------------------------------------------------------------------------
# Forecast metrics (explained in 05_time_series/evaluation.md)
# ---------------------------------------------------------------------------
def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in %, the 0–200% variant: mean(2|y-ŷ| / (|y|+|ŷ|)).
    Defined as 0 when both are 0."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred)
    ratio = np.where(denom == 0, 0.0, 2 * np.abs(y_true - y_pred) / np.where(denom == 0, 1, denom))
    return 100 * ratio.mean()


def mase(y_true, y_pred, y_train, m: int = 1) -> float:
    """Mean Absolute Scaled Error (Hyndman & Koehler, 2006).
    Scale = in-sample MAE of the seasonal-naive forecast with period m on y_train.
    MASE < 1 means better than the in-sample seasonal-naive one-step forecast."""
    y_train = np.asarray(y_train, float)
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))) / scale


# ---------------------------------------------------------------------------
# Rolling-origin evaluation (explained in 05_time_series/validation.md)
# ---------------------------------------------------------------------------
def rolling_origin_splits(n: int, initial: int, horizon: int, step: int,
                          window: int | None = None, gap: int = 0) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) for rolling-origin evaluation.

    initial : size of the first training set
    horizon : number of steps forecast from each origin
    step    : how far the origin moves between folds
    window  : None -> expanding window; int -> sliding window of that length
    gap     : steps skipped between the end of training and the first forecast
              (e.g. data latency: forecasts made on Monday with data up to Saturday)
    """
    origin = initial
    while origin + gap + horizon <= n:
        start = 0 if window is None else max(0, origin - window)
        yield np.arange(start, origin), np.arange(origin + gap, origin + gap + horizon)
        origin += step


ForecastFn = Callable[[pd.Series, int, "pd.DataFrame | None", "pd.DataFrame | None"], np.ndarray]


def evaluate_forecaster(y: pd.Series, forecaster: ForecastFn, initial: int, horizon: int, step: int,
                        X: pd.DataFrame | None = None, window: int | None = None, gap: int = 0,
                        season: int = 7, name: str = "model") -> pd.DataFrame:
    """Run `forecaster(y_train, h, X_train, X_future) -> array(h)` over rolling origins.

    Returns a long DataFrame: one row per (origin, horizon step) with y, yhat,
    the absolute error and the MASE scale computed on that origin's training data.
    """
    rows = []
    for tr, te in rolling_origin_splits(len(y), initial, horizon, step, window, gap):
        y_tr = y.iloc[tr]
        X_tr = None if X is None else X.iloc[tr]
        X_fu = None if X is None else X.iloc[te]
        yhat = np.asarray(forecaster(y_tr, len(te), X_tr, X_fu), float)
        scale = np.mean(np.abs(y_tr.values[season:] - y_tr.values[:-season]))
        rows.append(pd.DataFrame({
            "model": name, "origin": y.index[tr[-1]], "h": np.arange(1, len(te) + 1),
            "timestamp": y.index[te], "y": y.iloc[te].values, "yhat": yhat, "scale": scale}))
    out = pd.concat(rows, ignore_index=True)
    out["abs_err"] = (out["y"] - out["yhat"]).abs()
    return out


def summarize_forecasts(res: pd.DataFrame, by: str | list[str] = "model") -> pd.DataFrame:
    """Aggregate a long forecast-results table into MAE, RMSE, sMAPE, MASE (and n origins)."""
    def agg(g):
        e = g["y"] - g["yhat"]
        return pd.Series({"MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()),
                          "sMAPE_%": smape(g["y"], g["yhat"]),
                          "MASE": (e.abs() / g["scale"]).mean(),
                          "bias": -e.mean(),   # mean(ŷ − y): > 0 means over-forecasting
                          "n_origins": g["origin"].nunique()})
    return res.groupby(by).apply(agg, include_groups=False)


# ---------------------------------------------------------------------------
# Direct multi-horizon ML forecasting (explained in 05_time_series/feature_engineering.md
# and 05_time_series/multi_step_forecasting.md)
# ---------------------------------------------------------------------------
def make_direct_features(df: pd.DataFrame, H: int, target: str = "demand_mwh", n_lags: int = 14,
                         temp_col: str | None = "temperature_c", temp_noise=(0.8, 0.15),
                         events: dict | None = None, seed: int = 0) -> pd.DataFrame:
    """Build a 'pooled direct' training table: one row per (origin o, horizon h).

    Features use ONLY information available at origin o:
      * y_lag0..y_lag{n_lags-1}: y at o, o-1, ... (lag0 = last observed value)
      * rolling statistics of y ending at o
      * y_same_weekday: last observed value with the same weekday as the target day
    plus information about the TARGET day t = o + h that is known in advance:
      * calendar (day of week, Fourier terms of day of year), holiday flag, known events
      * a simulated temperature FORECAST for day t (actual + noise growing with h),
        because actual future temperature is not known at o.
    Target: y at t. Columns 'origin', 'h', 'target_time' identify the row.
    Requires a regular DatetimeIndex (daily).
    """
    rng = np.random.default_rng(seed)
    y = df[target]
    base = pd.DataFrame(index=df.index)
    for k in range(n_lags):
        base[f"y_lag{k}"] = y.shift(k)
    base["roll7_mean"] = y.rolling(7).mean()
    base["roll28_mean"] = y.rolling(28).mean()
    base["roll7_std"] = y.rolling(7).std()
    blocks = []
    for h in range(1, H + 1):
        X = base.copy()
        t = df.index + pd.Timedelta(days=h)
        X["h"] = h
        X["origin"] = df.index
        X["target_time"] = t
        k = int(7 * np.ceil(h / 7) - h)                       # lag (from origin) of the same weekday
        X["y_same_weekday"] = y.shift(k)
        X["dow"] = t.dayofweek
        doy = t.dayofyear.to_numpy()
        for j in (1, 2):
            X[f"sin_year{j}"] = np.sin(2 * np.pi * j * doy / 365.25)
            X[f"cos_year{j}"] = np.cos(2 * np.pi * j * doy / 365.25)
        X["is_holiday_t"] = df["is_holiday"].shift(-h).to_numpy() if "is_holiday" in df else 0
        if temp_col is not None:
            t_true = df[temp_col].shift(-h).to_numpy()
            sd = temp_noise[0] + temp_noise[1] * h
            temp_fc = t_true + rng.normal(0, sd, len(df))
            X["temp_fc"] = temp_fc
            X["heat_fc"] = np.clip(15 - temp_fc, 0, None)
            X["cool_fc"] = np.clip(temp_fc - 22, 0, None)
        for name, start in (events or {}).items():
            X[name] = (t >= pd.Timestamp(start)).astype(int)
        X["y_target"] = y.shift(-h).to_numpy()
        blocks.append(X)
    out = pd.concat(blocks).dropna().reset_index(drop=True)
    return out


def evaluate_direct_model(feats: pd.DataFrame, model_factory, origins, feature_cols, H: int,
                          y: pd.Series, season: int = 7, name: str = "model", target_transform=None) -> pd.DataFrame:
    """Rolling-origin evaluation of a pooled direct model.

    For each origin O: train on rows whose TARGET time is <= O (so no training target
    lies in the future of the origin), predict the H rows with origin == O.
    target_transform: None, or the name of a feature column (e.g. "roll28_mean", "y_same_weekday")
        whose value is subtracted from the target before fitting and added back to the prediction.
        Removing the recent level lets tree models cope with trends they cannot extrapolate.
    Returns the same long format as evaluate_forecaster.
    """
    res = []
    for O in origins:
        tr = feats[feats["target_time"] <= O]
        te = feats[feats["origin"] == O].sort_values("h")
        if len(te) < H:
            continue
        m = model_factory()
        if target_transform is not None:                      # model y_target - reference column
            ref_tr, ref_te = tr[target_transform], te[target_transform].to_numpy()
            m.fit(tr[feature_cols], tr["y_target"] - ref_tr)
            yhat = m.predict(te[feature_cols]) + ref_te
        else:
            m.fit(tr[feature_cols], tr["y_target"])
            yhat = m.predict(te[feature_cols])
        y_hist = y[y.index <= O].to_numpy()
        scale = np.mean(np.abs(y_hist[season:] - y_hist[:-season]))
        res.append(pd.DataFrame({"model": name, "origin": O, "h": te["h"].to_numpy(),
                                 "timestamp": te["target_time"].to_numpy(), "y": te["y_target"].to_numpy(),
                                 "yhat": yhat, "scale": scale}))
    out = pd.concat(res, ignore_index=True)
    out["abs_err"] = (out["y"] - out["yhat"]).abs()
    return out
