from __future__ import annotations

import math

import numpy as np
from scipy.stats import rankdata


EPS = 1e-6


def stable_sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x)
    out = np.where(arr >= 0, 1.0 / (1.0 + np.exp(-arr)), np.exp(arr) / (1.0 + np.exp(arr)))
    if np.isscalar(x):
        return float(out)
    return out


def safe_div(a: np.ndarray, b: np.ndarray | float) -> np.ndarray:
    return np.divide(a, np.asarray(b) + EPS)


def nan_to_num(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def rank_rows(x: np.ndarray) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(x.shape[0]):
        row = x[i]
        mask = np.isfinite(row)
        if mask.sum() < 2:
            continue
        out[i, mask] = rankdata(row[mask], method="average") / mask.sum()
    return out


def rowwise_spearman(x: np.ndarray, y: np.ndarray, min_count: int = 8) -> np.ndarray:
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    vals: list[float] = []
    for i in range(x.shape[0]):
        mask = np.isfinite(x[i]) & np.isfinite(y[i])
        if mask.sum() < min_count:
            vals.append(np.nan)
            continue
        xr = rankdata(x[i, mask], method="average")
        yr = rankdata(y[i, mask], method="average")
        xr = xr - xr.mean()
        yr = yr - yr.mean()
        denom = math.sqrt(float(np.dot(xr, xr) * np.dot(yr, yr)))
        vals.append(float(np.dot(xr, yr) / denom) if denom > EPS else np.nan)
    return np.asarray(vals, dtype=float)


def rowwise_pearson(x: np.ndarray, y: np.ndarray, min_count: int = 8) -> np.ndarray:
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    vals: list[float] = []
    for i in range(x.shape[0]):
        mask = np.isfinite(x[i]) & np.isfinite(y[i])
        if mask.sum() < min_count:
            vals.append(np.nan)
            continue
        xv = x[i, mask] - x[i, mask].mean()
        yv = y[i, mask] - y[i, mask].mean()
        denom = math.sqrt(float(np.dot(xv, xv) * np.dot(yv, yv)))
        vals.append(float(np.dot(xv, yv) / denom) if denom > EPS else np.nan)
    return np.asarray(vals, dtype=float)


def mean_ic_icir(daily_ic: np.ndarray) -> tuple[float, float, int]:
    finite = daily_ic[np.isfinite(daily_ic)]
    if finite.size == 0:
        return 0.0, 0.0, 0
    mean = float(finite.mean())
    std = float(finite.std())
    return mean, mean / max(std, EPS), int(finite.size)
