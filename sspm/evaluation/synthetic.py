from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sspm.core.math_utils import nan_to_num, rank_rows
from sspm.core.operators import Add, Delay, Delta, Div, Log, Sub, TsMean, TsStd


@dataclass(slots=True)
class MarketData:
    features: dict[str, np.ndarray]
    target: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.target.shape


def make_synthetic_market(n_days: int = 720, n_assets: int = 120, seed: int = 0) -> MarketData:
    """Create an OHLCV panel with persistent cross-sectional alpha structure."""

    rng = np.random.default_rng(seed)
    market = rng.normal(0.0, 0.008, size=(n_days, 1))
    style = rng.normal(0.0, 0.006, size=(1, n_assets))
    shocks = rng.normal(0.0, 0.018, size=(n_days, n_assets))
    returns = market + 0.15 * style + shocks
    close = 20.0 * np.exp(np.cumsum(returns, axis=0))
    open_ = close * (1.0 + rng.normal(0.0, 0.004, size=close.shape))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.006, 0.004, size=close.shape)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.006, 0.004, size=close.shape)))
    volume_base = rng.lognormal(mean=13.0, sigma=0.45, size=(1, n_assets))
    volume = volume_base * (1.0 + 6.0 * np.abs(returns) + rng.lognormal(0.0, 0.15, size=close.shape))
    vwap = (open_ + high + low + close) / 4.0

    features = {
        "open_": open_,
        "close": close,
        "high": high,
        "low": low,
        "volume": volume,
        "vwap": vwap,
    }

    ret1 = Div(Delta(close, 1), Add(Delay(close, 1), 1e-6))
    mom5 = Div(Delta(close, 5), Add(Delay(close, 5), 1e-6))
    reversal3 = -Div(Delta(close, 3), Add(Delay(close, 3), 1e-6))
    vol = -TsStd(ret1, 20)
    volume_spike = Delta(Log(Add(volume, 1.0)), 5)
    intraday = Div(Sub(close, open_), Add(Sub(high, low), 1e-6))
    smooth_pressure = TsMean(rank_rows(volume_spike) * rank_rows(intraday), 5)

    signal = (
        0.050 * rank_rows(mom5)
        + 0.045 * rank_rows(reversal3)
        + 0.035 * rank_rows(vol)
        + 0.035 * rank_rows(volume_spike)
        + 0.030 * rank_rows(intraday)
        + 0.025 * rank_rows(smooth_pressure)
    )
    finite = np.isfinite(signal)
    counts = finite.sum(axis=1, keepdims=True)
    row_means = np.nansum(signal, axis=1, keepdims=True) / np.maximum(counts, 1)
    signal = np.where(finite, signal - row_means, 0.0)
    noise = rng.normal(0.0, 0.18, size=signal.shape)
    target = nan_to_num(signal + noise)
    target[-1] = np.nan
    return MarketData(features=features, target=target)
