from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .synthetic import MarketData


@dataclass(frozen=True, slots=True)
class QlibMarketSpec:
    name: str
    qlib_market: str
    provider_uri: str
    instrument_file: str
    region: str
    benchmark: str
    template_config: str


@dataclass(slots=True)
class QlibPanel:
    spec: QlibMarketSpec
    dates: pd.DatetimeIndex
    instruments: list[str]
    features: dict[str, np.ndarray]
    active: np.ndarray


SELF_EVO_ROOT = Path(__file__).resolve().parents[3]
FACTOR_TEMPLATE_DIR = SELF_EVO_ROOT / "templates" / "qlib_factor_template"

MARKET_SPECS: dict[str, QlibMarketSpec] = {
    "csi500": QlibMarketSpec(
        name="csi500",
        qlib_market="csi500",
        provider_uri="~/.qlib/qlib_data/cn_data",
        instrument_file="csi500.txt",
        region="cn",
        benchmark="SH000905",
        template_config="conf_cn_combined_kdd_ver.yaml",
    ),
    "sp500": QlibMarketSpec(
        name="sp500",
        qlib_market="sp500",
        provider_uri="~/.qlib/qlib_data/us_data",
        instrument_file="sp500.txt",
        region="us",
        benchmark="^GSPC",
        template_config="conf_us_combined_kdd_ver.yaml",
    ),
}


def make_qlib_market_data(
    market: str = "csi500",
    start_time: str = "2016-01-01",
    end_time: str = "2020-12-31",
    provider_uri: str | None = None,
    label_days: int = 20,
) -> MarketData:
    panel = load_qlib_panel(market=market, start_time=start_time, end_time=end_time, provider_uri=provider_uri)
    target = future_return_label(panel.features["close"], label_days=label_days)
    return MarketData(features=panel.features, target=target)


def load_qlib_panel(
    market: str = "csi500",
    start_time: str = "2016-01-01",
    end_time: str = "2025-12-26",
    provider_uri: str | None = None,
) -> QlibPanel:
    spec = get_market_spec(market)
    provider = Path(provider_uri or spec.provider_uri).expanduser()
    if not provider.exists():
        raise FileNotFoundError(f"Qlib data directory does not exist: {provider}")

    all_dates = _read_calendar(provider)
    requested_start = pd.Timestamp(start_time)
    requested_end = pd.Timestamp(end_time)
    if requested_end > all_dates.max():
        raise ValueError(
            "Qlib data coverage is insufficient for the requested range: "
            f"requested [{requested_start.date()}, {requested_end.date()}], "
            f"available [{all_dates.min().date()}, {all_dates.max().date()}] in {provider}"
        )
    date_mask = (all_dates >= pd.Timestamp(start_time)) & (all_dates <= pd.Timestamp(end_time))
    if not date_mask.any():
        raise ValueError(f"no trading dates found in {provider} for [{start_time}, {end_time}]")

    instruments, intervals = _read_instruments(provider / "instruments" / spec.instrument_file)
    dates = all_dates[date_mask]
    date_indices = np.flatnonzero(date_mask)

    raw_fields = {
        "open_": "open",
        "close": "close",
        "high": "high",
        "low": "low",
        "volume": "volume",
    }
    features = {
        key: np.full((len(dates), len(instruments)), np.nan, dtype=np.float32)
        for key in [*raw_fields.keys(), "vwap", "ret"]
    }
    active = np.zeros((len(dates), len(instruments)), dtype=bool)

    for j, instrument in enumerate(instruments):
        instrument_active = _active_mask_for_intervals(all_dates, intervals[instrument])[date_mask]
        active[:, j] = instrument_active
        for out_name, file_name in raw_fields.items():
            full_values = _read_feature_bin(provider, instrument, file_name, len(all_dates))
            values = full_values[date_indices]
            features[out_name][:, j] = np.where(instrument_active, values, np.nan)

    features["vwap"] = (features["open_"] + features["high"] + features["low"] + features["close"]) / 4.0
    with np.errstate(divide="ignore", invalid="ignore"):
        features["ret"][1:] = features["close"][1:] / features["close"][:-1] - 1.0
    return QlibPanel(spec=spec, dates=dates, instruments=instruments, features=features, active=active)


def future_return_label(close: np.ndarray, label_days: int = 1) -> np.ndarray:
    """Return future holding-period return after the next tradable close.

    label_days=1 matches the current next-day protocol:
    Ref($close, -2) / Ref($close, -1) - 1.
    label_days=20 gives a 20-trading-day forward return:
    Ref($close, -21) / Ref($close, -1) - 1.
    """

    if label_days < 1:
        raise ValueError(f"label_days must be >= 1, got {label_days}")
    label = np.full_like(close, np.nan, dtype=float)
    shift = label_days + 1
    with np.errstate(divide="ignore", invalid="ignore"):
        if close.shape[0] > shift:
            label[:-shift] = close[shift:] / close[1:-label_days] - 1.0
    return label


def get_market_spec(market: str) -> QlibMarketSpec:
    key = market.lower()
    if key not in MARKET_SPECS:
        choices = ", ".join(sorted(MARKET_SPECS))
        raise ValueError(f"unknown Qlib market: {market}. choices: {choices}")
    return MARKET_SPECS[key]


def template_config_path(market: str) -> Path:
    spec = get_market_spec(market)
    path = FACTOR_TEMPLATE_DIR / spec.template_config
    if not path.exists():
        raise FileNotFoundError(f"missing Qlib template config: {path}")
    return path


def read_exp_res_path() -> Path:
    path = FACTOR_TEMPLATE_DIR / "read_exp_res.py"
    if not path.exists():
        raise FileNotFoundError(f"missing Qlib result reader: {path}")
    return path


def _read_calendar(provider: Path) -> pd.DatetimeIndex:
    path = provider / "calendars" / "day.txt"
    if not path.exists():
        raise FileNotFoundError(f"missing Qlib day calendar: {path}")
    return pd.to_datetime([line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


def _read_instruments(path: Path) -> tuple[list[str], dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]]:
    if not path.exists():
        raise FileNotFoundError(f"missing Qlib instrument file: {path}")
    intervals: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        instrument, start_time, end_time = parts[:3]
        intervals.setdefault(instrument, []).append((pd.Timestamp(start_time), pd.Timestamp(end_time)))
    return sorted(intervals), intervals


def _read_feature_bin(provider: Path, instrument: str, field: str, calendar_len: int) -> np.ndarray:
    path = provider / "features" / instrument.lower() / f"{field}.day.bin"
    out = np.full(calendar_len, np.nan, dtype=np.float32)
    if not path.exists():
        return out

    raw = np.fromfile(path, dtype="<f4")
    if raw.size == 0:
        return out
    start_index = int(raw[0])
    values = raw[1:]
    if start_index < 0:
        values = values[-start_index:]
        start_index = 0
    end_index = min(calendar_len, start_index + len(values))
    if end_index > start_index:
        out[start_index:end_index] = values[: end_index - start_index]
    return out


def _active_mask_for_intervals(
    dates: pd.DatetimeIndex,
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> np.ndarray:
    mask = np.zeros(len(dates), dtype=bool)
    for start_time, end_time in intervals:
        mask |= (dates >= start_time) & (dates <= end_time)
    return mask
