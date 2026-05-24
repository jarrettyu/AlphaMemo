#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


FIELDS = ("open", "high", "low", "close", "volume", "factor")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CSI500 Qlib dataset from Yahoo Finance.")
    parser.add_argument("--source-qlib", default="~/.qlib/qlib_data/cn_data")
    parser.add_argument("--target-dir", default="data/qlib/cn_data_nextday2025")
    parser.add_argument("--market", default="csi500")
    parser.add_argument("--benchmark", default="SH000905")
    parser.add_argument("--benchmark-yahoo", default="000905.SZ")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2025-12-27")
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--limit", type=int, default=0, help="Debug limit for number of market symbols.")
    args = parser.parse_args()

    source = Path(args.source_qlib).expanduser()
    target = Path(args.target_dir).expanduser()
    instruments_path = source / "instruments" / f"{args.market}.txt"
    intervals = read_instrument_intervals(instruments_path)
    intervals = extend_latest_intervals(intervals, end_date=pd.Timestamp(args.end) - pd.Timedelta(days=1))

    symbols = sorted(symbol for symbol in intervals if symbol != args.benchmark)
    if args.limit:
        symbols = symbols[: args.limit]
    download_symbols = symbols + [args.benchmark]

    frames: dict[str, pd.DataFrame] = {}
    for batch in chunks(download_symbols, args.batch_size):
        print(f"Downloading {len(batch)} symbols: {batch[0]} ... {batch[-1]}", flush=True)
        data = yf.download(
            [to_yahoo_symbol(symbol, args.benchmark, args.benchmark_yahoo) for symbol in batch],
            start=args.start,
            end=args.end,
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
        )
        frames.update(extract_batch(data, batch, args.benchmark, args.benchmark_yahoo))
        time.sleep(0.5)

    frames = {symbol: df for symbol, df in frames.items() if not df.empty}
    if args.benchmark not in frames:
        raise RuntimeError(f"benchmark {args.benchmark} was not downloaded successfully")

    calendar = sorted(set().union(*(set(df.index) for df in frames.values())))
    if not calendar:
        raise RuntimeError("no data downloaded")
    calendar = pd.DatetimeIndex(calendar)

    write_calendar(target, calendar)
    write_instruments(target, frames, intervals, args.market, args.benchmark)
    write_feature_bins(target, frames, calendar)
    print(f"Built Qlib CN dataset: {target}")
    print(f"Calendar: {calendar.min().date()} to {calendar.max().date()}, {len(calendar)} days")
    print(f"Symbols with data: {len(frames)}")


def read_instrument_intervals(path: Path) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    intervals: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        intervals.setdefault(parts[0], []).append((pd.Timestamp(parts[1]), pd.Timestamp(parts[2])))
    return intervals


def extend_latest_intervals(
    intervals: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]],
    end_date: pd.Timestamp,
) -> dict[str, list[tuple[str, str]]]:
    latest = max(right for rows in intervals.values() for _left, right in rows)
    out: dict[str, list[tuple[str, str]]] = {}
    for symbol, rows in intervals.items():
        out[symbol] = [
            (left.strftime("%Y-%m-%d"), (end_date if right == latest else right).strftime("%Y-%m-%d"))
            for left, right in rows
        ]
    return out


def chunks(items: list[str], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def to_yahoo_symbol(symbol: str, benchmark: str, benchmark_yahoo: str) -> str:
    if symbol == benchmark:
        return benchmark_yahoo
    exchange = symbol[:2]
    code = symbol[2:]
    if exchange == "SH":
        return f"{code}.SS"
    if exchange == "SZ":
        return f"{code}.SZ"
    raise ValueError(f"unsupported CN symbol: {symbol}")


def extract_batch(
    data: pd.DataFrame,
    symbols: list[str],
    benchmark: str,
    benchmark_yahoo: str,
) -> dict[str, pd.DataFrame]:
    out = {}
    for symbol in symbols:
        yahoo_symbol = to_yahoo_symbol(symbol, benchmark, benchmark_yahoo)
        try:
            if isinstance(data.columns, pd.MultiIndex):
                df = data[yahoo_symbol].copy()
            else:
                df = data.copy()
        except KeyError:
            continue
        df = df.rename(columns={col: str(col).lower().replace(" ", "_") for col in df.columns})
        keep = [name for name in ("open", "high", "low", "close", "volume") if name in df.columns]
        df = df[keep].dropna(how="all")
        if "close" not in df.columns or df["close"].dropna().empty:
            continue
        df["factor"] = 1.0
        for field in FIELDS:
            if field not in df.columns:
                df[field] = np.nan
        df = df[list(FIELDS)]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        out[symbol] = df.sort_index()
    return out


def write_calendar(target: Path, calendar: pd.DatetimeIndex) -> None:
    path = target / "calendars"
    path.mkdir(parents=True, exist_ok=True)
    text = "\n".join(day.strftime("%Y-%m-%d") for day in calendar) + "\n"
    (path / "day.txt").write_text(text, encoding="utf-8")


def write_instruments(
    target: Path,
    frames: dict[str, pd.DataFrame],
    intervals: dict[str, list[tuple[str, str]]],
    market: str,
    benchmark: str,
) -> None:
    path = target / "instruments"
    path.mkdir(parents=True, exist_ok=True)

    all_lines = []
    market_lines = []
    for symbol in sorted(frames):
        df = frames[symbol]
        start = df.index.min().strftime("%Y-%m-%d")
        end = df.index.max().strftime("%Y-%m-%d")
        all_lines.append(f"{symbol}\t{start}\t{end}")
        if symbol == benchmark:
            continue
        for left, right in intervals.get(symbol, [(start, end)]):
            market_lines.append(f"{symbol}\t{left}\t{right}")

    (path / "all.txt").write_text("\n".join(all_lines) + "\n", encoding="utf-8")
    (path / f"{market}.txt").write_text("\n".join(market_lines) + "\n", encoding="utf-8")


def write_feature_bins(target: Path, frames: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex) -> None:
    feat_root = target / "features"
    feat_root.mkdir(parents=True, exist_ok=True)
    date_to_index = {day: i for i, day in enumerate(calendar)}
    for symbol, df in frames.items():
        symbol_dir = feat_root / symbol.lower()
        symbol_dir.mkdir(parents=True, exist_ok=True)
        start_idx = date_to_index[df.index.min()]
        end_idx = date_to_index[df.index.max()]
        cal_slice = calendar[start_idx : end_idx + 1]
        aligned = df.reindex(cal_slice)
        for field in FIELDS:
            values = aligned[field].astype(np.float32).to_numpy()
            data = np.hstack([np.float32(start_idx), values]).astype("<f4")
            data.tofile(symbol_dir / f"{field}.day.bin")


if __name__ == "__main__":
    main()
