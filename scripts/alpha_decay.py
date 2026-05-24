#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sspm.core.math_utils import rowwise_pearson, rowwise_spearman
from sspm.core.operators import evaluate_formula
from sspm.evaluation.qlib_data import future_return_label, load_qlib_panel


METHOD_PATHS = {
    "AlphaMemo": ("", "alphamemo"),
}


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows: list[dict[str, str | float | int]] = []
    for market in args.markets:
        panel = load_qlib_panel(market=market, start_time=args.start, end_time=args.end, provider_uri=provider_uri(args, market))
        target = future_return_label(panel.features["close"], label_days=args.label_days)
        for method, (stage, strategy) in METHOD_PATHS.items():
            summary_path = root / market / strategy / "qlib" / "qlib_export_summary.json"
            if stage:
                summary_path = root / stage / market / strategy / "qlib" / "qlib_export_summary.json"
            if not summary_path.exists():
                continue
            formulas = load_selected_formulas(summary_path, args.max_factors)
            if not formulas:
                continue
            signal = factor_pool_signal(formulas, panel.features)
            daily_ic = rowwise_pearson(signal, target)
            daily_ric = rowwise_spearman(signal, target)
            for year in args.years:
                mask = panel.dates.year == year
                ic = finite_mean(daily_ic[mask])
                ric = finite_mean(daily_ric[mask])
                rows.append(
                    {
                        "method": method,
                        "market": market,
                        "year": year,
                        "IC": ic,
                        "RankIC": ric,
                        "n_days": int(np.isfinite(daily_ic[mask]).sum()),
                    }
                )

    out_dir = root / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "alpha_decay_annual.csv", rows)
    write_latex(out_dir / "alpha_decay_rows.tex", rows, args.years)
    print(f"wrote: {out_dir / 'alpha_decay_annual.csv'}")
    print(f"wrote: {out_dir / 'alpha_decay_rows.tex'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute annual IC/RankIC decay for selected factor pools.")
    parser.add_argument("--root", default="runs/main_20day2025")
    parser.add_argument("--markets", nargs="+", choices=["csi500", "sp500"], default=["csi500", "sp500"])
    parser.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2025-12-26")
    parser.add_argument("--label-days", type=int, default=20)
    parser.add_argument("--max-factors", type=int, default=50)
    parser.add_argument("--cn-provider-uri", default="data/qlib/cn_data_nextday2025")
    parser.add_argument("--us-provider-uri", default="data/qlib/us_data_nextday2025")
    return parser.parse_args()


def provider_uri(args: argparse.Namespace, market: str) -> str:
    return args.cn_provider_uri if market.startswith("csi") else args.us_provider_uri


def load_selected_formulas(summary_path: Path, max_factors: int) -> list[tuple[str, float]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    formulas = []
    for item in payload.get("selected_factors", [])[:max_factors]:
        formula = item.get("formula")
        if isinstance(formula, str) and formula.strip():
            ic = float(item.get("ic", 0.0) or 0.0)
            sign = -1.0 if ic < 0 else 1.0
            formulas.append((formula.strip(), sign))
    return formulas


def factor_pool_signal(formulas: list[tuple[str, float]], features: dict[str, np.ndarray]) -> np.ndarray:
    sums: np.ndarray | None = None
    counts: np.ndarray | None = None
    for formula, sign in formulas:
        values = evaluate_formula(formula, features)
        z = sign * row_zscore(values)
        finite = np.isfinite(z)
        if sums is None:
            sums = np.zeros_like(z, dtype=float)
            counts = np.zeros_like(z, dtype=float)
        sums += np.where(finite, z, 0.0)
        counts += finite.astype(float)
    assert sums is not None and counts is not None
    return np.where(counts > 0, sums / np.maximum(counts, 1.0), np.nan)


def row_zscore(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    counts = finite.sum(axis=1, keepdims=True)
    sums = np.where(finite, values, 0.0).sum(axis=1, keepdims=True)
    mean = np.divide(sums, counts, out=np.zeros_like(sums, dtype=float), where=counts > 0)
    centered = np.where(finite, values - mean, 0.0)
    var = np.divide((centered * centered).sum(axis=1, keepdims=True), counts, out=np.zeros_like(sums), where=counts > 1)
    std = np.sqrt(var)
    out = np.full_like(values, np.nan, dtype=float)
    valid = finite & (std > 1e-8)
    np.divide(values - mean, std, out=out, where=valid)
    out[~valid] = np.nan
    return out


def finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else 0.0


def write_csv(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    fields = ["method", "market", "year", "IC", "RankIC", "n_days"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_latex(path: Path, rows: list[dict[str, str | float | int]], years: list[int]) -> None:
    by_key: dict[tuple[str, str], dict[int, dict[str, str | float | int]]] = {}
    for row in rows:
        by_key.setdefault((str(row["method"]), str(row["market"])), {})[int(row["year"])] = row
    lines = []
    for method, market in sorted(by_key):
        vals = []
        for year in years:
            row = by_key[(method, market)].get(year)
            if row is None:
                vals.extend(["--", "--"])
            else:
                vals.extend([f"{float(row['IC']):.4f}", f"{float(row['RankIC']):.4f}"])
        lines.append(f"{method} & {market.upper()} & " + " & ".join(vals) + r" \\")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
