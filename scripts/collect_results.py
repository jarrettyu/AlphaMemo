#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path


METRIC_KEYS = {
    "IC": {"source": "IC", "scale": 1.0, "digits": 4},
    "ICIR": {"source": "ICIR", "scale": 1.0, "digits": 4},
    "RankIC": {"source": "Rank IC", "scale": 1.0, "digits": 4},
    "RankICIR": {"source": "Rank ICIR", "scale": 1.0, "digits": 4},
    "AR (%)": {"source": "portfolio.annualized_return", "scale": 100.0, "digits": 2},
    "MDD (%)": {"source": "portfolio.max_drawdown", "scale": 100.0, "digits": 2},
    "Sharpe": {"source": "portfolio.sharpe", "scale": 1.0, "digits": 4},
}

MAIN_METHOD_ORDER = ["AlphaMemo", "Search-Ledger", "Structured Search", "GP", "Random"]

ABLATION_METHOD_ORDER = [
    "Structured Search",
    "Search-Ledger Agent",
    "AlphaMemo-mem1.5",
    "AlphaMemo-warm10",
    "AlphaMemo-seed1",
    "AlphaMemo-seed2",
]

BASELINE_NAMES = {
    "alpha158": "Alpha158",
    "gp": "GP",
    "lightgbm": "LightGBM",
    "lstm": "LSTM",
}


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows: list[dict[str, str]] = []
    rows.extend(
        read_main_table(
            root / "main_table_summary.json",
            {
                "alphamemo": "AlphaMemo",
                "sspm": "AlphaMemo",
                "graph": "Search-Ledger",
                "structured": "Structured Search",
                "gp": "GP",
                "random": "Random",
            },
        )
    )
    rows.extend(read_variant_dirs(root))

    rows = dedup(rows)
    rows.sort(key=sort_key)

    out_dir = root / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "main_table_metrics.csv", rows)
    write_markdown(out_dir / "main_table_metrics.md", rows)
    extra_methods = sorted(
        {
            row["method"]
            for row in rows
            if row["method"] not in {*MAIN_METHOD_ORDER, *ABLATION_METHOD_ORDER}
        }
    )
    write_latex_rows(out_dir / "main_table_rows.tex", rows, MAIN_METHOD_ORDER)
    write_latex_rows(out_dir / "ablation_table_rows.tex", rows, ["AlphaMemo", *ABLATION_METHOD_ORDER, *extra_methods])
    print(f"wrote: {out_dir / 'main_table_metrics.csv'}")
    print(f"wrote: {out_dir / 'main_table_metrics.md'}")
    print(f"wrote: {out_dir / 'main_table_rows.tex'}")
    print(f"wrote: {out_dir / 'ablation_table_rows.tex'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect AlphaMemo experiment metrics.")
    parser.add_argument("--root", default="runs/main_20day2025")
    return parser.parse_args()


def read_baselines(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in data:
        metrics = dict(item.get("qlib_metrics") or {})
        metrics.update(read_workspace_metrics(item.get("workspace")))
        method = BASELINE_NAMES.get(str(item.get("baseline", "")), str(item.get("baseline", "")))
        rows.append(build_row(method, str(item.get("market", "")), metrics))
    return rows


def read_gp_baseline(root: Path) -> list[dict[str, str]]:
    rows = []
    for market in ("csi500", "sp500"):
        run_dir = root / market / "gp"
        metrics = read_workspace_metrics(run_dir / "qlib")
        if not metrics:
            metrics = read_search_summary(run_dir / "search.json")
        if metrics:
            rows.append(build_row("GP", market, metrics))
    return rows


def read_main_table(path: Path, names: dict[str, str]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in data:
        strategy = str(item.get("strategy", ""))
        method = names.get(strategy, strategy)
        metrics = dict(item.get("qlib_metrics") or {})
        metrics.update(read_workspace_metrics(item.get("qlib_workspace")))
        if not metrics:
            metrics = search_summary_to_metrics(item.get("search_summary") or {})
        rows.append(build_row(method, str(item.get("market", "")), metrics))
    return rows


def read_variant_dirs(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run_dir in sorted(root.iterdir() if root.exists() else []):
        if not run_dir.is_dir():
            continue
        if run_dir.name.startswith("alphamemo_"):
            rows.extend(read_main_table(run_dir / "main_table_summary.json", {"alphamemo": variant_label(run_dir.name)}))
        elif run_dir.name.startswith("graphmemo_"):
            rows.extend(read_main_table(run_dir / "main_table_summary.json", {"alphamemo": variant_label(run_dir.name)}))
        elif run_dir.name.startswith("sspm_"):
            rows.extend(read_main_table(run_dir / "main_table_summary.json", {"sspm": variant_label(run_dir.name)}))
        elif run_dir.name.startswith("veto_"):
            rows.extend(read_main_table(run_dir / "main_table_summary.json", {"veto": variant_label(run_dir.name)}))
    return rows


def variant_label(name: str) -> str:
    parts = name.split("_")
    if parts and parts[0] in {"alphamemo", "graphmemo"}:
        family = "AlphaMemo"
    elif parts and parts[0] == "veto":
        family = "APV"
    else:
        family = "SSPM"
    warm = next((p[1:] for p in parts if p.startswith("w") and len(p) > 1), "")
    mem = next((p[1:] for p in parts if p.startswith("m") and len(p) > 1), "")
    random = next((p[1:] for p in parts if p.startswith("r") and len(p) > 1), "")
    seed = next((p[1:] for p in parts if p.startswith("s") and p[1:].isdigit()), "")
    mem = f"{int(mem) / 100:.2f}" if mem.isdigit() else mem
    random = f"{int(random) / 100:.2f}" if random.isdigit() else random
    suffix = " ".join(
        item
        for item in (
            f"w{warm}" if warm else "",
            f"m{mem}" if mem else "",
            f"r{random}" if random else "",
            f"s{seed}" if seed else "",
        )
        if item
    )
    if family == "AlphaMemo":
        return f"AlphaMemo {suffix}".strip()
    return f"AlphaMemo-{family} {suffix}".strip()


def read_workspace_metrics(workspace) -> dict:
    if not workspace:
        return {}
    out = {}
    path = Path(workspace) / "qlib_res.csv"
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    out[row[0]] = row[1]
    out.update(read_portfolio_metrics(Path(workspace)))
    return out


def read_portfolio_metrics(workspace: Path) -> dict:
    ret_path = workspace / "ret.pkl"
    if not ret_path.exists():
        return {}
    try:
        with ret_path.open("rb") as f:
            report = pickle.load(f)
        returns = report["return"].dropna().astype(float)
    except Exception:
        return {}
    if len(returns) == 0:
        return {}

    annualized = float(returns.mean() * 252.0)
    std = float(returns.std(ddof=0))
    sharpe = annualized / (std * math.sqrt(252.0)) if std > 0 else None
    nav = (1.0 + returns).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    return {
        "portfolio.annualized_return": annualized,
        "portfolio.max_drawdown": float(drawdown.min()),
        "portfolio.sharpe": sharpe,
    }


def read_search_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return search_summary_to_metrics(payload.get("summary") or {})


def search_summary_to_metrics(summary: dict) -> dict:
    if not summary:
        return {}
    return {
        "IC": summary.get("mean_abs_ic_discovered") or summary.get("mean_abs_ic_ok"),
        "ICIR": summary.get("mean_abs_icir_discovered"),
        "Rank IC": summary.get("mean_abs_ric_discovered"),
        "Rank ICIR": summary.get("mean_abs_ricir_discovered"),
    }


def build_row(method: str, market: str, metrics: dict) -> dict[str, str]:
    row = {"method": method, "market": market}
    for out_key, spec in METRIC_KEYS.items():
        row[out_key] = format_metric(metrics.get(spec["source"]), scale=spec["scale"], digits=spec["digits"])
    return row


def format_metric(value, scale: float = 1.0, digits: int = 4) -> str:
    if value is None or value == "":
        return "--"
    try:
        return f"{float(value) * scale:.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def dedup(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = {}
    for row in rows:
        seen[(row["method"], row["market"])] = row
    return list(seen.values())


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["method", "market", *METRIC_KEYS.keys()]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["method", "market", *METRIC_KEYS.keys()]
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row.get(field, "--") for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_rows(path: Path, rows: list[dict[str, str]], method_order: list[str]) -> None:
    by_method: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], {})[row["market"]] = row
    lines = []
    for method in method_order:
        market_rows = by_method.get(method)
        if not market_rows:
            continue
        csi = market_rows.get("csi500", empty_row(method, "csi500"))
        sp = market_rows.get("sp500", empty_row(method, "sp500"))
        vals = [csi[key] for key in METRIC_KEYS] + [sp[key] for key in METRIC_KEYS]
        lines.append(method + " & " + " & ".join(vals) + r" \\")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def empty_row(method: str, market: str) -> dict[str, str]:
    return {"method": method, "market": market, **{key: "--" for key in METRIC_KEYS}}


def sort_key(row: dict[str, str]) -> tuple[int, str]:
    order = [*MAIN_METHOD_ORDER, *ABLATION_METHOD_ORDER]
    method = row["method"]
    return (order.index(method) if method in order else 999, row["market"])


if __name__ == "__main__":
    main()
