from __future__ import annotations

import json
import pickle
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from sspm.core.math_utils import mean_ic_icir, rowwise_pearson, rowwise_spearman
from sspm.core.operators import evaluate_formula
from sspm.seeds import default_seed_formulas

from .qlib_data import future_return_label, load_qlib_panel, read_exp_res_path, template_config_path


@dataclass(frozen=True, slots=True)
class QlibExportConfig:
    run_json: str
    market: str = "csi500"
    out_dir: str = "runs/qlib_export"
    max_factors: int = 50
    include_seed_factors: bool = False
    include_all_ok_candidates: bool = True
    selection_start: str = "2016-01-01"
    selection_end: str = "2020-12-31"
    valid_start: str = "2021-01-01"
    valid_end: str = "2021-12-31"
    test_start: str = "2022-01-01"
    test_end: str = "2025-12-26"
    backtest_start: str = "2016-01-01"
    backtest_end: str = "2025-12-26"
    provider_uri: str | None = None
    min_train_abs_icir: float = 0.0
    label_days: int = 20


def export_qlib_factor_workspace(config: QlibExportConfig) -> dict:
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    formulas = load_formulas_from_run(
        config.run_json,
        include_seed_factors=config.include_seed_factors,
        include_all_ok_candidates=config.include_all_ok_candidates,
    )
    if not formulas:
        raise ValueError(f"no discovered formulas found in {config.run_json}")

    panel = load_qlib_panel(
        market=config.market,
        start_time=config.backtest_start,
        end_time=config.backtest_end,
        provider_uri=config.provider_uri,
    )
    target = future_return_label(panel.features["close"], label_days=config.label_days)
    selection_mask = (panel.dates >= pd.Timestamp(config.selection_start)) & (
        panel.dates <= pd.Timestamp(config.selection_end)
    )

    factor_rows = []
    factor_arrays = []
    for formula in formulas:
        try:
            values = evaluate_formula(formula, panel.features)
            metrics = _factor_metrics(values[selection_mask], target[selection_mask])
        except Exception as exc:
            factor_rows.append({"formula": formula, "ok": False, "error": str(exc)})
            continue

        row = {"formula": formula, "ok": True, **metrics}
        factor_rows.append(row)
        if metrics["abs_icir"] >= config.min_train_abs_icir:
            factor_arrays.append((formula, values, metrics))

    factor_arrays.sort(key=lambda item: (item[2]["abs_icir"], item[2]["abs_ic"]), reverse=True)
    selected = factor_arrays[: config.max_factors]
    if not selected:
        raise ValueError("no valid factors survived Qlib train-period filtering")

    factor_df = _factor_dataframe(panel.dates, panel.instruments, selected)
    factor_path = out_dir / "combined_factors_df.pkl"
    with factor_path.open("wb") as f:
        pickle.dump(factor_df, f)

    _write_qlib_config(config, out_dir / "conf.yaml", panel.spec, panel.dates)
    shutil.copyfile(read_exp_res_path(), out_dir / "read_exp_res.py")

    summary = {
        "config": asdict(config),
        "market": panel.spec.name,
        "qlib_market": panel.spec.qlib_market,
        "provider_uri": panel.spec.provider_uri,
        "n_dates": len(panel.dates),
        "n_instruments": len(panel.instruments),
        "n_input_formulas": len(formulas),
        "n_valid_formulas": sum(1 for row in factor_rows if row.get("ok")),
        "n_selected_factors": len(selected),
        "factor_pickle": str(factor_path),
        "qlib_config": str(out_dir / "conf.yaml"),
        "result_reader": str(out_dir / "read_exp_res.py"),
        "selected_factors": [
            {"name": f"SSPM_{i:03d}", "formula": formula, **metrics}
            for i, (formula, _values, metrics) in enumerate(selected)
        ],
        "all_factor_diagnostics": factor_rows,
    }
    (out_dir / "qlib_export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "selected_formulas.txt").write_text(
        "\n".join(item["formula"] for item in summary["selected_factors"]) + "\n",
        encoding="utf-8",
    )
    return summary


def run_qlib_backtest(work_dir: str | Path, qrun_bin: str = "qrun") -> dict:
    work_path = Path(work_dir)
    if not (work_path / "conf.yaml").exists():
        raise FileNotFoundError(f"missing conf.yaml in {work_path}")
    qrun_path = _resolve_qrun(qrun_bin)
    if qrun_path is None:
        raise RuntimeError(
            f"{qrun_bin!r} is not available. Install pyqlib in this Python environment before running backtests."
        )

    subprocess.run([qrun_path, "conf.yaml"], cwd=work_path, check=True)
    subprocess.run([sys.executable, "read_exp_res.py"], cwd=work_path, check=True)
    return read_qlib_metrics(work_path)


def read_qlib_metrics(work_dir: str | Path) -> dict:
    path = Path(work_dir) / "qlib_res.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing Qlib metrics file: {path}")
    metrics = pd.read_csv(path, index_col=0).iloc[:, 0]
    return {str(k): _json_scalar(v) for k, v in metrics.items()}


def load_formulas_from_run(
    run_json: str | Path,
    include_seed_factors: bool = False,
    include_all_ok_candidates: bool = True,
) -> list[str]:
    payload = json.loads(Path(run_json).read_text(encoding="utf-8"))
    formulas: list[str] = []
    if include_seed_factors:
        formulas.extend(default_seed_formulas())
    for row in payload.get("discovered", []):
        formula = row.get("formula")
        if isinstance(formula, str) and formula.strip():
            formulas.append(formula.strip())
    if include_all_ok_candidates:
        for row in sorted(payload.get("events", []), key=lambda item: item.get("abs_icir", 0.0), reverse=True):
            formula = row.get("formula")
            if row.get("ok") and isinstance(formula, str) and formula.strip():
                formulas.append(formula.strip())
    return _deduplicate(formulas)


def _factor_metrics(values: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    daily_ic = rowwise_pearson(values, target)
    daily_ric = rowwise_spearman(values, target)
    ic, icir, n_days = mean_ic_icir(daily_ic)
    ric, ricir, n_rank_days = mean_ic_icir(daily_ric)
    return {
        "ic": float(ic),
        "icir": float(icir),
        "ric": float(ric),
        "ricir": float(ricir),
        "abs_ic": float(abs(ic)),
        "abs_icir": float(abs(icir)),
        "abs_ric": float(abs(ric)),
        "abs_ricir": float(abs(ricir)),
        "n_days": int(min(n_days, n_rank_days)),
    }


def _factor_dataframe(
    dates: pd.DatetimeIndex,
    instruments: list[str],
    selected: Iterable[tuple[str, np.ndarray, dict]],
) -> pd.DataFrame:
    names = []
    matrices = []
    for i, (_formula, values, _metrics) in enumerate(selected):
        names.append(f"SSPM_{i:03d}")
        matrices.append(np.asarray(values, dtype=np.float32).reshape(-1))

    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    data = np.column_stack(matrices)
    data[~np.isfinite(data)] = np.nan
    df = pd.DataFrame(data, index=index, columns=names)
    df = df.dropna(how="all")
    df.columns = pd.MultiIndex.from_product([["feature"], df.columns])
    return df.sort_index()


def _deduplicate(formulas: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for formula in formulas:
        if formula in seen:
            continue
        seen.add(formula)
        out.append(formula)
    return out


def _json_scalar(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _write_qlib_config(config: QlibExportConfig, out_path: Path, spec, calendar: pd.DatetimeIndex) -> None:
    template_path = template_config_path(config.market)
    with template_path.open("r", encoding="utf-8") as f:
        qlib_config = yaml.safe_load(f)

    qlib_config["qlib_init"]["provider_uri"] = _provider_uri_for_config(config.provider_uri or spec.provider_uri)
    qlib_config["qlib_init"]["region"] = spec.region
    qlib_config["market"] = spec.qlib_market
    qlib_config["benchmark"] = spec.benchmark

    handler = qlib_config["data_handler_config"]
    handler["start_time"] = config.backtest_start
    handler["end_time"] = config.backtest_end
    handler["instruments"] = spec.qlib_market
    _set_qlib_label(handler, _qlib_label_expr(config.label_days))

    segments = qlib_config["task"]["dataset"]["kwargs"]["segments"]
    segments["train"] = [config.selection_start, config.selection_end]
    segments["valid"] = [config.valid_start, config.valid_end]
    segments["test"] = [config.test_start, config.test_end]

    backtest = qlib_config["port_analysis_config"]["backtest"]
    backtest["start_time"] = config.test_start
    backtest["end_time"] = _safe_backtest_end(calendar, config.test_end)
    backtest["benchmark"] = spec.benchmark

    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(qlib_config, f, sort_keys=False)


def _resolve_qrun(qrun_bin: str) -> str | None:
    found = shutil.which(qrun_bin)
    if found is not None:
        return found
    local = Path.home() / ".local" / "bin" / qrun_bin
    if local.exists():
        return str(local)
    return None


def _provider_uri_for_config(provider_uri: str) -> str:
    if provider_uri.startswith("~"):
        return provider_uri
    return str(Path(provider_uri).expanduser().resolve())


def _safe_backtest_end(calendar: pd.DatetimeIndex, requested_end: str) -> str:
    """Qlib needs one calendar step after the portfolio end time."""

    ts = pd.Timestamp(requested_end)
    idx = int(calendar.searchsorted(ts, side="right")) - 1
    if idx < 0:
        raise ValueError(f"no trading calendar date found before requested test end: {requested_end}")
    if idx >= len(calendar) - 1:
        idx -= 1
    if idx < 0:
        raise ValueError("not enough trading calendar dates to run a Qlib backtest")
    return str(calendar[idx].date())


def _qlib_label_expr(label_days: int) -> str:
    if label_days < 1:
        raise ValueError(f"label_days must be >= 1, got {label_days}")
    return f"Ref($close, -{label_days + 1})/Ref($close, -1) - 1"


def _set_qlib_label(handler: dict, label_expr: str) -> None:
    data_loader = handler.get("data_loader", {})
    kwargs = data_loader.get("kwargs", {})
    loaders = kwargs.get("dataloader_l")
    if isinstance(loaders, list):
        for loader in loaders:
            config = loader.get("kwargs", {}).get("config")
            if isinstance(config, dict) and "label" in config:
                config["label"] = [[label_expr], ["LABEL0"]]
                return
    config = kwargs.get("config")
    if isinstance(config, dict):
        config["label"] = [[label_expr], ["LABEL0"]]
