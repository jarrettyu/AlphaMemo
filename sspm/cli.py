from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation.qlib_export import QlibExportConfig, export_qlib_factor_workspace, run_qlib_backtest
from .runner import RunConfig, run_benchmark, run_search


EXPERIMENT_PRESETS = {
    "paper2025": {
        "data_start": "2016-01-01",
        "data_end": "2020-12-31",
        "valid_start": "2021-01-01",
        "valid_end": "2021-12-31",
        "test_start": "2022-01-01",
        "test_end": "2025-12-26",
        "backtest_start": "2016-01-01",
        "backtest_end": "2025-12-26",
    },
    "nextday2025": {
        "data_start": "2016-01-01",
        "data_end": "2020-12-31",
        "valid_start": "2021-01-01",
        "valid_end": "2021-12-31",
        "test_start": "2022-01-01",
        "test_end": "2025-12-26",
        "backtest_start": "2016-01-01",
        "backtest_end": "2025-12-26",
    },
    "target2024": {
        "data_start": "2015-01-01",
        "data_end": "2019-12-31",
        "valid_start": "2020-01-01",
        "valid_end": "2020-12-31",
        "test_start": "2021-01-01",
        "test_end": "2024-12-30",
        "backtest_start": "2015-01-01",
        "backtest_end": "2024-12-30",
    },
    "local2020": {
        "data_start": "2015-01-01",
        "data_end": "2018-12-31",
        "valid_start": "2019-01-01",
        "valid_end": "2019-12-31",
        "test_start": "2020-01-01",
        "test_end": "2020-09-25",
        "backtest_start": "2015-01-01",
        "backtest_end": "2020-09-25",
    },
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sspm", description="Run AlphaMemo alpha-mining experiments.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run one online search loop.")
    _add_common_args(run_p)
    strategy_choices = ["alphamemo", "sspm", "veto", "structured", "graph", "gp", "random"]
    run_p.add_argument("--strategy", choices=strategy_choices, default="alphamemo")
    run_p.add_argument("--out", default="runs/run.json")

    bench_p = sub.add_parser("benchmark", help="Run several strategies/seeds.")
    _add_common_args(bench_p)
    bench_p.add_argument(
        "--strategies",
        nargs="+",
        choices=strategy_choices,
        default=["alphamemo", "structured", "random"],
    )
    bench_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    bench_p.add_argument("--out", default="runs/benchmark.json")

    export_p = sub.add_parser("export-qlib", help="Export discovered formulas as a Qlib factor workspace.")
    export_p.add_argument("--run-json", required=True)
    export_p.add_argument("--market", choices=["csi500", "sp500"], default="csi500")
    export_p.add_argument("--out-dir", default="runs/qlib_export")
    export_p.add_argument("--max-factors", type=int, default=50)
    export_p.add_argument("--include-seed-factors", action="store_true")
    export_p.add_argument("--discovered-only", action="store_true")
    export_p.add_argument("--selection-start", default="2016-01-01")
    export_p.add_argument("--selection-end", default="2020-12-31")
    export_p.add_argument("--valid-start", default="2021-01-01")
    export_p.add_argument("--valid-end", default="2021-12-31")
    export_p.add_argument("--test-start", default="2022-01-01")
    export_p.add_argument("--test-end", default="2025-12-26")
    export_p.add_argument("--backtest-start", default="2016-01-01")
    export_p.add_argument("--backtest-end", default="2025-12-26")
    export_p.add_argument("--provider-uri", default="")
    export_p.add_argument("--min-train-abs-icir", type=float, default=0.0)
    export_p.add_argument("--label-days", type=int, default=20)

    qlib_p = sub.add_parser("qlib-backtest", help="Run qrun inside an exported Qlib workspace.")
    qlib_p.add_argument("--work-dir", required=True)
    qlib_p.add_argument("--qrun-bin", default="qrun")

    main_p = sub.add_parser("main-table", help="Mine factors on real Qlib data and prepare/run main-table backtests.")
    _add_common_args(main_p)
    main_p.add_argument(
        "--strategies",
        nargs="+",
        choices=strategy_choices,
        default=["alphamemo"],
    )
    main_p.add_argument("--markets", nargs="+", choices=["csi500", "sp500"], default=["csi500", "sp500"])
    main_p.add_argument("--out-dir", default="runs/main_table")
    main_p.add_argument("--preset", choices=sorted(EXPERIMENT_PRESETS), default="paper2025")
    main_p.add_argument("--cn-provider-uri", default="", help="Qlib data directory for CSI markets.")
    main_p.add_argument("--us-provider-uri", default="", help="Qlib data directory for US markets.")
    main_p.add_argument("--valid-start", default="")
    main_p.add_argument("--valid-end", default="")
    main_p.add_argument("--test-start", default="")
    main_p.add_argument("--test-end", default="")
    main_p.add_argument("--backtest-start", default="")
    main_p.add_argument("--backtest-end", default="")
    main_p.add_argument("--max-factors", type=int, default=50)
    main_p.add_argument("--run-backtest", action="store_true")
    main_p.add_argument("--qrun-bin", default="qrun")

    args = parser.parse_args(argv)
    if args.cmd == "run":
        config = _config_from_args(args)
        config.strategy = args.strategy
        payload = run_search(config, out=args.out, verbose=not args.quiet)
        print(json.dumps(payload["summary"], indent=2))
        print(f"saved: {args.out}")
    elif args.cmd == "benchmark":
        config = _config_from_args(args)
        payload = run_benchmark(args.strategies, args.seeds, config, out=args.out, verbose=not args.quiet)
        print(_format_benchmark_table(payload["summary"]))
        print(json.dumps(payload["summary"], indent=2))
        print(f"saved: {args.out}")
    elif args.cmd == "export-qlib":
        summary = export_qlib_factor_workspace(_export_config_from_args(args))
        print(_format_export_summary(summary))
        print(f"saved: {args.out_dir}")
    elif args.cmd == "qlib-backtest":
        metrics = run_qlib_backtest(args.work_dir, qrun_bin=args.qrun_bin)
        print(json.dumps(metrics, indent=2))
    elif args.cmd == "main-table":
        summaries = _run_main_table(args)
        summary_path = Path(args.out_dir) / "main_table_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(json.dumps(summaries, indent=2))
        print(f"saved: {summary_path}")
    else:
        raise ValueError(args.cmd)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--evaluator", choices=["synthetic", "qlib"], default="synthetic")
    parser.add_argument("--market", choices=["csi500", "sp500"], default="csi500")
    parser.add_argument("--data-start", default="2016-01-01")
    parser.add_argument("--data-end", default="2020-12-31")
    parser.add_argument("--provider-uri", default="")
    parser.add_argument("--label-days", type=int, default=20)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--market-seed", type=int, default=123)
    parser.add_argument("--n-days", type=int, default=720)
    parser.add_argument("--n-assets", type=int, default=120)
    parser.add_argument("--success-ic", type=float, default=0.0)
    parser.add_argument("--success-icir", type=float, default=0.10)
    parser.add_argument("--corr-threshold", type=float, default=0.70)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--memory-weight", type=float, default=1.0)
    parser.add_argument("--motif-sample-size", type=int, default=4)
    parser.add_argument("--random-motif-prob", type=float, default=0.35)
    parser.add_argument("--generator", choices=["heuristic", "openrouter", "openai-compatible"], default="heuristic")
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--quiet", action="store_true")


def _format_benchmark_table(summary: dict) -> str:
    if not summary:
        return ""
    lines = [
        "| Strategy | Effective | vs Random | IC | ICIR | RIC | RICIR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy, stats in summary.items():
        lines.append(
            "| {strategy} | {eff:.2f}±{std:.2f} | {vs:.2f}x | {ic:.4f} | {icir:.4f} | {ric:.4f} | {ricir:.4f} |".format(
                strategy=strategy,
                eff=stats.get("mean_effective", 0.0),
                std=stats.get("std_effective", 0.0),
                vs=stats.get("vs_random", 0.0),
                ic=stats.get("mean_abs_ic", 0.0),
                icir=stats.get("mean_abs_icir", 0.0),
                ric=stats.get("mean_abs_ric", 0.0),
                ricir=stats.get("mean_abs_ricir", 0.0),
            )
        )
    return "\n".join(lines)


def _config_from_args(args: argparse.Namespace) -> RunConfig:
    preset = _preset_values(args)
    return RunConfig(
        evaluator=args.evaluator,
        market=args.market,
        data_start=preset.get("data_start", args.data_start),
        data_end=preset.get("data_end", args.data_end),
        provider_uri=args.provider_uri,
        label_days=args.label_days,
        budget=args.budget,
        batch_size=args.batch_size,
        seed=args.seed,
        market_seed=args.market_seed,
        n_days=args.n_days,
        n_assets=args.n_assets,
        success_ic=args.success_ic,
        success_icir=args.success_icir,
        corr_threshold=args.corr_threshold,
        warmup=args.warmup,
        memory_weight=args.memory_weight,
        motif_sample_size=args.motif_sample_size,
        random_motif_prob=args.random_motif_prob,
        generator=args.generator,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )


def _export_config_from_args(args: argparse.Namespace) -> QlibExportConfig:
    return QlibExportConfig(
        run_json=args.run_json,
        market=args.market,
        out_dir=args.out_dir,
        max_factors=args.max_factors,
        include_seed_factors=args.include_seed_factors,
        include_all_ok_candidates=not args.discovered_only,
        selection_start=args.selection_start,
        selection_end=args.selection_end,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
        test_start=args.test_start,
        test_end=args.test_end,
        backtest_start=args.backtest_start,
        backtest_end=args.backtest_end,
        provider_uri=args.provider_uri or None,
        min_train_abs_icir=args.min_train_abs_icir,
        label_days=args.label_days,
    )


def _run_main_table(args: argparse.Namespace) -> list[dict]:
    preset = _preset_values(args)
    summaries = []
    for market in args.markets:
        for strategy in args.strategies:
            run_dir = Path(args.out_dir) / market / strategy
            run_dir.mkdir(parents=True, exist_ok=True)
            provider_uri = _provider_uri_for_market(args, market)
            config = _config_from_args(args)
            config.evaluator = "qlib"
            config.market = market
            config.strategy = strategy
            config.provider_uri = provider_uri
            search_path = run_dir / "search.json"
            payload = run_search(config, out=search_path, verbose=not args.quiet)
            qlib_dir = run_dir / "qlib"
            export_summary = export_qlib_factor_workspace(
                QlibExportConfig(
                    run_json=str(search_path),
                    market=market,
                    out_dir=str(qlib_dir),
                    max_factors=args.max_factors,
                    selection_start=preset["data_start"],
                    selection_end=preset["data_end"],
                    valid_start=preset["valid_start"],
                    valid_end=preset["valid_end"],
                    test_start=preset["test_start"],
                    test_end=preset["test_end"],
                    backtest_start=preset["backtest_start"],
                    backtest_end=preset["backtest_end"],
                    provider_uri=provider_uri or None,
                    label_days=args.label_days,
                )
            )
            item = {
                "market": market,
                "strategy": strategy,
                "search_summary": payload["summary"],
                "qlib_workspace": str(qlib_dir),
                "n_selected_factors": export_summary["n_selected_factors"],
            }
            if args.run_backtest:
                item["qlib_metrics"] = run_qlib_backtest(qlib_dir, qrun_bin=args.qrun_bin)
            summaries.append(item)
    return summaries


def _provider_uri_for_market(args: argparse.Namespace, market: str) -> str:
    if market.startswith("csi") and getattr(args, "cn_provider_uri", ""):
        return args.cn_provider_uri
    if market == "sp500" and getattr(args, "us_provider_uri", ""):
        return args.us_provider_uri
    return args.provider_uri


def _preset_values(args: argparse.Namespace) -> dict[str, str]:
    preset_name = getattr(args, "preset", None)
    values = dict(EXPERIMENT_PRESETS.get(preset_name, {}))
    for key, attr in (
        ("valid_start", "valid_start"),
        ("valid_end", "valid_end"),
        ("test_start", "test_start"),
        ("test_end", "test_end"),
        ("backtest_start", "backtest_start"),
        ("backtest_end", "backtest_end"),
    ):
        value = getattr(args, attr, "")
        if value:
            values[key] = value
    if getattr(args, "data_start", ""):
        values.setdefault("data_start", args.data_start)
    if getattr(args, "data_end", ""):
        values.setdefault("data_end", args.data_end)
    return values


def _format_export_summary(summary: dict) -> str:
    lines = [
        f"market: {summary['market']} ({summary['qlib_market']})",
        f"dates/instruments: {summary['n_dates']} / {summary['n_instruments']}",
        f"formulas: input={summary['n_input_formulas']} valid={summary['n_valid_formulas']} selected={summary['n_selected_factors']}",
        f"factor pickle: {summary['factor_pickle']}",
        f"qlib config: {summary['qlib_config']}",
    ]
    if summary["selected_factors"]:
        best = summary["selected_factors"][0]
        lines.append(f"best train IC/ICIR: {best['ic']:.4f} / {best['icir']:.4f}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
