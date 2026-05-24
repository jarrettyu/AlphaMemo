from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from sspm.core.types import SearchEvent
from sspm.evaluation.formula_eval import FormulaEvaluator
from sspm.evaluation.qlib_data import make_qlib_market_data
from sspm.evaluation.synthetic import make_synthetic_market
from sspm.generation import HeuristicGenerator, OpenAICompatibleGenerator
from sspm.seeds import default_seed_formulas
from sspm.strategies import (
    GraphMemoryStrategy,
    GeneticProgrammingStrategy,
    RandomSearch,
    SSPMStrategy,
    StructuredSearchStrategy,
    VetoMemoryStrategy,
)
from sspm.strategies.base import SearchStrategy
from sspm.utils.env import load_dotenv


@dataclass(slots=True)
class RunConfig:
    strategy: str = "sspm"
    evaluator: str = "synthetic"
    market: str = "csi500"
    data_start: str = "2016-01-01"
    data_end: str = "2020-12-31"
    provider_uri: str = ""
    label_days: int = 20
    budget: int = 100
    batch_size: int = 5
    seed: int = 0
    market_seed: int = 123
    n_days: int = 720
    n_assets: int = 120
    success_ic: float = 0.0
    success_icir: float = 0.10
    corr_threshold: float = 0.70
    warmup: int = 30
    memory_weight: float = 1.0
    motif_sample_size: int = 4
    random_motif_prob: float = 0.35
    generator: str = "heuristic"
    model: str = ""
    base_url: str = ""
    api_key_env: str = ""


def make_generator(config: RunConfig, rng: np.random.Generator):
    if config.generator == "heuristic":
        return HeuristicGenerator(rng)
    if config.generator == "openrouter":
        return OpenAICompatibleGenerator(
            model=config.model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            base_url=config.base_url or "https://openrouter.ai/api/v1",
            api_key_env=config.api_key_env or "OPENROUTER_API_KEY",
            timeout=_env_float("OPENROUTER_TIMEOUT", 60.0),
            max_retries=_env_int("OPENROUTER_MAX_RETRIES", 5),
            retry_sleep=_env_float("OPENROUTER_RETRY_SLEEP", 2.0),
        )
    if config.generator == "openai-compatible":
        return OpenAICompatibleGenerator(
            model=config.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_url=config.base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            api_key_env=config.api_key_env or "OPENAI_API_KEY",
            timeout=_env_float("LLM_TIMEOUT", 60.0),
            max_retries=_env_int("LLM_MAX_RETRIES", 5),
            retry_sleep=_env_float("LLM_RETRY_SLEEP", 2.0),
        )
    raise ValueError(f"unknown generator: {config.generator}")


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def make_strategy(name: str, generator, rng: np.random.Generator, config: RunConfig) -> SearchStrategy:
    if name == "random":
        return RandomSearch(generator, rng)
    if name == "gp":
        return GeneticProgrammingStrategy(generator, rng)
    if name in {"graph", "structured"}:
        return StructuredSearchStrategy(generator, rng, name=name)
    if name in {"alphamemo", "graphmemo"}:
        return GraphMemoryStrategy(
            generator,
            rng,
            warmup=config.warmup,
            memory_weight=config.memory_weight,
            motif_sample_size=config.motif_sample_size,
            random_motif_prob=config.random_motif_prob,
        )
    if name == "veto":
        return VetoMemoryStrategy(generator, rng, warmup=config.warmup)
    if name == "sspm":
        return SSPMStrategy(generator, rng, warmup=config.warmup, memory_weight=config.memory_weight)
    raise ValueError(f"unknown strategy: {name}")


def run_search(config: RunConfig, out: str | Path | None = None, verbose: bool = True) -> dict:
    load_dotenv()
    rng = np.random.default_rng(config.seed)
    market_data = make_market_data(config)
    evaluator = FormulaEvaluator(market_data)
    generator = make_generator(config, rng)
    strategy = make_strategy(config.strategy, generator, rng, config)
    strategy.initialize(evaluator, default_seed_formulas())

    events: list[SearchEvent] = []
    discovered: list[dict] = []
    step = 0
    while step < config.budget:
        batch_n = min(config.batch_size, config.budget - step)
        candidates = strategy.propose(batch_n, step)
        for candidate in candidates:
            if step >= config.budget:
                break
            step += 1
            result = evaluator.evaluate(candidate.formula)
            success = _is_success(result, config)
            strategy.update(candidate, result, success, step)
            if success:
                evaluator.add_to_library(candidate.formula)
                discovered.append(
                    {
                        "step": step,
                        "formula": candidate.formula,
                        "ic": result.ic,
                        "icir": result.icir,
                        "ric": result.ric,
                        "ricir": result.ricir,
                        "abs_ic": result.abs_ic,
                        "abs_icir": result.abs_icir,
                        "abs_ric": result.abs_ric,
                        "abs_ricir": result.abs_ricir,
                        "category": candidate.category,
                        "motif": candidate.motif,
                        "parent_id": candidate.parent_id,
                    }
                )
            events.append(SearchEvent(step, strategy.name, candidate, result, success, len(discovered)))

        if verbose and (step % max(10, config.batch_size * 4) == 0 or step == config.budget):
            ok_count = sum(1 for e in events if e.result.ok)
            print(
                f"[{strategy.name}] step={step:4d}/{config.budget} "
                f"effective={len(discovered):3d} ok={ok_count:3d}"
            )

    payload = _build_payload(config, strategy, discovered, events)
    if out is not None:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def make_market_data(config: RunConfig):
    if config.evaluator == "synthetic":
        return make_synthetic_market(n_days=config.n_days, n_assets=config.n_assets, seed=config.market_seed)
    if config.evaluator == "qlib":
        return make_qlib_market_data(
            market=config.market,
            start_time=config.data_start,
            end_time=config.data_end,
            provider_uri=config.provider_uri or None,
            label_days=config.label_days,
        )
    raise ValueError(f"unknown evaluator: {config.evaluator}")


def run_benchmark(
    strategies: Iterable[str],
    seeds: Iterable[int],
    base_config: RunConfig,
    out: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    runs = []
    for seed in seeds:
        for strategy_name in strategies:
            cfg = RunConfig(**asdict(base_config))
            cfg.strategy = strategy_name
            cfg.seed = int(seed)
            cfg.market_seed = base_config.market_seed + int(seed)
            if verbose:
                print(f"\n=== strategy={strategy_name} seed={seed} ===")
            result = run_search(cfg, out=None, verbose=verbose)
            runs.append(result)

    summary: dict[str, dict] = {}
    for strategy_name in strategies:
        vals = [run["summary"]["n_effective"] for run in runs if run["config"]["strategy"] == strategy_name]
        if not vals:
            continue
        summary[strategy_name] = {
            "mean_effective": float(np.mean(vals)),
            "std_effective": float(np.std(vals)),
            "values": vals,
        }
        for metric in ("ic", "icir", "ric", "ricir"):
            metric_vals = [
                run["summary"][f"mean_abs_{metric}_discovered"]
                for run in runs
                if run["config"]["strategy"] == strategy_name
            ]
            summary[strategy_name][f"mean_abs_{metric}"] = float(np.mean(metric_vals)) if metric_vals else 0.0
    random_mean = max(summary.get("random", {}).get("mean_effective", 1e-6), 1e-6)
    for stats in summary.values():
        stats["vs_random"] = float(stats["mean_effective"] / random_mean)

    payload = {"summary": summary, "runs": runs}
    if out is not None:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _is_success(result, config: RunConfig) -> bool:
    return (
        result.ok
        and result.abs_ic >= config.success_ic
        and result.abs_icir >= config.success_icir
        and result.max_corr <= config.corr_threshold
    )


def _build_payload(config: RunConfig, strategy: SearchStrategy, discovered: list[dict], events: list[SearchEvent]) -> dict:
    ok_events = [event for event in events if event.result.ok]
    discovered_events = [event for event in events if event.success]
    mean_abs_ic_ok = float(np.mean([event.result.abs_ic for event in ok_events])) if ok_events else 0.0
    return {
        "config": asdict(config),
        "summary": {
            "strategy": strategy.name,
            "n_effective": len(discovered),
            "n_ok": len(ok_events),
            "budget": config.budget,
            "mean_abs_ic_ok": mean_abs_ic_ok,
            "mean_abs_ic_discovered": _mean_result_metric(discovered_events, "abs_ic"),
            "mean_abs_icir_discovered": _mean_result_metric(discovered_events, "abs_icir"),
            "mean_abs_ric_discovered": _mean_result_metric(discovered_events, "abs_ric"),
            "mean_abs_ricir_discovered": _mean_result_metric(discovered_events, "abs_ricir"),
        },
        "discovered": discovered,
        "curve": [(event.step, event.n_discovered) for event in events],
        "events": [event.to_dict() for event in events],
        "diagnostics": strategy.diagnostics(),
    }


def _mean_result_metric(events: list[SearchEvent], attr: str) -> float:
    if not events:
        return 0.0
    return float(np.mean([getattr(event.result, attr) for event in events]))
