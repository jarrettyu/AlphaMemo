# AlphaMemo

AlphaMemo is a research codebase for self-evolving formulaic alpha mining with Structured Search-Process Memory (SSPM).
The agent searches symbolic factors, records which parent-context and edit-motif transitions work or fail, and uses that memory as a calibrated residual guide during later evolution.

This repository contains the core AlphaMemo implementation and scripts for synthetic smoke tests and Qlib-based CSI500/S&P500 experiments.

## Setup

```bash
conda env create -f environment.yml
conda activate alphamemo
pip install -e .
```

Alternatively, the same Python dependencies are listed in `requirements.txt` for pip-based environments.

Local smoke tests do not require an LLM API key:

```bash
bash scripts/run_smoke_test.sh
```

For LLM-backed runs, copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`, or pass an OpenAI-compatible endpoint with environment variables.

## Main Experiment

The paper setting uses a 20-trading-day forward return with train/validation/test split:

- train: 2016-01-01 to 2020-12-31
- validation: 2021-01-01 to 2021-12-31
- test/backtest: 2022-01-01 to 2025-12-26

Expected Qlib data locations:

```text
data/qlib/cn_data_nextday2025
data/qlib/us_data_nextday2025
```

Run AlphaMemo on CSI500 and S&P500:

```bash
bash scripts/run_main.sh
```

Useful overrides:

```bash
# quick parser/search sanity check without LLM calls
GENERATOR=heuristic RUN_BACKTEST=0 BUDGET=50 bash scripts/run_main.sh

# one-market reproduction/debug run with the paper AlphaMemo configuration
MARKETS="csi500" BUDGET=500 MEMORY_WEIGHT=0.05 WARMUP=200 bash scripts/run_main.sh
```

The main AlphaMemo configuration used by the current draft is:

```text
strategy=alphamemo
budget=500
batch_size=10
label_days=20
warmup=200
memory_weight=0.05
motif_sample_size=4
random_motif_prob=0.35
max_factors=50
```

Collect generated result tables:

```bash
python scripts/collect_results.py --root runs/main_20day2025
```

Compute annual IC/RankIC decay diagnostics:

```bash
python scripts/alpha_decay.py --root runs/main_20day2025
```

## Data

If Qlib data is not already available, the helper scripts can build approximate Qlib-format OHLCV data from Yahoo Finance:

```bash
python scripts/build_cn_qlib_yfinance.py --target-dir data/qlib/cn_data_nextday2025 --start 2016-01-01 --end 2025-12-27
python scripts/build_us_qlib_yfinance.py --target-dir data/qlib/us_data_nextday2025 --start 2016-01-01 --end 2025-12-27
```

For final paper numbers, use a stable data snapshot and report the exact provider paths and date coverage.

## Code Map

- `sspm/core`: symbolic formulas, operators, search ledger, and edit motifs.
- `sspm/memory`: residual process memory and asymmetric failure veto.
- `sspm/strategies/graph_memory.py`: the main AlphaMemo strategy.
- `sspm/evaluation`: synthetic evaluator, Qlib loader, formula evaluator, and Qlib export.
- `scripts/run_main.sh`: Qlib experiment wrapper.
- `scripts/run_smoke_test.sh`: no-API sanity check.

## Notes

This code is intended for research reproduction. It is not financial advice and should not be used for live trading without independent validation.
