#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python -m sspm run \
  --strategy alphamemo \
  --generator heuristic \
  --budget "${BUDGET:-40}" \
  --batch-size "${BATCH_SIZE:-5}" \
  --seed "${SEED:-0}" \
  --out "${OUT:-runs/smoke_alphamemo.json}"

python -m sspm benchmark \
  --strategies alphamemo structured random \
  --generator heuristic \
  --budget "${BENCH_BUDGET:-40}" \
  --seeds 0 1 \
  --out "${BENCH_OUT:-runs/smoke_benchmark.json}"
