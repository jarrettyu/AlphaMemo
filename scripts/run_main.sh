#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

PRESET="${PRESET:-paper2025}"
MARKETS="${MARKETS:-csi500 sp500}"
CN_PROVIDER_URI="${CN_PROVIDER_URI:-data/qlib/cn_data_nextday2025}"
US_PROVIDER_URI="${US_PROVIDER_URI:-data/qlib/us_data_nextday2025}"
OUT_DIR="${OUT_DIR:-runs/main_20day2025}"

BUDGET="${BUDGET:-500}"
BATCH_SIZE="${BATCH_SIZE:-10}"
SUCCESS_ICIR="${SUCCESS_ICIR:-0.02}"
CORR_THRESHOLD="${CORR_THRESHOLD:-0.70}"
WARMUP="${WARMUP:-200}"
MEMORY_WEIGHT="${MEMORY_WEIGHT:-0.05}"
MOTIF_SAMPLE_SIZE="${MOTIF_SAMPLE_SIZE:-4}"
RANDOM_MOTIF_PROB="${RANDOM_MOTIF_PROB:-0.35}"
MAX_FACTORS="${MAX_FACTORS:-50}"
SEED="${SEED:-0}"
LABEL_DAYS="${LABEL_DAYS:-20}"

GENERATOR="${GENERATOR:-openrouter}"
MODEL="${MODEL:-${OPENROUTER_MODEL:-deepseek/deepseek-v4-flash}}"
RUN_BACKTEST="${RUN_BACKTEST:-1}"
QRUN_BIN="${QRUN_BIN:-scripts/qrun_alphamemo.sh}"

is_placeholder_key() {
  case "${1:-}" in
    ""|"sk-or-v1-your-key-here"|"sk-your-key-here"|"your-key-here") return 0 ;;
    *) return 1 ;;
  esac
}

if [[ "${GENERATOR}" == "openrouter" ]] && is_placeholder_key "${OPENROUTER_API_KEY:-}"; then
  echo "Missing OPENROUTER_API_KEY. Copy .env.example to .env and set your key, or use GENERATOR=heuristic for smoke tests." >&2
  exit 1
fi

cmd=(
  python -m sspm main-table
  --preset "${PRESET}"
  --strategies alphamemo
  --markets ${MARKETS}
  --cn-provider-uri "${CN_PROVIDER_URI}"
  --us-provider-uri "${US_PROVIDER_URI}"
  --budget "${BUDGET}"
  --batch-size "${BATCH_SIZE}"
  --success-icir "${SUCCESS_ICIR}"
  --corr-threshold "${CORR_THRESHOLD}"
  --warmup "${WARMUP}"
  --memory-weight "${MEMORY_WEIGHT}"
  --motif-sample-size "${MOTIF_SAMPLE_SIZE}"
  --random-motif-prob "${RANDOM_MOTIF_PROB}"
  --max-factors "${MAX_FACTORS}"
  --label-days "${LABEL_DAYS}"
  --seed "${SEED}"
  --generator "${GENERATOR}"
  --model "${MODEL}"
  --out-dir "${OUT_DIR}"
  --qrun-bin "${QRUN_BIN}"
)

if [[ "${RUN_BACKTEST}" == "1" ]]; then
  cmd+=(--run-backtest)
fi

echo "Running AlphaMemo"
echo "markets=${MARKETS}"
echo "generator=${GENERATOR}"
echo "model=${MODEL}"
echo "out_dir=${OUT_DIR}"
echo

"${cmd[@]}"
