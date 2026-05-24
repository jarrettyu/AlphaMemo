#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
ORIGINAL_HOME="${HOME:-}"
PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
USER_SITE="${ORIGINAL_HOME}/.local/lib/python${PYTHON_VERSION}/site-packages"

if [[ -d "${USER_SITE}" ]]; then
  export PYTHONPATH="${USER_SITE}:${PYTHONPATH:-}"
fi

RUNTIME_HOME="${QLIB_RUNTIME_HOME:-${PWD}/home}"
mkdir -p "${RUNTIME_HOME}/tmp" "${RUNTIME_HOME}/.config/matplotlib"
export HOME="${RUNTIME_HOME}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${RUNTIME_HOME}/.config/matplotlib}"

exec "${PYTHON_BIN}" -m qlib.cli.run "$@"
