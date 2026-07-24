#!/usr/bin/env bash
# Plotting step ONLY: reload persisted analysis artifacts and (re)generate figures.
# No eval, no aggregation — consumes records.jsonl (+ optional comparisons.json).
#
# Usage:
#   ./run_plots.sh --from outputs/analysis
#   ./run_plots.sh --from outputs/analysis --out figs --benchmarks faitheval,harness
#
# Thin wrapper over `python -m analysis.plot`; every plot flag is forwarded. Runs in
# Eval_master's own .venv (needs the `plot` extra: matplotlib).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -n "${VENV_ROOT:-}" ]]; then LAUNCHER_VENV="${VENV_ROOT}/Eval_master"
    else LAUNCHER_VENV="${ROOT_DIR}/.venv"; fi
    for candidate in "${LAUNCHER_VENV}/bin/python" "${LAUNCHER_VENV}/Scripts/python.exe"; do
        if [[ -x "${candidate}" ]]; then PYTHON="${candidate}"; break; fi
    done
fi
PYTHON="${PYTHON:-python}"

cd "${ROOT_DIR}"
exec "$PYTHON" -m analysis.plot "$@"
