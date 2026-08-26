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
    # Deliberately no fallback to a bare `python`. When the env was missing, that
    # fallback turned into an unattributable ": No such file or directory" out of the
    # exec below, naming neither the path expected nor the fix. $VENV_ROOT and
    # setup_envs_*.sh are the two halves that must agree on where the env lives.
    [[ -n "${PYTHON:-}" ]] || {
        echo "error: no interpreter under ${LAUNCHER_VENV} (VENV_ROOT=${VENV_ROOT:-<unset>})." >&2
        echo "       Create it: ./setup_envs_HPC.sh${VENV_ROOT:+ --venv-root \"${VENV_ROOT}\"}   (or ./setup_envs_local.sh)" >&2
        echo "       Or export \$PYTHON to an interpreter that has this project installed." >&2
        exit 1
    }
fi

cd "${ROOT_DIR}"
exec "$PYTHON" -m analysis.plot "$@"
