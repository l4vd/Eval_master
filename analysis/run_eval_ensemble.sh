#!/usr/bin/env bash
# Eval-ensemble step ONLY: aggregate an ensemble (and optionally drive the evals that
# produce it) into records.jsonl + aggregate.json — NO figures.
#
# Usage:
#   # aggregate existing run dirs (one arm), no comparison, no plots
#   ./run_eval_ensemble.sh --arm dpo='outputs/.../seed_*' --no-compare
#
#   # produce the runs first via the Hydra launcher --multirun, then aggregate
#   ./run_eval_ensemble.sh --run-evals --models id1,id2 --no-compare
#
# This is a thin wrapper over `python -m analysis.cli --no-plot`; every analysis flag
# (--arm, --benchmarks, --exclude, --reference, --primary-map, --run-evals, ...) is
# forwarded. Runs in Eval_master's own .venv (see run_all.sh for the same resolution).

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
exec "$PYTHON" -m analysis.cli --no-plot "$@"
