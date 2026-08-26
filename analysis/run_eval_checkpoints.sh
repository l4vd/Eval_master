#!/usr/bin/env bash
# Checkpoint-ensemble step: evaluate a TRAINING ensemble (..._ensemble/seed_*) through
# the five-benchmark launcher into an analysis-ready seed_<SEED>/ layout, then (optionally)
# aggregate + compare + plot. This is the bridge the analysis wrappers assume already ran:
# run_analysis.sh reads eval run dirs, it does NOT evaluate checkpoints itself.
#
# Usage:
#   # produce eval outputs for every seed checkpoint, then compare against base
#   ./run_eval_checkpoints.sh \
#       --checkpoints '/gpfs/.../08-48-52_sft_ensemble/seed_*' \
#       --out outputs/eval/sft_ensemble \
#       --analyze --name sft --arm base=/gpfs/.../20-38-30 --reference base
#
#   # just see what would run (no models loaded), restricted to two benchmarks
#   ./run_eval_checkpoints.sh --checkpoints '.../seed_*' --out outputs/eval/tmp \
#       --benchmarks faitheval,harness --dry-run
#
#   # re-submit after a walltime kill: seeds whose benchmarks all wrote a summary
#   # are skipped, so only the truncated tail is re-evaluated
#   ./run_eval_checkpoints.sh --checkpoints '.../seed_*' --out outputs/eval/dpo \
#       --benchmarks faitheval,halueval,harness --resume
#
# NOTE: nothing is checkpointed *within* a seed, and a seed that exits non-zero is
# reported but skipped over (--stop-on-error to abort instead). Prefer one scheduler
# job per seed: runtime is checkpoint-dependent (a model that ignores the length
# instruction can cost 10x a well-behaved one), so a walltime budget calibrated on
# a fast arm will silently truncate a slow one.
#
# Thin wrapper over `python -m analysis.eval_checkpoints`. Runs in Eval_master's own
# .venv (same resolution as run_all.sh / run_analysis.sh).

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
exec "$PYTHON" -m analysis.eval_checkpoints "$@"
