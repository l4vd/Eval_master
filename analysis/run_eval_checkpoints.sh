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
fi
PYTHON="${PYTHON:-python}"

cd "${ROOT_DIR}"
exec "$PYTHON" -m analysis.eval_checkpoints "$@"
