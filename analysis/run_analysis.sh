#!/usr/bin/env bash
# JOINED workflow: aggregate -> compare -> plot (the eval-ensemble run plus plotting
# composition). Optionally drives the Hydra launcher --multirun first via --run-evals.
#
# Usage:
#   # base vs SFT vs DPO ensembles, full compare + figures
#   ./run_analysis.sh \
#       --arm base=outputs/.../base_run \
#       --arm sft='outputs/.../*_sft_ensemble/seed_*' \
#       --arm dpo='outputs/.../*_dpo_ensemble/seed_*' \
#       --reference base --out outputs/analysis
#
#   # produce evals for a model list first, then analyse+plot
#   ./run_analysis.sh --run-evals --models id1,id2 --reference model0
#
# Thin wrapper over `python -m analysis.cli` (compare + plot enabled). Runs in
# Eval_master's own .venv (needs the `stats` + `plot` extras for Wilcoxon/figures).

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
exec "$PYTHON" -m analysis.cli "$@"
