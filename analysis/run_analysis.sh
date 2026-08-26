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
exec "$PYTHON" -m analysis.cli "$@"
