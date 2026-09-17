#!/usr/bin/env bash
# OPTIONAL overview (descriptive, not pre-registered): derive every CPU variant of an eval
# root — halueval.{parsed,lenient,decontam,constrained_decontam}, faitheval.{strict,wordmatch,
# contains} — into sibling <run>/<bench>.<variant>/ dirs. No GPU, no model. Original dirs are
# only read. The official decontamination workflow (run_decontam.sh -> $EVAL_ROOT_MOD) is
# unchanged; see RUNNING_NEW_OPTIONS.md "Optional: all variants in one run (overview)".
#
# Usage (from anywhere):
#   ./analysis/run_derive.sh --root outputs/eval/v3 [--variants all] [--dry-run]
#
# Thin wrapper over `python -m analysis.derive`, run in Eval_master's own .venv (which has
# the PyYAML the FaithEval re-scorer needs).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -n "${VENV_ROOT:-}" ]]; then LAUNCHER_VENV="${VENV_ROOT}/Eval_master"
    else LAUNCHER_VENV="${ROOT_DIR}/.venv"; fi
    for candidate in "${LAUNCHER_VENV}/bin/python" "${LAUNCHER_VENV}/Scripts/python.exe"; do
        if [[ -x "${candidate}" ]]; then PYTHON="${candidate}"; break; fi
    done
    [[ -n "${PYTHON:-}" ]] || {
        echo "error: no interpreter under ${LAUNCHER_VENV} (VENV_ROOT=${VENV_ROOT:-<unset>})." >&2
        echo "       Create it: ./setup_envs_HPC.sh (or ./setup_envs_local.sh), or export \$PYTHON." >&2
        exit 1
    }
fi

# --root is resolved against the caller's cwd before we cd.
args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --root) args+=("--root" "$(cd "$2" && pwd)"); shift 2 ;;
        *) args+=("$1"); shift ;;
    esac
done

cd "${ROOT_DIR}"
exec "$PYTHON" -m analysis.derive "${args[@]}"
