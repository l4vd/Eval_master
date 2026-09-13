#!/usr/bin/env bash
# Eval-side decontamination of FINISHED HaluEval runs — no GPU, no model.
#
# Scores every stored <root>/<arm>/<run>/halueval/<task>_*_results.json against the task's
# exclusion list and writes <task>_<label>[_constrained]_decontam_summary.json into the SAME
# <arm>/<run>/halueval/ layout under a separate root (plus run_metadata.json, so the analysis
# recovers each seed). The original root is only read: a decontaminated summary is a
# modified protocol, and next to the original runs it would break the §5.5 census and
# --resume (SP-DPO-Base/EXPERIMENT_PROCEDURE.md §5.2).
#
# Usage (from Eval_master/):
#   ./analysis/run_decontam.sh --root "$EVAL_ROOT" --out-root "$EVAL_ROOT_MOD"
#   ./analysis/run_decontam.sh --root outputs/eval/v3 --out-root outputs/eval/v3_modified \
#       --list decontamination/ragtruth/halueval__summarization.json
#
# Without --list, every decontamination/ragtruth/halueval__<task>.json is used. Constrained
# results (*_constrained_results.json) under --root are scored the same way.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

EVAL_ROOT_IN=""
OUT_ROOT=""
LISTS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --root) EVAL_ROOT_IN="$2"; shift 2 ;;
        --out-root) OUT_ROOT="$2"; shift 2 ;;
        --list) LISTS+=("$2"); shift 2 ;;
        -h|--help) sed -n '2,18p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "error: unknown argument '$1' (see --help)" >&2; exit 2 ;;
    esac
done
[[ -n "${EVAL_ROOT_IN}" && -n "${OUT_ROOT}" ]] || {
    echo "usage: $0 --root DIR --out-root DIR [--list LIST.json ...]" >&2; exit 2; }

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

cd "${ROOT_DIR}"
[[ -d "${EVAL_ROOT_IN}" ]] || { echo "error: --root ${EVAL_ROOT_IN} is not a directory" >&2; exit 1; }
mkdir -p "${OUT_ROOT}"
if [[ "$(cd "${EVAL_ROOT_IN}" && pwd -P)" == "$(cd "${OUT_ROOT}" && pwd -P)" ]]; then
    echo "error: --out-root must not be --root: decontaminated summaries belong in the modified-protocol root" >&2
    exit 1
fi

if [[ ${#LISTS[@]} -eq 0 ]]; then
    shopt -s nullglob
    LISTS=(decontamination/ragtruth/halueval__*.json)
    shopt -u nullglob
fi
[[ ${#LISTS[@]} -gt 0 ]] || {
    echo "error: no exclusion lists; write them with ./analysis/run_overlap.sh --write-exclusions" >&2; exit 1; }

SCORER="HaluEval-reproduce/evaluation/score_results.py"
written=0
for list in "${LISTS[@]}"; do
    task="$(basename "${list}" .json)"
    task="${task#halueval__}"
    shopt -s nullglob
    results_files=("${EVAL_ROOT_IN}"/*/*/halueval/"${task}"_*_results.json)
    shopt -u nullglob
    echo "== ${task}: ${#results_files[@]} results file(s) under ${EVAL_ROOT_IN} (list ${list})"
    for results in "${results_files[@]}"; do
        run_dir="$(dirname "$(dirname "${results}")")"
        rel="${run_dir#"${EVAL_ROOT_IN%/}"/}"
        out_run="${OUT_ROOT%/}/${rel}"
        mkdir -p "${out_run}/halueval"
        if [[ -f "${run_dir}/run_metadata.json" && ! -f "${out_run}/run_metadata.json" ]]; then
            cp "${run_dir}/run_metadata.json" "${out_run}/run_metadata.json"
        fi
        "${PYTHON}" "${SCORER}" "${results}" --exclude "${list}" --emit-summary "${out_run}/halueval" \
            | grep -E "wrote|seen - clean|INCOMPLETE" || true
        written=$((written + 1))
    done
done
echo "==> ${written} decontaminated summary file(s) under ${OUT_ROOT}"
