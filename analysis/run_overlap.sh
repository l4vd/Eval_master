#!/usr/bin/env bash
# Train <-> eval overlap check: index a training corpus and tier every benchmark row
# against it (exact / near-duplicate / partial), then write the report and, with
# --write-exclusions, the lists the eval-side decontamination reads.
#
# Usage:
#   # default train side: ../SP-DPO-Base/data/ragtruth; report under outputs/overlap/ragtruth/
#   ./analysis/run_overlap.sh --harness-samples outputs/eval/<arm>/seed_42/harness/samples.jsonl
#
#   # refresh the committed exclusion lists (decontamination/ragtruth/*.json)
#   ./analysis/run_overlap.sh --harness-samples <samples.jsonl> --write-exclusions
#
#   # an added corpus, before training on it (FUTURE_EXTENSIONS.md §1h)
#   ./analysis/run_overlap.sh --train faithdial='jsonl:data/faithdial/train.jsonl:context,chosen,rejected:source_id'
#
# Thin wrapper over `python -m analysis.overlap` (stdlib only). Runs in Eval_master's own
# .venv (same resolution as run_eval_ensemble.sh).

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
        echo "       Create it: ./setup_envs_HPC.sh${VENV_ROOT:+ --venv-root \"${VENV_ROOT}\"}   (or ./setup_envs_local.sh)" >&2
        echo "       Or export \$PYTHON to an interpreter that has this project installed." >&2
        exit 1
    }
fi

cd "${ROOT_DIR}"
exec "$PYTHON" -m analysis.overlap "$@"
