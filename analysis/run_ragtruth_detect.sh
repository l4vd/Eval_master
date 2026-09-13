#!/usr/bin/env bash
# Shared RAG-Truth detection job: after every checkpoint ran `ragtruth.stage=generate`, load the
# 13B detector ONCE and run Stage 2 over every <root>/*/seed_*/ragtruth/ and
# <root>/base/run_*/ragtruth/ that has generations.jsonl but no summary.json. Resumable:
# re-submit the same command after a walltime kill.
#
# Usage (from Eval_master/):
#   ./analysis/run_ragtruth_detect.sh --root "$EVAL_ROOT" --dtype bfloat16
#   ./analysis/run_ragtruth_detect.sh --root "$EVAL_ROOT" --dry-run     # list pending dirs only
#
# Detector id / batch size / seed default to conf/ragtruth/default.yaml; --detector-* flags
# override them. Thin wrapper over `python -m analysis.ragtruth_detect` in Eval_master's .venv.

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
exec "$PYTHON" -m analysis.ragtruth_detect "$@"
