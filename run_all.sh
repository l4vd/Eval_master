#!/usr/bin/env bash
# Central entry point: run some or all of the four *-reproduce benchmarks against
# one model, configured via Hydra.
#
# Usage:
#   ./run_all.sh                                   # all four, default tiny model
#   ./run_all.sh model.id=/path/to/final_checkpoint
#   ./run_all.sh run='[faitheval,ragtruth]' num_samples=5 model.dtype=float32
#   ./run_all.sh ragtruth.detector.id=/path/to/detector
#   ./run_all.sh dry_run=true                      # print the commands only
#
# Any Hydra override is forwarded (see conf/config.yaml and conf/<benchmark>/).
#
# Environment variables:
#   PYTHON   interpreter used to launch the launcher itself. Defaults to this
#            folder's own .venv (created by ./setup_envs_HPC.sh or
#            ./setup_envs_local.sh). No fallback: a missing env is an error.
#            Per-benchmark interpreters come from Hydra: each benchmark defaults to
#            its own `<folder>/.venv` (`python: auto`), overridable with
#            `<benchmark>.python=...` or globally with `python=...`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${PYTHON:-}" ]]; then
    # Mirrors setup_envs_*.sh: VENV_ROOT relocates the envs out of the repo.
    if [[ -n "${VENV_ROOT:-}" ]]; then
        LAUNCHER_VENV="${VENV_ROOT}/Eval_master"
    else
        LAUNCHER_VENV="${SCRIPT_DIR}/.venv"
    fi
    for candidate in "${LAUNCHER_VENV}/bin/python" "${LAUNCHER_VENV}/Scripts/python.exe"; do
        if [[ -x "${candidate}" ]]; then
            PYTHON="${candidate}"
            break
        fi
    done
    # Deliberately no fallback to a bare `python`: with the env missing, that fallback
    # surfaced only as ": No such file or directory" from the exec below. $VENV_ROOT and
    # setup_envs_*.sh are the two halves that must agree on where the env lives.
    [[ -n "${PYTHON:-}" ]] || {
        echo "error: no interpreter under ${LAUNCHER_VENV} (VENV_ROOT=${VENV_ROOT:-<unset>})." >&2
        echo "       Create it: ./setup_envs_HPC.sh${VENV_ROOT:+ --venv-root \"${VENV_ROOT}\"}   (or ./setup_envs_local.sh)" >&2
        echo "       Or export \$PYTHON to an interpreter that has this project installed." >&2
        exit 1
    }
fi

exec "$PYTHON" "${SCRIPT_DIR}/run_benchmarks.py" "$@"
