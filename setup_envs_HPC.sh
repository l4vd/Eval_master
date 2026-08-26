#!/usr/bin/env bash
# HPC-only env setup: `uv sync` the launcher, then each benchmark submodule,
# using each project's pyproject-HPC.toml (pinned cluster stack).
#
# Usage:
#   ./setup_envs_HPC.sh                      # envs in <project>/.venv
#   ./setup_envs_HPC.sh --venv-root DIR      # envs in DIR/<project> instead
#   VENV_ROOT=DIR ./setup_envs_HPC.sh        # same, via the environment
#
# VENV_ROOT relocates the environments out of the repo -- onto scratch, or off a
# quota'd project filesystem. It must match what the *runners* use: run_all.sh,
# analysis/run_*.sh and conf/config.yaml's `venv_root` all read $VENV_ROOT and
# look for the launcher env at "$VENV_ROOT/Eval_master". This script previously
# ignored VENV_ROOT entirely and always wrote <project>/.venv, so a shell with
# VENV_ROOT exported produced envs the runners could not find -- and the runners
# then silently fell back to a bare `python`.
#
# Requires `uv` (module load uv/0.10.2 on the cluster). Run this on a LOGIN node:
# the resolve below needs the PyPI mirror, and compute nodes have no network.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARKS=(FaithEval-reproduce TruthfulQA-reproduce HaluEval-reproduce RAGTruth-reproduce harness-eval)

VENV_ROOT="${VENV_ROOT:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv-root) [[ $# -ge 2 ]] || { echo "--venv-root needs a value" >&2; exit 2; }
                     VENV_ROOT="$2"; shift 2 ;;
        -h|--help)   sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv not found. 'module load uv' or install it (https://docs.astral.sh/uv/)." >&2
    exit 1
fi

# venv_dir <project-folder-name|Eval_master>
venv_dir() {
    if [[ -n "${VENV_ROOT}" ]]; then
        echo "${VENV_ROOT}/$1"
    elif [[ "$1" == "Eval_master" ]]; then
        echo "${SCRIPT_DIR}/.venv"          # the launcher's project root IS SCRIPT_DIR
    else
        echo "${SCRIPT_DIR}/$1/.venv"
    fi
}

# uv sync reads pyproject.toml from the project dir, so swap in the HPC pins
# (backing up the original once) before syncing each project. The base
# pyproject.toml's uv.lock is resolved against different (unpinned)
# dependencies, so it's dropped here to force a fresh resolve against the
# HPC pins instead of risking a stale/mismatched lock being reused.
sync_hpc() {
    local dir="$1" target="$2"; shift 2
    [[ -f "${dir}/pyproject-HPC.toml" ]] || { echo "error: no pyproject-HPC.toml in ${dir}" >&2; exit 1; }
    [[ -f "${dir}/pyproject.toml.orig" ]] || cp "${dir}/pyproject.toml" "${dir}/pyproject.toml.orig"
    cp "${dir}/pyproject-HPC.toml" "${dir}/pyproject.toml"
    rm -f "${dir}/uv.lock"
    # UV_PROJECT_ENVIRONMENT puts the env at `target` rather than the in-repo default.
    (cd "${dir}" && UV_PROJECT_ENVIRONMENT="${target}" uv sync "$@")
}

LAUNCHER_VENV="$(venv_dir Eval_master)"
echo "==> Launcher: ${SCRIPT_DIR} -> ${LAUNCHER_VENV}"
# `stats` (scipy) and `plot` (matplotlib) are lazy imports in analysis.stats /
# analysis.plot, but they are not optional in practice: --analyze, run_analysis.sh
# and run_plots.sh all need them. Installing them here rather than leaving them to
# a follow-up `uv sync` means a multi-hour ensemble eval cannot finish and *then*
# die on a missing scipy -- and the extras must be resolved on a login node anyway.
sync_hpc "${SCRIPT_DIR}" "${LAUNCHER_VENV}" --extra dev --extra stats --extra plot

for bench in "${BENCHMARKS[@]}"; do
    target="$(venv_dir "${bench}")"
    echo "==> ${bench} -> ${target}"
    sync_hpc "${SCRIPT_DIR}/${bench}" "${target}" --extra dev
done

echo ""
echo "==================== Environments ===================="
printf '  %-22s %s\n' "Eval_master" "${LAUNCHER_VENV}/bin/python"
for bench in "${BENCHMARKS[@]}"; do
    printf '  %-22s %s\n' "${bench}" "$(venv_dir "${bench}")/bin/python"
done
echo ""
if [[ -n "${VENV_ROOT}" ]]; then
    echo "Envs are outside the repo, so tell the runners where they are:"
    echo "    export VENV_ROOT=\"${VENV_ROOT}\""
    echo "    ./run_all.sh"
else
    echo "Run the suite with:  ./run_all.sh"
fi
