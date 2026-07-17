#!/usr/bin/env bash
# Create one virtualenv per benchmark (plus one for the launcher itself).
#
# The five benchmarks have incompatible dependency stacks, so each gets its own
# environment. `run_benchmarks.py` resolves them automatically (`python: auto` in
# conf/config.yaml), so after this script `./run_all.sh` needs no extra flags.
#
# Usage:
#   ./setup_envs.sh                      # envs in <benchmark>/.venv (Linux / cluster)
#   ./setup_envs.sh --hpc                # pinned cluster stack (pyproject-HPC.toml)
#   ./setup_envs.sh --venv-root DIR      # envs in DIR/<benchmark> instead
#   VENV_ROOT=DIR ./setup_envs.sh        # same, via the environment
#
# Each benchmark is installed with `uv sync` from its committed uv.lock, so every
# machine gets the identical dependency stack (this is what keeps runs comparable
# across time and between laptop and cluster). Note that --hpc swaps in
# pyproject-HPC.toml, which necessarily re-resolves and rewrites that folder's
# uv.lock for the cluster stack -- don't commit the result back.
#
# --venv-root is required on Windows when this repo sits deep in the filesystem
# (e.g. under OneDrive): `<benchmark>/.venv/Lib/site-packages/...` then exceeds the
# 260-character MAX_PATH limit and imports fail. Export the same VENV_ROOT before
# ./run_all.sh, or set `venv_root=DIR` on its command line. Suggested value:
#   ./setup_envs.sh --venv-root "$LOCALAPPDATA/eval-venvs"
#
# Requires `uv` (module load uv/0.10.2 on the cluster).

set -euo pipefail

uv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARKS=(FaithEval-reproduce TruthfulQA-reproduce HaluEval-reproduce RAGTruth-reproduce harness-eval)

HPC=false
VENV_ROOT="${VENV_ROOT:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hpc) HPC=true; shift ;;
        --venv-root) VENV_ROOT="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv not found. Install it (https://docs.astral.sh/uv/) or 'module load uv'." >&2
    exit 1
fi

# uv hardlinks from its cache by default, which fails on OneDrive-backed and some
# network filesystems ("Cloud operation ... incompatible hard links", os error 396).
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) export UV_LINK_MODE=copy ;;
esac

# venv_dir <benchmark>
venv_dir() {
    if [[ -n "${VENV_ROOT}" ]]; then
        echo "${VENV_ROOT}/$1"
    else
        echo "${SCRIPT_DIR}/$1/.venv"
    fi
}

# The launcher itself only needs Hydra; the benchmarks bring their own stacks.
# Mirrors run_all.sh's own resolution (SCRIPT_DIR *is* the launcher's project
# root, unlike the benchmarks which live in subfolders of it).
if [[ -n "${VENV_ROOT}" ]]; then
    LAUNCHER_VENV="${VENV_ROOT}/Eval_master"
else
    LAUNCHER_VENV="${SCRIPT_DIR}/.venv"
fi
echo "==> Launcher env: ${LAUNCHER_VENV}"

if $HPC; then
    [[ -f "${SCRIPT_DIR}/pyproject.toml.orig" ]] || cp "${SCRIPT_DIR}/pyproject.toml" "${SCRIPT_DIR}/pyproject.toml.orig"
    cp "${SCRIPT_DIR}/pyproject-HPC.toml" "${SCRIPT_DIR}/pyproject.toml"
fi

# `uv sync` (not `uv pip install`) so the committed uv.lock is honoured, same as
# every benchmark below.
(cd "${SCRIPT_DIR}" && UV_PROJECT_ENVIRONMENT="${LAUNCHER_VENV}" uv sync --extra dev)

for bench in "${BENCHMARKS[@]}"; do
    folder="${SCRIPT_DIR}/${bench}"
    target="$(venv_dir "${bench}")"
    echo "==> ${bench} -> ${target}"

    if $HPC; then
        # Each repo documents this swap: the HPC file pins the known-good cluster
        # stack (torch 2.2.2 / transformers 4.41 / Python 3.12).
        [[ -f "${folder}/pyproject.toml.orig" ]] || cp "${folder}/pyproject.toml" "${folder}/pyproject.toml.orig"
        cp "${folder}/pyproject-HPC.toml" "${folder}/pyproject.toml"
    fi

    # `uv sync` (not `uv pip install`) so the committed uv.lock is honoured and every
    # machine resolves the identical stack. UV_PROJECT_ENVIRONMENT puts the env at
    # `target` instead of the default in-repo `.venv`. `--extra dev` adds pytest, so
    # each benchmark can run its own test suite in its own env.
    (cd "${folder}" && UV_PROJECT_ENVIRONMENT="${target}" uv sync --extra dev)
done

echo ""
echo "==================== Environments ===================="
for bench in "${BENCHMARKS[@]}"; do
    target="$(venv_dir "${bench}")"
    for rel in "bin/python" "Scripts/python.exe"; do
        if [[ -x "${target}/${rel}" ]]; then
            printf '  %-22s %s\n' "${bench}" "${target}/${rel}"
            break
        fi
    done
done
echo ""
if [[ -n "${VENV_ROOT}" ]]; then
    echo "Envs are outside the repo, so tell the launcher where they are:"
    echo "    export VENV_ROOT=\"${VENV_ROOT}\""
    echo "    ./run_all.sh"
else
    echo "Run the suite with:  ./run_all.sh"
fi
