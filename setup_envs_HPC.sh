#!/usr/bin/env bash
# HPC-only env setup: `uv sync` the launcher, then each benchmark submodule,
# using each project's pyproject-HPC.toml (pinned cluster stack).
#
# Usage:
#   ./setup_envs_HPC.sh
#
# Requires `uv` (module load uv/0.10.2 on the cluster).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARKS=(FaithEval-reproduce TruthfulQA-reproduce HaluEval-reproduce RAGTruth-reproduce harness-eval)

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv not found. 'module load uv' or install it (https://docs.astral.sh/uv/)." >&2
    exit 1
fi

# uv sync reads pyproject.toml from the project dir, so swap in the HPC pins
# (backing up the original once) before syncing each project. The base
# pyproject.toml's uv.lock is resolved against different (unpinned)
# dependencies, so it's dropped here to force a fresh resolve against the
# HPC pins instead of risking a stale/mismatched lock being reused.
sync_hpc() {
    local dir="$1"
    [[ -f "${dir}/pyproject.toml.orig" ]] || cp "${dir}/pyproject.toml" "${dir}/pyproject.toml.orig"
    cp "${dir}/pyproject-HPC.toml" "${dir}/pyproject.toml"
    rm -f "${dir}/uv.lock"
    (cd "${dir}" && uv sync --extra dev)
}

echo "==> Launcher: ${SCRIPT_DIR}"
sync_hpc "${SCRIPT_DIR}"

for bench in "${BENCHMARKS[@]}"; do
    echo "==> ${bench}"
    sync_hpc "${SCRIPT_DIR}/${bench}"
done

echo ""
echo "==================== Environments ===================="
printf '  %-22s %s\n' "Eval_master" "${SCRIPT_DIR}/.venv/bin/python"
for bench in "${BENCHMARKS[@]}"; do
    printf '  %-22s %s\n' "${bench}" "${SCRIPT_DIR}/${bench}/.venv/bin/python"
done
echo ""
echo "Run the suite with:  ./run_all.sh"
