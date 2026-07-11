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
#   PYTHON   interpreter used to launch the launcher itself (default: python).
#            Per-benchmark interpreters are set in Hydra: `<benchmark>.python=...`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

exec "$PYTHON" "${SCRIPT_DIR}/run_benchmarks.py" "$@"
