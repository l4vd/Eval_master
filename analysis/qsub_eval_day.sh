#!/usr/bin/env bash
# Submit one PBS job per training ensemble of a single day (outputs/<DATE>/*/seed_*),
# each running run_eval_checkpoints.sh over that ensemble's seeds.
#
# Usage (from anywhere; job scripts and outputs are relative to Eval_master):
#   # 1) print: write the job scripts, show the qsub lines, submit nothing
#   ./analysis/qsub_eval_day.sh /gpfs/.../SP-DPO-Base/outputs/2026-09-10
#
#   # 2) check: also run every job script HERE with --dry-run (modules, .env, .venv,
#   #    checkpoint glob) -- no allocation, no models loaded
#   ./analysis/qsub_eval_day.sh /gpfs/.../outputs/2026-09-10 check
#
#   # 3) submit for real
#   QSUB_RES="-l select=1:ncpus=4:mem=32gb:ngpus=1 -l walltime=24:00:00 -A <project>" \
#       ./analysis/qsub_eval_day.sh /gpfs/.../outputs/2026-09-10 submit
#
# Env: OUT_ROOT (outputs/eval/v4-2), SUFFIX (""), BENCHES (faitheval,halueval,harness),
#      JOB_NAME (DevSession), QSUB_RES (required for submit), JOB_DIR (logs/qsub_jobs/<DATE>),
#      EXTRA (Hydra overrides forwarded via --launcher-extra, space-separated),
#      VARIANTS (unset; `all` or a comma list = the optional all-variants overview, see below).
#
# OPTIONAL overview (descriptive, not pre-registered): VARIANTS=all appends variants=[all] to
# EXTRA (VARIANTS=halueval.constrained,faitheval.mc appends that list); qsub_overview_day.sh
# wraps this with a base job and a CPU derive job. Every variant then lands in a sibling
# <run>/<bench>.<variant>/ dir of OUT_ROOT (never in an original dir), finished units are
# skipped, and the legacy variant keys below are refused in EXTRA. EVAL_MASTER_REV (from $OUT_ROOT/EVAL_BLOCK.txt, else git) is exported for
# the launch log <run>/launches.jsonl.
#   VARIANTS=all ./analysis/qsub_eval_day.sh /gpfs/.../outputs/2026-09-10 check
#
# Modified protocols (constrained HaluEval, decontam, counterfactual_mc) go to their OWN
# root: --resume and the census read any summary under the original root as an original
# run. So EXTRA carrying those keys is refused unless OUT_ROOT contains "modified":
#   OUT_ROOT=outputs/eval/v4-2_modified BENCHES=halueval,faitheval EXTRA="$EVAL_MODIFIED" \
#       ./analysis/qsub_eval_day.sh /gpfs/.../outputs/2026-09-10 check
#
# NOTE: one job evaluates ALL seeds of an ensemble sequentially, so the walltime must
# cover the slowest arm x 6 seeds. Jobs pass --resume: re-submitting after a walltime
# kill skips seeds whose benchmarks already wrote a summary.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=qsub_lib.sh
source "${SCRIPT_DIR}/qsub_lib.sh"

DAY="${1:?usage: $0 <SP-DPO-Base/outputs/DATE> [print|check|submit]}"
MODE="${2:-print}"
DAY="$(cd "$DAY" && pwd)"
OUT_ROOT="${OUT_ROOT:-outputs/eval/v4-2}"
SUFFIX="${SUFFIX:-}"
BENCHES="${BENCHES:-faitheval,halueval,harness}"
JOB_NAME="${JOB_NAME:-DevSession}"
QSUB_RES="${QSUB_RES:-}"
JOB_DIR="${JOB_DIR:-logs/qsub_jobs/$(basename "$DAY")}"

case "$MODE" in print|check|submit) ;; *) echo "error: mode must be print|check|submit" >&2; exit 2 ;; esac
[[ "$MODE" != submit || -n "$QSUB_RES" ]] || { echo "error: set QSUB_RES for submit" >&2; exit 2; }

EXTRA="${EXTRA:-}"
VARIANTS="${VARIANTS:-}"
mod_re='scoring=(constrained|both)|decontam\.enabled=true|counterfactual_mc'
variant_keys_re="${mod_re}|strict_match=true"
if [[ -n "$VARIANTS" ]]; then
    [[ "$VARIANTS" =~ ^[a-z_]+(\.[a-z_]+)?(,[a-z_]+(\.[a-z_]+)?)*$ ]] || {
        echo "error: VARIANTS must be 'all' or a comma list of variant names (analysis/variants.py)" >&2; exit 2; }
    if [[ "$EXTRA" =~ $variant_keys_re ]]; then
        echo "error: VARIANTS computes every variant in its own dir; drop the legacy variant" >&2
        echo "       overrides from EXTRA ($EXTRA)" >&2
        exit 2
    fi
    [[ "$EXTRA" != *variants=* ]] || { echo "error: set VARIANTS, not variants= in EXTRA" >&2; exit 2; }
    EXTRA="${EXTRA:+$EXTRA }variants=[$VARIANTS]"
fi
split_overrides "$EXTRA"
if [[ -z "$VARIANTS" && "$EXTRA" =~ $mod_re && "$OUT_ROOT" != *modified* ]]; then
    echo "error: EXTRA holds modified-protocol overrides; OUT_ROOT ($OUT_ROOT) must be a" >&2
    echo "       separate root containing 'modified', e.g. OUT_ROOT=${OUT_ROOT}_modified" >&2
    exit 2
fi

cd "$ROOT_DIR"
mkdir -p "$JOB_DIR"

# Code version for the overview's launches.jsonl: the procedure's EVAL_BLOCK.txt, else git.
EVAL_MASTER_REV="$(eval_master_rev "$OUT_ROOT")"

declare -A seen
n=0
for run in "$DAY"/*/; do
    run="${run%/}"
    compgen -G "$run/seed_*" >/dev/null || continue          # not an ensemble dir
    name="$(basename "$run")"; name="${name#??-??-??_}"      # drop HH-MM-SS_ prefix
    [[ -n "${seen[$name]:-}" ]] && name="$(basename "$run")" # same method twice: keep prefix
    seen[$name]=1
    job="$JOB_DIR/$(basename "$run").pbs"

    {
        printf '#!/bin/bash -l\n'
        printf 'CKPT=%q\nOUT=%q\nBENCHES=%q\n' "$run/seed_*" "$OUT_ROOT/$name$SUFFIX" "$BENCHES"
        printf 'EXTRA=(%s)\n' "${extra_words[*]@Q}"   # quoted: [counterfactual_mc] is a glob
        printf 'export EVAL_MASTER_REV=%q\n' "$EVAL_MASTER_REV"
        cat <<'EOF'
set -eo pipefail
module load uv/0.10.2  gcc/13.2.0 Openssl/1.1.1t Tcl/8.6.11 Tk/8.6.13 Python/3.12.3 CUDA/11.7.1
cd /gpfs/project/ladan102/git-source/SP_DPO/Eval_master/

dos2unix .env
set -a; source .env; set +a
source .venv/bin/activate
export USER_NAME=ladan102
export PYTHONUNBUFFERED=1   # PBS stdout is a file: without this, Python's progress lines
                            # land after (or never before) a child's stderr error

echo "host=$(hostname) job=${PBS_JOBID:-local} ckpt=$CKPT out=$OUT"
# --launcher-extra swallows everything after it, so it goes after "$@" (e.g. --dry-run)
if (( ${#EXTRA[@]} )); then set -- "$@" --launcher-extra "${EXTRA[@]}"; fi
./analysis/run_eval_checkpoints.sh --checkpoints "$CKPT" --out "$OUT" \
    --benchmarks "$BENCHES" --resume "$@"
EOF
    } > "$job"

    n=$((n + 1))
    echo "== $(basename "$run") -> $OUT_ROOT/$name$SUFFIX${EXTRA:+  [extra: $EXTRA]}"
    case "$MODE" in
        print)  echo "   qsub -N $JOB_NAME ${QSUB_RES:-<QSUB_RES>} $job" ;;
        check)  bash -l "$job" --dry-run ;;
        submit) qsub -N "$JOB_NAME" $QSUB_RES "$job" ;;   # QSUB_RES unquoted: split into flags
    esac
done

(( n > 0 )) || { echo "error: no */seed_* dirs under $DAY" >&2; exit 1; }
echo "$n job(s), mode=$MODE, scripts in $JOB_DIR/"
