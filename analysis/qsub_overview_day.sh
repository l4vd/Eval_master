#!/usr/bin/env bash
# OPTIONAL all-variants overview (descriptive, not pre-registered) for one training day:
# the overview counterpart of qsub_eval_day.sh. Not part of EXPERIMENT_PROCEDURE §5.4-§5.8.
#
# Writes up to three kinds of PBS jobs:
#   1. one GPU job per training ensemble (outputs/<DATE>/*/seed_*): qsub_eval_day.sh with
#      VARIANTS set, so every variant lands in <OUT_ROOT>/<METHOD>_ensemble<SUFFIX>/seed_*/
#      <bench>.<variant>/ and finished units are skipped (resume per unit);
#   2. with BASE_MODEL set, one GPU job for the base model into <OUT_ROOT>/<BASE_DIR>;
#   3. one CPU job deriving every CPU variant over the WHOLE of OUT_ROOT (run_derive.sh),
#      which also covers seeds evaluated on other days. Safe to run while GPU jobs are
#      running: a unit being recomputed has no summary, so its derivations wait.
#
# Usage (from anywhere; job scripts and outputs are relative to Eval_master):
#   # 1) print: write the job scripts, show the qsub lines, submit nothing
#   ./analysis/qsub_overview_day.sh /gpfs/.../SP-DPO-Base/outputs/2026-09-10
#
#   # 2) check: run every job script HERE in dry-run mode (plans, status matrices, no models)
#   BASE_MODEL=/gpfs/.../Qwen2.5-0.5B-Instruct ./analysis/qsub_overview_day.sh /gpfs/.../2026-09-10 check
#
#   # 3) submit (CPU resources for the derive job via QSUB_RES_derive)
#   QSUB_RES="-l select=1:ncpus=4:mem=32gb:ngpus=1 -l walltime=24:00:00 -A <project>" \
#   QSUB_RES_derive="-l select=1:ncpus=2:mem=16gb -l walltime=04:00:00 -A <project>" \
#       ./analysis/qsub_overview_day.sh /gpfs/.../outputs/2026-09-10 submit
#
#   # CPU variants only, no GPU at all (the DAY argument is not needed then)
#   DERIVE_ONLY=1 OUT_ROOT=outputs/eval/v4-2 ./analysis/qsub_overview_day.sh - submit
#
# Env: VARIANTS (all; or a comma list, analysis/variants.py), OUT_ROOT (outputs/eval/v4-2),
#      SUFFIX (""), BENCHES (faitheval,halueval,harness), EXTRA (Hydra overrides, e.g.
#      "model.dtype=bfloat16 harness.batch_size=8" -- the SAME as the originals were run with,
#      or every GPU unit reads as stale), RECOMPUTE_STALE (unset; 1 = re-run stale GPU units),
#      BASE_MODEL (unset = no base job), BASE_DIR (base/run_0), DERIVE (1; 0 = no derive job),
#      DERIVE_ONLY (unset), JOB_NAME (Overview), JOB_DIR (logs/qsub_jobs/overview/<DATE>),
#      QSUB_RES (required for GPU submits), QSUB_RES_base / QSUB_RES_derive (default QSUB_RES),
#      EVAL_MASTER_DIR (where the jobs cd to; default this checkout).
#
# NAMING: VARIANTS here are the overview's scoring variants. They are unrelated to
# scripts/qsub_train.sh's VARIANT (the materialized difficulty profile, e.g. ups0_k8); give a
# profile variant its own eval root or SUFFIX (e.g. SUFFIX=-ups0_k8) instead.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=qsub_lib.sh
source "${SCRIPT_DIR}/qsub_lib.sh"

DAY="${1:?usage: $0 <SP-DPO-Base/outputs/DATE | - with DERIVE_ONLY=1> [print|check|submit]}"
MODE="${2:-print}"
VARIANTS="${VARIANTS:-all}"
OUT_ROOT="${OUT_ROOT:-outputs/eval/v4-2}"
SUFFIX="${SUFFIX:-}"
BENCHES="${BENCHES:-faitheval,halueval,harness}"
EXTRA="${EXTRA:-}"
RECOMPUTE_STALE="${RECOMPUTE_STALE:-}"
BASE_MODEL="${BASE_MODEL:-}"
BASE_DIR="${BASE_DIR:-base/run_0}"
DERIVE="${DERIVE:-1}"
DERIVE_ONLY="${DERIVE_ONLY:-}"
JOB_NAME="${JOB_NAME:-Overview}"
QSUB_RES="${QSUB_RES:-}"
QSUB_RES_base="${QSUB_RES_base:-$QSUB_RES}"
QSUB_RES_derive="${QSUB_RES_derive:-$QSUB_RES}"

case "$MODE" in print|check|submit) ;; *) echo "error: mode must be print|check|submit" >&2; exit 2 ;; esac
if [[ -n "$DERIVE_ONLY" ]]; then
    DAY_NAME="derive"
else
    DAY="$(cd "$DAY" && pwd)"
    DAY_NAME="$(basename "$DAY")"
fi
JOB_DIR="${JOB_DIR:-logs/qsub_jobs/overview/$DAY_NAME}"
[[ "$MODE" != submit || -n "$DERIVE_ONLY" || -n "$QSUB_RES" ]] || {
    echo "error: set QSUB_RES for submit" >&2; exit 2; }
[[ "$MODE" != submit || "$DERIVE" == 0 || -n "$QSUB_RES_derive" ]] || {
    echo "error: set QSUB_RES_derive (or QSUB_RES) for submit" >&2; exit 2; }
[[ -z "$RECOMPUTE_STALE" || "$RECOMPUTE_STALE" == 0 ]] || EXTRA="${EXTRA:+$EXTRA }recompute_stale=true"

cd "$ROOT_DIR"
mkdir -p "$JOB_DIR"
EVAL_MASTER_REV="$(eval_master_rev "$OUT_ROOT")"

submit_line() {   # submit_line JOB RESOURCES [CHECK_ARGS...]
    local job=$1 res=$2; shift 2
    case "$MODE" in
        print)  echo "   qsub -N $JOB_NAME ${res:-<QSUB_RES>} $job" ;;
        check)  bash -l "$job" "$@" ;;
        submit) qsub -N "$JOB_NAME" $res "$job" ;;   # unquoted: split into flags
    esac
}

jobs=0
# --- 1. GPU jobs per ensemble: qsub_eval_day.sh does the writing and the guards -----------
if [[ -z "$DERIVE_ONLY" ]]; then
    VARIANTS="$VARIANTS" EXTRA="$EXTRA" OUT_ROOT="$OUT_ROOT" SUFFIX="$SUFFIX" BENCHES="$BENCHES" \
        JOB_NAME="$JOB_NAME" QSUB_RES="$QSUB_RES" JOB_DIR="$JOB_DIR" EVAL_MASTER_REV="$EVAL_MASTER_REV" \
        "$SCRIPT_DIR/qsub_eval_day.sh" "$DAY" "$MODE"
    for run in "$DAY"/*/; do
        if compgen -G "${run}seed_*" >/dev/null; then jobs=$((jobs + 1)); fi
    done
fi

# --- 2. base model (fixed point), same variants ------------------------------------------
if [[ -z "$DERIVE_ONLY" && -n "$BASE_MODEL" ]]; then
    split_overrides "${EXTRA:+$EXTRA }variants=[$VARIANTS] resume=true"
    job="$JOB_DIR/base.pbs"
    {
        hpc_header "CUDA/11.7.1"
        printf 'export EVAL_MASTER_REV=%q\n' "$EVAL_MASTER_REV"
        printf 'MODEL=%q\nOUT=%q\nBENCHES=%q\n' "$BASE_MODEL" "$OUT_ROOT/$BASE_DIR" "$BENCHES"
        printf 'EXTRA=(%s)\n' "${extra_words[*]@Q}"
        cat <<'EOF'
DRY=(); [[ "${1:-}" == --dry-run ]] && DRY=(dry_run=true)
echo "host=$(hostname) job=${PBS_JOBID:-local} base=$MODEL out=$OUT"
./run_all.sh model.id="$MODEL" output_dir="$OUT" hydra.run.dir="$OUT" \
    run="[$BENCHES]" "${EXTRA[@]}" "${DRY[@]}"
EOF
    } > "$job"
    echo "== base ($BASE_MODEL) -> $OUT_ROOT/$BASE_DIR  [variants=[$VARIANTS]]"
    submit_line "$job" "$QSUB_RES_base" --dry-run
    jobs=$((jobs + 1))
fi

# --- 3. CPU derivations over the whole root ------------------------------------------------
if [[ "$DERIVE" != 0 ]]; then
    job="$JOB_DIR/derive.pbs"
    {
        hpc_header
        # The status table goes next to the job scripts: the eval root only ever gains
        # dotted sibling dirs.
        printf 'ROOT=%q\nVARIANTS=%q\nSTATUS=%q\n' "$OUT_ROOT" "$VARIANTS" "$JOB_DIR/overview_status.tsv"
        cat <<'EOF'
DRY=(); [[ "${1:-}" == --dry-run ]] && DRY=(--dry-run)
echo "host=$(hostname) job=${PBS_JOBID:-local} derive root=$ROOT"
if [[ ! -d "$ROOT" ]]; then echo "no eval root $ROOT yet; nothing to derive"; exit 0; fi
./analysis/run_derive.sh --root "$ROOT" --variants "$VARIANTS" "${DRY[@]}"
python -m analysis.variants --root "$ROOT" --variants "$VARIANTS" > "$STATUS"
grep -A20 '^== totals' "$STATUS" || echo "(no units under $ROOT)"
EOF
    } > "$job"
    echo "== derive (CPU) over $OUT_ROOT  [variants=$VARIANTS]"
    submit_line "$job" "$QSUB_RES_derive" --dry-run
    jobs=$((jobs + 1))
fi

echo "overview: $jobs job(s) incl. ensembles, mode=$MODE, scripts in $JOB_DIR/"
