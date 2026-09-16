#!/usr/bin/env bash
# Follow-up to qsub_eval_day.sh: one CPU PBS job running the comparisons of
# SP-DPO-Base/EXPERIMENT_PROCEDURE.md §5.7 over the eval dirs qsub_eval_day.sh wrote.
#
# qsub_eval_day.sh writes <OUT_ROOT>/<METHOD>_ensemble<SUFFIX>/seed_*/, so each procedure arm
# (§1 METHOD, e.g. off_sp_dpo_asc) is looked for there, and base at <OUT_ROOT>/<BASE_DIR>.
# A comparison whose arms are not there is left out of the job and reported.
#
# Usage (from anywhere; job script and outputs are relative to Eval_master):
#   # 1) print: write the job script, list missing arms, show the qsub line
#   OUT_ROOT=outputs/eval/v4-2 SUFFIX=-2epochs ./analysis/qsub_comparisons.sh
#
#   # 2) check: also show every command the job will run
#   OUT_ROOT=outputs/eval/v4-2 SUFFIX=-2epochs ./analysis/qsub_comparisons.sh check
#
#   # 3) submit for real
#   QSUB_RES="-A DialSys -l select=1:ncpus=2:mem=16gb -l walltime=02:00:00" \
#       OUT_ROOT=outputs/eval/v4-2 SUFFIX=-2epochs ./analysis/qsub_comparisons.sh submit
#
# Env: OUT_ROOT (outputs/eval/v4-2), SUFFIX (""), BENCHES (faitheval,halueval,harness),
#      BASE_DIR (base/run_0), ANALYSIS_ROOT (outputs/analysis/<basename of OUT_ROOT>),
#      JOB_NAME (DevSession), QSUB_RES (required for submit), JOB_DIR (logs/qsub_jobs/compare).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="${1:-print}"
OUT_ROOT="${OUT_ROOT:-outputs/eval/v4-2}"
SUFFIX="${SUFFIX:-}"
BENCHES="${BENCHES:-faitheval,halueval,harness}"
BASE_DIR="${BASE_DIR:-base/run_0}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-outputs/analysis/$(basename "$OUT_ROOT")}"
JOB_NAME="${JOB_NAME:-DevSession}"
QSUB_RES="${QSUB_RES:-}"
JOB_DIR="${JOB_DIR:-logs/qsub_jobs/compare}"

case "$MODE" in print|check|submit) ;; *) echo "error: mode must be print|check|submit" >&2; exit 2 ;; esac
[[ "$MODE" != submit || -n "$QSUB_RES" ]] || { echo "error: set QSUB_RES for submit" >&2; exit 2; }

cd "$ROOT_DIR"
[[ -d "$OUT_ROOT" ]] || { echo "error: no eval root $OUT_ROOT" >&2; exit 1; }
mkdir -p "$JOB_DIR"

arm_dir() { if [[ "$1" == base ]]; then echo "$OUT_ROOT/$BASE_DIR"; else echo "$OUT_ROOT/${1}_ensemble$SUFFIX"; fi; }
seeds_of() { (cd "$1" 2>/dev/null && ls -d seed_* 2>/dev/null | sort); }

PRIMARY="$ANALYSIS_ROOT/primary_confirmatory.yaml"   # written by the job (§5.7)

declare -A MISSING=()
has() {        # has ARM...: true if all were evaluated; the missing ones are noted
    local a ok=0
    for a; do [[ -d "$(arm_dir "$a")" ]] || { MISSING[$a]=1; ok=1; }; done
    return $ok
}

CMDS=() SKIPPED=()
add() { CMDS+=("run $(printf '%q ' "$@")"); }   # add OUT_SUBDIR ARGS...: one analysis call

# --- Claim A, direction, Claim B: deciding mc2 test + all-benchmark context (§5.7) -------
COMPARISONS="
claimA_dpo   off_sp_dpo_asc   off_sp_dpo_shuf
claimA_orpo  off_sp_orpo_asc  off_sp_orpo_shuf
sign_dpo     off_sp_dpo_asc   off_sp_dpo_desc
sign_orpo    off_sp_orpo_asc  off_sp_orpo_desc
claimB_dpo   off_sp_dpo_asc   dpo
claimB_orpo  off_sp_orpo_asc  orpo
"
while read -r NAME ARM REF; do
    [[ -n "$NAME" ]] || continue
    has "$ARM" "$REF" || { SKIPPED+=("$NAME"); continue; }
    S=(--arm "$ARM=$(arm_dir "$ARM")" --arm "$REF=$(arm_dir "$REF")" --reference "$REF")
    add "$NAME/mc2" "${S[@]}" --benchmarks harness --primary-map "$PRIMARY"
    add "$NAME/all" "${S[@]}" --benchmarks "$BENCHES"
done <<< "$COMPARISONS"

# --- Claim C and LR-schedule sensitivity: descriptive, all only ---------------------------
# SHORT may have only the seed prefix (§4d); the other arm is cut to SHORT's seeds by a
# folder of links, as in §5.7.
DESCRIPTIVE="
claimC_dpo       off_sp_dpo_asc   off_sp_dpo_legacy  off_sp_dpo_legacy
lrsens_dpo       dpo_cosine       dpo                dpo_cosine
lrsens_orpo      orpo_cosine      orpo               orpo_cosine
claimB_cos_dpo   off_sp_dpo_asc   dpo_cosine         dpo_cosine
claimB_cos_orpo  off_sp_orpo_asc  orpo_cosine        orpo_cosine
"
while read -r NAME ARM REF SHORT; do
    [[ -n "$NAME" ]] || continue
    has "$ARM" "$REF" || { SKIPPED+=("$NAME"); continue; }
    if [[ "$ARM" == "$SHORT" ]]; then OTHER=$REF; else OTHER=$ARM; fi
    OTHER_DIR=$(arm_dir "$OTHER")
    if [[ "$(seeds_of "$OTHER_DIR")" != "$(seeds_of "$(arm_dir "$SHORT")")" ]]; then
        SUB="$ANALYSIS_ROOT/_subsets/${OTHER}_as_${SHORT}"
        CMDS+=("mkdir -p $(printf '%q' "$SUB")")
        for s in $(seeds_of "$(arm_dir "$SHORT")"); do
            CMDS+=("ln -sfn $(printf '%q %q' "$(realpath -m "$OTHER_DIR/$s")" "$SUB/$s")")
        done
        OTHER_DIR=$SUB
    fi
    add "$NAME/all" --arm "$SHORT=$(arm_dir "$SHORT")" --arm "$OTHER=$OTHER_DIR" \
        --reference "$REF" --benchmarks "$BENCHES"
done <<< "$DESCRIPTIVE"

# --- Reference tables, one per objective, relative to base --------------------------------
table() {      # table NAME ARM...: base plus every evaluated ARM
    local name=$1 a; shift
    has base || { SKIPPED+=("$name"); return 0; }
    local S=(--arm "base=$(arm_dir base)")
    for a; do if has "$a"; then S+=(--arm "$a=$(arm_dir "$a")"); fi; done
    add "$name" --reference base --benchmarks "$BENCHES" "${S[@]}"
}
table table_dpo_family  sft dpo  dpo_cosine  off_sp_dpo_asc  off_sp_dpo_shuf  off_sp_dpo_desc off_sp_dpo_legacy
table table_orpo_family sft orpo orpo_cosine off_sp_orpo_asc off_sp_orpo_shuf off_sp_orpo_desc

# --- Job script ---------------------------------------------------------------------------
job="$JOB_DIR/$(basename "$OUT_ROOT")$SUFFIX.pbs"
{
    printf '#!/bin/bash -l\n'
    printf 'ANALYSIS_ROOT=%q\nPRIMARY=%q\n' "$ANALYSIS_ROOT" "$PRIMARY"
    cat <<'EOF'
set -o pipefail
module load uv/0.10.2  gcc/13.2.0 Openssl/1.1.1t Tcl/8.6.11 Tk/8.6.13 Python/3.12.3
cd /gpfs/project/ladan102/git-source/SP_DPO/Eval_master/

dos2unix .env
set -a; source .env; set +a
source .venv/bin/activate
export USER_NAME=ladan102
export PYTHONUNBUFFERED=1

echo "host=$(hostname) job=${PBS_JOBID:-local} out=$ANALYSIS_ROOT"
mkdir -p "$ANALYSIS_ROOT"
[[ -f "$PRIMARY" ]] || cat > "$PRIMARY" <<'YAML'
# Pre-registered in §5.1: the one deciding metric. Never edit after results exist.
harness: ["truthfulqa_mc2:acc"]
YAML

FAILED=()
run() {   # run OUT_SUBDIR ARGS...
    local out=$1; shift
    echo "== $out"
    ./analysis/run_analysis.sh "$@" --out "$ANALYSIS_ROOT/$out" || { echo "!! $out failed"; FAILED+=("$out"); }
}

EOF
    printf '%s\n' "${CMDS[@]}"
    cat <<'EOF'

if (( ${#FAILED[@]} )); then echo "!! failed: ${FAILED[*]}"; exit 1; fi
echo "done: $ANALYSIS_ROOT"
EOF
} > "$job"

# --- Report -------------------------------------------------------------------------------
echo "== $OUT_ROOT -> $ANALYSIS_ROOT"
for a in "${!MISSING[@]}"; do echo "   missing arm: $a (expected $(arm_dir "$a"))"; done | sort
(( ${#SKIPPED[@]} == 0 )) || echo "   left out:    ${SKIPPED[*]}"
echo "   $(printf '%s\n' "${CMDS[@]}" | grep -c '^run ') analysis call(s) in $job"
case "$MODE" in
    print)  echo "   qsub -N $JOB_NAME ${QSUB_RES:-<QSUB_RES>} $job" ;;
    check)  bash -n "$job" && grep -E '^(run|mkdir|ln) ' "$job" ;;
    submit) qsub -N "$JOB_NAME" $QSUB_RES "$job" ;;   # QSUB_RES unquoted: split into flags
esac
