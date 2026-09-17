#!/usr/bin/env bash
# OPTIONAL all-variants overview (descriptive, not pre-registered): the overview counterpart
# of qsub_comparisons.sh. One CPU PBS job that brings the CPU variants of OUT_ROOT up to date
# (run_derive.sh) and then writes, per arm set, the overview figures and table
# (run_analysis.sh --variants-overview --no-compare): every arm under every variant, no tests.
# The §5.7 comparisons and the §5.8 table stay with qsub_comparisons.sh; this job writes to
# its own ANALYSIS_ROOT and never a comparisons.json.
#
# Arms are found exactly as in qsub_comparisons.sh: <OUT_ROOT>/<METHOD>_ensemble<SUFFIX>/ and
# base at <OUT_ROOT>/<BASE_DIR>. An arm set whose arms are missing is left out and reported.
#
# Arm sets (SETS, default "claims families"):
#   claims    one overview per §5.7 pair (claimA_*, sign_*, claimB_*): ARM against REF, i.e.
#             REF is the dashed line and Δ/SD is ARM relative to REF under every variant
#   families  base + every evaluated arm of one objective (dpo family, orpo family)
#   all       base + every evaluated arm (more than 8 arms fold to grey)
#
# Usage (from anywhere; job script and outputs are relative to Eval_master):
#   OUT_ROOT=outputs/eval/v4-2 SUFFIX=-2epochs ./analysis/qsub_overview_compare.sh          # print
#   OUT_ROOT=outputs/eval/v4-2 SUFFIX=-2epochs ./analysis/qsub_overview_compare.sh check    # + commands
#   QSUB_RES="-A DialSys -l select=1:ncpus=2:mem=16gb -l walltime=04:00:00" \
#       OUT_ROOT=outputs/eval/v4-2 SUFFIX=-2epochs ./analysis/qsub_overview_compare.sh submit
#
# Env: OUT_ROOT (outputs/eval/v4-2), SUFFIX (""), BENCHES (faitheval,halueval,harness),
#      BASE_DIR (base/run_0), SETS (claims families), VARIANTS (all; for the derive step),
#      DERIVE (1; 0 = skip the derive step), PLOT (1; 0 = tables only),
#      ANALYSIS_ROOT (outputs/analysis/<basename of OUT_ROOT>_overview),
#      JOB_NAME (Overview), QSUB_RES (required for submit), JOB_DIR (logs/qsub_jobs/overview),
#      EVAL_MASTER_DIR (where the job cd's to; default this checkout).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=qsub_lib.sh
source "${SCRIPT_DIR}/qsub_lib.sh"

MODE="${1:-print}"
OUT_ROOT="${OUT_ROOT:-outputs/eval/v4-2}"
SUFFIX="${SUFFIX:-}"
BENCHES="${BENCHES:-faitheval,halueval,harness}"
BASE_DIR="${BASE_DIR:-base/run_0}"
SETS="${SETS:-claims families}"
VARIANTS="${VARIANTS:-all}"
DERIVE="${DERIVE:-1}"
PLOT="${PLOT:-1}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-outputs/analysis/$(basename "$OUT_ROOT")_overview}"
JOB_NAME="${JOB_NAME:-Overview}"
QSUB_RES="${QSUB_RES:-}"
JOB_DIR="${JOB_DIR:-logs/qsub_jobs/overview}"

case "$MODE" in print|check|submit) ;; *) echo "error: mode must be print|check|submit" >&2; exit 2 ;; esac
[[ "$MODE" != submit || -n "$QSUB_RES" ]] || { echo "error: set QSUB_RES for submit" >&2; exit 2; }
for s in $SETS; do
    case "$s" in claims|families|all) ;; *) echo "error: unknown arm set '$s' (claims|families|all)" >&2; exit 2 ;; esac
done

cd "$ROOT_DIR"
[[ -d "$OUT_ROOT" ]] || { echo "error: no eval root $OUT_ROOT" >&2; exit 1; }
mkdir -p "$JOB_DIR"

arm_dir() { if [[ "$1" == base ]]; then echo "$OUT_ROOT/$BASE_DIR"; else echo "$OUT_ROOT/${1}_ensemble$SUFFIX"; fi; }

declare -A MISSING=()
has() {        # has ARM...: true if all were evaluated; the missing ones are noted
    local a ok=0
    for a; do [[ -d "$(arm_dir "$a")" ]] || { MISSING[$a]=1; ok=1; }; done
    return $ok
}

CMDS=() SKIPPED=()
add() { CMDS+=("run $(printf '%q ' "$@")"); }   # add OUT_SUBDIR ARGS...: one overview call
wanted() { [[ " $SETS " == *" $1 "* ]]; }

# --- claims: the §5.7 pairs, ARM under every variant relative to REF ---------------------
if wanted claims; then
    PAIRS="
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
        add "$NAME" --arm "$ARM=$(arm_dir "$ARM")" --arm "$REF=$(arm_dir "$REF")" --reference "$REF"
    done <<< "$PAIRS"
fi

# --- families / all: base plus every evaluated arm -----------------------------------------
table() {      # table NAME ARM...
    local name=$1 a n=0; shift
    has base || { SKIPPED+=("$name"); return 0; }
    local S=(--arm "base=$(arm_dir base)")
    for a; do if has "$a"; then S+=(--arm "$a=$(arm_dir "$a")"); n=$((n + 1)); fi; done
    (( n > 0 )) || { SKIPPED+=("$name"); return 0; }
    add "$name" --reference base "${S[@]}"
}
DPO_FAMILY="sft dpo dpo_cosine off_sp_dpo_asc off_sp_dpo_shuf off_sp_dpo_desc off_sp_dpo_legacy"
ORPO_FAMILY="sft orpo orpo_cosine off_sp_orpo_asc off_sp_orpo_shuf off_sp_orpo_desc"
if wanted families; then
    # shellcheck disable=SC2086
    table family_dpo $DPO_FAMILY
    # shellcheck disable=SC2086
    table family_orpo $ORPO_FAMILY
fi
if wanted all; then
    # shellcheck disable=SC2086
    table all_arms $(printf '%s\n' $DPO_FAMILY $ORPO_FAMILY | awk '!seen[$0]++')
fi

# --- Job script ------------------------------------------------------------------------------
job="$JOB_DIR/$(basename "$OUT_ROOT")$SUFFIX.pbs"
{
    hpc_header
    printf 'ROOT=%q\nANALYSIS_ROOT=%q\nBENCHES=%q\nVARIANTS=%q\nDERIVE=%q\nPLOT=%q\n' \
        "$OUT_ROOT" "$ANALYSIS_ROOT" "$BENCHES" "$VARIANTS" "$DERIVE" "$PLOT"
    cat <<'EOF'
DRY=0; [[ "${1:-}" == --dry-run ]] && DRY=1
echo "host=$(hostname) job=${PBS_JOBID:-local} root=$ROOT out=$ANALYSIS_ROOT"
mkdir -p "$ANALYSIS_ROOT"
if [[ "$DERIVE" != 0 ]]; then
    if (( DRY )); then ./analysis/run_derive.sh --root "$ROOT" --variants "$VARIANTS" --dry-run | tail -n 6
    else ./analysis/run_derive.sh --root "$ROOT" --variants "$VARIANTS" || echo "!! derive reported failures"
    fi
fi
python -m analysis.variants --root "$ROOT" --variants "$VARIANTS" > "$ANALYSIS_ROOT/status.tsv"

FAILED=()
run() {   # run OUT_SUBDIR ARGS...: one descriptive overview
    local out=$1; shift
    local extra=(--variants-overview --no-compare --benchmarks "$BENCHES")
    [[ "$PLOT" != 0 ]] || extra+=(--no-plot)
    echo "== $out"
    if (( DRY )); then echo "   ./analysis/run_analysis.sh $* ${extra[*]} --out $ANALYSIS_ROOT/$out"; return; fi
    ./analysis/run_analysis.sh "$@" "${extra[@]}" --out "$ANALYSIS_ROOT/$out" \
        || { echo "!! $out failed"; FAILED+=("$out"); }
}

EOF
    printf '%s\n' "${CMDS[@]}"
    cat <<'EOF'

if (( ${#FAILED[@]} )); then echo "!! failed: ${FAILED[*]}"; exit 1; fi
echo "done: $ANALYSIS_ROOT (overview - descriptive, not pre-registered)"
EOF
} > "$job"

# --- Report -----------------------------------------------------------------------------------
echo "== $OUT_ROOT -> $ANALYSIS_ROOT  [sets: $SETS]"
for a in "${!MISSING[@]}"; do echo "   missing arm: $a (expected $(arm_dir "$a"))"; done | sort
(( ${#SKIPPED[@]} == 0 )) || echo "   left out:    ${SKIPPED[*]}"
echo "   ${#CMDS[@]} overview call(s) in $job"
case "$MODE" in
    print)  echo "   qsub -N $JOB_NAME ${QSUB_RES:-<QSUB_RES>} $job" ;;
    check)  bash -n "$job" && bash -l "$job" --dry-run ;;
    submit) qsub -N "$JOB_NAME" $QSUB_RES "$job" ;;   # QSUB_RES unquoted: split into flags
esac
