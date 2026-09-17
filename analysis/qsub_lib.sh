# Shared helpers for the analysis/qsub_*.sh job writers. Source it; it does nothing on its own.

# split_overrides STRING: fill the array `extra_words` with the Hydra overrides in STRING,
# splitting on whitespace OUTSIDE brackets/quotes only. A plain `read -ra` cut
# `faitheval.tasks=[a, b]` into `faitheval.tasks=[a,` + `b]`; Hydra rejected those, so every
# seed failed at once, leaving only run_metadata.json behind.
split_overrides() {
    local s="$1" word="" c i depth=0 quote=""
    extra_words=()
    for ((i = 0; i < ${#s}; i++)); do
        c="${s:i:1}"
        if [[ -n "$quote" ]]; then
            word+="$c"; [[ "$c" == "$quote" ]] && quote=""
            continue
        fi
        case "$c" in
            \'|\")     quote="$c"; word+="$c" ;;
            \[|\{|\()  depth=$((depth + 1)); word+="$c" ;;
            \]|\}|\))  depth=$((depth - 1)); word+="$c" ;;
            ' '|$'\t') if (( depth > 0 )); then word+="$c"
                       elif [[ -n "$word" ]]; then extra_words+=("$word"); word=""; fi ;;
            *)         word+="$c" ;;
        esac
    done
    [[ -z "$word" ]] || extra_words+=("$word")
    if (( depth != 0 )) || [[ -n "$quote" ]]; then
        echo "error: unbalanced brackets/quotes in EXTRA: $s" >&2; exit 2
    fi
}

# hpc_header [MODULES...]: the environment block every Eval_master PBS job starts with.
# EVAL_MASTER_DIR (default: this checkout) is where the job cd's to.
hpc_header() {
    printf '#!/bin/bash -l\nset -eo pipefail\n'
    printf 'module load uv/0.10.2  gcc/13.2.0 Openssl/1.1.1t Tcl/8.6.11 Tk/8.6.13 Python/3.12.3%s\n' \
        "${1:+ $1}"
    printf 'cd %q\n' "${EVAL_MASTER_DIR:-$ROOT_DIR}"
    cat <<'EOF'
if [[ -f .env ]]; then dos2unix -q .env; set -a; source .env; set +a; fi
source .venv/bin/activate
export USER_NAME="${USER_NAME:-ladan102}"
export PYTHONUNBUFFERED=1   # PBS stdout is a file: keep progress lines in order
EOF
}

# eval_master_rev OUT_ROOT: the code version for launches.jsonl — $EVAL_MASTER_REV, else the
# `eval_master` line of OUT_ROOT/EVAL_BLOCK.txt, else git HEAD, else empty.
eval_master_rev() {
    local rev="${EVAL_MASTER_REV:-}"
    if [[ -z "$rev" && -f "$1/EVAL_BLOCK.txt" ]]; then
        rev="$(awk '$1 == "eval_master" {print $2; exit}' "$1/EVAL_BLOCK.txt")"
    fi
    [[ -n "$rev" ]] || rev="$(git rev-parse HEAD 2>/dev/null || true)"
    printf '%s' "$rev"
}
