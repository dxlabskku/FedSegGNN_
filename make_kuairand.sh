#!/bin/bash
set -euo pipefail

# KuaiRand preprocessing wrapper (1K / 27K).
# Configure via environment variables, e.g.:
#   DOMAIN_PREFIX=Tablong MAX_SEQ_LEN=0 MIN_SEQ_LEN=1 MIN_ITEM_COUNT=1 MIN_USER_COUNT=1 ./make_kuairand.sh
#   DOMAINS=1,4,0,2 DOMAIN_PREFIX=Tabcustom ./make_kuairand.sh

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefer 27K if it exists, otherwise fall back to 1K.
DEFAULT_INPUT_DIR="$SCRIPT_DIR/data/KuaiRand-1K/data"
if [ -d "$SCRIPT_DIR/data/KuaiRand-27K/KuaiRand-27K/data" ]; then
  DEFAULT_INPUT_DIR="$SCRIPT_DIR/data/KuaiRand-27K/KuaiRand-27K/data"
elif [ -d "$SCRIPT_DIR/data/KuaiRand-27K/data" ]; then
  DEFAULT_INPUT_DIR="$SCRIPT_DIR/data/KuaiRand-27K/data"
fi

INPUT_DIR="${INPUT_DIR:-$DEFAULT_INPUT_DIR}"

DEFAULT_OUTPUT_DIR="$SCRIPT_DIR/data/kuairand"
if [[ "$INPUT_DIR" == *"KuaiRand-27K"* ]]; then
  DEFAULT_OUTPUT_DIR="$SCRIPT_DIR/data/kuairand27"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"

# Domain selection
SELECTION="${SELECTION:-top_interactions}"   # top_interactions | explicit
NUM_DOMAINS="${NUM_DOMAINS:-4}"
DOMAINS="${DOMAINS:-}"                       # comma-separated domain ids, e.g. "1,4,0,2"
DOMAIN_PREFIX="${DOMAIN_PREFIX:-abalation_20}"        # folder name prefix, e.g. Tablong -> Tablong1
DOMAIN_FIELD="${DOMAIN_FIELD:-tab}"          # tab | is_rand

# Positive signal
POSITIVE_COL="${POSITIVE_COL:-is_click}"
POSITIVE_VALUE="${POSITIVE_VALUE:-1}"

# Filtering / sequence lengths
MIN_SEQ_LEN="${MIN_SEQ_LEN:-10}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-20}"             # 0 or negative => no truncation
MIN_ITEM_COUNT="${MIN_ITEM_COUNT:-20}"
MIN_USER_COUNT="${MIN_USER_COUNT:-30}"

# Log selection
INCLUDE_RANDOM="${INCLUDE_RANDOM:-false}"    # true/false
LOG_FILES="${LOG_FILES:-}"                   # optional, comma-separated filenames
SAMPLE_RATIO="${SAMPLE_RATIO:-}"             # optional, e.g. 0.01

# Auto-detect log files when LOG_FILES is not provided.
if [ -z "$LOG_FILES" ]; then
  if [ -f "$INPUT_DIR/log_standard_4_08_to_4_21_27k_part1.csv" ]; then
    LOG_FILES="log_standard_4_08_to_4_21_27k_part1.csv,log_standard_4_08_to_4_21_27k_part2.csv,log_standard_4_22_to_5_08_27k_part1.csv,log_standard_4_22_to_5_08_27k_part2.csv"
    if [[ "${INCLUDE_RANDOM,,}" =~ ^(1|true|yes|on)$ ]] && [ -f "$INPUT_DIR/log_random_4_22_to_5_08_27k.csv" ]; then
      LOG_FILES="${LOG_FILES},log_random_4_22_to_5_08_27k.csv"
    fi
  fi
fi

ARGS=(
  --input_dir "$INPUT_DIR"
  --output_dir "$OUTPUT_DIR"
  --domain_prefix "$DOMAIN_PREFIX"
  --domain_field "$DOMAIN_FIELD"
  --positive_col "$POSITIVE_COL"
  --positive_value "$POSITIVE_VALUE"
  --min_seq_len "$MIN_SEQ_LEN"
  --max_seq_len "$MAX_SEQ_LEN"
  --min_item_count "$MIN_ITEM_COUNT"
  --min_user_count "$MIN_USER_COUNT"
)

if [ -n "$DOMAINS" ]; then
  ARGS+=(--selection explicit --domains "$DOMAINS")
else
  ARGS+=(--selection "$SELECTION" --num_domains "$NUM_DOMAINS")
fi

if [ -n "$LOG_FILES" ]; then
  ARGS+=(--log_files "$LOG_FILES")
fi

if [ -n "$SAMPLE_RATIO" ]; then
  ARGS+=(--sample_ratio "$SAMPLE_RATIO")
fi

case "${INCLUDE_RANDOM,,}" in
  1|true|yes|on) ARGS+=(--include_random) ;;
esac

python "$SCRIPT_DIR/data/kuairand/preprocess_kuairand.py" "${ARGS[@]}"
