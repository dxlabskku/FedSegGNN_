#!/bin/bash
# Run multiple SegFedGNN trials and report mean/std metrics.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$SCRIPT_DIR/run.sh}"

RUNS=${RUNS:-2}
# If not provided, pick a different base seed per experiment run.
BASE_SEED=${BASE_SEED:-$(( $(date +%s) + RANDOM ))}
LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/log}"

# ============ Generate experiment info tags ============
DATA_DIST=${DATA_DIST:-custom}
# Resolve DOMAIN_SCENARIO using run.sh so log folders follow run.sh defaults.
resolve_domain_scenario() {
    if [ -n "${DOMAIN_SCENARIO:-}" ]; then
        echo "$DOMAIN_SCENARIO"
        return 0
    fi
    local resolved
    resolved=$(PRINT_DOMAIN_SCENARIO=1 DATA_DIST="$DATA_DIST" bash "$TRAIN_SCRIPT" 2>/dev/null || true)
    echo "$resolved"
}

resolve_method() {
    if [ -n "${METHOD:-}" ]; then
        echo "$METHOD"
        return 0
    fi
    # Parse METHOD default from TRAIN_SCRIPT (e.g., METHOD=${METHOD:-FedDuoRec})
    local resolved
    resolved=$(sed -n 's/^[[:space:]]*METHOD=\${METHOD:-\([^}]*\)}.*/\1/p' "$TRAIN_SCRIPT" | head -n 1)
    echo "$resolved"
}

_DOMAIN_SCENARIO="$(resolve_domain_scenario)"
if [ -z "$_DOMAIN_SCENARIO" ]; then
    echo "[error] DOMAIN_SCENARIO is empty; set DOMAIN_SCENARIO or update run.sh default." >&2
    exit 1
fi

TARGET_METHOD="$(resolve_method)"
if [ -z "$TARGET_METHOD" ]; then
    echo "[error] Failed to resolve METHOD from env or $TRAIN_SCRIPT." >&2
    exit 1
fi

# Default log prefix should follow the active method.
LOG_PREFIXES="${LOG_PREFIXES:-$TARGET_METHOD}"

# Generate tags (concise format)
_DIST_TAG=$(echo "$DATA_DIST" | tr ' ' '-' | tr '[:upper:]' '[:lower:]')
_DOM_TAG=$(echo "$_DOMAIN_SCENARIO" | tr ' ' '-' | tr '/' '-' | tr '[:upper:]' '[:lower:]' | cut -c1-30)
DATASET_TAG="${DATASET_TAG:-}"
if [ -z "$DATASET_TAG" ]; then
    _FIRST_DOMAIN="${_DOMAIN_SCENARIO%% *}"
    if [[ "$_FIRST_DOMAIN" == */* ]]; then
        DATASET_TAG="${_FIRST_DOMAIN%%/*}"
    else
        DATASET_TAG="$_FIRST_DOMAIN"
    fi
    DATASET_TAG=$(echo "$DATASET_TAG" | tr '[:upper:]' '[:lower:]')
fi
_TIMESTAMP=$(date +%y%m%d_%H%M%S)

# Generate EXP_ID: DIST_DOM_TIMESTAMP
EXP_ID="${EXP_ID:-${_DIST_TAG}_${_DOM_TAG}_${_TIMESTAMP}}"
# ================================================================

# Log structure: LOG_ROOT/DATASET/multi_runs/exp_${EXP_ID}
EXP_DIR="$LOG_ROOT/${DATASET_TAG}/multi_runs/exp_${EXP_ID}"
RUNS_FILE="$EXP_DIR/runs.tsv"
PARAM_LOG="$EXP_DIR/parameters.log"
SUMMARY_FILE="${SUMMARY_FILE:-$EXP_DIR/summary.txt}"
mkdir -p "$EXP_DIR"

METRICS_FILE="$EXP_DIR/metrics.tsv"
DOM_METRICS_FILE="$EXP_DIR/metrics_domains.tsv"
: > "$METRICS_FILE"
: > "$DOM_METRICS_FILE"
: > "$RUNS_FILE"
: > "$SUMMARY_FILE"

{
    echo "timestamp: $(date -Is)"
    echo "experiment_id: $EXP_ID"
    echo "train_script: $TRAIN_SCRIPT"
    echo "runs: $RUNS"
    echo "base_seed: $BASE_SEED"
    echo "log_root: $LOG_ROOT"
    echo "target_method: $TARGET_METHOD"
    echo "log_prefixes: $LOG_PREFIXES"
    echo "exp_dir: $EXP_DIR"
    echo "cmd_args: $*"
    echo
    echo "# Config captured from first log will be appended below."
} > "$PARAM_LOG"

echo "[info] experiment id=$EXP_ID (dir=$EXP_DIR)"

for ((i = 0; i < RUNS; i++)); do
    seed=$((BASE_SEED + i))
    label="multi${i}"
    echo "=== Run $((i + 1))/$RUNS (seed=$seed, label=$label) ==="
    SEED=$seed bash "$TRAIN_SCRIPT" "$@" | tee "$EXP_DIR/run_${i}.out"

    LOG_SEARCH_ROOT="${LOG_SEARCH_ROOT:-$LOG_ROOT/$DATASET_TAG}"
    python - "$LOG_SEARCH_ROOT" "$label" "$METRICS_FILE" "$DOM_METRICS_FILE" "$i" "$LOG_PREFIXES" "$RUNS_FILE" "$PARAM_LOG" "$seed" "$TARGET_METHOD" <<'PY'
import sys
import pathlib
import re

log_root = pathlib.Path(sys.argv[1])
label = sys.argv[2]
metrics_path = pathlib.Path(sys.argv[3])
domain_metrics_path = pathlib.Path(sys.argv[4])
run_idx = int(sys.argv[5])
log_prefixes = [p for p in sys.argv[6].split(",") if p]
runs_path = pathlib.Path(sys.argv[7])
param_log_path = pathlib.Path(sys.argv[8])
seed = int(sys.argv[9])
expected_method = sys.argv[10].strip()

def find_log():
    def most_recent(glob):
        matches = list(log_root.rglob(glob))
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matches

    def select_best(candidates):
        # Prefer logs whose config block matches expected method/seed.
        for cand in candidates:
            try:
                cand_lines = cand.read_text(errors="ignore").splitlines()
            except Exception:
                continue

            parsed_method = None
            parsed_seed = None
            for ln in cand_lines:
                if parsed_method is None:
                    m = re.search(r"^\s*method\s*:\s*(\S+)", ln)
                    if m:
                        parsed_method = m.group(1).strip()
                if parsed_seed is None:
                    m = re.search(r"^\s*seed\s*:\s*(\d+)", ln)
                    if m:
                        parsed_seed = int(m.group(1))
                if parsed_method is not None and parsed_seed is not None:
                    break

            if expected_method and parsed_method and parsed_method != expected_method:
                continue
            if parsed_seed is not None and parsed_seed != seed:
                continue
            return cand, cand_lines
        return None, None

    candidates = []
    seed_token = f"seed{seed}"

    # 1) Most strict: expected method + label + seed
    if expected_method:
        candidates.extend(most_recent(f"{expected_method}_*{label}*{seed_token}*.log"))

    # 2) Prefix list + label + seed (user-overridden prefixes still supported)
    if not candidates:
        for prefix in log_prefixes:
            candidates.extend(most_recent(f"{prefix}_*{label}*{seed_token}*.log"))

    # 3) Label + seed regardless of prefix
    if not candidates:
        candidates.extend(most_recent(f"*{label}*{seed_token}*.log"))

    # 4) Relaxed fallback with label only
    if not candidates:
        if expected_method:
            candidates.extend(most_recent(f"{expected_method}_*{label}*.log"))
        for prefix in log_prefixes:
            candidates.extend(most_recent(f"{prefix}_*{label}*.log"))
    if not candidates:
        candidates.extend(most_recent(f"*{label}*.log"))
    if not candidates:
        candidates.extend(most_recent("*.log"))

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return select_best(candidates)

log_path, lines = find_log()
if log_path is None or lines is None:
    print(f"[warn] no log file found for run {run_idx}", file=sys.stderr)
    sys.exit(0)

last_test_idx = None
for idx, line in enumerate(lines):
    if "Test:" in line:
        last_test_idx = idx

if last_test_idx is None:
    print(f"[warn] no Test block in {log_path}", file=sys.stderr)
    sys.exit(0)

mrr = hr1 = hr5 = hr10 = hr20 = hr50 = ndcg5 = ndcg10 = ndcg20 = None
for line in lines[last_test_idx + 1:]:
    m = re.search(r"MRR:\s*([0-9.]+)", line)
    if m:
        mrr = float(m.group(1))
        continue
    m = re.search(r"HR @1\|5\|10\|20\|50:\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", line)
    if m:
        hr1, hr5, hr10, hr20, hr50 = map(float, m.groups())
        continue
    m = re.search(r"NDCG @5\|10\|20:\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", line)
    if m:
        ndcg5, ndcg10, ndcg20 = map(float, m.groups())
        continue
    if all(v is not None for v in (mrr, hr1, hr5, hr10, hr20, hr50, ndcg5, ndcg10, ndcg20)):
        break

if any(v is None for v in (mrr, hr1, hr5, hr10, hr20, hr50, ndcg5, ndcg10, ndcg20)):
    missing = [n for n, v in [('MRR', mrr), ('HR@1', hr1), ('HR@5', hr5), ('HR@10', hr10), ('HR@20', hr20), ('HR@50', hr50), ('NDCG@5', ndcg5), ('NDCG@10', ndcg10), ('NDCG@20', ndcg20)] if v is None]
    print(f"[warn] metrics missing in {log_path}: {', '.join(missing)}", file=sys.stderr)
    sys.exit(0)

with metrics_path.open("a") as f:
    f.write(f"{run_idx}\t{log_path}\t{mrr}\t{hr1}\t{hr5}\t{hr10}\t{hr20}\t{hr50}\t{ndcg5}\t{ndcg10}\t{ndcg20}\n")

with runs_path.open("a") as f:
    f.write(f"{run_idx}\t{seed}\t{label}\t{log_path}\n")

# Capture the config block from the first available log (once)
def append_config_once():
    if param_log_path.exists() and "Captured config from log" in param_log_path.read_text():
        return
    config_lines = []
    capture = False
    for ln in lines:
        if "Running with the following configs" in ln:
            capture = True
            config_lines.append(ln)
            continue
        if capture:
            if ln[:1].isspace():
                config_lines.append(ln)
                continue
            # Stop at the first non-indented line after the config block
            break
    if config_lines:
        with param_log_path.open("a") as f:
            f.write("\nCaptured config from log:\n")
            f.write("\n".join(config_lines))
            f.write(f"\n\nlog_file: {log_path}\n")

append_config_once()

domain_block = "\n".join(lines[last_test_idx + 1:])
domain_rows = []
pat = re.compile(
    r"\|\s*([^\|]+?)\s+MRR:\s*([0-9.]+).*?AUC:\s*([0-9.]+).*?HR @1:\s*([0-9.]+).*?HR @5:\s*([0-9.]+).*?HR @10:\s*([0-9.]+).*?HR @20:\s*([0-9.]+).*?HR @50:\s*([0-9.]+).*?NDCG @5:\s*([0-9.]+).*?NDCG @10:\s*([0-9.]+).*?NDCG @20:\s*([0-9.]+)",
    re.DOTALL,
)
for m in pat.finditer(domain_block):
    dom, dmrr, dauc, dhr1, dhr5, dhr10, dhr20, dhr50, dndcg5, dndcg10, dndcg20 = m.groups()
    domain_rows.append(
        (dom.strip(), float(dmrr), float(dauc), float(dhr1), float(dhr5),
         float(dhr10), float(dhr20), float(dhr50), float(dndcg5), float(dndcg10), float(dndcg20))
    )

if domain_rows:
    with domain_metrics_path.open("a") as f:
        for dom, dmrr, dauc, dhr1, dhr5, dhr10, dhr20, dhr50, dndcg5, dndcg10, dndcg20 in domain_rows:
            f.write(
                f"{run_idx}\t{log_path}\t{dom}\t{dmrr}\t{dauc}\t{dhr1}\t{dhr5}\t{dhr10}\t{dhr20}\t{dhr50}\t{dndcg5}\t{dndcg10}\t{dndcg20}\n"
            )

print(f"[info] captured metrics from {log_path} (domains={len(domain_rows)})")
PY
done

{
echo "=== Averages over $RUNS runs ==="
python - "$METRICS_FILE" "$DOM_METRICS_FILE" <<'PY'
import sys
import pathlib
import numpy as np

metrics_path = pathlib.Path(sys.argv[1])
if not metrics_path.exists() or metrics_path.stat().st_size == 0:
    print("No metrics captured.")
    sys.exit(0)

rows = []
for line in metrics_path.read_text().splitlines():
    parts = line.split("\t")
    if len(parts) < 11:
        continue
    rows.append([float(x) for x in parts[2:]])

if not rows:
    print("No metrics captured.")
    sys.exit(0)

arr = np.array(rows)
mean = arr.mean(axis=0)
std = arr.std(axis=0)

names = ["MRR", "HR@1", "HR@5", "HR@10", "HR@20", "HR@50", "NDCG@5", "NDCG@10", "NDCG@20"]
print("metric\tmean\tstd")
for name, m, s in zip(names, mean, std):
    print(f"{name}\t{m:.6f}\t{s:.6f}")

# Per-domain aggregation
dom_metrics_path = pathlib.Path(sys.argv[2])
dom_names = ["MRR", "AUC", "HR@1", "HR@5", "HR@10", "HR@20", "HR@50", "NDCG@5", "NDCG@10", "NDCG@20"]
if dom_metrics_path.exists() and dom_metrics_path.stat().st_size > 0:
    domain_rows = {}
    for line in dom_metrics_path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 13:
            continue
        run_idx_p, log_p, dom, dmrr, dauc, dhr1, dhr5, dhr10, dhr20, dhr50, dndcg5, dndcg10, dndcg20 = parts[:13]
        domain_rows.setdefault(dom, []).append(
            [float(dmrr), float(dauc), float(dhr1), float(dhr5), float(dhr10), float(dhr20), float(dhr50), float(dndcg5), float(dndcg10), float(dndcg20)]
        )

    if domain_rows:
        print("\n[per-domain]")
        print("domain\tmetric\tmean\tstd")
        for dom, vals in domain_rows.items():
            arr = np.array(vals)
            m = arr.mean(axis=0)
            s = arr.std(axis=0)
            for name, mv, sv in zip(dom_names, m, s):
                print(f"{dom}\t{name}\t{mv:.6f}\t{sv:.6f}")
PY
} | tee "$SUMMARY_FILE"
