#!/bin/bash

# If invoked via `sh run.sh`, re-exec with bash because this script uses
# bash-only features (arrays, += appends, ${var,,}).
if [ -z "${BASH_VERSION:-}" ]; then
    exec /bin/bash "$0" "$@"
fi

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIST=${DATA_DIST:-custom}
DOMAIN_SCENARIO=${DOMAIN_SCENARIO:-"kuairand27/abalation_1000 kuairand27/abalation_1001 kuairand27/abalation_1002 kuairand27/abalation_1004"}
#DOMAIN_SCENARIO=${DOMAIN_SCENARIO:-"kuairand27/abalation_200 kuairand27/abalation_201 kuairand27/abalation_202 kuairand27/abalation_204"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoint}
EPOCHS=${EPOCHS:-20}
LOCAL_EPOCH=${LOCAL_EPOCH:-1}
BATCH_SIZE=8
LR=${LR:-0.001}
GPU=${GPU:-2}
FRAC=${FRAC:-1.0}
PREP=${PREP:-false}
SEED=${SEED:-42}
LOG_ROOT=${LOG_ROOT:-log_abalation_100/abalation/knn}

# Training method
# Federated: FedDCSR, FedVGSAN, FedSASRec, FedVSAN, FedContrastVAE, FedCL4SRec,
#            FedDuoRec, SegFedGNN, FedDisenProto
# Local:     LocalVGSAN, LocalSASRec, LocalVSAN, LocalContrastVAE, LocalCL4SRec,
#            LocalDuoRec, LocalDisenProto
METHOD=${METHOD:-SegFedGNN}

# ---------------- SegFedGNN & FedDCSR Parameter ---------------- #
MAX_SEQ_LEN=${MAX_SEQ_LEN:-200}
SDSS_NUM_SEGMENTS=${SDSS_NUM_SEGMENTS:-64} # should be 128
#SDSS_NUM_SEGMENTS=${SDSS_NUM_SEGMENTS:-32}

# sdss , direct
SDSS_BRANCH_TYPE=${SDSS_BRANCH_TYPE:-sdss} # direct: directly fuse SDSS branch with main branch; separate: keep SDSS branch separate and only fuse at prediction layer
SDSS_DUAL_EXCLUSIVE_WEIGHT=${SDSS_DUAL_EXCLUSIVE_WEIGHT:-1.0} # L_exclusive: exclusive KL loss to encourage domain-specificity

# ---------------- SegFedGNN GNN Parameter ---------------- #
DOMAIN_GNN_HIDDEN=${DOMAIN_GNN_HIDDEN:-64}
DOMAIN_GNN_LAYERS=${DOMAIN_GNN_LAYERS:-4}
DOMAIN_KNN_K=${DOMAIN_KNN_K:-2}
HYPER_RANK=${HYPER_RANK:-1}

# ---------------- Abalation ---------------- #
SDSS_DUAL_DISABLE_SDSS_BRANCH=${SDSS_DUAL_DISABLE_SDSS_BRANCH:-false} 
SDSS_DUAL_DISABLE_PRED_FUSION=${SDSS_DUAL_DISABLE_PRED_FUSION:-false}
DISABLE_HYPERHEAD=${DISABLE_HYPERHEAD:-false}

PROTO_USE_GATED_FUSION=${PROTO_USE_GATED_FUSION:-true}

# ---------------------------Additional-------------------------- #
SDSS_AUX_RECON_WEIGHT=${SDSS_AUX_RECON_WEIGHT:-0.0}       # L_recon: reconstruction loss
SDSS_AUX_BOUNDARY_WEIGHT=${SDSS_AUX_BOUNDARY_WEIGHT:-0.00} # L_boundary: boundary KL loss
SDSS_AUX_COMPACT_WEIGHT=${SDSS_AUX_COMPACT_WEIGHT:-0.0}   # L_compact: compactness loss

# Data distribution scenarios
case "${DATA_DIST,,}" in
    s0)
        DOMAIN_SCENARIO=${DOMAIN_SCENARIO:-"Food Kitchen Clothing Beauty"}
        ;;
    s1|s1_base)
        DOMAIN_SCENARIO="Food Kitchen Clothing Beauty"
        ;;
    s1_life)
        DOMAIN_SCENARIO="Food Kitchen Home Garden"
        ;;
    s1_ent)
        DOMAIN_SCENARIO="Movies Books Games Sports"
        ;;
    s2)
        DOMAIN_SCENARIO="Food Kitchen Clothing Beauty Movies Books Games Sports Garden Home"
        ;;
    *)
        DOMAIN_SCENARIO=${DOMAIN_SCENARIO:-"Food Kitchen Clothing Beauty"}
        ;;
esac

# Tokenize domains once to avoid shell interpreting them as commands
read -r -a DOMAINS <<< "$DOMAIN_SCENARIO"
if [ ${#DOMAINS[@]} -eq 0 ]; then
    echo "[error] DOMAIN_SCENARIO is empty; please set DATA_DIST/DOMAIN_SCENARIO." >&2
    exit 1
fi

# Utility: allow other scripts to query the resolved domain scenario without running training.
if [ "${PRINT_DOMAIN_SCENARIO:-}" = "1" ]; then
    echo "$DOMAIN_SCENARIO"
    exit 0
fi

is_true() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

BRANCH_FLAG=$(is_true "$SDSS_DUAL_DISABLE_SDSS_BRANCH" && echo 1 || echo 0)
PRED_FUSE_FLAG=$(is_true "$SDSS_DUAL_DISABLE_PRED_FUSION" && echo 0 || echo 1)
HYPERHEAD_FLAG=$(is_true "$DISABLE_HYPERHEAD" && echo 0 || echo 1)
GATE_FLAG=$(is_true "$PROTO_USE_GATED_FUSION" && echo 1 || echo 0)

LABEL_FLAGS="ex${SDSS_DUAL_EXCLUSIVE_WEIGHT}_branch${BRANCH_FLAG}"
LABEL_FLAGS="${LABEL_FLAGS}_bt${SDSS_BRANCH_TYPE}"
LABEL_FLAGS="${LABEL_FLAGS}_pfuse${PRED_FUSE_FLAG}"
LABEL_FLAGS="${LABEL_FLAGS}_dh${HYPER_RANK}"
LABEL_FLAGS="${LABEL_FLAGS}_hh${HYPERHEAD_FLAG}"

# Add dgl_streeam flags if using that method
if [[ "$METHOD" == *"DisenProto"* ]] || [[ "$METHOD" == "dgl_streeam" ]]; then
    LABEL_FLAGS="${LABEL_FLAGS}_gate${GATE_FLAG}"
fi
DIST_TAG=$(echo "$DATA_DIST" | tr ' ' '-' | tr '[:upper:]' '[:lower:]')
DOM_TAG=$(echo "$DOMAIN_SCENARIO" | tr ' ' '-' | tr '/' '-' | tr '[:upper:]' '[:lower:]')
RUN_TAG="dist${DIST_TAG}_dom${DOM_TAG}_lr${LR}_seed${SEED}_len${MAX_SEQ_LEN}_seg${SDSS_NUM_SEGMENTS}"

DATASET_TAG="${DATASET_TAG:-}"
if [ -z "$DATASET_TAG" ]; then
    first_domain="${DOMAINS[0]}"
    if [[ "$first_domain" == */* ]]; then
        DATASET_TAG="${first_domain%%/*}"
    else
        DATASET_TAG="$first_domain"
    fi
    DATASET_TAG=$(echo "$DATASET_TAG" | tr '[:upper:]' '[:lower:]')
fi

# Log structure: LOG_ROOT/DATASET/METHOD/RUN_TAG/LABEL_FLAGS (LOG_DIR override allowed)
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/${DATASET_TAG}/${METHOD}/${RUN_TAG}/${LABEL_FLAGS}}"

if [ -n "${1:-}" ]; then
    GPU="$1"
fi

# Compose CLI args for main.py
ARGS=(
    --epochs "$EPOCHS"
    --local_epoch "$LOCAL_EPOCH"
    --batch_size "$BATCH_SIZE"
    --lr "$LR"
    --seed "$SEED"
    --gpu "$GPU"
    --frac "$FRAC"
    --max_seq_len "$MAX_SEQ_LEN"
    --data_dist "$DATA_DIST"
    --sdss_num_segments "$SDSS_NUM_SEGMENTS"
    --sdss_branch_type "$SDSS_BRANCH_TYPE"
    --method "$METHOD"
    --log_dir "$LOG_DIR"
    --checkpoint_dir "$CHECKPOINT_DIR"
    --sdss_dual_exclusive_weight "$SDSS_DUAL_EXCLUSIVE_WEIGHT"
    --sdss_aux_recon_weight "$SDSS_AUX_RECON_WEIGHT"
    --sdss_aux_boundary_weight "$SDSS_AUX_BOUNDARY_WEIGHT"
    --sdss_aux_compact_weight "$SDSS_AUX_COMPACT_WEIGHT"
    --use_domain_hyper
    --domain_gnn_hidden "$DOMAIN_GNN_HIDDEN"
    --domain_gnn_layers "$DOMAIN_GNN_LAYERS"
    --domain_knn_k "$DOMAIN_KNN_K"
    --hyper_rank "$HYPER_RANK"
)

if is_true "$PREP"; then
    ARGS+=(--load_prep)
fi

if is_true "$DISABLE_HYPERHEAD"; then
    ARGS+=(--disable_hyperhead)
fi

if is_true "$SDSS_DUAL_DISABLE_SDSS_BRANCH"; then
    ARGS+=(--sdss_dual_disable_sdss_branch)
fi

if is_true "$SDSS_DUAL_DISABLE_PRED_FUSION"; then
    ARGS+=(--sdss_dual_disable_pred_fusion)
fi

if is_true "$PROTO_USE_GATED_FUSION"; then
    ARGS+=(--proto_use_gated_fusion)
else
    ARGS+=(--proto_disable_gated_fusion)
fi

python -u "$SCRIPT_DIR/main.py" "${ARGS[@]}" "${DOMAINS[@]}"
