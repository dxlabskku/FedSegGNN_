#!/bin/bash

# ============================================
# Baseline Methods Training Script
# ============================================
# This script trains baseline methods (FedSASRec, FedVSAN, FedContrastVAE, FedCL4SRec, FedDuoRec)

# Activate conda environment
source /home/dxlab/anaconda3/etc/profile.d/conda.sh
conda activate fed

# ============================================
# Training Configuration
# ============================================

# --epochs: Number of total training iterations (default: 40)
EPOCHS=40

# --local_epoch: Number of local training epochs per round (default: 3)
LOCAL_EPOCH=3

# --eval_interval: Interval for evaluation (default: 1)
EVAL_INTERVAL=1

# --frac: Fraction of participating clients (default: 1.0, range: 0.0-1.0)
FRAC=1.0

# --batch_size: Training batch size (default: 256)
BATCH_SIZE=256

# --log_dir: Directory to save logs (default: log, can be set externally)
LOG_DIR=${LOG_DIR:-log}

# --method: Training method
# Federated Options: FedSASRec, FedVSAN, FedVGSAN, FedContrastVAE, FedCL4SRec, FedDuoRec
# Local Options: LocalSASRec, LocalVSAN, LocalVGSAN, LocalContrastVAE, LocalCL4SRec, LocalDuoRec
METHOD=FedContrastVAE

# --anneal_cap: KL annealing cap for variational methods
# Recommended: 1.0 for FKCB, 0.01 for MBG and SGH
# Note: Only applies to variational methods (FedContrastVAE, LocalContrastVAE, FedDCSR, etc.)
ANNEAL_CAP=1.0

# --lr: Learning rate (default: 0.001)
LR=0.001

# --seed: Random seed for reproducibility (default: 42)
SEED=42

# --optimizer: Optimizer choice
# Options: sgd, adagrad, adam, adamax (default: adam)
OPTIMIZER=adam

# --lr_decay: Learning rate decay rate (default: 1)
LR_DECAY=1

# --weight_decay: Weight decay (L2 regularization) (default: 5e-4)
WEIGHT_DECAY=5e-4

# --decay_epoch: Decay learning rate after this epoch (default: 10)
DECAY_EPOCH=10

# --es_patience: Early stopping patience (default: 5)
ES_PATIENCE=5

# --ld_patience: Learning rate decay patience (default: 1)
LD_PATIENCE=1

# --max_seq_len: Maximum sequence length (default: 16)
MAX_SEQ_LEN=16

# --temperature: Temperature for contrastive learning (default: 1.0)
# Note: Only applies to contrastive methods (FedCL4SRec, LocalCL4SRec, FedDuoRec, LocalDuoRec)
TEMPERATURE=1.0

# --sim: Similarity measure for contrastive learning (default: dot)
SIM=dot

# --checkpoint_dir: Directory to save model checkpoints (default: checkpoint)
CHECKPOINT_DIR=checkpoint

# --id: Model ID for saving (default: 00)
MODEL_ID=00

# --mu: Hyperparameter for FedProx (default: 0)
MU=0

# --gpu: GPU ID to use (default: 0)
GPU=0

# --load_prep: Load preprocessed data (add flag to enable)
# Uncomment the line below to use preprocessed data
# LOAD_PREP="--load_prep"
LOAD_PREP=""

# ============================================
# Domain Selection
# ============================================
# Choose one of the following scenarios:
# 1. Food Kitchen Clothing Beauty (FKCB) - recommended anneal_cap: 1.0
# 2. Movies Books Games (MBG) - recommended anneal_cap: 0.01
# 3. Sports Garden Home (SGH) - recommended anneal_cap: 0.01

# Current scenario: FKCB
DOMAINS="Food Kitchen Clothing Beauty"

# Uncomment below for MBG scenario (and change ANNEAL_CAP to 0.01)
# DOMAINS="Movies Books Games"

# Uncomment below for SGH scenario (and change ANNEAL_CAP to 0.01)
# DOMAINS="Sports Garden Home"

# ============================================
# Run Training
# ============================================
python -u main.py \
    --epochs $EPOCHS \
    --local_epoch $LOCAL_EPOCH \
    --eval_interval $EVAL_INTERVAL \
    --frac $FRAC \
    --batch_size $BATCH_SIZE \
    --log_dir $LOG_DIR \
    --method $METHOD \
    --anneal_cap $ANNEAL_CAP \
    --lr $LR \
    --seed $SEED \
    --optimizer $OPTIMIZER \
    --lr_decay $LR_DECAY \
    --weight_decay $WEIGHT_DECAY \
    --decay_epoch $DECAY_EPOCH \
    --es_patience $ES_PATIENCE \
    --ld_patience $LD_PATIENCE \
    --max_seq_len $MAX_SEQ_LEN \
    --temperature $TEMPERATURE \
    --sim $SIM \
    --checkpoint_dir $CHECKPOINT_DIR \
    --id $MODEL_ID \
    --mu $MU \
    --gpu $GPU \
    $LOAD_PREP \
    $DOMAINS
