# -*- coding: utf-8 -*-
# SDSS-SASRec Configuration - auto-set by train_fedsdsssasrec.sh

# Model dimensions
hidden_size = 128
segment_hidden_size = 128

# Self-Attention settings
num_heads = 2
num_blocks = 2
dropout_rate = 0.1
hidden_act = "relu"

# SDSS Summarizer settings
num_segments = 32
cnn_kernel_size = 3
use_local_transformer = False

# Auxiliary loss weights
alpha_recon = 0.0
beta_boundary = 0.01
gamma_compact = 0.1

# Boundary prediction
boundary_temp = 1.0
use_gumbel_softmax = False

# Optional regularizers
segment_alignment_weight = 0.0
