# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config

###################################
#       Activation Function       #
###################################

def gelu(x):
    """Implementation of the gelu activation function."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


###################################
#     Point-wise Feed Forward     #
###################################

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        if isinstance(config.hidden_act, str):
            self.feedforward_act_fn = ACT2FN[config.hidden_act]
        else:
            self.feedforward_act_fn = config.hidden_act
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.feedforward_act_fn(
            self.dropout1(self.conv1(inputs.transpose(-1, -2))))
        outputs = self.dropout2(self.conv2(outputs))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


###################################
#      SDSS Summarizer Module     #
###################################

class SDSSSummarizer(nn.Module):
    """
    SDSS Summarizer (Segment-based Domain-Shared Stream Summarizer)
    Converts raw sequence to fixed-length segment representations.
    This module is LOCAL-ONLY (not shared in FL).
    """
    def __init__(self, num_items, args):
        super(SDSSSummarizer, self).__init__()
        self.num_items = num_items
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"
        self.num_segments = config.num_segments  # K
        self.hidden_size = config.hidden_size
        self.segment_hidden_size = config.segment_hidden_size

        # 1.1 Step Embedding (Item + Position)
        # Item embedding is passed from main model

        # 1.2 Local Pattern Extraction (1D CNN)
        self.local_cnn = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.segment_hidden_size,
            kernel_size=config.cnn_kernel_size,
            padding=config.cnn_kernel_size // 2
        )
        self.local_ln = nn.LayerNorm(config.segment_hidden_size, eps=1e-8)
        self.local_dropout = nn.Dropout(config.dropout_rate)

        # 1.3 Segment Boundary Predictor
        self.boundary_predictor = nn.Linear(config.segment_hidden_size, 1)

        # 1.4 Segment Attention Pooling
        self.segment_attention = nn.Linear(config.segment_hidden_size, 1)

        # Projection for reconstruction
        self.reconstruction_decoder = nn.Linear(
            config.segment_hidden_size, config.hidden_size)

    def forward(self, seq_embeddings, seq_mask, **kwargs):
        """
        Args:
            seq_embeddings: (batch_size, seq_len, hidden_size) - h_t^(0)
            seq_mask: (batch_size, seq_len) - True for padding positions

        Returns:
            segment_reps: (batch_size, K, segment_hidden_size) - Z_u
            boundary_probs: (batch_size, seq_len) - p_t for auxiliary loss
            step_hidden: (batch_size, seq_len, segment_hidden_size) - h_tilde for aux loss
            segment_indices: list of segment assignments for each position
        """
        batch_size, seq_len, _ = seq_embeddings.size()

        # 1.2 Local Pattern Extraction via 1D CNN
        # Input: (batch_size, seq_len, hidden_size)
        # Conv1d expects: (batch_size, channels, seq_len)
        h_cnn = seq_embeddings.transpose(1, 2)  # (B, H, T)
        h_tilde = self.local_cnn(h_cnn)  # (B, H', T)
        h_tilde = h_tilde.transpose(1, 2)  # (B, T, H')
        h_tilde = self.local_ln(h_tilde)
        h_tilde = F.relu(h_tilde)
        h_tilde = self.local_dropout(h_tilde)

        # Mask padding positions
        h_tilde = h_tilde * (~seq_mask).unsqueeze(-1).float()

        # 1.3 Segment Boundary Prediction
        # p_t = sigmoid(w_b^T * h_tilde_t + b_b)
        boundary_logits = self.boundary_predictor(h_tilde).squeeze(-1)  # (B, T)

        # Mask padding positions with large negative value
        boundary_logits = boundary_logits.masked_fill(seq_mask, -1e9)
        boundary_probs = torch.sigmoid(boundary_logits)  # (B, T)

        # 1.3 Top-K Boundary Selection (K-1 boundaries for K segments)
        # Use Straight-Through Estimator for differentiability
        segment_reps, segment_indices = self._compute_segment_representations(
            h_tilde, boundary_probs, seq_mask)

        return segment_reps, boundary_probs, h_tilde, segment_indices

    def _compute_segment_representations(self, h_tilde, boundary_probs, seq_mask):
        """
        Compute segment representations using fixed-size segmentation
        Vectorized implementation for efficiency
        """
        batch_size, seq_len, hidden_dim = h_tilde.size()
        K = self.num_segments

        # Use fixed-size segmentation for efficiency
        # Divide sequence into K equal segments
        segment_size = seq_len // K

        segment_reps_list = []

        for k in range(K):
            start_idx = k * segment_size
            if k == K - 1:
                # Last segment takes remaining positions
                end_idx = seq_len
            else:
                end_idx = (k + 1) * segment_size

            # Get segment hidden states: (B, seg_len, H')
            segment_h = h_tilde[:, start_idx:end_idx, :]

            # Attention pooling for each segment
            attn_scores = self.segment_attention(segment_h).squeeze(-1)  # (B, seg_len)

            # Mask padding positions
            segment_mask = seq_mask[:, start_idx:end_idx]
            attn_scores = attn_scores.masked_fill(segment_mask, -1e9)

            attn_weights = F.softmax(attn_scores, dim=1)  # (B, seg_len)

            # z_k = sum(alpha_t * h_tilde_t)
            seg_rep = (attn_weights.unsqueeze(-1) * segment_h).sum(dim=1)  # (B, H')
            segment_reps_list.append(seg_rep)

        segment_reps = torch.stack(segment_reps_list, dim=1)  # (B, K, H')

        # Return empty segment_indices (not used in simplified version)
        segment_indices_list = [[] for _ in range(batch_size)]

        return segment_reps, segment_indices_list

    def get_shared_params(self):
        shared = {
            "local_cnn.weight": self.local_cnn.weight,
            "local_cnn.bias": self.local_cnn.bias,
            "boundary_predictor.weight": self.boundary_predictor.weight,
            "boundary_predictor.bias": self.boundary_predictor.bias,
            "segment_attention.weight": self.segment_attention.weight,
            "segment_attention.bias": self.segment_attention.bias,
        }
        return shared

    def compute_reconstruction_loss(self, h_original, h_tilde, segment_reps,
                                    segment_indices, seq_mask):
        """
        Compute reconstruction loss: L_recon = sum_t ||h_t^(0) - h_hat_t||^2
        where h_hat_t = g(z_seg(t))
        """
        batch_size, seq_len, _ = h_original.size()
        total_loss = 0.0
        count = 0

        for b in range(batch_size):
            valid_len = (~seq_mask[b]).sum().item()
            start_idx = seq_len - valid_len

            for k, indices in enumerate(segment_indices[b]):
                if not indices:
                    continue

                # Decode segment representation
                h_hat = self.reconstruction_decoder(segment_reps[b, k])

                # Compare with original embeddings
                for local_idx in indices:
                    global_idx = start_idx + local_idx
                    h_orig = h_original[b, global_idx]
                    total_loss += F.mse_loss(h_orig, h_hat)
                    count += 1

        if count > 0:
            return total_loss / count
        return torch.tensor(0.0, device=h_original.device)

    def compute_boundary_loss(self, boundary_probs, seq_mask):
        """
        Boundary regularization using KL divergence
        L_boundary = sum_t KL(p_t || K/T)
        """
        batch_size, seq_len = boundary_probs.size()
        total_loss = 0.0

        for b in range(batch_size):
            valid_len = (~seq_mask[b]).sum().item()
            if valid_len == 0:
                continue

            # Target boundary probability
            target_prob = self.num_segments / valid_len
            target_prob = min(target_prob, 0.99)  # Clip for numerical stability

            # Get valid boundary probs
            valid_probs = boundary_probs[b, seq_len - valid_len:]

            # Binary KL divergence
            # KL(p || q) = p*log(p/q) + (1-p)*log((1-p)/(1-q))
            eps = 1e-8
            kl = valid_probs * torch.log((valid_probs + eps) / (target_prob + eps)) + \
                 (1 - valid_probs) * torch.log((1 - valid_probs + eps) / (1 - target_prob + eps))
            total_loss += kl.mean()

        return total_loss / batch_size

    def compute_compactness_loss(self, h_tilde, segment_reps, segment_indices, seq_mask):
        """
        Compactness constraint: L_compact = sum_k sum_t ||h_tilde_t - z_k||^2
        for t in segment k
        """
        batch_size, seq_len, _ = h_tilde.size()
        total_loss = 0.0
        count = 0

        for b in range(batch_size):
            valid_len = (~seq_mask[b]).sum().item()
            start_idx = seq_len - valid_len
            valid_h = h_tilde[b, start_idx:]

            for k, indices in enumerate(segment_indices[b]):
                if not indices:
                    continue

                z_k = segment_reps[b, k]

                for local_idx in indices:
                    h_t = valid_h[local_idx]
                    total_loss += F.mse_loss(h_t, z_k)
                    count += 1

        if count > 0:
            return total_loss / count
        return torch.tensor(0.0, device=h_tilde.device)


###################################
#    Segment-level Self-Attention #
###################################

class SegmentSelfAttention(nn.Module):
    """
    Self-Attention for segment-level sequential modeling
    This is the GLOBAL module (shared in FL)
    """
    def __init__(self, num_segments, args):
        super(SegmentSelfAttention, self).__init__()
        self.num_segments = num_segments
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(config.segment_hidden_size, eps=1e-8)

        for _ in range(config.num_blocks):
            new_attn_layernorm = nn.LayerNorm(config.segment_hidden_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = nn.MultiheadAttention(
                config.segment_hidden_size,
                config.num_heads,
                config.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(config.segment_hidden_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(
                config.segment_hidden_size, config.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def forward(self, segment_reps):
        """
        Args:
            segment_reps: (batch_size, K, segment_hidden_size)

        Returns:
            segment_features: (batch_size, K, segment_hidden_size)
        """
        seqs = segment_reps
        tl = seqs.shape[1]  # K segments

        # Causal attention mask for autoregressive prediction
        attention_mask = ~torch.tril(torch.ones(
            (tl, tl), dtype=torch.bool)).to(self.device)

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)  # (K, B, H')
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)  # (B, K, H')

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs)

        return log_feats
