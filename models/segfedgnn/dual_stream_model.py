# -*- coding: utf-8 -*-
"""
SegFedGNN (Dual-Stream Variant)

Pipeline:
- DGL-Stream (Disentangled Graph Learning): dgl_streeam on item sequences
- SDSS (Segment-based Domain-Shared Stream): SDSS Summarizer -> Segment Encoder
- Gated fusion for final prediction

Shared params (Fed): DGL encoder_s + SDSS segment encoder.
Local params: item embeddings/heads, SDSS Summarizer.
"""

import torch
import torch.nn as nn

from .dgl.dgl_stream import dgl_streeam
from .sdss.modules import SDSSSummarizer, SegmentSelfAttention
from .sdss import config as sdss_config
from .dgl import config as dgl_config
from .domain_hyper import SdssDomainEncoder


class SegFedGNN(nn.Module):
    def __init__(self, num_items, args):
        super().__init__()
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"
        self.num_items = num_items
        self.num_segments = sdss_config.num_segments
        self.gate_min = getattr(args, "sdss_dual_gate_min", 0.2)
        self.gate_mode = getattr(args, "sdss_gate_mode", "sigmoid")
        self.gate_temp = getattr(args, "sdss_gate_softmax_temp", 1.0)
        self.branch_type = getattr(args, "sdss_branch_type", "sdss")
        self.use_domain_hyper = getattr(args, "use_domain_hyper", True)
        self.use_sdss_branch = not getattr(
            args, "sdss_dual_disable_sdss_branch", False)
        self.disable_sdss_pred_fusion = bool(
            getattr(args, "sdss_dual_disable_pred_fusion", False))

        # DGL-Stream (SegFedGNN always uses base dgl_streeam)
        self.disen = dgl_streeam(num_items, args)

        # SDSS Summarizer and Segment Encoder
        self.sdss_summarizer = SDSSSummarizer(num_items, args)
        # Item/position embeddings for summarizer input
        self.item_emb = nn.Embedding(
            num_items + 1, sdss_config.hidden_size, padding_idx=num_items)
        self.pos_emb = nn.Embedding(args.max_seq_len, sdss_config.hidden_size)
        self.step_layernorm = nn.LayerNorm(sdss_config.hidden_size, eps=1e-12)
        self.step_dropout = nn.Dropout(sdss_config.dropout_rate)
        self.seg2pred = nn.Linear(sdss_config.segment_hidden_size,
                                  dgl_config.hidden_size)
        # Segment Encoder (SDSS self-attention)
        self.segment_encoder = SegmentSelfAttention(self.num_segments, args)
        self.segment_pos_emb = nn.Embedding(
            self.num_segments, sdss_config.segment_hidden_size)
        # Alternative backbones
        if self.branch_type == "sasrec":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=sdss_config.hidden_size,
                nhead=sdss_config.num_heads,
                dim_feedforward=sdss_config.hidden_size * 4,
                dropout=sdss_config.dropout_rate,
                activation="relu",
                batch_first=False,
            )
            self.sasrec_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=sdss_config.num_blocks)
            self.seq2hidden = nn.Linear(sdss_config.hidden_size, dgl_config.hidden_size)
            self.sas_ln = nn.LayerNorm(sdss_config.hidden_size, eps=1e-12)
        elif self.branch_type == "mlp":
            self.mlp_agg = nn.Sequential(
                nn.Linear(sdss_config.hidden_size, dgl_config.hidden_size),
                nn.ReLU(),
                nn.Linear(dgl_config.hidden_size, dgl_config.hidden_size),
                nn.LayerNorm(dgl_config.hidden_size, eps=1e-12),
            )
        elif self.branch_type == "direct":
            self.direct_seq2hidden = nn.Linear(
                sdss_config.hidden_size, dgl_config.hidden_size)
        # Gated fusion between DGL-Stream and SDSS
        # Scalar gate per time step controlling SDSS contribution
        self.gate_mlp = nn.Linear(dgl_config.hidden_size * 2, 1)
        self.gate_mlp_softmax = nn.Linear(dgl_config.hidden_size * 2, 2)
        self.gate_dropout = nn.Dropout(dgl_config.dropout_rate)
        nn.init.constant_(self.gate_mlp.bias, 1.0)  # open gate at init

        if self.use_domain_hyper:
            self.domain_encoder = SdssDomainEncoder(
                item_emb_dim=sdss_config.hidden_size,
                hidden_dim=sdss_config.segment_hidden_size,
                domain_emb_dim=getattr(args, "domain_emb_dim", 64),
                num_heads=sdss_config.num_heads,
            )

    def add_segment_pos(self, segs, seg_embeddings):
        position_ids = torch.arange(
            segs.size(1), dtype=torch.long, device=segs.device).unsqueeze(0)
        position_ids = position_ids.expand_as(segs)
        pos_emb = self.segment_pos_emb(position_ids)
        return seg_embeddings + pos_emb

    def add_step_pos(self, seqs, seq_embeddings):
        seq_length = seqs.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=seqs.device)
        position_ids = position_ids.unsqueeze(0).expand_as(seqs)
        pos_emb = self.pos_emb(position_ids)
        seq_embeddings = seq_embeddings + pos_emb
        seq_embeddings = self.step_layernorm(seq_embeddings)
        seq_embeddings = self.step_dropout(seq_embeddings)
        return seq_embeddings

    def compute_domain_embedding(self, seqs):
        """Compute domain embedding from raw item sequences."""
        if not getattr(self, "domain_encoder", None):
            return None
        with torch.no_grad():
            seqs = seqs.to(self.device)
            seq_mask = (seqs == self.num_items)
            seqs_emb = self.item_emb(seqs)
            seqs_emb = seqs_emb * (self.item_emb.embedding_dim ** 0.5)
            seqs_emb = self.add_step_pos(seqs, seqs_emb)
            domain_emb = self.domain_encoder(seqs_emb, seq_mask)
        return domain_emb.detach().cpu()

    def forward(self, seqs, neg_seqs=None, aug_seqs=None):
        # Ensure neg/aug placeholders to satisfy dgl_streeam signature
        if neg_seqs is None:
            neg_seqs = seqs
        if aug_seqs is None:
            aug_seqs = seqs

        # DGL-Stream outputs (dgl_streeam)
        disen_out = self.disen(seqs, neg_seqs, aug_seqs)
        dgl_gate_stats = {
            "alpha_gate_mean": None,
            "alpha_gate_min": None,
            "alpha_gate_max": None,
        }
        # Handle train/eval return shapes
        if isinstance(disen_out, tuple):
            # dgl_streeam training:
            # (result, result_exclusive, z_g, z_l, aug_z_l, z_l_proj, aug_z_l_proj,
            #  gate, mu_g, logvar_g, mu_l, logvar_l)
            (disen_result, result_exclusive, z_g, z_l, aug_z_l, _z_l_proj,
             _aug_z_l_proj, _dgl_gate, mu_g, logvar_g, mu_l, logvar_l) = disen_out
            if _dgl_gate is not None:
                dgl_gate_stats = {
                    "alpha_gate_mean": _dgl_gate.mean().detach().item(),
                    "alpha_gate_min": _dgl_gate.min().detach().item(),
                    "alpha_gate_max": _dgl_gate.max().detach().item(),
                }
            z_s = z_g
            z_e = z_l
            mu_s = mu_g
            logvar_s = logvar_g
            mu_e = mu_l
            logvar_e = logvar_l
            neg_z_e = None
            aug_z_e = aug_z_l
            z_sum = z_s + z_e
        else:
            # eval mode: only logits
            disen_result = disen_out
            z_s = None
            z_e = None
            z_sum = None

        if not self.use_sdss_branch:
            aux = None
            if isinstance(disen_out, tuple):
                aux = {
                    "result_exclusive": result_exclusive,
                    "mu_s": mu_s,
                    "logvar_s": logvar_s,
                    "mu_e": mu_e,
                    "logvar_e": logvar_e,
                    "neg_z_e": neg_z_e,
                    "aug_z_e": aug_z_e,
                    "gate_mean": None,
                    "gate_min": None,
                    "gate_max": None,
                    "alpha_gate_mean": dgl_gate_stats.get("alpha_gate_mean"),
                    "alpha_gate_min": dgl_gate_stats.get("alpha_gate_min"),
                    "alpha_gate_max": dgl_gate_stats.get("alpha_gate_max"),
                }
            return disen_result, z_s, z_e, z_sum, aux

        # SDSS: Segment-based Domain-Shared Stream
        seq_mask = (seqs == self.num_items)
        seqs_emb = self.item_emb(seqs)
        seqs_emb = seqs_emb * (self.item_emb.embedding_dim ** 0.5)
        seqs_emb = self.add_step_pos(seqs, seqs_emb)

        # Initialize SDSS auxiliary outputs
        sdss_boundary_probs = None
        sdss_h_tilde = None
        sdss_segment_indices = None
        sdss_h_original = seqs_emb  # Original embeddings for reconstruction loss

        if self.branch_type == "sdss":
            segment_reps_raw, sdss_boundary_probs, sdss_h_tilde, sdss_segment_indices = self.sdss_summarizer(
                seqs_emb, seq_mask)
            segment_reps = self.add_segment_pos(
                seqs.new_zeros(seqs.size(0), segment_reps_raw.size(1)), segment_reps_raw)
            segment_reps = self.segment_encoder(segment_reps)
            segment_reps = self.seg2pred(segment_reps)  # (B, K, H)
            summary_vec = segment_reps.mean(dim=1)  # (B, H)
        elif self.branch_type == "sasrec":
            # Transformer encoder over step embeddings (mask padding)
            src = seqs_emb.transpose(0, 1)  # (T, B, Hc)
            encoded = self.sasrec_encoder(
                src, src_key_padding_mask=seq_mask)  # (T, B, Hc)
            encoded = encoded.transpose(0, 1)  # (B, T, Hc)
            encoded = self.sas_ln(encoded)
            # mean over valid steps
            valid_mask = (~seq_mask).unsqueeze(-1).float()
            denom = valid_mask.sum(dim=1).clamp_min(1.0)
            summary_raw = (encoded * valid_mask).sum(dim=1) / denom
            summary_vec = self.seq2hidden(summary_raw)  # (B, H)
            segment_reps = summary_vec.unsqueeze(1)  # dummy for aux
        elif self.branch_type == "mlp":
            valid_mask = (~seq_mask).unsqueeze(-1).float()
            denom = valid_mask.sum(dim=1).clamp_min(1.0)
            mean_emb = (seqs_emb * valid_mask).sum(dim=1) / denom
            summary_vec = self.mlp_agg(mean_emb)  # (B, H)
            segment_reps = summary_vec.unsqueeze(1)
        elif self.branch_type == "direct":
            # Direct sequence -> embedding ablation (no fixed-size segmentation).
            valid_mask = (~seq_mask).unsqueeze(-1).float()
            denom = valid_mask.sum(dim=1).clamp_min(1.0)
            mean_emb = (seqs_emb * valid_mask).sum(dim=1) / denom
            summary_vec = self.direct_seq2hidden(mean_emb)  # (B, H)
            segment_reps = summary_vec.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported sdss_branch_type: {self.branch_type}")

        # SDSS-based logits from sequence-level summary.
        sdss_hidden = summary_vec
        sdss_logits = torch.matmul(sdss_hidden, self.disen.linear.weight.t())  # (B, num_items)
        sdss_logits = sdss_logits.unsqueeze(1)  # (B, 1, num_items)

        # Fuse: add segment_rep to disen decoder output at prediction step
        # result: (B, T, num_items+1), we align K to last position
        # Use SDSS summary for last position, broadcast
        sdss_hidden_time = sdss_hidden.unsqueeze(1)
        if sdss_hidden_time.size(1) != disen_result.size(1):
            sdss_hidden_time = sdss_hidden_time.expand(-1, disen_result.size(1), -1)

        # Split logits and pad
        logits = disen_result[:, :, :-1]
        pad_logit = disen_result[:, :, -1:]
        sdss_logits_full = torch.cat(
            [sdss_logits.expand(-1, logits.size(1), -1), pad_logit], dim=-1)

        # Ablation: keep SDSS branch active, but exclude SDSS prediction logits from final fusion.
        if self.disable_sdss_pred_fusion:
            fused = disen_result
            gate_stats = {
                "gate_mean": None,
                "gate_min": None,
                "gate_max": None,
                "gate_entropy": None
            }
        else:
            # Gated fusion: control how much segment stream contributes
            z_sum_full = z_sum if z_sum is not None else torch.zeros_like(sdss_hidden_time)
            gate_in = torch.cat([z_sum_full, sdss_hidden_time], dim=-1)
            gate_stats = {}
            if self.gate_mode == "softmax":
                gate_logits = self.gate_mlp_softmax(gate_in)  # (B, T, 2)
                if self.gate_temp != 1.0:
                    gate_logits = gate_logits / self.gate_temp
                gate_weights = torch.softmax(gate_logits, dim=-1)
                # weight[... ,0]=Disen, [...,1]=SDSS
                w_sdss = gate_weights[..., 1:2]
                w_disen = gate_weights[..., 0:1]
                logits = w_disen * logits + w_sdss * sdss_logits.expand(-1, logits.size(1), -1)
                fused = torch.cat([logits, pad_logit], dim=-1)
                # entropy for regularization/logging
                entropy = -(gate_weights * (gate_weights.clamp_min(1e-8).log())).sum(dim=-1).mean()
                gate_stats = {
                    "gate_mean": w_sdss.mean().detach().item(),
                    "gate_min": w_sdss.min().detach().item(),
                    "gate_max": w_sdss.max().detach().item(),
                    "gate_entropy": entropy.detach().item()
                }
            else:
                gate = torch.sigmoid(self.gate_mlp(gate_in))  # (B, T, 1)
                gate = gate.clamp(min=self.gate_min)  # avoid full shutoff early
                logits = gate * logits + (1 - gate) * sdss_logits.expand(-1, logits.size(1), -1)
                fused = torch.cat([logits, pad_logit], dim=-1)
                gate_stats = {
                    "gate_mean": gate.mean().detach().item(),
                    "gate_min": gate.min().detach().item(),
                    "gate_max": gate.max().detach().item(),
                    "gate_entropy": None
                }

        aux = None
        if isinstance(disen_out, tuple):
            aux = {
                "result_exclusive": result_exclusive,
                "mu_s": mu_s,
                "logvar_s": logvar_s,
                "mu_e": mu_e,
                "logvar_e": logvar_e,
                "neg_z_e": neg_z_e,
                "aug_z_e": aug_z_e,
                "gate_mean": gate_stats.get("gate_mean"),
                "gate_min": gate_stats.get("gate_min"),
                "gate_max": gate_stats.get("gate_max"),
                "gate_entropy": gate_stats.get("gate_entropy"),
                "alpha_gate_mean": dgl_gate_stats.get("alpha_gate_mean"),
                "alpha_gate_min": dgl_gate_stats.get("alpha_gate_min"),
                "alpha_gate_max": dgl_gate_stats.get("alpha_gate_max"),
                "seg_logits": sdss_logits_full,
                "segment_reps": segment_reps,
                "summary_vec": summary_vec,
                "sdss_hidden": sdss_hidden,
                # SDSS auxiliary data for summarizer losses
                "sdss_boundary_probs": sdss_boundary_probs,
                "sdss_h_tilde": sdss_h_tilde,
                "sdss_h_original": sdss_h_original,
                "sdss_segment_indices": sdss_segment_indices,
                "sdss_seq_mask": seq_mask,
            }

        return fused, z_s, z_e, z_sum, aux
