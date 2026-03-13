# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from .modules import SelfAttention
from .gnn import GCNLayer
from . import config


def resolve_disen_encoder_type(method):
    """Resolve encoder type from training method (no runtime CLI switch)."""
    # FedDCSR keeps VAE encoder; others use base.
    if method == "FedDCSR":
        return "vae"
    return "base"


class BaseEncoder(nn.Module):
    def __init__(self, num_items, args):
        super().__init__()
        self.encoder = SelfAttention(num_items, args)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.ln = nn.LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, seqs, seqs_data):
        feat = self.encoder(seqs, seqs_data)
        feat = self.proj(feat)
        feat = self.ln(feat)
        return feat


class GatedFusion(nn.Module):
    """Gated fusion layer for combining global and local representations."""
    def __init__(self, hidden_size):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size, eps=1e-12),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, z_g, z_l):
        """
        z_g: global representation (B, T, H)
        z_l: local representation (B, T, H)
        """
        gate = self.gate(torch.cat([z_g, z_l], dim=-1))  # (B, T, H)
        fused = gate * z_g + (1 - gate) * z_l
        fused = self.proj(fused)
        fused = self.ln(fused)
        return fused, gate


class dgl_streeam(nn.Module):

    def __init__(self, num_items, args):
        super().__init__()
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"
        self.num_items = num_items
        self.encoder_type = resolve_disen_encoder_type(
            getattr(args, "method", ""))
        self.use_gated_fusion = getattr(args, 'proto_use_gated_fusion', True)

        # Item embeddings (domain-specific, not shared)
        self.item_emb_g = nn.Embedding(
            num_items + 1, config.hidden_size, padding_idx=num_items)
        self.item_emb_l = nn.Embedding(
            num_items + 1, config.hidden_size, padding_idx=num_items)

        # Position embeddings
        self.pos_emb_g = nn.Embedding(args.max_seq_len, config.hidden_size)
        self.pos_emb_l = nn.Embedding(args.max_seq_len, config.hidden_size)

        # GNN for item graph
        self.GNN_encoder_g = GCNLayer(args)
        self.GNN_encoder_l = GCNLayer(args)

        # Encoders (global and local)
        if self.encoder_type == 'base':
            self.encoder_g = BaseEncoder(num_items, args)
            self.encoder_l = BaseEncoder(num_items, args)
        else:
            # VAE-style encoder (for comparison)
            from .disen_vgsan_model import Encoder
            self.encoder_g = Encoder(num_items, args)
            self.encoder_l = Encoder(num_items, args)

        # Gated fusion (optional)
        if self.use_gated_fusion:
            self.gated_fusion = GatedFusion(config.hidden_size)
        else:
            self.gated_fusion = None

        # Prediction head
        self.linear = nn.Linear(config.hidden_size, num_items)
        self.linear_pad = nn.Linear(config.hidden_size, 1)

        # LayerNorm and Dropout
        self.LayerNorm_g = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.LayerNorm_l = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Contrastive projection head
        self.contrastive_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size // 2)
        )

    def my_index_select_embedding(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = torch.index_select(memory.weight, 0, index)
        ans = ans.view(tmp)
        return ans

    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def graph_convolution(self, adj):
        adj = adj.detach() if adj.requires_grad else adj
        self.item_index_g = torch.arange(
            0, self.item_emb_g.num_embeddings, 1).to(self.device)
        self.item_index_l = torch.arange(
            0, self.item_emb_l.num_embeddings, 1).to(self.device)
        item_embs_g = self.my_index_select_embedding(
            self.item_emb_g, self.item_index_g)
        item_embs_l = self.my_index_select_embedding(
            self.item_emb_l, self.item_index_l)
        self.item_graph_embs_g = self.GNN_encoder_g(item_embs_g, adj)
        self.item_graph_embs_l = self.GNN_encoder_l(item_embs_l, adj)

    def get_position_ids(self, seqs):
        seq_length = seqs.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=seqs.device)
        position_ids = position_ids.unsqueeze(0).expand_as(seqs)
        return position_ids

    def add_position_embedding_g(self, seqs, seq_embeddings):
        position_ids = self.get_position_ids(seqs)
        position_embeddings = self.pos_emb_g(position_ids)
        seq_embeddings = seq_embeddings + position_embeddings
        seq_embeddings = self.LayerNorm_g(seq_embeddings)
        seq_embeddings = self.dropout(seq_embeddings)
        return seq_embeddings

    def add_position_embedding_l(self, seqs, seq_embeddings):
        position_ids = self.get_position_ids(seqs)
        position_embeddings = self.pos_emb_l(position_ids)
        seq_embeddings = seq_embeddings + position_embeddings
        seq_embeddings = self.LayerNorm_l(seq_embeddings)
        seq_embeddings = self.dropout(seq_embeddings)
        return seq_embeddings

    def forward(self, seqs, neg_seqs=None, aug_seqs=None):
        # Get item embeddings with GNN
        seqs_emb_g = self.my_index_select(
            self.item_graph_embs_g, seqs) + self.item_emb_g(seqs)
        seqs_emb_l = self.my_index_select(
            self.item_graph_embs_l, seqs) + self.item_emb_l(seqs)

        seqs_emb_g = seqs_emb_g * (self.item_emb_g.embedding_dim ** 0.5)
        seqs_emb_l = seqs_emb_l * (self.item_emb_l.embedding_dim ** 0.5)

        seqs_emb_g = self.add_position_embedding_g(seqs, seqs_emb_g)
        seqs_emb_l = self.add_position_embedding_l(seqs, seqs_emb_l)

        # Augmented sequences for contrastive learning
        if self.training and aug_seqs is not None:
            aug_seqs_emb = self.my_index_select(
                self.item_graph_embs_l, aug_seqs) + self.item_emb_l(aug_seqs)
            aug_seqs_emb = aug_seqs_emb * (self.item_emb_l.embedding_dim ** 0.5)
            aug_seqs_emb = self.add_position_embedding_l(aug_seqs, aug_seqs_emb)

        # Encode
        mu_g, logvar_g = None, None
        mu_l, logvar_l = None, None
        if self.encoder_type == 'vae':
            mu_g, logvar_g = self.encoder_g(seqs_emb_g, seqs)
            z_g = self._reparameterization(mu_g, logvar_g)
            mu_l, logvar_l = self.encoder_l(seqs_emb_l, seqs)
            z_l = self._reparameterization(mu_l, logvar_l)
        else:
            z_g = self.encoder_g(seqs_emb_g, seqs)
            z_l = self.encoder_l(seqs_emb_l, seqs)

        if self.training and aug_seqs is not None:
            if self.encoder_type == 'vae':
                aug_mu_l, aug_logvar_l = self.encoder_l(aug_seqs_emb, aug_seqs)
                aug_z_l = self._reparameterization(aug_mu_l, aug_logvar_l)
            else:
                aug_z_l = self.encoder_l(aug_seqs_emb, aug_seqs)
        else:
            aug_z_l = None


        # Combine global and local
        if self.gated_fusion is not None:
            z_combined, gate = self.gated_fusion(z_g, z_l)
        else:
            z_combined = z_g + z_l
            gate = None

        # Prediction
        result = self.linear(z_combined)
        result_pad = self.linear_pad(z_combined)
        result = torch.cat((result, result_pad), dim=-1)

        # Exclusive prediction (for auxiliary loss)
        result_exclusive = self.linear(z_l)
        result_exclusive_pad = self.linear_pad(z_l)
        result_exclusive = torch.cat((result_exclusive, result_exclusive_pad), dim=-1)

        if self.training:
            # Contrastive features for InfoNCE loss
            z_l_proj = self.contrastive_proj(z_l[:, -1, :])  # (B, H/2)
            aug_z_l_proj = self.contrastive_proj(aug_z_l[:, -1, :]) if aug_z_l is not None else None

            return (result, result_exclusive, z_g, z_l, aug_z_l,
                    z_l_proj, aug_z_l_proj, gate, mu_g, logvar_g, mu_l, logvar_l)
        else:
            return result

    @property
    def encoder_s(self):
        return self.encoder_g

    @property
    def encoder_e(self):
        return self.encoder_l

    def _reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        if self.training:
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu
