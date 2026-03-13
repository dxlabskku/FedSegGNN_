# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class SdssDomainEncoder(nn.Module):
    """Encode a batch of sequences into a single domain embedding using CNN+MHA."""

    def __init__(self, item_emb_dim, hidden_dim, domain_emb_dim, num_heads=4):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=item_emb_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, domain_emb_dim),
        )

    def forward(self, seq_emb_batch, seq_mask=None):
        """
        Args:
            seq_emb_batch: [B, T, E]
            seq_mask: [B, T] bool, True for padding (optional)
        Returns:
            domain_emb: [domain_emb_dim]
        """
        x = seq_emb_batch.transpose(1, 2)  # [B, E, T]
        x = self.conv(x)                   # [B, H, T]
        x = x.transpose(1, 2)              # [B, T, H]
        if seq_mask is not None:
            attn_out, _ = self.attn(x, x, x, key_padding_mask=seq_mask)
        else:
            attn_out, _ = self.attn(x, x, x)

        seq_repr = attn_out
        if seq_mask is not None:
            valid = (~seq_mask).float().unsqueeze(-1)
            denom = valid.sum(dim=1).clamp_min(1.0)
            seq_repr = (seq_repr * valid).sum(dim=1) / denom
        else:
            seq_repr = seq_repr.mean(dim=1)

        domain_repr = seq_repr.mean(dim=0)  # [H]
        domain_emb = self.proj(domain_repr)
        return domain_emb


class DomainGNN(nn.Module):
    """Lightweight MLP-style GNN for domain smoothing on cosine graph."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        layers = []
        if num_layers <= 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Linear(hidden_dim, out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, V, A):
        """
        Args:
            V: [D, F_in]
            A: [D, D] normalized adjacency
        Returns:
            H: [D, F_out]
        """
        H = V
        for i, layer in enumerate(self.layers):
            H = A @ H
            H = layer(H)
            if i < len(self.layers) - 1:
                H = F.relu(H)
        return H


class HyperHead(nn.Module):
    """Generate low-rank adapters for the prediction head from domain embeddings."""

    def __init__(self, dom_emb_dim, hidden_dim, num_items, rank=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_items = num_items
        self.rank = rank
        self.fc_A = nn.Sequential(
            nn.Linear(dom_emb_dim, dom_emb_dim),
            nn.ReLU(),
            nn.Linear(dom_emb_dim, hidden_dim * rank),
        )
        self.fc_B = nn.Sequential(
            nn.Linear(dom_emb_dim, dom_emb_dim),
            nn.ReLU(),
            nn.Linear(dom_emb_dim, num_items * rank),
        )

    def forward(self, g_d):
        """
        Args:
            g_d: [dom_emb_dim]
        Returns:
            delta_w [hidden_dim, num_items]
        """
        A_flat = self.fc_A(g_d)
        B_flat = self.fc_B(g_d)
        A = A_flat.view(self.hidden_dim, self.rank)  # [H, r]
        B = B_flat.view(self.num_items, self.rank)   # [N, r]

        delta_w = torch.matmul(A, B.t())  # [H, N]
        return delta_w

    def compute_delta_logits(self, z, g_d):
        """Compute delta logits from hidden states and domain embedding.

        Args:
            z: [B, T, H] input representations
            g_d: [dom_emb_dim] domain embedding
        Returns:
            delta_logits: [B, T, N]
        """
        delta_w = self.forward(g_d)
        delta_logits = torch.matmul(z, delta_w)  # [B, T, N]
        return delta_logits

    def get_shared_params(self):
        """Return fc_A parameters for FedAvg sharing.
        fc_A is domain-agnostic (only depends on H), so it can be shared.
        fc_B depends on num_items (N), so it must remain local.
        """
        return {'fc_A': self.fc_A.state_dict()}

    def load_shared_params(self, shared_state):
        """Load shared fc_A parameters from FedAvg aggregation."""
        if 'fc_A' in shared_state:
            self.fc_A.load_state_dict(shared_state['fc_A'])
