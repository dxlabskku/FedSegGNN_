# -*- coding: utf-8 -*-
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vgsan.disen_vgsan_model import DisenVGSAN
from models.vgsan.dgl_stream import dgl_streeam
from models.vgsan import config
from models.segfedgnn.sdss import config as sdss_config
from models.vgsan.vgsan_model import VGSAN
from models.sasrec.sasrec_model import SASRec
from models.vsan.vsan_model import VSAN
from models.contrastvae.contrastvae_model import ContrastVAE
from models.cl4srec.cl4srec_model import CL4SRec
from models.duorec.duorec_model import DuoRec
from models.segfedgnn.dual_stream_model import SegFedGNN
from models.segfedgnn.domain_hyper import HyperHead, DomainGNN
from utils import train_utils
from losses import NCELoss, HingeLoss, JSDLoss, Discriminator, priorKL


class Trainer(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def train_batch(self, *args, **kwargs):
        raise NotImplementedError

    def test_batch(self, *args, **kwargs):
        raise NotImplementedError

    def update_lr(self, new_lr):
        train_utils.change_lr(self.optimizer, new_lr)


class ModelTrainer(Trainer):
    def __init__(self, args, num_items, max_seq_len):
        self.args = args
        self.method = args.method
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"
        self.summary_graph_payload = None
        self.local_summary_gnn = None
        # Disable anomaly detection for better performance
        torch.autograd.set_detect_anomaly(False)
        if self.method == "FedDCSR":
            self.model = DisenVGSAN(num_items, args).to(self.device)
            # Here we set `self.z_s[:], self.z_g = [None], [None]` so that
            # we can use `self.z_s[:] = ...`, `self.z_g[:] = ...` to modify
            # them later.
            # Note that if we set `self.z_s, self.z_g = None, None`,
            # then `self.z_s = obj` / `self.z_g = obj` will just refer to a
            # new object `obj`, rather than modify `self.z_s` / `self.z_g`
            # itself
            self.z_s, self.z_g = [None], [None]
            self.discri = Discriminator(
                config.hidden_size, max_seq_len).to(self.device)
        elif "VGSAN" in self.method:
            self.model = VGSAN(num_items, args).to(self.device)
        elif "VSAN" in self.method:
            self.model = VSAN(num_items, args).to(self.device)
        elif "SASRec" in self.method:
            self.model = SASRec(num_items, args).to(self.device)
        elif "CL4SRec" in self.method:
            self.model = CL4SRec(num_items, args).to(self.device)
        elif "DuoRec" in self.method:
            self.model = DuoRec(num_items, args).to(self.device)
        elif "ContrastVAE" in self.method:
            self.model = ContrastVAE(num_items, args).to(self.device)
        elif "DisenProto" in self.method or self.method == "dgl_streeam":
            # dgl_streeam: VAE-free + Gated Fusion + Contrastive model
            self.model = dgl_streeam(num_items, args).to(self.device)
        elif "SegFedGNN" in self.method:
            # SegFedGNN: upload summary_vec; server builds graph payload for local GNN.
            self.model = SegFedGNN(num_items, args).to(self.device)
            # Buffer to store summary_vec (for server transmission)
            self.last_summary_vec = None
            self.hyper_head = None
            if not getattr(args, "disable_hyperhead", False):
                # HyperHead for domain adaptation (based on summary_vec)
                # summary_vec has hidden_size dimensions
                self.hyper_in_dim = config.hidden_size
                self.hyper_head = HyperHead(
                    dom_emb_dim=self.hyper_in_dim,
                    hidden_dim=config.hidden_size,
                    num_items=num_items,
                    rank=getattr(args, "hyper_rank", 4),
                ).to(self.device)
            if self.hyper_head is not None:
                seg_hidden = getattr(args, "domain_gnn_hidden", config.hidden_size)
                seg_layers = getattr(args, "domain_gnn_layers", 2)
                self.local_summary_gnn = DomainGNN(
                    config.hidden_size, seg_hidden, config.hidden_size, seg_layers
                ).to(self.device)

        self.bce_criterion = nn.BCEWithLogitsLoss(
            reduction="none").to(self.device)
        self.cs_criterion = nn.CrossEntropyLoss(
            reduction="none").to(self.device)
        self.cl_criterion = NCELoss(
            temperature=args.temperature).to(self.device)
        self.jsd_criterion = JSDLoss().to(self.device)
        self.hinge_criterion = HingeLoss(margin=0.3).to(self.device)
        # Prototype buffer for SDSS segment representations
        self.segment_proto = None

        if args.method == "FedDCSR":
            self.params = list(self.model.parameters()) + \
                list(self.discri.parameters())
        elif self.method == "SegFedGNN":
            self.params = list(self.model.parameters())
            if getattr(self, "hyper_head", None) is not None:
                self.params += list(self.hyper_head.parameters())
            if getattr(self, "local_summary_gnn", None) is not None:
                self.params += list(self.local_summary_gnn.parameters())
        else:
            self.params = list(self.model.parameters())
        self.optimizer = train_utils.get_optimizer(
            args.optimizer, self.params, args.lr)
        self.step = 0

    def set_summary_graph_payload(self, payload):
        """Set server summary-graph payload for local DomainGNN mode."""
        if payload is None:
            self.summary_graph_payload = None
            return
        self.summary_graph_payload = {
            "V": payload.get("V", None),
            "A": payload.get("A", None),
            "idx": int(payload.get("idx", -1)),
        }

    def _resolve_hyper_domain_vec(self):
        """Resolve g_d for HyperHead via local DomainGNN payload."""
        if getattr(self, "local_summary_gnn", None) is not None:
            payload = getattr(self, "summary_graph_payload", None)
            if payload is not None:
                V = payload.get("V", None)
                idx = payload.get("idx", -1)
                if V is not None and 0 <= idx < V.size(0):
                    V = V.to(self.device)
                    A = payload.get("A", None)
                    if A is None:
                        A = torch.eye(V.size(0), device=self.device)
                    else:
                        A = A.to(self.device)
                    G = self.local_summary_gnn(V, A)
                    return G[idx]
            return None
        return None

    def train_batch(self, sessions, adj, num_items, args, global_params=None):
        """Trains the model for one batch.

        Args:
            sessions: Input user sequences.
            adj: Adjacency matrix of the local graph.
            num_items: Number of items in the current domain.
            args: Other arguments for training.
            global_params: Global model parameters used in `FedProx` method.
        """
        self.optimizer.zero_grad()

        if (self.method == "FedDCSR") or ("VGSAN" in self.method) or ("SegFedGNN" in self.method) or ("DisenProto" in self.method) or (self.method == "dgl_streeam"):
            # Here the items are first sent to GNN for convolution, and then
            # the resulting embeddings are sent to the self-attention module.
            # Note that each batch must be convolved once, and the
            # item_embeddings input to the convolution layer are updated from
            # the previous batch.
            if self.method == "SegFedGNN":
                self.model.disen.graph_convolution(adj)
            elif self.method == "FedDCSR":
                self.model.graph_convolution(adj)
            elif "VGSAN" in self.method:
                self.model.graph_convolution(adj)
            elif "DisenProto" in self.method or self.method == "dgl_streeam":
                self.model.graph_convolution(adj)

        sessions = [torch.LongTensor(x).to(self.device) for x in sessions]

        if self.method == "FedDCSR":
            # seq: (batch_size, seq_len), ground: (batch_size, seq_len),
            # ground_mask:  (batch_size, seq_len),
            # js_neg_seqs: (batch_size, seq_len),
            # contrast_aug_seqs: (batch_size, seq_len)
            # Here `js_neg_seqs` is used for computing similarity loss,
            # `contrast_aug_seqs` is used for computing contrastive infomax
            # loss
            seq, ground, ground_mask, js_neg_seqs, contrast_aug_seqs = sessions
            result, result_exclusive, mu_s, logvar_s, self.z_s[0], mu_e, \
                logvar_e, z_e, neg_z_e, aug_z_e = self.model(
                    seq,
                neg_seqs=js_neg_seqs,
                aug_seqs=contrast_aug_seqs)
            # Broadcast in last dim. it well be used to compute `z_g` by
            # federated aggregation later
            self.z_s[0] *= ground_mask.unsqueeze(-1)
            loss = self.disen_vgsan_loss_fn(result, result_exclusive, mu_s,
                                            logvar_s,  mu_e, logvar_e,
                                            ground, self.z_s[0], self.z_g[0],
                                            z_e, neg_z_e, aug_z_e, ground_mask,
                                            num_items, self.step)

        elif "VGSAN" in self.method:
            seq, ground, ground_mask = sessions
            result, mu, logvar = self.model(seq)
            loss = self.vgsan_loss_fn(
                result, mu, logvar, ground, ground_mask, num_items, self.step)
            if "Fed" in self.method and args.mu:
                loss += self.prox_reg(
                    [dict(self.model.encoder.named_parameters())],
                    global_params, args.mu)

        elif "VSAN" in self.method:
            seq, ground, ground_mask = sessions
            result, mu, logvar = self.model(seq)
            loss = self.vsan_loss_fn(
                result, mu, logvar, ground, ground_mask, num_items, self.step)
            if self.method == "FedVSAN" and args.mu:
                loss += self.prox_reg(
                    [dict(self.model.encoder.named_parameters())],
                    global_params, args.mu)

        elif "DisenProto" in self.method or self.method == "dgl_streeam":
            # dgl_streeam: VAE-free + Gated Fusion + Contrastive model
            seq, ground, ground_mask, js_neg_seqs, contrast_aug_seqs = sessions
            model_out = self.model(seq, neg_seqs=js_neg_seqs, aug_seqs=contrast_aug_seqs)
            # Unpack outputs
            (result, result_exclusive, z_s, z_e, aug_z_e,
             z_e_proj, aug_z_e_proj, gate, mu_s, logvar_s, mu_e, logvar_e) = model_out

            loss = self.dgl_stream_loss_fn(
                result, result_exclusive, z_s, z_e, aug_z_e,
                z_e_proj, aug_z_e_proj, gate, mu_s, logvar_s, mu_e, logvar_e,
                ground, ground_mask, num_items, self.step)

            if "Fed" in self.method and args.mu:
                loss += self.prox_reg(
                    [dict(self.model.encoder_g.named_parameters())],
                    global_params, args.mu)

        elif "SegFedGNN" in self.method:
            # SegFedGNN: server payload -> local GNN -> HyperHead.
            seq, ground, ground_mask = sessions
            self.last_summary_vec = None

            model_out = self.model(seq, neg_seqs=None, aug_seqs=None)
            if len(model_out) == 5:
                result, z_s, z_e, z, aux = model_out
            else:
                result, z_s, z_e, z = model_out
                aux = None

            if aux and aux.get("summary_vec") is not None:
                # Generate domain representative vector via batch mean
                self.last_summary_vec = aux["summary_vec"].mean(dim=0).detach().cpu()

            # Apply HyperHead: use global_summary_vec (GNN-processed result from server)
            if self.hyper_head is not None and z is not None:
                g_d = self._resolve_hyper_domain_vec()
                if g_d is not None:
                    delta_logits = self.hyper_head.compute_delta_logits(z, g_d)  # (B, T, N)
                    base_logits = result[:, :, :-1]
                    base_logits = base_logits + delta_logits
                    result = torch.cat([base_logits, result[:, :, -1:].clone()], dim=-1)

            aux_data = {
                'seq_mask': (seq == num_items),
                'local_boundary_probs': None,
                'global_boundary_probs': None
            }
            loss = self.sdsssasrec_loss_fn(result, ground, ground_mask,
                                           num_items, aux_data)

            # Optional extra losses from DGL-Stream
            if aux:
                if self.step % 100 == 0 and aux.get("gate_mean") is not None:
                    logging.info(
                        "SegFedGNN Gate (step %d): mean=%.4f min=%.4f max=%.4f",
                        self.step, aux["gate_mean"], aux["gate_min"], aux["gate_max"])
                if self.step % 100 == 0 and aux.get("alpha_gate_mean") is not None:
                    logging.info(
                        "SegFedGNN Alpha Gate (step %d): mean=%.4f min=%.4f max=%.4f",
                        self.step, aux["alpha_gate_mean"], aux["alpha_gate_min"], aux["alpha_gate_max"])

                if getattr(args, "sdss_dual_exclusive_weight", 0) > 0 and \
                        aux.get("result_exclusive") is not None:
                    recons_loss_exclusive = self.cs_criterion(
                        aux["result_exclusive"].reshape(-1, num_items + 1),
                        ground.reshape(-1))
                    recons_loss_exclusive = (
                        recons_loss_exclusive * (ground_mask.reshape(-1))).mean()
                    loss = loss + args.sdss_dual_exclusive_weight * recons_loss_exclusive

                # SDSS Summarizer auxiliary losses
                sdss_summarizer = getattr(self.model, "sdss_summarizer", None)
                if sdss_summarizer is not None:
                    # L_boundary: Boundary KL loss
                    boundary_weight = getattr(args, "sdss_aux_boundary_weight", None)
                    if boundary_weight is None:
                        boundary_weight = sdss_config.beta_boundary
                    if boundary_weight > 0 and aux.get("sdss_boundary_probs") is not None:
                        boundary_loss = sdss_summarizer.compute_boundary_loss(
                            aux["sdss_boundary_probs"], aux["sdss_seq_mask"])
                        loss = loss + boundary_weight * boundary_loss

                    # L_compact: Compactness loss
                    compact_weight = getattr(args, "sdss_aux_compact_weight", None)
                    if compact_weight is None:
                        compact_weight = sdss_config.gamma_compact
                    if compact_weight > 0 and aux.get("sdss_h_tilde") is not None:
                        compact_loss = sdss_summarizer.compute_compactness_loss(
                            aux["sdss_h_tilde"], aux["segment_reps"],
                            aux["sdss_segment_indices"], aux["sdss_seq_mask"])
                        loss = loss + compact_weight * compact_loss

                    # L_recon: Reconstruction loss
                    recon_weight = getattr(args, "sdss_aux_recon_weight", None)
                    if recon_weight is None:
                        recon_weight = sdss_config.alpha_recon
                    if recon_weight > 0 and aux.get("sdss_h_original") is not None:
                        recon_loss = sdss_summarizer.compute_reconstruction_loss(
                            aux["sdss_h_original"], aux["sdss_h_tilde"],
                            aux["segment_reps"], aux["sdss_segment_indices"],
                            aux["sdss_seq_mask"])
                        loss = loss + recon_weight * recon_loss

            # FedProx regularization
            if args.mu:
                if getattr(self.model, "use_sdss_branch", True) and \
                        getattr(self.model, "branch_type", "sdss") == "sdss":
                    loss += self.prox_reg(
                        [dict(self.model.disen.encoder_s.named_parameters()),
                         dict(self.model.segment_encoder.named_parameters())],
                        global_params, args.mu)
                else:
                    loss += self.prox_reg(
                        [dict(self.model.disen.encoder_s.named_parameters())],
                        global_params, args.mu)

        elif "SASRec" in self.method:
            seq, ground, ground_mask = sessions
            # result： (batch_size, seq_len, hidden_size)
            result = self.model(seq)
            loss = self.sasrec_loss_fn(result, ground, ground_mask, num_items)
            if "Fed" in self.method and args.mu:
                loss += self.prox_reg(
                    [dict(self.model.encoder.named_parameters())],
                    global_params, args.mu)

        elif "CL4SRec" in self.method:
            seq, ground, ground_mask, aug_seqs1, aug_seqs2 = sessions
            # result： (batch_size, seq_len, hidden_size)
            result, aug_seqs_fea1, aug_seqs_fea2 = self.model(
                seq, aug_seqs1=aug_seqs1, aug_seqs2=aug_seqs2)
            loss = self.cl4srec_loss_fn(
                result, aug_seqs_fea1, aug_seqs_fea2, ground, ground_mask,
                num_items)
            if "Fed" in self.method and args.mu:
                loss += self.prox_reg(
                    [dict(self.model.encoder.named_parameters())],
                    global_params, args.mu)

        elif "DuoRec" in self.method:
            seq, ground, ground_mask, aug_seqs = sessions
            # result： (batch_size, seq_len, hidden_size)
            result, seqs_fea, aug_seqs_fea = self.model(seq, aug_seqs=aug_seqs)
            loss = self.duorec_loss_fn(
                result, seqs_fea, aug_seqs_fea, ground, ground_mask, num_items)
            if "Fed" in self.method and args.mu:
                loss += self.prox_reg(
                    [dict(self.model.encoder.named_parameters())],
                    global_params, args.mu)

        elif "ContrastVAE" in self.method:
            seq, ground, ground_mask, aug_seqs = sessions
            # result： (batch_size, seq_len, hidden_size)
            result, aug_result, mu, logvar, z, aug_mu, aug_logvar, aug_z, \
                alpha = self.model(seq, aug_seqs=aug_seqs)
            loss = self.contrastvae_loss_fn(result, aug_result, mu, logvar, z,
                                            aug_mu, aug_logvar, aug_z, alpha,
                                            ground, ground_mask, num_items,
                                            self.step)
            if "Fed" in self.method and args.mu:
                loss += self.prox_reg(
                    [dict(self.model.encoder.named_parameters())],
                    global_params, args.mu)

        loss.backward()
        self.optimizer.step()
        self.step += 1
        return loss.item()

    def disen_vgsan_loss_fn(self, result, result_exclusive, mu_s, logvar_s,
                            mu_e, logvar_e, ground, z_s, z_g, z_e, neg_z_e,
                            aug_z_e, ground_mask, num_items, step):
        """Overall loss function of FedDCSR (our method).
        """
        disable_sim = getattr(self.args, "dcsr_disable_sim", False)
        disable_exclusive = getattr(self.args, "dcsr_disable_exclusive", False)
        disable_contrastive = getattr(self.args, "dcsr_disable_contrastive", False)
        disable_kld = getattr(self.args, "dcsr_disable_kld", False)

        def sim_loss_fn(self, z_s, z_g, neg_z_e, ground_mask):
            pos = self.discri(z_s, z_g, ground_mask)
            neg = self.discri(neg_z_e, z_g, ground_mask)

            # pos_label, neg_label = torch.ones(pos.size()).to(self.device), \
            #     torch.zeros(neg.size()).to(self.device)
            # sim_loss = self.bce_criterion(pos, pos_label) \
            #     + self.bce_criterion(neg, neg_label)

            # sim_loss = self.jsd_criterion(pos, neg)
            sim_loss = self.hinge_criterion(pos, neg)

            sim_loss = sim_loss.mean()

            return sim_loss

        recons_loss = self.cs_criterion(
            result.reshape(-1, num_items + 1),
            ground.reshape(-1))  # (batch_size * seq_len, )
        recons_loss = (recons_loss * (ground_mask.reshape(-1))).mean()

        recons_loss_exclusive = self.cs_criterion(
            result_exclusive.reshape(-1, num_items + 1),
            ground.reshape(-1))  # (batch_size * seq_len, )
        recons_loss_exclusive = (
            recons_loss_exclusive * (ground_mask.reshape(-1))).mean()

        kld_loss_s = -0.5 * \
            torch.sum(1 + logvar_s - mu_s ** 2 -
                      logvar_s.exp(), dim=-1).reshape(-1)
        kld_loss_s = (kld_loss_s * (ground_mask.reshape(-1))).mean()

        kld_loss_e = -0.5 * \
            torch.sum(1 + logvar_e - mu_e ** 2 -
                      logvar_e.exp(), dim=-1).reshape(-1)
        kld_loss_e = (kld_loss_e * (ground_mask.reshape(-1))).mean()

        # If it is the first training round
        if (z_g is not None) and (not disable_sim):
            sim_loss = sim_loss_fn(self, z_s, z_g, neg_z_e, ground_mask)
        else:
            sim_loss = 0

        alpha = 1.0  # 1.0 for all scenarios

        kld_weight = 0 if disable_kld else self.kl_anneal_function(
            self.args.anneal_cap, step, self.args.total_annealing_step)

        beta = 0 if disable_sim else 2.0  # 2.0 for FKCB, 0.5 for BMG and SGH

        gamma = 0 if disable_exclusive else 1.0  # 1.0 for all scenarios

        lam = 0 if disable_contrastive else 1.0  # 1.0 for FKCB and BMG, 0.1 for SGH

        user_representation1 = z_e[:, -1, :]
        user_representation2 = aug_z_e[:, -1, :]
        contrastive_loss = self.cl_criterion(
            user_representation1, user_representation2)
        contrastive_loss = contrastive_loss.mean()

        loss = alpha * (recons_loss + kld_weight * kld_loss_s + kld_weight
                        * kld_loss_e) \
            + beta * sim_loss \
            + gamma * recons_loss_exclusive \
            + lam * contrastive_loss

        return loss

    def vgsan_loss_fn(self, result, mu, logvar, ground, ground_mask, num_items,
                      step):
        """Compute kl divergence, reconstruction.
        result: (batch_size, seq_len, hidden_size),
        mu: (batch_size, seq_len, hidden_size),
        log_var: (batch_size, seq_len, hidden_size),
        ground: (batch_size, seq_len)
        ground_mask: (batch_size, seq_len)
        """
        recons_loss = self.cs_criterion(
            result.reshape(-1, num_items + 1),
            ground.reshape(-1))  # (batch_size * seq_len, )
        recons_loss = (recons_loss * (ground_mask.reshape(-1))).mean()

        kld_loss = -0.5 * torch.sum(1 + logvar -
                                    mu ** 2 - logvar.exp(), dim=-1).reshape(-1)
        kld_loss = (kld_loss * (ground_mask.reshape(-1))).mean()

        kld_weight = self.kl_anneal_function(
            self.args.anneal_cap, step, self.args.total_annealing_step)

        loss = recons_loss + kld_weight * kld_loss
        return loss

    def vsan_loss_fn(self, result, mu, logvar, ground, ground_mask, num_items,
                     step):
        """Compute kl divergence, reconstruction.
        result: (batch_size, seq_len, hidden_size),
        mu: (batch_size, seq_len, hidden_size),
        log_var: (batch_size, seq_len, hidden_size),
        ground: (batch_size, seq_len),
        ground_mask: (batch_size, seq_len)
        """
        recons_loss = self.cs_criterion(
            result.reshape(-1, num_items + 1),
            ground.reshape(-1))  # (batch_size * seq_len, )
        recons_loss = (recons_loss * (ground_mask.reshape(-1))).mean()

        kld_loss = -0.5 * torch.sum(1 + logvar -
                                    mu ** 2 - logvar.exp(), dim=-1).reshape(-1)
        kld_loss = (kld_loss * (ground_mask.reshape(-1))).mean()

        kld_weight = self.kl_anneal_function(
            self.args.anneal_cap, step, self.args.total_annealing_step)

        loss = recons_loss + kld_weight * kld_loss
        return loss

    def dgl_stream_loss_fn(self, result, result_exclusive, z_s, z_e, aug_z_e,
                           z_e_proj, aug_z_e_proj,
                           gate, mu_s, logvar_s, mu_e, logvar_e,
                           ground, ground_mask, num_items, step):
        """Loss function for dgl_streeam (VAE-free + Contrastive).

        Components:
        1. Reconstruction loss (main prediction)
        2. Exclusive reconstruction loss (z_e only)
        3. InfoNCE contrastive loss (z_e vs aug_z_e)
        4. Optional KL loss (if using VAE encoder)
        """
        # 1. Main reconstruction loss
        recons_loss = self.cs_criterion(
            result.reshape(-1, num_items + 1),
            ground.reshape(-1))
        recons_loss = (recons_loss * ground_mask.reshape(-1)).mean()

        # 2. Exclusive reconstruction loss
        recons_loss_exclusive = self.cs_criterion(
            result_exclusive.reshape(-1, num_items + 1),
            ground.reshape(-1))
        recons_loss_exclusive = (recons_loss_exclusive * ground_mask.reshape(-1)).mean()

        # 3. InfoNCE contrastive loss
        contrastive_loss = torch.tensor(0.0, device=result.device)
        if z_e_proj is not None and aug_z_e_proj is not None:
            # Normalize for cosine similarity
            z_e_proj_norm = F.normalize(z_e_proj, p=2, dim=-1)
            aug_z_e_proj_norm = F.normalize(aug_z_e_proj, p=2, dim=-1)
            contrastive_loss = self.cl_criterion(z_e_proj_norm, aug_z_e_proj_norm)
            contrastive_loss = contrastive_loss.mean()

        # 4. Optional KL loss (only if using VAE encoder)
        kld_loss = torch.tensor(0.0, device=result.device)
        if mu_s is not None and logvar_s is not None:
            kld_loss_s = -0.5 * torch.sum(
                1 + logvar_s - mu_s ** 2 - logvar_s.exp(), dim=-1).reshape(-1)
            kld_loss_s = (kld_loss_s * ground_mask.reshape(-1)).mean()

            kld_loss_e = -0.5 * torch.sum(
                1 + logvar_e - mu_e ** 2 - logvar_e.exp(), dim=-1).reshape(-1)
            kld_loss_e = (kld_loss_e * ground_mask.reshape(-1)).mean()

            kld_weight = self.kl_anneal_function(
                self.args.anneal_cap, step, self.args.total_annealing_step)
            kld_loss = kld_weight * (kld_loss_s + kld_loss_e)

        # Loss weights
        alpha = 1.0  # main reconstruction
        gamma = getattr(self.args, 'sdss_dual_exclusive_weight', 1.0)  # exclusive
        lam = 0.1  # contrastive (fixed)

        loss = (alpha * recons_loss +
                gamma * recons_loss_exclusive +
                lam * contrastive_loss +
                kld_loss)

        return loss

    def sasrec_loss_fn(self, result, ground, ground_mask, num_items):
        """Compute cross entropy loss for next item prediction.
        result: (batch_size, seq_len, hidden_size),
        ground: (batch_size, seq_len),
        ground_mask: (batch_size, seq_len)
        """
        loss = self.cs_criterion(
            result.reshape(-1, num_items + 1),
            ground.reshape(-1))  # （batch_size * seq_len, ）
        loss = 1.0 * (loss * (ground_mask.reshape(-1))).mean()
        return loss

    def duorec_loss_fn(self, result, seqs_fea, aug_seqs_fea, ground,
                       ground_mask, num_items):
        recons_loss = self.cs_criterion(
            result.reshape(-1, num_items + 1),
            ground.reshape(-1))  # (batch_size * seq_len, )
        recons_loss = (recons_loss * (ground_mask.reshape(-1))).mean()

        user_representation1 = seqs_fea[:, -1, :]
        user_representation2 = aug_seqs_fea[:, -1, :]
        contrastive_loss = self.cl_criterion(
            user_representation1, user_representation2)
        contrastive_loss = contrastive_loss.mean()

        lam = 0.1  # 0.1 is the best
        loss = recons_loss + lam * contrastive_loss
        return loss

    def cl4srec_loss_fn(self, result, aug_seqs_fea1, aug_seqs_fea2, ground,
                        ground_mask, num_items):
        recons_loss = self.cs_criterion(
            result.reshape(-1, num_items + 1),
            ground.reshape(-1))  # (batch_size * seq_len, )
        recons_loss = (recons_loss * (ground_mask.reshape(-1))).mean()

        user_representation1 = aug_seqs_fea1[:, -1, :]
        user_representation2 = aug_seqs_fea2[:, -1, :]
        contrastive_loss = self.cl_criterion(
            user_representation1, user_representation2)
        contrastive_loss = contrastive_loss.mean()

        lam = 0.1  # 0.1 is the best
        loss = recons_loss + lam * contrastive_loss
        return loss

    def contrastvae_loss_fn(self, result, aug_result, mu, logvar, z, aug_mu,
                            aug_logvar, aug_z, alpha, ground, ground_mask,
                            num_items, step):
        recons_loss = self.cs_criterion(
            result.reshape(-1, num_items + 1),
            ground.reshape(-1))  # (batch_size * seq_len, )
        recons_loss = (recons_loss * (ground_mask.reshape(-1))).mean()
        aug_recons_loss = self.cs_criterion(
            aug_result.reshape(-1, num_items + 1),
            ground.reshape(-1))  # (batch_size * seq_len, )
        aug_recons_loss = (aug_recons_loss * (ground_mask.reshape(-1))).mean()

        kld_loss = -0.5 * torch.sum(1 + logvar -
                                    mu ** 2 - logvar.exp(), dim=-1).reshape(-1)
        kld_loss = (kld_loss * (ground_mask.reshape(-1))).mean()
        aug_kld_loss = -0.5 * \
            torch.sum(1 + aug_logvar - aug_mu ** 2 -
                      aug_logvar.exp(), dim=-1).reshape(-1)
        aug_kld_loss = (aug_kld_loss * (ground_mask.reshape(-1))).mean()

        kld_weight = self.kl_anneal_function(
            self.args.anneal_cap, step, self.args.total_annealing_step)

        lam = 1.0  # 1.0 is the best

        mask = ground_mask.float().sum(-1).unsqueeze(-1)\
            .repeat(1, ground_mask.size(-1))
        mask = 1 / mask
        mask = ground_mask * mask  # For mean
        mask = ground_mask.unsqueeze(-1).repeat(1, 1, z.size(-1))
        # user representation1: (batch_size, hidden_size)
        # user_representation2: (batch_size, hidden_size)
        user_representation1 = (z * mask).sum(1)
        user_representation2 = (aug_z * mask).sum(1)

        contrastive_loss = self.cl_criterion(
            user_representation1, user_representation2)
        contrastive_loss = contrastive_loss.mean()

        loss = recons_loss + aug_recons_loss + kld_weight * (kld_loss
                                                             + aug_kld_loss) \
            + lam * contrastive_loss
        # Compute priorKL loss
        if alpha:
            adaptive_alpha_loss = priorKL(alpha).mean()
            loss += adaptive_alpha_loss
        return loss

    def sdsssasrec_loss_fn(self, result, ground, ground_mask, num_items, aux_data):
        """Compute loss for SDSS-based models (SegFedGNN).

        result: (batch_size, K, num_items+1) - prediction for each segment
        ground: (batch_size, seq_len) - ground truth items
        ground_mask: (batch_size, seq_len) - mask for valid positions
        num_items: number of items in current domain
        aux_data: auxiliary data for computing SDSS losses
        """
        batch_size, num_segments, _ = result.size()
        seq_len = ground.size(1)

        # Create segment-level ground truth by taking last item of each segment
        segment_size = seq_len // num_segments
        segment_ground = []
        segment_mask = []

        for k in range(num_segments):
            if k == num_segments - 1:
                end_idx = seq_len
            else:
                end_idx = (k + 1) * segment_size
            # Take the last item in each segment as target
            segment_ground.append(ground[:, end_idx - 1])
            segment_mask.append(ground_mask[:, end_idx - 1])

        segment_ground = torch.stack(segment_ground, dim=1)  # (B, K)
        segment_mask = torch.stack(segment_mask, dim=1)  # (B, K)

        # Compute loss over all segments (like FedDCSR)
        recons_loss = self.cs_criterion(
            result.reshape(-1, num_items + 1),  # (B*K, num_items+1)
            segment_ground.reshape(-1))  # (B*K,)
        recons_loss = (recons_loss * segment_mask.reshape(-1)).mean()

        # Auxiliary losses from SDSS (skip if unsupported or not provided)
        aux_loss = 0.0
        if hasattr(self.model, "compute_auxiliary_losses"):
            has_local = aux_data.get("local_boundary_probs") is not None if aux_data else False
            has_single = aux_data.get("boundary_probs") is not None if aux_data else False
            if aux_data is not None and (has_local or has_single):
                try:
                    aux_loss, aux_loss_dict = self.model.compute_auxiliary_losses(
                        aux_data, getattr(self, "args", None))
                except TypeError:
                    aux_loss, aux_loss_dict = self.model.compute_auxiliary_losses(aux_data)
        # Total loss
        loss = recons_loss + aux_loss

        return loss

    def kl_anneal_function(self, anneal_cap, step, total_annealing_step):
        """
        step: increment by 1 for every forward-backward step.
        total annealing steps: pre-fixed parameter control the speed of
        anealing.
        """
        # borrows from https://github.com/timbmg/Sentence-VAE/blob/master/train.py
        return min(anneal_cap, step / total_annealing_step)

    @ staticmethod
    def flatten(source):
        return torch.cat([value.flatten() for value in source])

    def prox_reg(self, params1, params2, mu):
        params1_values, params2_values = [], []
        # Record the model parameter aggregation results of each branch
        # separately
        for branch_params1, branch_params2 in zip(params1, params2):
            branch_params2 = [branch_params2[key]
                              for key in branch_params1.keys()]
            params1_values.extend(branch_params1.values())
            params2_values.extend(branch_params2)

        # Multidimensional parameters should be compressed into one dimension
        # using the flatten function
        s1 = self.flatten(params1_values)
        s2 = self.flatten(params2_values)
        return mu/2 * torch.norm(s1 - s2)

    def test_batch(self, sessions):
        """Tests the model for one batch.

        Args:
            sessions: Input user sequences.

        Returns:
            List of dicts with 'rank', 'auc', 'num_negs' for each sample.
        """
        sessions = [torch.LongTensor(x).to(self.device) for x in sessions]

        # seq: (batch_size, seq_len), ground_truth: (batch_size, ),
        # neg_list: (batch_size, num_test_neg)
        seq, ground_truth, neg_list = sessions

        pred = []

        # Special handling for SegFedGNN variants that return segment logits
        if "DisenProto" in self.method or self.method == "dgl_streeam":
            # dgl_streeam returns just logits in eval mode
            result = self.model(seq)
            logits = result[:, -1]  # (batch_size, num_items)
        elif self.method == "SegFedGNN":
            # Models return tuple (result, z_s, z_e, z); take result
            result_tuple = self.model(seq, neg_seqs=None, aug_seqs=None)
            logits_full = result_tuple[0]
            # Apply HyperHead: use global_summary_vec
            if len(result_tuple) >= 4:
                z_sum = result_tuple[3]
                if getattr(self, "hyper_head", None) is not None:
                    g_d = self._resolve_hyper_domain_vec()
                    if z_sum is not None and g_d is not None:
                        delta_logits = self.hyper_head.compute_delta_logits(z_sum, g_d)
                        logits_full = logits_full.clone()
                        logits_full[:, :, :-1] += delta_logits
            if logits_full.dim() == 3:
                # (B, K/T, num_items+1), use last position
                logits = logits_full[:, -1, :]
            else:
                logits = logits_full  # fallback
        else:
            # result: (batch_size, seq_len, num_items)
            result = self.model(seq)
            logits = result[:, -1]  # (batch_size, num_items)

        # Compute rank and AUC for each sample
        for id in range(len(logits)):
            score = logits[id]
            pos_score = score[ground_truth[id]].item()
            neg_scores = score[neg_list[id]].data.cpu().numpy()
            num_negs = len(neg_scores)

            # Rank: number of negatives with higher score + 1
            score_larger = (neg_scores > pos_score).sum()
            true_item_rank = score_larger + 1

            # AUC: proportion of negatives with lower score than positive
            # AUC = (num_negs - score_larger) / num_negs
            auc = (num_negs - score_larger) / num_negs if num_negs > 0 else 0.0

            pred.append({
                'rank': int(true_item_rank),
                'auc': float(auc),
                'num_negs': num_negs
            })

        return pred
