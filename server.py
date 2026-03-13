# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn.functional as F


class Server(object):
    def __init__(self, args, init_global_params):
        self.args = args
        self.global_params = init_global_params
        if args.method == "FedDCSR":
            self.global_reps = None
        # SegFedGNN: server builds graph payload (V, A, idx) for local GNN.
        if args.method == "SegFedGNN":
            self.summary_vecs = None
            self.summary_adj = None
            self._summary_vec_cid_map = {}

    def aggregate_params(self, clients, random_cids):
        """Sums up parameters of models shared by all active clients at each
        epoch.

        Args:
            clients: A list of clients instances.
            random_cids: Randomly selected client ID in each training round.
        """
        # Record the model parameter aggregation results of each branch
        # separately
        num_branchs = len(self.global_params)
        for branch_idx in range(num_branchs):
            client_params_sum = None
            for c_id in random_cids:
                # Obtain current client's parameters
                current_client_params = clients[c_id].get_params()[branch_idx]
                # Sum it up with weights
                if client_params_sum is None:
                    client_params_sum = dict((key, value
                                              * clients[c_id].train_weight)
                                             for key, value
                                             in current_client_params.items())
                else:
                    for key in client_params_sum.keys():
                        client_params_sum[key] += clients[c_id].train_weight \
                            * current_client_params[key]
            self.global_params[branch_idx] = client_params_sum

    def aggregate_reps(self, clients, random_cids):
        """Sums up representations of user sequences shared by all active
        clients at each epoch.

        Args:
            clients: A list of clients instances.
            random_cids: Randomly selected client ID in each training round.
        """
        # Record the user sequence aggregation results of each branch
        # separately
        client_reps_sum = None
        for c_id in random_cids:
            # Obtain current client's user sequence representations
            current_client_reps = clients[c_id].get_reps_shared()
            # Sum it up with weights
            if client_reps_sum is None:
                client_reps_sum = current_client_reps * \
                    clients[c_id].train_weight
            else:
                client_reps_sum += clients[c_id].train_weight * \
                    current_client_reps
        self.global_reps = client_reps_sum

    def choose_clients(self, n_clients, ratio=1.0):
        """Randomly chooses some clients.
        """
        choose_num = math.ceil(n_clients * ratio)
        return np.random.permutation(n_clients)[:choose_num]

    def get_global_params(self):
        """Returns a reference to the parameters of the global model.
        """
        return self.global_params

    def get_global_reps(self):
        """Returns a reference to the parameters of the global model.
        """
        return self.global_reps

    # -------- SegFedGNN: summary graph payload -------- #
    def aggregate_summary_vecs(self, clients, random_cids):
        """Aggregate summary vectors and build graph payload for local GNN."""
        vec_list = []
        valid_cids = []
        for c_id in random_cids:
            vec = clients[c_id].get_summary_vec()
            if vec is not None:
                vec_list.append(vec)
                valid_cids.append(c_id)

        if not vec_list:
            self.summary_vecs = None
            self.summary_adj = None
            self._summary_vec_cid_map = {}
            return

        # Stack all summary_vecs: [num_clients, hidden_dim]
        V = torch.stack(vec_list, dim=0)
        self.summary_vecs = V.cpu()
        A = self._build_adj(V, k=getattr(self.args, "domain_knn_k", 2))
        self.summary_adj = A.cpu() if A is not None else None
        self._summary_vec_cid_map = {cid: i for i, cid in enumerate(valid_cids)}

    def get_summary_graph_payload(self, c_id):
        """Return payload (V, A, idx) for local DomainGNN execution on clients."""
        if self.summary_vecs is None:
            return None
        idx = getattr(self, '_summary_vec_cid_map', {}).get(c_id, None)
        if idx is None or idx >= self.summary_vecs.size(0):
            return None
        return {
            "V": self.summary_vecs,
            "A": self.summary_adj,
            "idx": idx,
        }

    # -------- Domain hypernetwork utilities -------- #
    def _build_adj(self, V, k=2):
        """Build cosine k-NN adjacency, row-normalized."""
        if V is None or V.size(0) == 0:
            return None
        norm = F.normalize(V, p=2, dim=-1)
        sim = torch.matmul(norm, norm.t())  # [D, D]
        D = sim.size(0)
        topk = torch.topk(sim, k=min(k + 1, D), dim=-1).indices  # include self
        rows = []
        for i in range(D):
            idx = topk[i]
            rows.append(F.one_hot(idx, num_classes=D).sum(dim=0))
        A = torch.stack(rows, dim=0).float()
        # remove self-loop overcount if needed
        A = A / A.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return A
