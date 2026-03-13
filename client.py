# -*- coding: utf-8 -*-
import os
import gc
import copy
import logging
import numpy as np
import torch
from dataloader import SeqDataloader
from utils.io_utils import ensure_dir


class Client:
    def __init__(self, model_fn, c_id, args, adj, train_dataset, valid_dataset, test_dataset):
        # Used for computing the mask in self-attention module
        self.num_items = train_dataset.num_items
        self.domain = train_dataset.domain
        # A readable client identifier for logging/aggregation
        self.name = getattr(train_dataset, "client_name", self.domain)
        # Used for computing the positional embeddings
        self.max_seq_len = args.max_seq_len
        self.trainer = model_fn(args, self.num_items, self.max_seq_len)
        self.model = self.trainer.model
        # Provide domain index to trainer for optional domain adversarial loss
        try:
            self.trainer.domain_idx = c_id
        except Exception:
            pass
        self.method = args.method
        self.checkpoint_dir = args.checkpoint_dir
        self.model_id = args.id if len(args.id) > 1 else "0" + args.id
        self.label_suffix = ""
        if args.method == "FedDCSR":
            self.z_s = self.trainer.z_s
            self.z_g = self.trainer.z_g
        self.c_id = c_id
        self.args = args
        self.adj = adj

        self.train_dataloader = SeqDataloader(
            train_dataset, batch_size=args.batch_size, shuffle=True)
        self.valid_dataloader = SeqDataloader(
            valid_dataset, batch_size=args.batch_size, shuffle=False)
        self.test_dataloader = SeqDataloader(
            test_dataset, batch_size=args.batch_size, shuffle=False)

        # Compute the number of samples for each client
        self.n_samples_train = len(train_dataset)
        self.n_samples_valid = len(valid_dataset)
        self.n_samples_test = len(test_dataset)
        # The aggretation weight
        self.train_pop, self.valid_weight, self.test_weight = 0.0, 0.0, 0.0
        # Model evaluation results
        self.MRR, self.NDCG_5, self.NDCG_10, self.HR_1, self.HR_5, self.HR_10 \
            = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        # Extended metrics
        self.HR_20, self.HR_50, self.NDCG_20, self.AUC = 0.0, 0.0, 0.0, 0.0
        # For segment alignment (SegFedGNN)
        self.segment_proto = None

    def train_epoch(self, round, args, global_params=None):
        """Trains one client with its own training data for one epoch.

        Args:
            round: Training round.
            args: Other arguments for training.
            global_params: Global model parameters used in `FedProx` method.
        """
        self.trainer.model.train()
        for _ in range(args.local_epoch):
            loss = 0
            step = 0
            for _, sessions in self.train_dataloader:
                if ("Fed" in args.method) and args.mu:
                    batch_loss = self.trainer.train_batch(
                        sessions, self.adj, self.num_items, args,
                        global_params=global_params)
                else:
                    batch_loss = self.trainer.train_batch(
                        sessions, self.adj, self.num_items, args)
                loss += batch_loss
                step += 1

            gc.collect()
        logging.info("Epoch {}/{} - client {} -  Training Loss: {:.3f}".format(
            round, args.epochs, self.c_id, loss / step))
        return self.n_samples_train

    def evaluation(self, mode="valid"):
        """Evaluates one client with its own valid/test data for one epoch.

        Args:
            mode: `valid` or `test`.
        """
        if mode == "valid":
            dataloader = self.valid_dataloader
        elif mode == "test":
            dataloader = self.test_dataloader

        self.trainer.model.eval()
        with torch.no_grad():
            if (self.method == "FedDCSR") or ("VGSAN" in self.method) or (self.method == "SegFedGNN"):
                if self.method == "SegFedGNN":
                    self.trainer.model.disen.graph_convolution(self.adj)
                else:
                    self.trainer.model.graph_convolution(self.adj)
            pred = []
            for _, sessions in dataloader:
                predictions = self.trainer.test_batch(sessions)
                pred = pred + predictions

        gc.collect()
        metrics = self.cal_test_score(pred)
        self.MRR = metrics["MRR"]
        self.NDCG_5 = metrics["NDCG @5"]
        self.NDCG_10 = metrics["NDCG @10"]
        self.NDCG_20 = metrics["NDCG @20"]
        self.HR_1 = metrics["HR @1"]
        self.HR_5 = metrics["HR @5"]
        self.HR_10 = metrics["HR @10"]
        self.HR_20 = metrics["HR @20"]
        self.HR_50 = metrics["HR @50"]
        self.AUC = metrics["AUC"]
        return metrics

    def get_old_eval_log(self):
        """Returns the evaluation result of the lastest epoch.
        """
        return {"MRR": self.MRR, "HR @1": self.HR_1, "HR @5": self.HR_5,
                "HR @10": self.HR_10, "HR @20": self.HR_20, "HR @50": self.HR_50,
                "NDCG @5": self.NDCG_5, "NDCG @10": self.NDCG_10,
                "NDCG @20": self.NDCG_20, "AUC": self.AUC}

    @ staticmethod
    def cal_test_score(predictions):
        """Calculate evaluation metrics from predictions.

        Args:
            predictions: List of dicts with 'rank', 'auc', 'num_negs' keys.

        Returns:
            Dict with all evaluation metrics.
        """
        MRR = 0.0
        HR_1 = 0.0
        HR_5 = 0.0
        HR_10 = 0.0
        HR_20 = 0.0
        HR_50 = 0.0
        NDCG_5 = 0.0
        NDCG_10 = 0.0
        NDCG_20 = 0.0
        AUC_sum = 0.0
        valid_entity = 0.0

        for pred_info in predictions:
            # Handle both old format (int) and new format (dict)
            if isinstance(pred_info, dict):
                pred = pred_info['rank']
                auc = pred_info.get('auc', 0.0)
            else:
                pred = pred_info
                auc = 0.0

            valid_entity += 1
            MRR += 1 / pred
            AUC_sum += auc

            # HR@K: hit if rank <= K
            if pred <= 1:
                HR_1 += 1
            if pred <= 5:
                HR_5 += 1
            if pred <= 10:
                HR_10 += 1
            if pred <= 20:
                HR_20 += 1
            if pred <= 50:
                HR_50 += 1

            # NDCG@K: only count if rank <= K
            if pred <= 5:
                NDCG_5 += 1 / np.log2(pred + 1)
            if pred <= 10:
                NDCG_10 += 1 / np.log2(pred + 1)
            if pred <= 20:
                NDCG_20 += 1 / np.log2(pred + 1)

        n = valid_entity if valid_entity > 0 else 1
        return {
            "MRR": MRR / n,
            "HR @1": HR_1 / n,
            "HR @5": HR_5 / n,
            "HR @10": HR_10 / n,
            "HR @20": HR_20 / n,
            "HR @50": HR_50 / n,
            "NDCG @5": NDCG_5 / n,
            "NDCG @10": NDCG_10 / n,
            "NDCG @20": NDCG_20 / n,
            "AUC": AUC_sum / n
        }

    def get_params(self):
        """Returns the model parameters that need to be shared between clients.
        """
        if self.method == "FedDCSR":
            return copy.deepcopy([self.model.encoder_s.state_dict()])
        elif "VGSAN" in self.method:
            return copy.deepcopy([self.model.encoder.state_dict()])
        elif self.method == "SegFedGNN":
            # Share DGL encoder_s; include segment-encoder params only for SDSS branch.
            if getattr(self.model, "use_sdss_branch", True) and \
                    getattr(self.model, "branch_type", "sdss") == "sdss":
                shared_extras = {
                    'seg2pred.weight': self.model.seg2pred.weight.data,
                    'seg2pred.bias': self.model.seg2pred.bias.data,
                    'segment_pos_emb.weight': self.model.segment_pos_emb.weight.data,
                }
                params = [
                    self.model.disen.encoder_s.state_dict(),
                    self.model.segment_encoder.state_dict(),
                    shared_extras
                ]
            else:
                # SDSS disabled: only DGL encoder is federated
                params = [self.model.disen.encoder_s.state_dict()]
            return copy.deepcopy(params)
        elif "SASRec" in self.method:
            return copy.deepcopy([self.model.encoder.state_dict()])
        elif "VSAN" in self.method:
            return copy.deepcopy([self.model.encoder.state_dict(),
                                  self.model.decoder.state_dict()])
        elif "ContrastVAE" in self.method:
            return copy.deepcopy([self.model.encoder.state_dict(),
                                  self.model.decoder.state_dict()])
        elif "CL4SRec" in self.method:
            return copy.deepcopy([self.model.encoder.state_dict()])
        elif "DuoRec" in self.method:
            return copy.deepcopy([self.model.encoder.state_dict()])

    def get_reps_shared(self):
        """Returns the user sequence representations that need to be shared
        between clients.
        """
        assert (self.method == "FedDCSR")
        return copy.deepcopy(self.z_s[0].detach())

    def set_global_params(self, global_params):
        """Assign the local shared model parameters with global model
        parameters.
        """
        assert (self.method in ["FedDCSR", "FedVGSAN", "FedSASRec", "FedVSAN",
                                "FedContrastVAE", "FedCL4SRec", "FedDuoRec",
                                "SegFedGNN"])
        if self.method == "FedDCSR":
            self.model.encoder_s.load_state_dict(global_params[0])
        elif self.method == "FedVGSAN":
            self.model.encoder.load_state_dict(global_params[0])
            # self.model.decoder.load_state_dict(global_params[1])
        elif self.method == "FedSASRec":
            self.model.encoder.load_state_dict(global_params[0])
        elif self.method == "FedVSAN":
            self.model.encoder.load_state_dict(global_params[0])
            self.model.decoder.load_state_dict(global_params[1])
        elif self.method == "FedContrastVAE":
            self.model.encoder.load_state_dict(global_params[0])
            self.model.decoder.load_state_dict(global_params[1])
        elif self.method == "FedCL4SRec":
            self.model.encoder.load_state_dict(global_params[0])
        elif self.method == "FedDuoRec":
            self.model.encoder.load_state_dict(global_params[0])
        elif self.method == "SegFedGNN":
            self.model.disen.encoder_s.load_state_dict(global_params[0])
            if getattr(self.model, "use_sdss_branch", True) and \
                    getattr(self.model, "branch_type", "sdss") == "sdss" and \
                    len(global_params) >= 3:
                self.model.segment_encoder.load_state_dict(global_params[1])
                self.model.seg2pred.weight.data = global_params[2]['seg2pred.weight']
                self.model.seg2pred.bias.data = global_params[2]['seg2pred.bias']
                self.model.segment_pos_emb.weight.data = global_params[2]['segment_pos_emb.weight']

    def set_global_reps(self, global_rep):
        """Copy global user sequence representations to local.
        """
        assert (self.method == "FedDCSR")
        self.z_g[0] = copy.deepcopy(global_rep)

    # ===== SegFedGNN: directly transmit summary_vec =====
    def get_summary_vec(self):
        """Return latest summary_vec for server graph construction (SegFedGNN)."""
        if self.method == "SegFedGNN" and hasattr(self.trainer, "last_summary_vec"):
            vec = self.trainer.last_summary_vec
            if vec is None:
                return None
            return vec.detach().cpu()
        return None

    def set_summary_graph_payload(self, payload):
        """Set server payload (V, A, idx) for local DomainGNN execution."""
        if self.method == "SegFedGNN" and hasattr(self.trainer, "set_summary_graph_payload"):
            self.trainer.set_summary_graph_payload(payload)

    def save_params(self):
        method_ckpt_path = os.path.join(self.checkpoint_dir,
                                        "domain_" +
                                        "".join([domain[0]
                                                for domain
                                                 in self.args.domains]),
                                        self.method + "_" + self.model_id + self.label_suffix)
        ensure_dir(method_ckpt_path, verbose=True)
        ckpt_filename = os.path.join(
            method_ckpt_path, "client%d.pt" % self.c_id)
        checkpoint = {"model": self.trainer.model.state_dict()}
        if getattr(self.trainer, "hyper_head", None) is not None:
            checkpoint["hyper_head"] = self.trainer.hyper_head.state_dict()
        if getattr(self.trainer, "local_summary_gnn", None) is not None:
            checkpoint["local_summary_gnn"] = self.trainer.local_summary_gnn.state_dict()
        try:
            # Write to a temp file first, then atomically replace to avoid
            # partially written checkpoints on interruption.
            tmp_file = ckpt_filename + ".tmp"
            torch.save(checkpoint, tmp_file)
            os.replace(tmp_file, ckpt_filename)
            print("Model saved to {}".format(ckpt_filename))
        except IOError:
            print("[ Warning: Saving failed... continuing anyway. ]")
        except Exception as e:
            print(f"[ Warning: Saving failed ({e})... continuing anyway. ]")

    def load_params(self):
        ckpt_filename = os.path.join(self.checkpoint_dir,
                                     "domain_" +
                                     "".join([domain[0]
                                            for domain in self.args.domains]),
                                     self.method + "_" + self.model_id + self.label_suffix,
                                     "client%d.pt" % self.c_id)
        try:
            # Load checkpoint tensors on CPU first for robustness:
            # - avoids CUDA deserialization failures when GPU visibility changes
            # - avoids temporary GPU memory spikes during checkpoint read
            checkpoint = torch.load(ckpt_filename, map_location="cpu")
        except (IOError, RuntimeError, FileNotFoundError) as e:
            logging.warning("Cannot load model from %s (%s).", ckpt_filename, e)
            print(f"[ Fail: Cannot load model from {ckpt_filename} ({e}). ]")
            return False

        # Backward compatibility:
        # - old checkpoints: raw model state_dict
        # - new checkpoints: dict with model/hyper_head/local_summary_gnn
        if isinstance(checkpoint, dict) and "model" in checkpoint and \
                isinstance(checkpoint["model"], dict):
            model_state = checkpoint["model"]
        else:
            model_state = checkpoint

        if self.trainer.model is not None and model_state is not None:
            # Validate key/shape compatibility before loading to avoid
            # partially mutating in-memory weights on mismatch.
            current_state = self.trainer.model.state_dict()
            ckpt_keys = set(model_state.keys())
            current_keys = set(current_state.keys())
            if ckpt_keys != current_keys:
                missing = sorted(current_keys - ckpt_keys)[:5]
                unexpected = sorted(ckpt_keys - current_keys)[:5]
                logging.warning(
                    "Checkpoint/model key mismatch for %s (missing=%s, unexpected=%s).",
                    ckpt_filename, missing, unexpected
                )
                print(
                    f"[ Fail: Checkpoint/model key mismatch in {ckpt_filename} "
                    f"(missing={len(current_keys - ckpt_keys)}, "
                    f"unexpected={len(ckpt_keys - current_keys)}). ]"
                )
                return False

            for key, tensor in current_state.items():
                loaded_tensor = model_state.get(key, None)
                if loaded_tensor is None or tensor.shape != loaded_tensor.shape:
                    logging.warning(
                        "Checkpoint/model shape mismatch at %s in %s: "
                        "expected %s, got %s.",
                        key, ckpt_filename, tuple(tensor.shape),
                        None if loaded_tensor is None else tuple(loaded_tensor.shape)
                    )
                    print(
                        f"[ Fail: Checkpoint/model shape mismatch at {key} in "
                        f"{ckpt_filename}. ]"
                    )
                    return False
            try:
                self.trainer.model.load_state_dict(model_state)
            except RuntimeError as e:
                logging.warning("Cannot load model state from %s (%s).", ckpt_filename, e)
                print(f"[ Fail: Cannot load model state from {ckpt_filename} ({e}). ]")
                return False

        if isinstance(checkpoint, dict):
            if getattr(self.trainer, "hyper_head", None) is not None and "hyper_head" in checkpoint:
                try:
                    self.trainer.hyper_head.load_state_dict(checkpoint["hyper_head"])
                except RuntimeError as e:
                    logging.warning("Cannot load hyper_head from %s (%s).", ckpt_filename, e)
                    print(f"[ Warning: Cannot load hyper_head from {ckpt_filename} ({e}). ]")

            if getattr(self.trainer, "local_summary_gnn", None) is not None and \
                    "local_summary_gnn" in checkpoint:
                try:
                    self.trainer.local_summary_gnn.load_state_dict(
                        checkpoint["local_summary_gnn"])
                except RuntimeError as e:
                    logging.warning("Cannot load local_summary_gnn from %s (%s).", ckpt_filename, e)
                    print(f"[ Warning: Cannot load local_summary_gnn from {ckpt_filename} ({e}). ]")
        return True
