# -*- coding: utf-8 -*-
import os
import numpy as np
import random
import argparse
import torch
from trainer import ModelTrainer
import logging
from client import Client
from server import Server
from utils.data_utils import load_dataset, load_single_domain_dataset, \
    init_clients_weight, apply_data_dist_defaults
from utils.io_utils import save_config, ensure_dir
from fl import run_fl


def resolve_device(args):
    """Validate requested CUDA device and fall back gracefully."""
    if not args.cuda or not torch.cuda.is_available():
        args.cuda = False
        args.gpu = "cpu"
        return torch.device("cpu")

    try:
        gpu_idx = int(str(args.gpu).split(",")[0])
    except ValueError:
        print(f"Invalid GPU id `{args.gpu}`. Falling back to CPU.")
        args.cuda = False
        args.gpu = "cpu"
        return torch.device("cpu")

    visible_gpus = torch.cuda.device_count()
    if visible_gpus == 0:
        print("CUDA requested but no GPUs detected. Using CPU instead.")
        args.cuda = False
        args.gpu = "cpu"
        return torch.device("cpu")

    if gpu_idx < 0 or gpu_idx >= visible_gpus:
        fallback_idx = 0
        print(
            f"Requested GPU {gpu_idx} is out of range (0-{visible_gpus - 1}). "
            f"Using GPU {fallback_idx} instead."
        )
        args.gpu = str(fallback_idx)
        return torch.device(f"cuda:{fallback_idx}")

    return torch.device(f"cuda:{gpu_idx}")


def arg_parse():
    parser = argparse.ArgumentParser(conflict_handler="resolve")

    # Dataset part
    parser.add_argument(dest="domains", metavar="domains", nargs="*",
                        help="`Food Kitchen Clothing Beauty` or "
                        "`Movies Books Games` or `Sports Garden Home`")
    parser.add_argument("--single_domain", type=str, default=None,
                        help="Use a single domain and split it into multiple clients")
    parser.add_argument("--num_clients", type=int, default=None,
                        help="Number of clients when using --single_domain")
    parser.add_argument("--split_alpha", type=float, default=1.0,
                        help="Dirichlet alpha for random client splits (single-domain mode)")
    parser.add_argument("--split_seed", type=int, default=None,
                        help="Random seed for client splits (defaults to --seed)")
    parser.add_argument("--min_client_size", type=int, default=10,
                        help="Minimum number of training sessions per client when splitting a single domain")
    parser.add_argument("--load_prep", dest="load_prep", action="store_true",
                        default=False,
                        help="Whether need to load preprocessed the data. If "
                        "you want to load preprocessed data, add it")
    parser.add_argument("--max_seq_len", type=int,
                        default=16, help="maxisum sequence length")
    parser.add_argument("--data_dist", type=str.lower,
                        choices=["custom", "s0", "s1", "s1_base", "s1_life",
                                 "s1_ent", "s2"],
                        default="custom",
                        help="Data distribution scenario. `s1`: low heterogeneity "
                             "(Food/Kitchen/Clothing/Beauty). `s1_life`: "
                             "Food/Kitchen/Home/Garden. `s1_ent`: Movies/Books/"
                             "Games/Sports. `s2`: full 10 domains. "
                             "`s0`: base/original distribution using provided domains.")

    # Training part
    parser.add_argument("--method", type=str, default="FedDCSR",
                        help="method, possible are `FedDCSR`(ours), "
                        "`FedVGSAN`, `LocalVGSAN`, `FedSASRec`, "
                        "`LocalSASRec`, `FedVSAN`, `LocalVSAN`, "
                        "`FedContrastVAE`, `LocalContrastVAE`, `FedCL4SRec`, "
                        "`LocalCL4SRec`, `FedDuoRec`, `LocalDuoRec`, "
                        "`SegFedGNN`, "
                        "`dgl_streeam`, `FedDisenProto`, `LocalDisenProto`")
    parser.add_argument("--log_dir", type=str,
                        default="log", help="directory of logs")
    parser.add_argument("--cuda", type=bool, default=torch.cuda.is_available())
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use")
    parser.add_argument("--epochs", type=int, default=40,
                        help="Number of total training iterations.")
    parser.add_argument("--local_epoch", type=int, default=3,
                        help="Number of local training epochs.")
    parser.add_argument("--optimizer", choices=["sgd", "adagrad", "adam",
                                                "adamax"], default="adam",
                        help="Optimizer: sgd, adagrad, adam or adamax.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Applies to sgd and adagrad.")  # 0.001
    parser.add_argument("--lr_decay", type=float, default=1,
                        help="Learning rate decay rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight decay (L2 loss on parameters).")
    parser.add_argument("--decay_epoch", type=int, default=10,
                        help="Decay learning rate after this epoch.")
    parser.add_argument("--batch_size", type=int,
                        default=256, help="Training batch size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_interval", type=int,
                        default=1, help="Interval of evalution")
    parser.add_argument("--frac", type=float, default=1,
                        help="Fraction of participating clients")
    parser.add_argument("--mu", type=float, default=0,
                        help="hyper parameter for FedProx")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="checkpoint", help="Checkpoint Dir")
    parser.add_argument("--id", type=str, default="00",
                        help="Model ID under which to save models.")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--es_patience", type=int,
                        default=5, help="Early stop patience.")
    parser.add_argument("--ld_patience", type=int, default=1,
                        help="Learning rate decay patience.")

    # KL annealing arguments for variantional method (including ours)
    parser.add_argument("--anneal_cap", type=float, default=1.0, help="KL "
                        "annealing arguments for variantional method "
                        "(including ours). 1.0 for FKCB is the best, 0.01 for "
                        "MBG and SGH is the best")
    parser.add_argument("--total_annealing_step", type=int, default=10000)

    # SDSS auxiliary loss weights (seg input / summarizer quality)
    parser.add_argument("--sdss_aux_recon_weight", type=float, default=None,
                        help="Weight for summarizer reconstruction loss (defaults to config.alpha_recon)")
    parser.add_argument("--sdss_aux_boundary_weight", type=float, default=None,
                        help="Weight for summarizer boundary KL loss (defaults to config.beta_boundary)")
    parser.add_argument("--sdss_aux_compact_weight", type=float, default=None,
                        help="Weight for summarizer compactness loss (defaults to config.gamma_compact)")
    parser.add_argument("--sdss_num_segments", type=int, default=None,
                        help="Override number of SDSS segments (defaults to config.num_segments)")
    parser.add_argument("--sdss_dual_exclusive_weight", type=float, default=0.0,
                        help="Weight for exclusive reconstruction loss in SDSS-DCSR dual")
    parser.add_argument("--sdss_dual_aux_ce_weight", type=float, default=0.0,
                        help="Weight for auxiliary CE loss on SDSS logits in dual model")
    parser.add_argument("--sdss_dual_aux_per_ce_weight", type=float, default=0.0,
                        help="Weight for per-position auxiliary CE on raw SDSS logits in dual model")
    parser.add_argument("--sdss_dual_sdss_contrastive_weight", type=float, default=0.0,
                        help="Weight for contrastive alignment between Disen and SDSS compressed aug")
    parser.add_argument("--sdss_dual_sdss_noise_std", type=float, default=0.1,
                        help="Noise std for SDSS-derived augmentation view")
    parser.add_argument("--sdss_dual_gate_min", type=float, default=0.2,
                        help="Minimum gate value for SDSS contribution in dual model (clamp)")
    parser.add_argument("--sdss_dual_disable_sdss_branch", action="store_true",
                        help="Disable SDSS in dual model (use DGL-Stream only)")
    parser.add_argument("--sdss_dual_disable_pred_fusion", action="store_true",
                        help="Keep SDSS branch active but exclude SDSS prediction logits from final fusion")
    parser.add_argument("--sdss_multiview_contrastive_weight", type=float, default=0.0,
                        help="Weight for segment↔sequence multiview contrastive loss (optional)")
    parser.add_argument("--sdss_gate_mode", choices=["sigmoid", "softmax"],
                        default="sigmoid",
                        help="Gate mode for combining DGL-Stream and SDSS")
    parser.add_argument("--sdss_gate_softmax_temp", type=float, default=1.0,
                        help="Temperature for softmax gate (softmax mode only)")
    parser.add_argument("--sdss_gate_entropy_weight", type=float, default=0.0,
                        help="Entropy regularization weight for softmax gate")
    parser.add_argument("--sdss_branch_type", choices=["sdss", "sasrec", "mlp", "direct"],
                        default="sdss",
                        help="Backbone for prototype branch: SDSS (CNN+Attn), SASRec-style encoder, simple MLP, or direct sequence pooling")
    parser.add_argument("--use_domain_hyper", action="store_true",
                        help="Enable domain encoder + GNN + hypernetwork head adapters")
    parser.add_argument("--domain_emb_dim", type=int, default=64,
                        help="Dimension for domain embeddings (SDSS domain encoder + GNN)")
    parser.add_argument("--domain_gnn_hidden", type=int, default=64,
                        help="Hidden dim for domain GNN smoothing")
    parser.add_argument("--domain_gnn_layers", type=int, default=2,
                        help="Number of layers in domain GNN")
    parser.add_argument("--domain_knn_k", type=int, default=2,
                        help="k-NN for domain similarity graph")
    parser.add_argument("--hyper_rank", type=int, default=4,
                        help="Low-rank adapter rank for hypernetwork head")
    parser.add_argument("--disable_hyperhead", action="store_true",
                        help="Disable HyperHead logit adaptation from global summary vectors")
    parser.add_argument("--domain_encoder_batches", type=int, default=1,
                        help="Number of train batches to summarize per round for domain encoder")

    # dgl_streeam arguments (method-specific encoder + Gated Fusion + Contrastive)
    # USE_GATED_FUSION: true (default) | false
    parser.add_argument("--proto_use_gated_fusion", action="store_true", default=True,
                        help="Use gated fusion for shared/exclusive representations")
    parser.add_argument("--proto_disable_gated_fusion", action="store_true",
                        help="Disable gated fusion")

    # Contrastive arguments for contrastive method (including ours)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Contrastive arguments for contrastive method "
                        "(including ours)")
    parser.add_argument("--sim", type=str, default="dot")

    # Optional toggles for FedDCSR loss terms
    parser.add_argument("--dcsr_disable_sim", action="store_true",
                        help="Disable similarity (hinge) loss between z_s and z_g")
    parser.add_argument("--dcsr_disable_exclusive", action="store_true",
                        help="Disable exclusive reconstruction loss from z_e")
    parser.add_argument("--dcsr_disable_contrastive", action="store_true",
                        help="Disable contrastive loss between z_e and aug_z_e")
    parser.add_argument("--dcsr_disable_kld", action="store_true",
                        help="Disable KL divergence losses on z_s/z_e")

    args = parser.parse_args()
    assert (args.method in ["FedDCSR", "FedVGSAN", "LocalVGSAN", "FedSASRec",
                            "LocalSASRec", "FedVSAN", "LocalVSAN",
                            "FedContrastVAE", "LocalContrastVAE", "FedCL4SRec",
                            "LocalCL4SRec", "FedDuoRec", "LocalDuoRec",
                            "SegFedGNN",
                            "dgl_streeam", "FedDisenProto", "LocalDisenProto"])

    # Handle disable flags for dgl_streeam
    if getattr(args, 'proto_disable_gated_fusion', False):
        args.proto_use_gated_fusion = False

    # Override SDSS segment count if provided
    if getattr(args, "sdss_num_segments", None):
        try:
            from models.segfedgnn.sdss import config as sdss_config
            sdss_config.num_segments = args.sdss_num_segments
        except Exception:
            pass
    return args


def seed_everything(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_logger(args):
    """Init a file logger that opens the file periodically and write to it.
    """
    dom_tag = "domain_" + "".join([domain[0] for domain in args.domains])
    # Avoid duplicating run tags: keep directory rich, keep filename short
    log_path = os.path.join(args.log_dir, dom_tag)
    ensure_dir(log_path, verbose=True)

    model_id = args.id if len(args.id) > 1 else "0" + args.id
    # File name keeps method + id
    log_file = os.path.join(log_path, f"{args.method}_{model_id}.log")

    logging.basicConfig(
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filename=log_file,
        filemode="w+"
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)


def main():
    args = arg_parse()

    apply_data_dist_defaults(args)
    args.device = resolve_device(args)

    # Configure domain list for single-domain split experiments
    if args.single_domain:
        if args.num_clients is None:
            args.num_clients = 1
        args.domains = [args.single_domain for _ in range(args.num_clients)]
        if args.split_seed is None:
            args.split_seed = args.seed
    elif len(args.domains) == 0:
        raise ValueError("Please provide domains or use --single_domain to split one domain across clients.")

    seed_everything(args)

    init_logger(args)

    if args.single_domain:
        train_datasets, valid_datasets, test_datasets, adjs = load_single_domain_dataset(args)
    else:
        train_datasets, valid_datasets, test_datasets, adjs = load_dataset(args)

    n_clients = len(args.domains)
    clients = [Client(ModelTrainer, c_id, args, adjs[c_id],
                      train_datasets[c_id], valid_datasets[c_id],
                      test_datasets[c_id]) for c_id in range(n_clients)]
    # Initialize the aggretation weight
    init_clients_weight(clients)

    # Save the config of input arguments
    save_config(args)

    server = Server(args, clients[0].get_params())

    run_fl(clients, server, args)


if __name__ == "__main__":
    main()
