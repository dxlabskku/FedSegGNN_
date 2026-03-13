# -*- coding: utf-8 -*-
import numpy as np
import torch
from dataset import SeqDataset
from local_graph import LocalGraph

# Preset domain groupings for reproducible data distribution scenarios
DATA_DIST_SCENARIOS = {
    "s0": None,  # Base/original distribution, do not override provided domains
    "s1": ["Food", "Kitchen", "Clothing", "Beauty"],
    "s1_base": ["Food", "Kitchen", "Clothing", "Beauty"],
    "s1_life": ["Food", "Kitchen", "Home", "Garden"],
    "s1_ent": ["Movies", "Books", "Games", "Sports"],
    "s2": ["Food", "Kitchen", "Clothing", "Beauty",
           "Movies", "Books", "Games", "Sports", "Garden", "Home"],
}


def build_data_dist_config(args):
    """Build a config dict (and cache suffix) for the requested data scenario."""
    scenario = (getattr(args, "data_dist", "custom") or "custom").lower()
    args.data_dist = scenario
    config = {"name": scenario, "seed": getattr(args, "seed", 42)}
    suffix = None

    if scenario in {"s0", "s1", "s1_base", "s1_life", "s1_ent", "s2"}:
        suffix = f"dist_{scenario}"
    return config, suffix


def apply_data_dist_defaults(args):
    """Mutate args with domain selections and config derived from --data_dist."""
    scenario = (getattr(args, "data_dist", "custom") or "custom").lower()
    args.data_dist = scenario

    # Do not override explicit single-domain split mode
    # Map legacy alias s1 -> s1_base internally
    if scenario == "s1":
        scenario = "s1_base"
        args.data_dist = scenario
    predef = DATA_DIST_SCENARIOS.get(scenario)
    if not getattr(args, "single_domain", None) and predef:
        args.domains = predef

    config, suffix = build_data_dist_config(args)
    args.data_dist_config = config
    args.data_dist_suffix = suffix


def load_dataset(args):
    client_train_datasets = []
    client_valid_datasets = []
    client_test_datasets = []
    data_dist = getattr(args, "data_dist", "custom")
    data_dist_config = getattr(args, "data_dist_config", None)
    prep_suffix = getattr(args, "data_dist_suffix", None)

    for domain in args.domains:
        if args.method == "FedDCSR":
            model = "DisenVGSAN"
        elif args.method == "SegFedGNN":
            # SegFedGNN uses same data format as SDSSDCSR_DUAL
            model = "SDSSDCSR_DUAL"
        else:
            model = args.method.replace("Fed", "")
            model = model.replace("Local", "")

        train_dataset = SeqDataset(
            domain, model, mode="train", max_seq_len=args.max_seq_len,
            load_prep=args.load_prep, prep_suffix=prep_suffix,
            data_dist=data_dist, data_dist_seed=args.seed,
            data_dist_config=data_dist_config)
        valid_dataset = SeqDataset(
            domain, model, mode="valid", max_seq_len=args.max_seq_len,
            load_prep=args.load_prep, prep_suffix=prep_suffix,
            data_dist=data_dist, data_dist_seed=args.seed,
            data_dist_config=data_dist_config)
        test_dataset = SeqDataset(
            domain, model, mode="test", max_seq_len=args.max_seq_len,
            load_prep=args.load_prep, prep_suffix=prep_suffix,
            data_dist=data_dist, data_dist_seed=args.seed,
            data_dist_config=data_dist_config)

        client_train_datasets.append(train_dataset)
        client_valid_datasets.append(valid_dataset)
        client_test_datasets.append(test_dataset)

    adjs = []
    for train_dataset, domain in zip(client_train_datasets, args.domains):
        local_graph = LocalGraph(
            args, domain, train_dataset.num_items,
            train_sessions=train_dataset.sessions)
        adjs.append(local_graph.adj)
        print("%s graph loaded!" % domain)

    device = torch.device(
        getattr(args, "device", None)
        if getattr(args, "device", None) is not None
        else ("cuda:%s" % args.gpu if args.cuda and torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    for idx, adj in enumerate(adjs):
        adjs[idx] = adj.to(device)

    return client_train_datasets, client_valid_datasets, \
        client_test_datasets, adjs


def _dirichlet_split_sessions(user_ids, sessions, num_clients, alpha=1.0,
                              seed=42, min_size=1, mode_label="train",
                              max_retries=50):
    """Randomly split sessions into multiple clients with a Dirichlet prior.

    Ensures that each client receives at least `min_size` sessions. Raises
    an error if the dataset is too small for the requested number of clients.
    """
    total = len(sessions)
    if total < num_clients:
        raise ValueError(
            f"Not enough {mode_label} sessions ({total}) to split across "
            f"{num_clients} clients. Reduce --num_clients or use a larger dataset."
        )

    # Clamp the minimum size to a feasible number
    min_size = min(min_size, max(1, total // num_clients))

    rng = np.random.default_rng(seed)
    indices = np.arange(total)

    for _ in range(max_retries):
        client_indices = [[] for _ in range(num_clients)]
        rng.shuffle(indices)
        probs = rng.dirichlet([alpha] * num_clients)
        for idx in indices:
            cid = rng.choice(num_clients, p=probs)
            client_indices[cid].append(idx)

        if min(len(idx_list) for idx_list in client_indices) >= min_size:
            break
    else:
        # Fallback to near-even split to avoid training failures
        client_indices = np.array_split(indices, num_clients)
        client_indices = [list(arr) for arr in client_indices]

    client_user_ids, client_sessions = [], []
    for cid_indices in client_indices:
        cid_user_ids = [user_ids[i] for i in cid_indices] if user_ids is not None else list(range(len(cid_indices)))
        cid_sessions = [sessions[i] for i in cid_indices]
        client_user_ids.append(cid_user_ids)
        client_sessions.append(cid_sessions)

    return client_user_ids, client_sessions


def load_single_domain_dataset(args):
    """Load one domain and randomly split it into multiple clients."""
    domain = args.single_domain
    num_clients = args.num_clients
    split_seed = args.split_seed if getattr(args, "split_seed", None) is not None else args.seed
    split_alpha = getattr(args, "split_alpha", 1.0)
    data_dist = getattr(args, "data_dist", "custom")
    data_dist_config = getattr(args, "data_dist_config", None)
    dist_suffix = getattr(args, "data_dist_suffix", None)

    if domain is None:
        raise ValueError("--single_domain must be provided when using single-domain split mode")
    if num_clients is None or num_clients < 1:
        raise ValueError("--num_clients must be a positive integer for single-domain split mode")

    if args.method == "FedDCSR":
        model = "DisenVGSAN"
    elif args.method == "SegFedGNN":
        # SegFedGNN uses same data format as SDSSDCSR_DUAL
        model = "SDSSDCSR_DUAL"
    else:
        model = args.method.replace("Fed", "")
        model = model.replace("Local", "")

    # Load full domain once per split to reuse raw sessions
    train_full = SeqDataset(domain, model, mode="train",
                            max_seq_len=args.max_seq_len,
                            load_prep=args.load_prep,
                            prep_suffix=dist_suffix,
                            data_dist=data_dist,
                            data_dist_seed=args.seed,
                            data_dist_config=data_dist_config)
    valid_full = SeqDataset(domain, model, mode="valid",
                            max_seq_len=args.max_seq_len,
                            load_prep=args.load_prep,
                            prep_suffix=dist_suffix,
                            data_dist=data_dist,
                            data_dist_seed=args.seed,
                            data_dist_config=data_dist_config)
    test_full = SeqDataset(domain, model, mode="test",
                           max_seq_len=args.max_seq_len,
                           load_prep=args.load_prep,
                           prep_suffix=dist_suffix,
                           data_dist=data_dist,
                           data_dist_seed=args.seed,
                           data_dist_config=data_dist_config)

    num_items = max(train_full.num_items, valid_full.num_items, test_full.num_items)

    # Split sessions for each mode
    train_users, train_sessions = _dirichlet_split_sessions(
        train_full.user_ids, train_full.sessions, num_clients,
        alpha=split_alpha, seed=split_seed,
        min_size=getattr(args, "min_client_size", 1), mode_label="train")

    valid_users, valid_sessions = _dirichlet_split_sessions(
        valid_full.user_ids, valid_full.sessions, num_clients,
        alpha=split_alpha, seed=split_seed + 1,
        min_size=1, mode_label="valid")

    test_users, test_sessions = _dirichlet_split_sessions(
        test_full.user_ids, test_full.sessions, num_clients,
        alpha=split_alpha, seed=split_seed + 2,
        min_size=1, mode_label="test")

    client_train_datasets, client_valid_datasets, client_test_datasets = [], [], []
    adjs = []

    alpha_tag = str(split_alpha).replace('.', 'p')
    for cid in range(num_clients):
        prep_suffix = f"c{cid}_n{num_clients}_a{alpha_tag}_s{split_seed}"
        if dist_suffix:
            prep_suffix = f"{prep_suffix}_{dist_suffix}"

        train_dataset = SeqDataset(
            domain, model, mode="train", max_seq_len=args.max_seq_len,
            load_prep=args.load_prep, sessions_override=train_sessions[cid],
            user_ids_override=train_users[cid], num_items_override=num_items,
            prep_suffix=prep_suffix, data_dist=data_dist,
            data_dist_seed=args.seed, data_dist_config=data_dist_config)
        valid_dataset = SeqDataset(
            domain, model, mode="valid", max_seq_len=args.max_seq_len,
            load_prep=args.load_prep, sessions_override=valid_sessions[cid],
            user_ids_override=valid_users[cid], num_items_override=num_items,
            prep_suffix=prep_suffix, data_dist=data_dist,
            data_dist_seed=args.seed, data_dist_config=data_dist_config)
        test_dataset = SeqDataset(
            domain, model, mode="test", max_seq_len=args.max_seq_len,
            load_prep=args.load_prep, sessions_override=test_sessions[cid],
            user_ids_override=test_users[cid], num_items_override=num_items,
            prep_suffix=prep_suffix, data_dist=data_dist,
            data_dist_seed=args.seed, data_dist_config=data_dist_config)

        # Attach a readable client name for logging
        client_name = f"{domain}_c{cid}"
        train_dataset.client_name = client_name
        valid_dataset.client_name = client_name
        test_dataset.client_name = client_name

        client_train_datasets.append(train_dataset)
        client_valid_datasets.append(valid_dataset)
        client_test_datasets.append(test_dataset)

        local_graph = LocalGraph(args, domain, num_items,
                                 train_sessions=train_sessions[cid])
        adjs.append(local_graph.adj)
        print(f"Client {cid} graph built for {client_name}!")

    device = torch.device(
        getattr(args, "device", None)
        if getattr(args, "device", None) is not None
        else ("cuda:%s" % args.gpu if args.cuda and torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    for idx, adj in enumerate(adjs):
        adjs[idx] = adj.to(device)

    return client_train_datasets, client_valid_datasets, client_test_datasets, adjs


def init_clients_weight(clients):
    """Initialize the aggretation weight, which is the ratio of the number of
    samples per client to the total number of samples.
    """
    client_n_samples_train = [client.n_samples_train for client in clients]

    samples_sum_train = np.sum(client_n_samples_train)
    for client in clients:
        client.train_weight = client.n_samples_train / samples_sum_train
        client.valid_weight = 1 / len(clients)
        client.test_weight = 1 / len(clients)
