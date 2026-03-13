# -*- coding: utf-8 -*-
import logging
from tqdm import tqdm
from utils.train_utils import EarlyStopping, LRDecay


def evaluation_logging(eval_logs, round, weights, mode="valid"):
    if mode == "valid":
        logging.info("Epoch%d Valid:" % round)
    else:
        logging.info("Test:")

    avg_eval_log = {}
    for metric_name in list(eval_logs.values())[0].keys():
        avg_eval_val = 0
        for client_key in eval_logs.keys():
            avg_eval_val += eval_logs[client_key][metric_name] * weights[client_key]
        avg_eval_log[metric_name] = avg_eval_val

    # Core metrics
    logging.info("MRR: %.6f" % avg_eval_log["MRR"])
    logging.info("AUC: %.6f" % avg_eval_log.get("AUC", 0.0))
    logging.info("HR @1|5|10|20|50: %.6f \t %.6f \t %.6f \t %.6f \t %.6f" %
                 (avg_eval_log["HR @1"], avg_eval_log["HR @5"],
                  avg_eval_log["HR @10"], avg_eval_log.get("HR @20", 0.0),
                  avg_eval_log.get("HR @50", 0.0)))
    logging.info("NDCG @5|10|20: %.6f \t %.6f \t %.6f" %
                 (avg_eval_log["NDCG @5"], avg_eval_log["NDCG @10"],
                  avg_eval_log.get("NDCG @20", 0.0)))

    for client_name, eval_log in eval_logs.items():
        # Print per-client/domain breakdown with extended metrics
        logging.info(
            "%s MRR: %.6f \t AUC: %.6f \t HR @1: %.6f \t HR @5: %.6f \t HR @10: %.6f \t "
            "HR @20: %.6f \t HR @50: %.6f \t NDCG @5: %.6f \t NDCG @10: %.6f \t NDCG @20: %.6f"
            % (client_name, eval_log["MRR"], eval_log.get("AUC", 0.0),
               eval_log.get("HR @1", 0.0), eval_log.get("HR @5", 0.0),
               eval_log.get("HR @10", 0.0), eval_log.get("HR @20", 0.0),
               eval_log.get("HR @50", 0.0), eval_log.get("NDCG @5", 0.0),
               eval_log.get("NDCG @10", 0.0), eval_log.get("NDCG @20", 0.0)))

    return avg_eval_log


def load_and_eval_model(n_clients, clients, args, server=None):
    eval_logs = {}
    for c_id in tqdm(range(n_clients), ascii=True):
        ok = clients[c_id].load_params()
        if ok is False:
            logging.warning(
                "Skipping checkpoint load for client %s; evaluating current in-memory weights.",
                clients[c_id].name,
            )
        if server is not None and args.method == "SegFedGNN":
            clients[c_id].set_summary_graph_payload(
                server.get_summary_graph_payload(c_id))
        eval_log = clients[c_id].evaluation(mode="test")
        eval_logs[clients[c_id].name] = eval_log
    weights = dict((client.name, client.test_weight) for client in clients)
    evaluation_logging(eval_logs, 0, weights, mode="test")


def run_fl(clients, server, args):
    n_clients = len(clients)
    if args.do_eval:
        load_and_eval_model(n_clients, clients, args, server=None)
    else:
        early_stopping = EarlyStopping(
            args.checkpoint_dir, patience=args.es_patience, verbose=True)
        lr_decay = LRDecay(args.lr, args.decay_epoch,
                           args.optimizer, args.lr_decay,
                           patience=args.ld_patience, verbose=True)
        for round in range(1, args.epochs + 1):
            random_cids = server.choose_clients(n_clients, args.frac)

            # Train with these clients
            for c_id in tqdm(random_cids, ascii=True):
                if "Fed" in args.method:
                    # Restore global parameters to client's model
                    clients[c_id].set_global_params(server.get_global_params())
                    if args.method == "FedDCSR":
                        clients[c_id].set_global_reps(server.get_global_reps())
                    elif args.method == "SegFedGNN":
                        clients[c_id].set_summary_graph_payload(
                            server.get_summary_graph_payload(c_id))

                # Train one client
                clients[c_id].train_epoch(
                    round, args, global_params=server.global_params)

            if "Fed" in args.method or args.method == "SegFedGNN":
                server.aggregate_params(clients, random_cids)
                if args.method == "FedDCSR":
                    server.aggregate_reps(clients, random_cids)
                elif args.method == "SegFedGNN":
                    server.aggregate_summary_vecs(clients, random_cids)
                    for cid in random_cids:
                        clients[cid].set_summary_graph_payload(
                            server.get_summary_graph_payload(cid))

            if round % args.eval_interval == 0:
                eval_logs = {}
                for c_id in tqdm(range(n_clients), ascii=True):
                    if "Fed" in args.method or args.method == "SegFedGNN":
                        clients[c_id].set_global_params(
                            server.get_global_params())
                        if args.method == "SegFedGNN":
                            clients[c_id].set_summary_graph_payload(
                                server.get_summary_graph_payload(c_id))
                    if c_id in random_cids:
                        eval_log = clients[c_id].evaluation(mode="valid")
                    else:
                        eval_log = clients[c_id].get_old_eval_log()
                    eval_logs[clients[c_id].name] = eval_log

                weights = dict((client.name, client.valid_weight)
                               for client in clients)
                avg_eval_log = evaluation_logging(
                    eval_logs, round, weights, mode="valid")

                # Early Stopping. Here only compare the current results with
                # the best results
                early_stopping(avg_eval_log, clients)
                if early_stopping.early_stop:
                    logging.info("Early stopping")
                    break

                # Learning rate decay. Here only compare the current results
                # with the latest results
                lr_decay(round, avg_eval_log, clients)

        load_and_eval_model(n_clients, clients, args, server)
