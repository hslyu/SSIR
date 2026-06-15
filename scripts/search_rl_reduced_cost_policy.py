import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from ssir import basestations as bs
from ssir.pathfinder import montecarlo, rl_reduced_cost_policy


def load_density_graph(env_dir: Path, exp_id: int, density: float, target_type: str):
    graph = bs.IABRelayGraph()
    graph.load_graph(str(env_dir / f"exp_{exp_id:03d}" / "graph.pkl"), pkl=True)
    for bs_node in graph.basestations:
        if bs_node.basestation_type.name == target_type:
            bs_node.basestation_type.config.eavesdropper_density = density
    return graph


def time_graph(fn, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    start = time.perf_counter()
    graph = fn()
    elapsed = time.perf_counter() - start
    return float(graph.compute_network_throughput()), elapsed


def summarize(rows):
    summary = {}
    for scheme in sorted({row["scheme"] for row in rows}):
        scheme_rows = [row for row in rows if row["scheme"] == scheme]
        throughputs = np.array([row["throughput"] for row in scheme_rows])
        seconds = np.array([row["seconds"] for row in scheme_rows])
        ratios = np.array([row["ratio_to_montecarlo"] for row in scheme_rows])
        summary[scheme] = {
            "mean_throughput": float(throughputs.mean()),
            "median_throughput": float(np.median(throughputs)),
            "mean_seconds": float(seconds.mean()),
            "median_ratio": float(np.median(ratios)),
            "mean_clipped_ratio": float(np.clip(ratios, 0.0, 10.0).mean()),
            "wins": int(np.sum(ratios > 1.0)),
            "count": int(len(scheme_rows)),
        }
    return summary


def _parse_alpha_list(value: str):
    return tuple(float(item) for item in value.split(",") if item.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-dir",
        "--results-dir",
        dest="env_dir",
        default="/fast/hslyu/results_mmf_vs_density/env",
    )
    parser.add_argument("--density", type=float, default=4.64e-3)
    parser.add_argument("--target-type", default=bs.BaseStationType.MARITIME.name)
    parser.add_argument("--work-dir", default="tmp/rl_reduced_cost_search")
    parser.add_argument("--variant", default="base")
    parser.add_argument("--train-start", type=int, default=850)
    parser.add_argument("--train-graphs", "--train-count", dest="train_graphs", type=int, default=5)
    parser.add_argument("--eval-start", type=int, default=800)
    parser.add_argument("--eval-graphs", "--eval-count", dest="eval_graphs", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-predecessors", "--baseline-p", dest="num_predecessors", type=int, default=100)
    parser.add_argument("--num-rounds", "--baseline-r", dest="num_rounds", type=int, default=1)
    parser.add_argument("--num-trials", "--baseline-t", dest="num_trials", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-negatives", type=int, default=96)
    parser.add_argument("--alpha", type=float, default=20.0)
    parser.add_argument("--dual-kl-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--direct-rl", action="store_true")
    parser.add_argument("--rl-reward-temperature", type=float, default=0.15)
    parser.add_argument("--rl-train-learned-mix", type=float, default=0.5)
    parser.add_argument("--rl-dual-kl-weight", type=float, default=0.02)
    parser.add_argument(
        "--dual-anchor",
        choices=["mix", "logit_residual"],
        default="mix",
    )
    parser.add_argument(
        "--policy-arch",
        choices=["mlp", "gnn", "future_gnn_mlp"],
        default="mlp",
    )
    parser.add_argument("--gnn-layers", type=int, default=3)
    parser.add_argument("--alpha-list", type=_parse_alpha_list, default=(20.0,))
    parser.add_argument("--num-policy-paths", type=int, default=16)
    parser.add_argument("--noise-scale", type=float, default=0.15)
    parser.add_argument("--learned-mix", type=float, default=0.25)
    parser.add_argument("--max-hop", type=int, default=12)
    parser.add_argument("--hop-slack", type=int, default=3)
    parser.add_argument(
        "--dual-mode",
        choices=["smooth", "barrier", "lookahead", "lookahead_mix", "active_set"],
        default="smooth",
    )
    parser.add_argument("--lookahead-weight", type=float, default=1.0)
    parser.add_argument("--barrier-weight", type=float, default=1.0)
    parser.add_argument("--catastrophe-threshold", type=float, default=0.0)
    parser.add_argument(
        "--user-order",
        choices=["farthest", "nearest", "random", "constrained"],
        default="farthest",
    )
    parser.add_argument("--zero-shot", action="store_true")
    parser.add_argument("--no-classic-paths", action="store_true")
    parser.add_argument("--no-exact-commit", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    env_dir = Path(args.env_dir)
    work_dir = Path(args.work_dir) / args.variant
    work_dir.mkdir(parents=True, exist_ok=True)
    model_path = work_dir / "model.pt"
    report_path = work_dir / "report.json"

    history = []
    active_model_path = None
    train_graphs = None
    if not args.zero_shot:
        train_ids = range(args.train_start, args.train_start + args.train_graphs)
        train_graphs = [
            load_density_graph(env_dir, exp_id, args.density, args.target_type)
            for exp_id in train_ids
        ]

    if not args.zero_shot:
        if args.direct_rl:
            config = rl_reduced_cost_policy.DirectRLConfig(
                hidden_dim=args.hidden_dim,
                lr=args.lr,
                epochs=args.epochs,
                policy_device=device,
                seed=args.seed,
                alpha=args.alpha,
                alpha_list=args.alpha_list,
                dual_kl_weight=args.rl_dual_kl_weight,
                temperature=args.temperature,
                reward_temperature=args.rl_reward_temperature,
                num_policy_paths=args.num_policy_paths,
                noise_scale=args.noise_scale,
                learned_mix=args.rl_train_learned_mix,
                max_hop=args.max_hop,
                hop_slack=args.hop_slack,
                user_order=args.user_order,
                dual_mode=args.dual_mode,
                lookahead_weight=args.lookahead_weight,
                barrier_weight=args.barrier_weight,
                include_classic_paths=not args.no_classic_paths,
                policy_arch=args.policy_arch,
                gnn_layers=args.gnn_layers,
                dual_anchor=args.dual_anchor,
            )
            _, history = rl_reduced_cost_policy.train_policy_direct_rl(
                train_graphs,
                save_path=str(model_path),
                config=config,
                log_every_epoch=True,
            )
        else:
            config = rl_reduced_cost_policy.RLReducedCostConfig(
                hidden_dim=args.hidden_dim,
                lr=args.lr,
                epochs=args.epochs,
                num_predecessors=args.num_predecessors,
                policy_device=device,
                seed=args.seed,
                max_negatives=args.max_negatives,
                alpha=args.alpha,
                dual_kl_weight=args.dual_kl_weight,
                temperature=args.temperature,
            )
            _, history = rl_reduced_cost_policy.train_policy(
                train_graphs,
                save_path=str(model_path),
                config=config,
                log_every_epoch=True,
            )
        active_model_path = str(model_path)

    rows = []
    eval_ids = range(args.eval_start, args.eval_start + args.eval_graphs)
    for exp_id in eval_ids:
        baseline, baseline_seconds = time_graph(
            lambda exp_id=exp_id: montecarlo.get_solution_graph(
                load_density_graph(env_dir, exp_id, args.density, args.target_type),
                num_predecessors=args.num_predecessors,
                num_rounds=args.num_rounds,
                num_trials=args.num_trials,
            ),
            seed=exp_id,
        )
        rows.append(
            {
                "exp_id": exp_id,
                "scheme": "montecarlo_cpu",
                "throughput": baseline,
                "seconds": baseline_seconds,
                "ratio_to_montecarlo": 1.0,
            }
        )

        throughput, seconds = time_graph(
            lambda exp_id=exp_id: rl_reduced_cost_policy.get_solution_graph(
                load_density_graph(env_dir, exp_id, args.density, args.target_type),
                model_path=active_model_path,
                num_rounds=args.num_rounds,
                num_trials=args.num_trials,
                policy_device=device,
                alpha_list=args.alpha_list,
                num_policy_paths=args.num_policy_paths,
                noise_scale=args.noise_scale,
                learned_mix=args.learned_mix,
                max_hop=args.max_hop,
                hop_slack=args.hop_slack,
                user_order=args.user_order,
                include_classic_paths=not args.no_classic_paths,
                exact_commit=not args.no_exact_commit,
                dual_mode=args.dual_mode,
                lookahead_weight=args.lookahead_weight,
                barrier_weight=args.barrier_weight,
                catastrophe_threshold=args.catastrophe_threshold,
                dual_anchor=args.dual_anchor,
            ),
            seed=exp_id,
        )
        rows.append(
            {
                "exp_id": exp_id,
                "scheme": (
                    "rl_reduced_cost_zero_shot"
                    if args.zero_shot
                    else (
                        "rl_reduced_cost_direct_rl"
                        if args.direct_rl
                        else "rl_reduced_cost"
                    )
                ),
                "throughput": throughput,
                "seconds": seconds,
                "ratio_to_montecarlo": throughput / baseline if baseline > 0 else 0.0,
            }
        )

    report = {
        "args": vars(args),
        "device": device,
        "history": history,
        "rows": rows,
        "summary": summarize(rows),
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["summary"], indent=2))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
