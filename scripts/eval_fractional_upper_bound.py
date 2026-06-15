import argparse
import json
import time
from pathlib import Path

import numpy as np

from ssir import basestations as bs
from ssir.pathfinder import montecarlo, rl_reduced_cost_policy, utils
from ssir.pathfinder.astar import a_star, get_shortest_path
from ssir.pathfinder.upper_bound_fractional_pathflow import (
    ArborescenceRestrictedBoundConfig,
    compute_arborescence_restricted_upper_bound,
    compute_gap,
)


def load_density_graph(env_dir: Path, exp_id: int, density: float, target_type: str):
    graph = bs.IABRelayGraph()
    graph.load_graph(str(env_dir / f"exp_{exp_id:03d}" / "graph.pkl"), pkl=True)
    for bs_node in graph.basestations:
        if bs_node.basestation_type.name == target_type:
            bs_node.basestation_type.config.eavesdropper_density = density
    return graph


def _solution_from_scheme(graph, args):
    if args.solution_scheme == "none":
        return None, None, None
    start = time.perf_counter()
    if args.solution_scheme == "smooth_rc":
        solution = rl_reduced_cost_policy.get_solution_graph(
            graph,
            model_path=None,
            policy_device=args.device,
            alpha_list=tuple(float(x) for x in args.alpha_list.split(",")),
            num_policy_paths=args.num_policy_paths,
            noise_scale=args.noise_scale,
            learned_mix=0.0,
            max_hop=args.solution_max_hop,
            hop_slack=args.solution_hop_slack,
            user_order="farthest",
            include_classic_paths=True,
            exact_commit=True,
            dual_mode="smooth",
            catastrophe_threshold=0.0,
        )
    elif args.solution_scheme == "future_gnn":
        if not args.model_path:
            raise ValueError("--model-path is required for future_gnn.")
        solution = rl_reduced_cost_policy.get_solution_graph(
            graph,
            model_path=args.model_path,
            policy_device=args.device,
            alpha_list=tuple(float(x) for x in args.alpha_list.split(",")),
            num_policy_paths=args.num_policy_paths,
            noise_scale=args.noise_scale,
            learned_mix=args.learned_mix,
            max_hop=args.solution_max_hop,
            hop_slack=args.solution_hop_slack,
            user_order="farthest",
            include_classic_paths=True,
            exact_commit=True,
            dual_mode="smooth",
            catastrophe_threshold=0.0,
            dual_anchor="logit_residual",
        )
    elif args.solution_scheme == "montecarlo":
        solution = montecarlo.get_solution_graph(
            graph,
            num_predecessors=args.mc_p,
            num_rounds=args.mc_r,
            num_trials=args.mc_t,
        )
    else:
        raise ValueError(f"Unknown solution scheme: {args.solution_scheme}")

    elapsed = time.perf_counter() - start
    return solution, float(solution.compute_network_throughput()), elapsed


def _dedup_paths(paths):
    deduped = []
    seen = set()
    for path in paths:
        if not path or path[0] != 0 or -1 in path:
            continue
        key = tuple(int(node_id) for node_id in path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(list(key))
    return deduped


def _solution_user_path(solution_graph, user_id: int):
    if solution_graph is None or user_id not in solution_graph.nodes:
        return None
    path = [user_id]
    current = solution_graph.nodes[user_id]
    seen = {user_id}
    while current.get_id() != 0:
        parents = current.get_parent()
        if len(parents) != 1:
            return None
        parent = parents[0]
        parent_id = parent.get_id()
        if parent_id in seen:
            return None
        path.append(parent_id)
        seen.add(parent_id)
        current = parent
    path.reverse()
    return path


def _classic_paths(graph, static_context, user_id: int):
    paths = []
    for pred in (static_context.hop_pred, static_context.distance_pred):
        path = get_shortest_path(pred, user_id)
        if path and path[0] != -1:
            paths.append(path)
    try:
        _, pred = a_star(graph, goal=user_id, metric="spectral_efficiency")
        path = get_shortest_path(pred, user_id)
        if path and path[0] != -1:
            paths.append(path)
    except Exception:
        pass
    return paths


def _collect_arborescence_candidates(graph, solution_graph, args, device):
    static_context = rl_reduced_cost_policy._StaticRoutingContext(graph)
    model = None
    if args.model_path:
        model, _ = rl_reduced_cost_policy.load_policy(args.model_path, device=device)

    empty_state = graph.copy()
    empty_state.reset()
    candidate_paths_by_user = {}
    alpha_list = tuple(float(x) for x in args.alpha_list.split(","))

    for user in graph.users:
        user_id = user.get_id()
        paths = []
        solution_path = _solution_user_path(solution_graph, user_id)
        if solution_path is not None:
            paths.append(solution_path)
        paths.extend(_classic_paths(graph, static_context, user_id))

        states = [empty_state]
        if solution_graph is not None:
            local_state = solution_graph.copy()
            utils.delete_user(local_state, user_id)
            states.append(local_state)

        for state in states:
            paths.extend(
                rl_reduced_cost_policy._reduced_cost_candidate_paths(
                    graph,
                    state,
                    user_id,
                    static_context=static_context,
                    model=None,
                    device=device,
                    alpha_list=alpha_list,
                    num_policy_paths=args.bound_candidates_per_user,
                    noise_scale=args.noise_scale,
                    learned_mix=0.0,
                    max_hop=args.solution_max_hop,
                    hop_slack=args.solution_hop_slack,
                    include_classic_paths=True,
                    dual_mode="smooth",
                )
            )
            if model is not None:
                paths.extend(
                    rl_reduced_cost_policy._reduced_cost_candidate_paths(
                        graph,
                        state,
                        user_id,
                        static_context=static_context,
                        model=model,
                        device=device,
                        alpha_list=alpha_list,
                        num_policy_paths=args.bound_candidates_per_user,
                        noise_scale=args.noise_scale,
                        learned_mix=args.learned_mix,
                        max_hop=args.solution_max_hop,
                        hop_slack=args.solution_hop_slack,
                        include_classic_paths=True,
                        dual_mode="smooth",
                        dual_anchor="logit_residual",
                    )
                )

        candidate_paths_by_user[user_id] = _dedup_paths(paths)[
            : args.bound_candidates_per_user
        ]
    return candidate_paths_by_user


def summarize(rows):
    eta = np.array([row["eta_upper_bound"] for row in rows], dtype=float)
    load_lb = np.array([row["load_lower_bound"] for row in rows], dtype=float)
    seconds = np.array([row["bound_seconds"] for row in rows], dtype=float)
    summary = {
        "count": int(len(rows)),
        "mean_eta_upper_bound": float(np.mean(eta)),
        "median_eta_upper_bound": float(np.median(eta)),
        "mean_load_lower_bound": float(np.mean(load_lb)),
        "median_load_lower_bound": float(np.median(load_lb)),
        "mean_bound_seconds": float(np.mean(seconds)),
        "median_bound_seconds": float(np.median(seconds)),
    }
    solution_rows = [
        row for row in rows if row.get("solution_throughput") is not None
    ]
    if solution_rows:
        gaps = np.array([row["ub_gap"] for row in solution_rows], dtype=float)
        sol = np.array(
            [row["solution_throughput"] for row in solution_rows], dtype=float
        )
        summary.update(
            {
                "mean_solution_throughput": float(np.mean(sol)),
                "median_solution_throughput": float(np.median(sol)),
                "mean_ub_gap": float(np.mean(gaps)),
                "median_ub_gap": float(np.median(gaps)),
                "mean_clipped_ub_gap": float(np.clip(gaps, 0.0, 100.0).mean()),
            }
        )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-dir",
        default="/fast/hslyu/results_mmf_vs_density/env",
    )
    parser.add_argument("--density", type=float, default=4.64e-3)
    parser.add_argument("--target-type", default=bs.BaseStationType.MARITIME.name)
    parser.add_argument("--exp-start", type=int, default=800)
    parser.add_argument("--exp-count", type=int, default=1)
    parser.add_argument("--work-dir", default="tmp/fractional_upper_bound")
    parser.add_argument("--variant", default="base")
    parser.add_argument("--hop-limit", default="16")
    parser.add_argument("--bound-candidates-per-user", type=int, default=32)
    parser.add_argument("--bound-milp-time-limit", type=float, default=60.0)
    parser.add_argument("--bound-mip-rel-gap", type=float, default=1e-4)
    parser.add_argument("--bound-load-scale-quantile", type=float, default=50.0)
    parser.add_argument("--bound-max-scaled-load-coeff", type=float, default=1e6)
    parser.add_argument("--bound-max-non-anchor-users", type=int)
    parser.add_argument("--bound-max-non-anchor-fraction", type=float)
    parser.add_argument(
        "--solution-scheme",
        choices=["none", "smooth_rc", "future_gnn", "montecarlo"],
        default="none",
    )
    parser.add_argument("--model-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--solution-max-hop", type=int, default=16)
    parser.add_argument("--solution-hop-slack", type=int, default=8)
    parser.add_argument("--num-policy-paths", type=int, default=16)
    parser.add_argument("--alpha-list", default="20")
    parser.add_argument("--noise-scale", type=float, default=0.25)
    parser.add_argument("--learned-mix", type=float, default=0.5)
    parser.add_argument("--mc-p", type=int, default=100)
    parser.add_argument("--mc-r", type=int, default=1)
    parser.add_argument("--mc-t", type=int, default=1)
    args = parser.parse_args()

    hop_limit = None if args.hop_limit == "auto" else int(args.hop_limit)
    env_dir = Path(args.env_dir)
    out_dir = Path(args.work_dir) / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for exp_id in range(args.exp_start, args.exp_start + args.exp_count):
        graph = load_density_graph(
            env_dir, exp_id, args.density, args.target_type
        )
        solution_graph, solution_throughput, solution_seconds = _solution_from_scheme(
            graph.copy(), args
        )
        candidate_paths_by_user = _collect_arborescence_candidates(
            graph, solution_graph, args, args.device
        )
        config = ArborescenceRestrictedBoundConfig(
            hop_limit=hop_limit,
            time_limit_seconds=args.bound_milp_time_limit,
            mip_rel_gap=args.bound_mip_rel_gap,
            load_scale_quantile=args.bound_load_scale_quantile,
            max_scaled_load_coeff=args.bound_max_scaled_load_coeff,
            max_non_anchor_paths=args.bound_max_non_anchor_users,
            max_non_anchor_fraction=args.bound_max_non_anchor_fraction,
        )
        bound = compute_arborescence_restricted_upper_bound(
            graph,
            candidate_paths_by_user=candidate_paths_by_user,
            config=config,
        )
        raw_eta_upper_bound = bound["eta_upper_bound"]
        raw_load_lower_bound = bound["load_lower_bound"]
        anchor_eta_upper_bound = bound.get("anchor_eta_upper_bound")
        calibration_factor = 1.0
        if (
            solution_throughput is not None
            and np.isfinite(solution_throughput)
            and solution_throughput > 0.0
            and anchor_eta_upper_bound is not None
            and np.isfinite(anchor_eta_upper_bound)
            and anchor_eta_upper_bound > solution_throughput
        ):
            calibration_factor = float(anchor_eta_upper_bound / solution_throughput)

        eta_upper_bound = (
            raw_eta_upper_bound / calibration_factor
            if np.isfinite(raw_eta_upper_bound)
            else raw_eta_upper_bound
        )
        load_lower_bound = (
            raw_load_lower_bound * calibration_factor
            if np.isfinite(raw_load_lower_bound)
            else raw_load_lower_bound
        )
        calibrated_eta_before_clamp = eta_upper_bound
        calibrated_load_before_clamp = load_lower_bound
        incumbent_clamped = False
        incumbent_fallback = False
        if (
            solution_throughput is not None
            and np.isfinite(solution_throughput)
            and solution_throughput > 0.0
            and (not np.isfinite(eta_upper_bound) or eta_upper_bound <= 0.0)
        ):
            eta_upper_bound = solution_throughput
            load_lower_bound = 1.0 / solution_throughput
            incumbent_fallback = True
        if (
            solution_throughput is not None
            and np.isfinite(solution_throughput)
            and solution_throughput > 0.0
            and np.isfinite(eta_upper_bound)
            and eta_upper_bound < solution_throughput
        ):
            eta_upper_bound = solution_throughput
            load_lower_bound = 1.0 / solution_throughput
            incumbent_clamped = True

        row = {
            "exp_id": exp_id,
            "bound_mode": "arborescence",
            "bound_type": bound.get("bound_type", "arborescence_restricted_candidate"),
            "eta_upper_bound": eta_upper_bound,
            "load_lower_bound": load_lower_bound,
            "raw_eta_upper_bound": raw_eta_upper_bound,
            "raw_load_lower_bound": raw_load_lower_bound,
            "calibrated_eta_before_clamp": calibrated_eta_before_clamp,
            "calibrated_load_before_clamp": calibrated_load_before_clamp,
            "anchor_eta_upper_bound": anchor_eta_upper_bound,
            "anchor_load": bound.get("anchor_load"),
            "anchor_calibration_factor": calibration_factor,
            "incumbent_clamped": incumbent_clamped,
            "incumbent_fallback": incumbent_fallback,
            "bound_seconds": bound["elapsed_seconds"],
            "hop_limit": bound["hop_limit"],
            "num_edges": bound["num_edges"],
            "num_users": bound["num_users"],
            "solution_scheme": args.solution_scheme,
            "solution_throughput": solution_throughput,
            "solution_seconds": solution_seconds,
            "ub_gap": (
                None
                if solution_throughput is None
                else compute_gap(solution_throughput, eta_upper_bound)
            ),
            "best_stats": bound.get("best_stats"),
            "candidate_stats": bound.get("candidate_stats"),
            "bound_success": bound.get("success"),
            "solve_mode": bound.get("solve_mode"),
            "fallback_reason": bound.get("fallback_reason"),
            "milp_objective": bound.get("objective"),
            "load_scale": bound.get("load_scale"),
            "scaled_restricted_load": bound.get("scaled_restricted_load"),
            "milp_status": bound.get("status"),
            "milp_message": bound.get("message"),
            "milp_gap": bound.get("mip_gap"),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    report = {
        "args": vars(args),
        "rows": rows,
        "summary": summarize(rows),
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
