#!/usr/bin/env python3
import json
import multiprocessing as mp
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# SSIR‑related imports
from ssir import basestations as bs
from ssir.pathfinder import astar, bruteforce, genetic, greedy, montecarlo

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_exp_graph(exp_id, env_dir):
    """
    Load existing config and graph for exp_id, or generate new ones if not found.
    """
    exp_dir = os.path.join(env_dir, f"exp_{exp_id:03d}")
    os.makedirs(exp_dir, exist_ok=True)

    config_path = os.path.join(exp_dir, "config.json")
    graph_path = os.path.join(exp_dir, "graph.pkl")

    # Load existing config and graph
    with open(config_path, "r") as f:
        config = json.load(f)
    loaded_graph = bs.IABRelayGraph()
    loaded_graph.load_graph(graph_path, pkl=True)
    return config, loaded_graph


# ---------------------------------------------------------------------------
# Optimisation schemes
# ---------------------------------------------------------------------------


def run_schemes(
    graph: bs.IABRelayGraph,
) -> Dict[str, Tuple[bs.IABRelayGraph, float]]:
    """
    Execute every path‑finding / optimisation scheme on *graph*.

    Returned dict maps scheme names to ``(solution_graph, throughput)``.
    """
    results: Dict[str, Tuple[bs.IABRelayGraph, float]] = {}

    # ----- A* (distance) ----------------------------------------------------
    _, predecessors = astar.a_star(graph, metric="distance")
    g_distance = astar.get_solution_graph(graph, predecessors)
    results["astar_distance"] = (g_distance, g_distance.compute_network_throughput())

    # Abort if any user is unreachable -------------------------------------
    unreachable = any(
        astar.get_shortest_path(predecessors, user.get_id())[0] == -1
        for user in graph.users
    )
    if unreachable:
        for key in [
            "astar_distance",
            "astar_hop",
            "astar_spectral_efficiency",
            "genetic",
            "montecarlo",
            "greedy",
            "bruteforce",
        ]:
            results[key] = (graph, 0.0)
        return results

    # ----- A* (hop) ---------------------------------------------------------
    _, predecessors = astar.a_star(graph, metric="hop")
    g_hop = astar.get_solution_graph(graph, predecessors)
    results["astar_hop"] = (g_hop, g_hop.compute_network_throughput())

    # ----- A* (spectral efficiency) ----------------------------------------
    _, predecessors = astar.a_star(graph, metric="spectral_efficiency")
    g_se = astar.get_solution_graph(graph, predecessors)
    results["astar_spectral_efficiency"] = (
        g_se,
        g_se.compute_network_throughput(),
    )

    # ----- Genetic algorithm ----------------------------------------------
    g_genetic, thr_genetic = genetic.get_solution_graph(graph)
    results["genetic"] = (g_genetic, thr_genetic)

    # ----- Monte Carlo -----------------------------------------------------
    g_mc = montecarlo.get_solution_graph(graph, 100, 5, 20, verbose=False)
    results["montecarlo"] = (g_mc, g_mc.compute_network_throughput())

    # ----- Greedy ----------------------------------------------------------
    g_greedy = greedy.get_solution_graph(graph, 50, verbose=False)
    results["greedy"] = (g_greedy, g_greedy.compute_network_throughput())

    # ----- Brute force -----------------------------------------------------
    g_bf = bruteforce.get_solution_graph(graph, 2_000, verbose=False)
    results["bruteforce"] = (g_bf, g_bf.compute_network_throughput())

    return results


# ---------------------------------------------------------------------------
# Single‑experiment routine
# ---------------------------------------------------------------------------


def run_single_experiment(
    density: float,
    exp_id: int,
    base_dir: Path,
    env_dir: Path,
    target_bs_type: List[str],
) -> Tuple[float, Dict[str, float], str]:
    """Run one experiment instance for the specified *density* and *exp_id*."""
    # 1. Load base topology --------------------------------------------------
    _, graph = load_exp_graph(exp_id, env_dir)

    # 2. Override eavesdropper density for *every* BaseStation --------------
    for bs_node in graph.basestations:
        if bs_node.basestation_type.name in target_bs_type:
            bs_node.basestation_type.config.eavesdropper_density = density

    # 3. Re‑connect and optimise -------------------------------------------
    graph.reset()
    graph.connect_reachable_nodes()
    scheme_results = run_schemes(graph)

    # 4. Persist results ----------------------------------------------------
    out_dir = base_dir / f"density_{density:.2e}" / f"exp_{exp_id:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5. Load existing results if any
    result_path = out_dir / "result.json"

    throughputs: Dict[str, float] = {}
    if result_path.exists():
        with open(result_path, "r") as fp:
            throughputs = json.load(fp)

    msg_parts: List[str] = []
    for name, (sol_graph, tp) in scheme_results.items():
        sol_graph.save_graph(out_dir / f"solution_{name}.pkl", pkl=True)
        throughputs[name] = tp
        msg_parts.append(f"{name}={tp:.2f}")

    # Write JSON summary
    with open(out_dir / "result.json", "w") as fp:
        json.dump(throughputs, fp, indent=4)

    summary_msg = f"Density={density:.2e}, Exp={exp_id:03d} | " + " ".join(msg_parts)
    return density, throughputs, summary_msg


# Wrapper is required because Pool.imap_unordered expects a single argument
def run_task(args):
    # Unpack arguments and run a single experiment
    return run_single_experiment(*args)


# ---------------------------------------------------------------------------
# Script entry‑point
# ---------------------------------------------------------------------------


def main() -> None:
    target_bs_type = [bs.BaseStationType.MARITIME.name]
    eaves_densities: np.ndarray = np.logspace(-5, -1, 13, base=10)

    start: int = 0
    num_experiments: int = 1000

    base_dir = Path("./results_mmf_vs_density").resolve()
    env_dir = base_dir / "env"
    env_dir.mkdir(parents=True, exist_ok=True)

    # Create task list ------------------------------------------------------
    tasks = [
        (dens, exp, base_dir, env_dir, target_bs_type)
        for exp in range(start, start + num_experiments)
        for dens in eaves_densities
    ]
    total_tasks = len(tasks)

    # Run with multiprocessing --------------------------------------------
    progress_bar = tqdm(total=total_tasks, desc="Overall progress", position=0)

    density_to_results: defaultdict = defaultdict(list)
    completed: int = 0

    with mp.Pool(processes=os.cpu_count() - 4, maxtasksperchild=1) as pool:
        for dens, tp_dict, summary in pool.imap_unordered(run_task, tasks):
            completed += 1
            progress_bar.update(1)
            tqdm.write(f"[{completed}/{total_tasks}] {summary}")

            density_to_results[dens].append(tp_dict)
            if len(density_to_results[dens]) == num_experiments:
                # Aggregate mean throughput per scheme --------------------
                aggregate: Dict[str, float] = defaultdict(float)
                for one_run in density_to_results[dens]:
                    for scheme, val in one_run.items():
                        aggregate[scheme] += val
                avg: Dict[str, float] = {
                    k: v / num_experiments for k, v in aggregate.items()
                }
                avg_str = " ".join(f"{k}={v:.2f}" for k, v in avg.items())
                tqdm.write(f"[Summary] Eve density={dens:.2e} | {avg_str}")

    progress_bar.close()
    print("All experiments finished.")


if __name__ == "__main__":  # pragma: no cover
    main()
