import json
import multiprocessing
import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

# ssir-related imports
from ssir import basestations as bs
from ssir.pathfinder import (
    astar,
    bruteforce,
    genetic,
    greedy,
    montecarlo,
)

"""
This script performs experiments by sweeping power_level from 0.0 to 0.9,
while keeping the graph structure fixed. Instead of modifying basestation.py,
we override each BaseStation node's config directly within this experimental script.
"""


def save_config(config, dir_path, filename="config.json"):
    # Save config dictionary as a JSON file
    os.makedirs(dir_path, exist_ok=True)
    filepath = os.path.join(dir_path, filename)
    with open(filepath, "w") as f:
        json.dump(config, f, indent=4)


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


def run_schemes(graph):
    """
    Run all pathfinding/optimization schemes on the given graph.
    Returns a dictionary: {scheme_name: (solution_graph, throughput)}.
    """
    scheme_results = {}

    # A* using distance metric
    costs, predecessors = astar.a_star(graph, metric="distance")
    g_distance = astar.get_solution_graph(graph, predecessors)
    scheme_results["astar_distance"] = (
        g_distance,
        g_distance.compute_network_throughput(),
    )

    # Check if the graph is feasible
    for user in graph.users:
        path = astar.get_shortest_path(predecessors, user.get_id())
        if path[0] == -1:
            graph.reset()
            scheme_results["astar_distance"] = (graph, 0)
            scheme_results["astar_hop"] = (graph, 0)
            scheme_results["astar_spectral_efficiency"] = (graph, 0)
            scheme_results["genetic"] = (graph, 0)
            scheme_results["montecarlo"] = (graph, 0)
            scheme_results["greedy"] = (graph, 0)
            scheme_results["bruteforce"] = (graph, 0)
            return scheme_results

    # A* using hop count
    costs, predecessors = astar.a_star(graph, metric="hop")
    g_hop = astar.get_solution_graph(graph, predecessors)
    scheme_results["astar_hop"] = (g_hop, g_hop.compute_network_throughput())

    # A* using spectral efficiency
    costs, predecessors = astar.a_star(graph, metric="spectral_efficiency")
    g_spectral = astar.get_solution_graph(graph, predecessors)
    scheme_results["astar_spectral_efficiency"] = (
        g_spectral,
        g_spectral.compute_network_throughput(),
    )

    # Genetic algorithm
    g_genetic, genetic_throughput = genetic.get_solution_graph(graph)
    scheme_results["genetic"] = (g_genetic, genetic_throughput)

    # Monte Carlo method
    g_montecarlo = montecarlo.get_solution_graph(graph, 100, 5, 20, verbose=False)
    scheme_results["montecarlo"] = (
        g_montecarlo,
        g_montecarlo.compute_network_throughput(),
    )

    # Greedy algorithm
    g_greedy = greedy.get_solution_graph(graph, 50, verbose=False)
    scheme_results["greedy"] = (
        g_greedy,
        g_greedy.compute_network_throughput(),
    )

    # Brute-force search
    g_bruteforce = bruteforce.get_solution_graph(graph, 2000, verbose=False)
    scheme_results["bruteforce"] = (
        g_bruteforce,
        g_bruteforce.compute_network_throughput(),
    )

    return scheme_results


def run_one_experiment_with_power(power_level, exp_id, base_dir, env_dir):
    """
    Run one experiment for a given power level and exp_id.

    power_level (float): scaling factor (e.g., 0.0 ~ 0.9)
    exp_id (int): experiment ID
    base_dir (str): base output directory
    env_dir (str): directory to store or load environment graphs
    """
    # 1) Load master graph and config
    config, graph = load_exp_graph(exp_id, env_dir)

    # 2) Adjust power capacity of each base station node
    for bs_node in graph.basestations:
        # old_cfg = bs_node.basestation_type.config
        # new_cfg = replace(old_cfg, minimum_transit_power_ratio=power_level)
        bs_node.basestation_type.config.minimum_transit_power_ratio = power_level

    graph.reset()
    graph.connect_reachable_nodes()

    # 3) Run optimization schemes
    scheme_results = run_schemes(graph)

    # 4) Save results to disk
    out_dir = os.path.join(base_dir, f"power_{power_level:.1f}", f"exp_{exp_id:03d}")
    os.makedirs(out_dir, exist_ok=True)
    throughput_dict = {}
    result_str_parts = []
    for scheme_name, (scheme_graph, throughput) in scheme_results.items():
        # Save solution graph
        scheme_graph.save_graph(
            os.path.join(out_dir, f"solution_{scheme_name}.pkl"), pkl=True
        )
        throughput_dict[scheme_name] = throughput
        result_str_parts.append(f"{scheme_name}={throughput:.2f}")

    # Save throughput metrics
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(throughput_dict, f, indent=4)

    summary_str = f"Power={power_level:.2f}, Exp={exp_id:03d} | " + " ".join(
        result_str_parts
    )
    return power_level, throughput_dict, summary_str


def run_task(args):
    # Unpack arguments and run a single experiment
    return run_one_experiment_with_power(*args)


def main_experiment():
    # Define power levels to test (0.0 to 0.9 with step size 0.1)
    power_levels_to_test = np.arange(0.95, 0.49, -0.05)
    print(power_levels_to_test)
    start = 0
    num_experiments = 10

    base_dir = "./results_mmf_vs_power"
    os.makedirs(base_dir, exist_ok=True)

    env_dir = os.path.join(base_dir, "env")
    os.makedirs(env_dir, exist_ok=True)

    # Generate all (power_level, exp_id) task combinations
    tasks = [
        (p, e, base_dir, env_dir)
        for p in power_levels_to_test
        for e in range(start, num_experiments + start)
    ]
    total_tasks = len(tasks)
    power_results_map = defaultdict(list)
    completed = 0

    pbar = tqdm(total=total_tasks, desc="Overall Progress", position=0, leave=True)

    # Run experiments in parallel using multiprocessing
    with multiprocessing.Pool(processes=os.cpu_count(), maxtasksperchild=1) as pool:
        for result in pool.imap_unordered(run_task, tasks):
            completed += 1
            pbar.update(1)
            power_level, throughput_dict, summary_str = result
            tqdm.write(f"[{completed}/{total_tasks}] {summary_str}")

            power_results_map[power_level].append(throughput_dict)

            # Print summary after collecting all runs for a power level
            if len(power_results_map[power_level]) == num_experiments:
                aggregate = {}
                for td in power_results_map[power_level]:
                    for scheme, val in td.items():
                        aggregate[scheme] = aggregate.get(scheme, 0) + val
                avg_summary = {k: v / num_experiments for k, v in aggregate.items()}
                avg_str_parts = [f"{k}={v:.2f}" for k, v in avg_summary.items()]
                tqdm.write(
                    f"[Summary] Power={power_level:.2f} | " + " ".join(avg_str_parts)
                )
    pbar.close()
    print("All experiments finished.")


if __name__ == "__main__":
    main_experiment()
