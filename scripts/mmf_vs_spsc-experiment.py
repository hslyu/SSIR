import json
import multiprocessing
import os
import time
from collections import defaultdict

import numpy as np
from tqdm import tqdm

# ssir-related imports
import ssir.environment as env
from ssir import basestations as bs
from ssir.pathfinder import (
    astar,
    bruteforce,
    bruteforce_fast,
    genetic,
    greedy,
    montecarlo,
)


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

    if os.path.isfile(config_path) and os.path.isfile(graph_path):
        # Load existing config and graph
        with open(config_path, "r") as f:
            config = json.load(f)
        loaded_graph = bs.IABRelayGraph()
        loaded_graph.load_graph(graph_path, pkl=True)
        return config, loaded_graph
    else:
        # Create new config and graph
        config = env.generate_config(exp_id)
        save_config(config, exp_dir, "config.json")

        dm = env.DataManager(**config)
        new_graph = dm.generate_master_graph()
        new_graph.save_graph(graph_path, pkl=True)
        return config, new_graph


def run_schemes(graph):
    """
    Run all pathfinding/optimization schemes on the given graph.
    Returns a dict: {scheme_name: (scheme_graph, throughput)}.
    """
    scheme_results = {}

    # A* distance
    costs, predecessors = astar.a_star(graph, metric="distance")
    g_distance = astar.get_solution_graph(graph, predecessors)
    scheme_results["astar_distance"] = (
        g_distance,
        g_distance.compute_network_throughput(),
    )

    # A* hop
    costs, predecessors = astar.a_star(graph, metric="hop")
    g_hop = astar.get_solution_graph(graph, predecessors)
    scheme_results["astar_hop"] = (g_hop, g_hop.compute_network_throughput())

    # A* spectral_efficiency
    costs, predecessors = astar.a_star(graph, metric="spectral_efficiency")
    g_spectral = astar.get_solution_graph(graph, predecessors)
    scheme_results["astar_spectral_efficiency"] = (
        g_spectral,
        g_spectral.compute_network_throughput(),
    )

    # Genetic
    g_genetic, genetic_throughput = genetic.get_solution_graph(graph)
    scheme_results["genetic"] = (g_genetic, genetic_throughput)

    # Montecarlo
    g_montecarlo = montecarlo.get_solution_graph(graph, 100, 5, 20, verbose=False)
    scheme_results["montecarlo"] = (
        g_montecarlo,
        g_montecarlo.compute_network_throughput(),
    )

    # Greedy
    g_greedy = greedy.get_solution_graph(graph, 50, verbose=False)
    scheme_results["greedy"] = (
        g_greedy,
        g_greedy.compute_network_throughput(),
    )

    # Bruteforce
    g_bruteforce = bruteforce.get_solution_graph(graph, 2000, verbose=False)
    scheme_results["bruteforce"] = (
        g_bruteforce,
        g_bruteforce.compute_network_throughput(),
    )

    return scheme_results


def run_one_experiment(threshold, exp_id, base_dir, env_dir):
    """
    Run a single experiment for a given (threshold, exp_id).
    Returns (threshold, throughput_dict, summary_str).
    """
    # Load or create config and graph
    config, graph = load_exp_graph(exp_id, env_dir)

    # Set SPSC_probability
    bs.environmental_variables.SPSC_probability = threshold

    # Run pathfinding schemes
    scheme_results = run_schemes(graph)

    # Save results under spsc_{threshold}/exp_{exp_id}/
    threshold_dir = os.path.join(base_dir, f"spsc_{threshold:.4f}")
    exp_dir = os.path.join(threshold_dir, f"exp_{exp_id:03d}")
    os.makedirs(exp_dir, exist_ok=True)

    throughput_dict = {}
    result_str_parts = []
    for scheme_name, (scheme_graph, throughput, t) in scheme_results.items():
        # Save the graph as solution_<scheme>.pkl
        graph_path = os.path.join(exp_dir, f"solution_{scheme_name}.pkl")
        scheme_graph.save_graph(graph_path, pkl=True)

        throughput_dict[scheme_name] = throughput
        result_str_parts.append(f"{scheme_name}={throughput:.2f}, {int(t)}s |")

    # Save the throughput results in result.json
    result_path = os.path.join(exp_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(throughput_dict, f, indent=4)

    # Build summary string including experiment index
    summary_str = f"[Exp] Threshold={threshold:.4f}, Exp={exp_id:03d} | " + " ".join(
        result_str_parts
    )
    return threshold, throughput_dict, summary_str


def run_task(args):
    # Unpack arguments and execute a single experiment task
    return run_one_experiment(*args)


def main_experiment():
    """
    Run experiments in parallel using multiprocessing Pool with maxtasksperchild=1.
    After each task finishes, print progress [completed/total] and the experiment summary.
    The tqdm progress bar is fixed at the top, and outputs are printed below.
    """
    raw_logspace = np.logspace(-1, -4, 10, base=10)
    thresholds_to_test = 1 - raw_logspace
    start = 0
    num_experiments = 200

    base_dir = "./results_mmf_vs_spsc"
    os.makedirs(base_dir, exist_ok=True)

    env_dir = os.path.join(base_dir, "env")
    os.makedirs(env_dir, exist_ok=True)

    # Prepare all tasks: (threshold, exp_id, base_dir, env_dir)
    tasks = [
        (t, e, base_dir, env_dir)
        for t in thresholds_to_test
        for e in range(start, num_experiments + start)
    ]
    total_tasks = len(tasks)
    threshold_results_map = defaultdict(list)
    completed = 0

    # Set up tqdm progress bar fixed at the top (position=0)
    pbar = tqdm(total=total_tasks, desc="Overall Progress", position=0, leave=True)

    # Create a multiprocessing Pool with maxtasksperchild=1 to mitigate memory leakage
    with multiprocessing.Pool(processes=15, maxtasksperchild=1) as pool:
        # Use imap_unordered to process tasks as they complete
        for result in pool.imap_unordered(run_task, tasks):
            completed += 1
            pbar.update(1)
            threshold, throughput_dict, summary_str = result

            # Print the progress and summary; tqdm.write ensures the progress bar stays fixed
            tqdm.write(f"[{completed}/{total_tasks}] {summary_str}")

            # Accumulate results per threshold
            threshold_results_map[threshold].append(throughput_dict)
            if len(threshold_results_map[threshold]) == num_experiments:
                aggregate = {}
                for td in threshold_results_map[threshold]:
                    for scheme, val in td.items():
                        aggregate[scheme] = aggregate.get(scheme, 0) + val
                avg_summary = {k: v / num_experiments for k, v in aggregate.items()}
                avg_str_parts = [f"{k}={v:.2f}" for k, v in avg_summary.items()]
                tqdm.write(
                    f"[Summary] Threshold={threshold:.4f} | " + " ".join(avg_str_parts)
                )

    pbar.close()
    print("All experiments finished.")


if __name__ == "__main__":
    main_experiment()
