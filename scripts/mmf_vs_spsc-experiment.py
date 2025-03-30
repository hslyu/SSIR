import concurrent.futures
import json
import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

# ssir-related imports
import ssir.environment as env
from ssir import basestations as bs
from ssir.pathfinder import astar, bruteforce, genetic, montecarlo


def save_config(config, dir_path, filename="config.json"):
    """Save config dictionary as a JSON file."""
    os.makedirs(dir_path, exist_ok=True)
    filepath = os.path.join(dir_path, filename)
    with open(filepath, "w") as f:
        json.dump(config, f, indent=4)


def load_exp_graph(exp_id, env_dir):
    """
    For a given exp_id, check if config.json and graph.pkl exist in env_dir/exp_{exp_id:03d}.
    If they do, load them. Otherwise, generate a new config & graph, then save them.
    Returns (config, graph).
    """
    exp_dir = os.path.join(env_dir, f"exp_{exp_id:03d}")
    os.makedirs(exp_dir, exist_ok=True)

    config_path = os.path.join(exp_dir, "config.json")
    graph_path = os.path.join(exp_dir, "graph.pkl")

    if os.path.isfile(config_path) and os.path.isfile(graph_path):
        # Load existing config
        with open(config_path, "r") as f:
            config = json.load(f)
        # Load existing graph
        loaded_graph = bs.IABRelayGraph()
        loaded_graph.load_graph(graph_path, pkl=True)
        return config, loaded_graph
    else:
        # Create a new config
        config = env.generate_config(exp_id)
        save_config(config, exp_dir, "config.json")

        # Create a new graph
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
    g_genetic, _ = genetic.get_solution_graph(graph, verbose=False)
    scheme_results["genetic"] = (g_genetic, g_genetic.compute_network_throughput())

    # Montecarlo
    g_montecarlo = montecarlo.get_solution_graph(graph)
    scheme_results["montecarlo"] = (
        g_montecarlo,
        g_montecarlo.compute_network_throughput(),
    )

    # Bruteforce
    g_bruteforce = bruteforce.get_solution_graph(graph)
    scheme_results["bruteforce"] = (
        g_bruteforce,
        g_bruteforce.compute_network_throughput(),
    )

    return scheme_results


def run_one_experiment(threshold, exp_id, base_dir, env_dir):
    """
    Single experiment for a given (threshold, exp_id).
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

    # Collect throughput for saving and printing
    throughput_dict = {}
    result_str_parts = []
    for scheme_name, (scheme_graph, throughput) in scheme_results.items():
        # Save graph as solution_<scheme>.pkl
        graph_path = os.path.join(exp_dir, f"solution_{scheme_name}.pkl")
        scheme_graph.save_graph(graph_path, pkl=True)

        throughput_dict[scheme_name] = throughput
        result_str_parts.append(f"{scheme_name}={throughput:.2f}")

    # Save all throughputs in one result.json
    result_path = os.path.join(exp_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(throughput_dict, f, indent=4)

    # Build a summary string
    summary_str = f"[Exp] Threshold={threshold:.4f}, Exp={exp_id:03d} | " + " ".join(
        result_str_parts
    )

    return threshold, throughput_dict, summary_str


def main_experiment():
    """
    Demonstrates real-time partial results using concurrent.futures.ProcessPoolExecutor.
    - Each (threshold, exp_id) is submitted as an individual task.
    - We gather results as they complete (as_completed).
    - We keep a global progress bar (tqdm), incremented once per completed task.
    - Whenever all tasks for a threshold have arrived, print an average summary.
    """
    raw_logspace = np.logspace(-1, -4, 15)
    thresholds_to_test = 1 - raw_logspace
    num_experiments = 10

    base_dir = "./results_mmf_vs_spsc"
    os.makedirs(base_dir, exist_ok=True)

    env_dir = os.path.join(base_dir, "env")
    os.makedirs(env_dir, exist_ok=True)

    # Prepare all tasks
    tasks = [(t, e) for t in thresholds_to_test for e in range(num_experiments)]
    total_tasks = len(tasks)

    # This dict accumulates results for each threshold
    # threshold -> [throughput_dict, throughput_dict, ...] up to num_experiments
    threshold_results_map = defaultdict(list)

    print(f"Total tasks: {total_tasks}")
    with tqdm(total=total_tasks, desc="Overall Progress") as pbar:
        # Use a process pool
        with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
            # Submit all tasks
            future_map = {
                executor.submit(run_one_experiment, t, e, base_dir, env_dir): (t, e)
                for (t, e) in tasks
            }

            # as_completed gives us futures in the order they finish
            for future in concurrent.futures.as_completed(future_map):
                pbar.update(1)
                threshold, throughput_dict, summary_str = future.result()

                current_count = pbar.n
                total_count = pbar.total

                # Print partial results immediately
                print(f"[{current_count}/{total_count}] {summary_str}")

                # Accumulate results
                threshold_results_map[threshold].append(throughput_dict)

                # If threshold is fully done, we can compute average
                if len(threshold_results_map[threshold]) == num_experiments:
                    aggregate = {}
                    for td in threshold_results_map[threshold]:
                        for scheme, val in td.items():
                            aggregate[scheme] = aggregate.get(scheme, 0) + val
                    avg_summary = {k: v / num_experiments for k, v in aggregate.items()}
                    avg_str_parts = [f"{k}={v:.2f}" for k, v in avg_summary.items()]
                    print(
                        f"[Summary] Threshold={threshold:.4f} | "
                        + " ".join(avg_str_parts)
                    )

    print("All experiments finished.")


if __name__ == "__main__":
    main_experiment()
