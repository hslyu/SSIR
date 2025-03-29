import json
import os

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

import ssir.environment as env
from ssir import basestations as bs
from ssir.pathfinder import astar, bruteforce, genetic, montecarlo


def save_config(config, dir_path, filename="config.json"):
    """Save config dictionary as a JSON file"""
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

    g_bruteforce = bruteforce.get_solution_graph(graph)
    scheme_results["bruteforce"] = (
        g_bruteforce,
        g_bruteforce.compute_network_throughput(),
    )

    return scheme_results


def run_one_experiment(exp_id, threshold, base_dir, env_dir):
    # Load or create config and graph for this exp_id
    config, graph = load_exp_graph(exp_id, env_dir)

    # Set SPSC_probability
    bs.environmental_variables.SPSC_probability = threshold

    # Run pathfinding schemes on the same graph
    scheme_results = run_schemes(graph)

    # Save results under spsc_{threshold}/exp_{exp_id}/scheme_name
    threshold_dir = os.path.join(base_dir, f"spsc_{threshold:.4f}")
    exp_dir = os.path.join(threshold_dir, f"exp_{exp_id:03d}")
    os.makedirs(exp_dir, exist_ok=True)

    # Collect throughput for printing
    result_str_parts = []
    for scheme_name, (scheme_graph, throughput) in scheme_results.items():
        # Save throughput in result.json
        scheme_dir = os.path.join(exp_dir, scheme_name)
        os.makedirs(scheme_dir, exist_ok=True)

        result_path = os.path.join(scheme_dir, "result.json")
        with open(result_path, "w") as f:
            json.dump({"throughput": throughput}, f, indent=4)

        # Save the scheme-specific graph in pickle
        graph_path = os.path.join(scheme_dir, "solution.pkl")
        scheme_graph.save_graph(graph_path, pkl=True)

        result_str_parts.append(f"{scheme_name}={throughput:.2f}")

    # Print summary for this experiment
    summary_str = f"[Exp] Threshold={threshold:.4f}, Exp={exp_id:03d} | " + " ".join(
        result_str_parts
    )
    return summary_str


def main_experiment():
    """
    This experiment will:
      1) Swap loop order: outer loop on thresholds, inner loop on experiments.
      2) Parallelize the experiment loop using joblib.
      3) For each (threshold, exp_id), load or create config+graph from base_dir/env/exp_{exp_id:03d}.
      4) Run all schemes, save throughput in result.json, and solution graph as solution.pkl.
    """
    # Define thresholds
    raw_logspace = np.logspace(-1, -4, 15)
    thresholds_to_test = 1 - raw_logspace

    # Number of experiments
    num_experiments = 30  # could be 500 or any large number

    # Base directory for all results
    base_dir = "./results_mmf_vs_spsc"
    os.makedirs(base_dir, exist_ok=True)

    # Directory to store environment config and graph
    env_dir = os.path.join(base_dir, "env")
    os.makedirs(env_dir, exist_ok=True)

    for threshold in tqdm(thresholds_to_test, desc="Threshold Loop"):
        # Parallel execution for each exp_id
        summaries = Parallel(n_jobs=-1)(
            delayed(run_one_experiment)(exp_id, threshold, base_dir, env_dir)
            for exp_id in range(num_experiments)
        )

        # Print results from all experiments
        for summary_str in summaries:
            print(summary_str)


if __name__ == "__main__":
    main_experiment()
