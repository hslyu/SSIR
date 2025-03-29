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

    # Collect throughput for saving and printing
    throughput_dict = {}
    result_str_parts = []
    for scheme_name, (scheme_graph, throughput) in scheme_results.items():
        # Save graph as solution_<scheme>.pkl
        graph_path = os.path.join(exp_dir, f"solution_{scheme_name}.pkl")
        scheme_graph.save_graph(graph_path, pkl=True)

        # Collect throughput
        throughput_dict[scheme_name] = throughput
        result_str_parts.append(f"{scheme_name}={throughput:.2f}")

    # Save all throughputs in one result.json
    result_path = os.path.join(exp_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(throughput_dict, f, indent=4)

    # Print summary for this experiment
    summary_str = f"[Exp] Threshold={threshold:.4f}, Exp={exp_id:03d} | " + " ".join(
        result_str_parts
    )
    return throughput_dict, summary_str


def main_experiment():
    """
    This experiment will:
      1) Use a single TQDM progress bar for all threshold * experiment tasks.
      2) For each threshold, run experiments in parallel.
      3) Print partial progress in real time, and after finishing the chunk of experiments for a threshold, print average summary.
    """
    # Define thresholds
    raw_logspace = np.logspace(-1, -4, 15)
    thresholds_to_test = 1 - raw_logspace

    # Number of experiments
    num_experiments = 10  # could be 500 or any large number

    # Base directory for all results
    base_dir = "./results_mmf_vs_spsc"
    os.makedirs(base_dir, exist_ok=True)

    # Directory to store environment config and graph
    env_dir = os.path.join(base_dir, "env")
    os.makedirs(env_dir, exist_ok=True)

    # Calculate total tasks for progress bar
    total_tasks = len(thresholds_to_test) * num_experiments
    with tqdm(total=total_tasks, desc="Overall Progress") as pbar:
        for threshold in thresholds_to_test:
            # Run all experiments for this threshold in parallel
            results = Parallel(n_jobs=-1)(
                delayed(run_one_experiment)(exp_id, threshold, base_dir, env_dir)
                for exp_id in range(num_experiments)
            )

            # Print partial results as soon as each experiment returns
            # (Since joblib gathers after all tasks are done in parallel for this chunk,
            #  we print them after the chunk completes. But we still update the progress bar for each.)
            for _, summary_str in results:
                pbar.update(1)
                # Show how many tasks are done out of total
                current_count = pbar.n
                total_count = pbar.total
                print(f"[{current_count}/{total_count}] {summary_str}")

            # After finishing all experiments for this threshold, compute average summary
            aggregate = {}
            for throughput_dict, _ in results:
                for scheme, val in throughput_dict.items():
                    aggregate[scheme] = aggregate.get(scheme, 0) + val

            avg_summary = {k: v / num_experiments for k, v in aggregate.items()}
            avg_str = f"[Summary] Threshold={threshold:.4f} | " + " ".join(
                [f"{k}={v:.2f}" for k, v in avg_summary.items()]
            )
            print(avg_str)


if __name__ == "__main__":
    main_experiment()
