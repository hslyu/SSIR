#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import multiprocessing
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from ssir import basestations as bs
from ssir.pathfinder import bruteforce

graph_table = {}


def get_exp_graph(exp_id, env_dir):
    if exp_id not in graph_table:
        exp_dir = os.path.join(env_dir, f"exp_{exp_id:03d}")
        os.makedirs(exp_dir, exist_ok=True)
        graph_path = os.path.join(exp_dir, "graph.pkl")
        g = bs.IABRelayGraph()
        g.load_graph(graph_path, pkl=True)
        graph_table[exp_id] = g
    return graph_table[exp_id]


def run_schemes(graph):
    scheme_results = {}
    g_bf_5k = bruteforce.get_solution_graph(graph, 5000, verbose=False, early_stop=1000)
    scheme_results["bruteforce_5000"] = g_bf_5k.compute_network_throughput()
    g_bf_50k = bruteforce.get_solution_graph(graph, 50000, verbose=True, early_stop=1000)
    scheme_results["bruteforce_20000"] = g_bf_50k.compute_network_throughput()
    return scheme_results


def run_one_experiment(threshold, exp_id, env_dir):
    graph = get_exp_graph(exp_id, env_dir)
    bs.environmental_variables.SPSC_probability = threshold
    return threshold, run_schemes(graph)


def run_task(args):
    return run_one_experiment(*args)


def main_experiment():
    raw_logspace = np.concatenate(
    (np.logspace(-5, -4, 7, base=10)[:-1], np.logspace(-4, 0, 13, base=10)[:-6])
    )
    thresholds_to_test = 1 - raw_logspace
    start = 0
    num_experiments = 100

    env_dir = "/fast/hslyu/mmf_vs_spsc/mmf_result_1/env"
    if not os.path.isdir(env_dir):
        print(f"Skipping bruteforce_test: result directory not found: {env_dir}")
        return

    has_graph = any(
        os.path.isfile(os.path.join(env_dir, f"exp_{exp_id:03d}", "graph.pkl"))
        for exp_id in range(start, start + num_experiments)
    )
    if not has_graph:
        print(f"Skipping bruteforce_test: no graph.pkl files found under {env_dir}")
        return

    tasks = [(t, e, env_dir) for t in thresholds_to_test for e in range(start, start + num_experiments)]
    total_tasks = len(tasks)

    threshold_results_map = defaultdict(list)
    completed = 0

    pbar = tqdm(total=total_tasks, desc="Overall Progress", position=0, leave=True)

    with multiprocessing.Pool(processes=1) as pool:
        for result in pool.imap_unordered(run_task, tasks):
            completed += 1
            pbar.update(1)
            threshold, throughput_dict = result
            threshold_results_map[threshold].append(throughput_dict)

    pbar.close()

    # --- Aggregate results in memory ---
    avg_results_by_threshold = {}
    for threshold, results in threshold_results_map.items():
        aggregate = defaultdict(float)
        for td in results:
            for scheme, val in td.items():
                aggregate[scheme] += val
        avg_summary = {k: v / num_experiments for k, v in aggregate.items()}
        avg_results_by_threshold[threshold] = avg_summary

    # --- Plotting in memory ---
    fontsize = 16
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['axes.labelsize'] = fontsize
    plt.rcParams['xtick.labelsize'] = fontsize
    plt.rcParams['ytick.labelsize'] = fontsize

    marker_list = ['v', 'o']
    color_list = ["#3FB3B7", "#2991D1"]

    plt.figure(figsize=(8, 4.5))
    for i, scheme in enumerate(["bruteforce_5000", "bruteforce_50000"]):
        avg_throughputs = []
        for thresh in thresholds_to_test:
            val = avg_results_by_threshold.get(thresh, {}).get(scheme, np.nan)
            avg_throughputs.append(val / 1000 if val is not None else np.nan)

        plt.plot(
            raw_logspace,
            avg_throughputs,
            linewidth=1.75,
            marker=marker_list[i],
            markersize=8,
            markerfacecolor="w",
            markeredgewidth=1.5,
            color=color_list[i],
            label=scheme,
        )

    plt.xlim(min(raw_logspace), max(raw_logspace))
    plt.xscale("log")
    plt.ylim(0, 8.7)
    plt.yticks(list(range(0, 9, 2)))
    plt.xlabel("1 - Threshold (log scale)")
    plt.ylabel("Average MMF (Kbps)")
    plt.legend(fontsize=12, loc="lower right")
    plt.grid(True, which='both', linestyle=(0, (5, 5)), linewidth=0.5, color="#e0e0e0")
    plt.tight_layout()
    plt.savefig("bruteforce_test.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main_experiment()
