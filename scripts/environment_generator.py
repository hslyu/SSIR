import json
import multiprocessing
import os

import numpy as np
from tqdm import tqdm

# ssir-related imports
import ssir.environment as env
from ssir import basestations as bs


def save_config(config, dir_path, filename="config.json"):
    # Save config dictionary as a JSON file
    os.makedirs(dir_path, exist_ok=True)
    filepath = os.path.join(dir_path, filename)
    with open(filepath, "w") as f:
        json.dump(config, f, indent=4)


def generate_environment(exp_id, env_dir):
    """
    Generate config and graph, and save directly under env_dir/exp_{id}
    """
    exp_dir = os.path.join(env_dir, f"exp_{exp_id:03d}")
    os.makedirs(exp_dir, exist_ok=True)

    config = env.generate_config(exp_id)
    save_config(config, exp_dir, "config.json")

    dm = env.DataManager(**config)
    graph = dm.generate_master_graph()
    graph.save_graph(os.path.join(exp_dir, "graph.pkl"), pkl=True)

    return exp_id


def run_env_task(args):
    return generate_environment(*args)


def main_generate_environments():
    """
    Generate all experiment environments in parallel, with tqdm and multiprocessing.
    """
    start = 0
    num_experiments = 1000

    env_dir = "./results_mmf_vs_spsc/env"
    os.makedirs(env_dir, exist_ok=True)

    tasks = [(e, env_dir) for e in range(start, num_experiments + start)]
    total_tasks = len(tasks)
    completed = 0
    pbar = tqdm(total=total_tasks, desc="Generating Environments", position=0)

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        for result in pool.imap_unordered(run_env_task, tasks):
            completed += 1
            pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    main_generate_environments()
