import json
import multiprocessing
import os

from tqdm import tqdm

import ssir.environment as env
from ssir.pathfinder import astar, genetic


def save_config(config, dir, filename: str = "config.json"):
    if not os.path.exists(dir):
        os.makedirs(dir)
    filename = os.path.join(dir, filename)

    with open(filename, "w") as f:
        json.dump(config, f, indent=4)


def run_experiment(exp_index, progress_queue):
    # Generate the environment according to the index
    config = env.generate_config(exp_index)
    dm = env.DataManager(**config)
    graph = dm.generate_master_graph()

    # TODO: Save the configuration
    # result dir architecture:
    # results/exp_index/{config.json, graph_1.json, graph_2.json, ...}
    parent_dir = f"results/{exp_index}"
    os.makedirs(parent_dir, exist_ok=True)
    save_config(config, parent_dir)

    try:
        # A* algorithm with distacne
        _, predecessors = astar.a_star(graph, metric="distance")
        graph_astar_distance = astar.get_solution_graph(graph, predecessors)
        graph_astar_distance.save_graph(
            os.path.join(parent_dir, "graph_astar_distance.json")
        )

        # A* algorithm with hop
        _, predecessors = astar.a_star(graph, metric="hop")
        graph_astar_hop = astar.get_solution_graph(graph, predecessors)
        graph_astar_hop.save_graph(os.path.join(parent_dir, "graph_astar_hop.json"))

        # Genetic algorithm
        graph_genetic, _ = genetic.get_solution_graph(graph)
        graph_genetic.save_graph(os.path.join(parent_dir, "graph_genetic.json"))
        progress_queue.put(1)
    except Exception as e:
        print(f"Error in experiment {exp_index}: {e}")
        progress_queue.put(1)


# %%
num_experiments = 5000
num_workers = 14

# 진행률 표시를 위한 큐와 tqdm 설정
manager = multiprocessing.Manager()
progress_queue = manager.Queue()
progress_bar = tqdm(total=num_experiments, desc="Experiments")


# 진행률 업데이트 프로세스
def update_progress():
    for _ in range(num_experiments):
        progress_queue.get()
        progress_bar.update(1)


progress_process = multiprocessing.Process(target=update_progress)
progress_process.start()

# 멀티프로세싱 실행
with multiprocessing.Pool(processes=num_workers) as pool:
    pool.starmap(run_experiment, [(i, progress_queue) for i in range(num_experiments)])

# 진행률 업데이트 종료
progress_process.join()
progress_bar.close()
