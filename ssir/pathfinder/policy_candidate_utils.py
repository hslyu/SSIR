import random

import numpy as np
import torch

from ssir.pathfinder import utils
from ssir.pathfinder.astar import a_star, get_shortest_path


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dedup_paths(path_list):
    return [list(path) for path in dict.fromkeys(tuple(path) for path in path_list)]


def valid_candidate_paths(path_list):
    return [
        path
        for path in dedup_paths(path_list)
        if path and path[0] != -1 and -1 not in path
    ]


def prepare_candidates(graph, num_predecessors: int):
    _, pred = a_star(graph, metric="hop")
    uid_list = [user.get_id() for user in graph.users]
    hop_list = [len(get_shortest_path(pred, user.get_id())) for user in graph.users]
    sorted_id_list = sorted(
        uid_list,
        key=lambda x: hop_list[uid_list.index(x)],
        reverse=True,
    )

    metrics = ["hop", "distance"] + ["random"] * num_predecessors
    predecessors_list = []
    for metric in metrics:
        _, preds = a_star(graph, metric=metric)
        predecessors_list.append(preds)

    all_shortest_paths = utils.get_all_shortest_paths(graph, predecessors_list)
    all_shortest_paths = {
        user_id: dedup_paths(path_list)
        for user_id, path_list in all_shortest_paths.items()
    }
    return sorted_id_list, all_shortest_paths
