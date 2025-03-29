import random
from typing import Dict, List

import numpy as np

from ssir import basestations as bs
from ssir.pathfinder.astar import a_star, get_shortest_path


def get_solution_graph(
    graph: bs.IABRelayGraph, n_trial: int = 5000, num_predecessors: int = 20
):
    count = 0
    best_graph: bs.IABRelayGraph = None
    best_throughput = float("-inf")
    while count < n_trial:
        metrics = ["hop", "distance"] + ["random"] * num_predecessors
        predecessors_list = []
        for metric in metrics:
            _, preds = a_star(graph, metric=metric)
            predecessors_list.append(preds)

        all_shortest_paths = get_all_shortest_paths(graph, predecessors_list)

        # Get the best path for each user
        new_graph = graph.copy()
        new_graph.reset()
        user_list = graph.users[:]
        random.shuffle(user_list)
        for user in user_list:
            candidate_graph_list = []
            for path in all_shortest_paths[user.get_id()]:
                candidate_graph = get_aborescence_graph(new_graph, path)
                candidate_graph_list.append(candidate_graph)

            throughput_list = [
                g.compute_network_throughput() for g in candidate_graph_list
            ]
            best_index = np.argmax(throughput_list)
            new_graph = candidate_graph_list[best_index]

        throughput = new_graph.compute_network_throughput()
        if throughput > best_throughput:
            best_graph = new_graph
            best_throughput = throughput

        count += 1
        print(
            f"[{count}/{n_trial}] Best throughput: {best_throughput:.2f}",
            end="\r",
            flush=True,
        )

    return best_graph


def get_aborescence_graph(graph: bs.IABRelayGraph, path: List[int]):
    """
    Get the aborescence graph from the path.

    args:
    - graph [IABRelayGraph]: the graph to which the path will be added
    - path [List[int]]: the path to be added
    """
    aborescence_graph = graph.copy()
    child = path.pop()
    while path:
        child_parent = graph.nodes[child].get_parent()
        if len(child_parent) == 0:
            parent = path.pop()
            aborescence_graph.add_edge(parent, child)
        elif len(child_parent) == 1:
            break
        else:
            raise ValueError("Multiple parents detected.")
        child = parent

    return aborescence_graph


def get_all_shortest_paths(
    graph: bs.IABRelayGraph, predecessors_list: List[dict]
) -> Dict[int, List[List[int]]]:
    """
    Compute all shortest paths for all users and all predecessors.

    return:
        paths_dict: {user_id: [ path_from_predecessors_0, ..., path_from_predecessors_n ]}
    """
    paths_dict: Dict[int, List[List[int]]] = {}
    user_ids = [u.get_id() for u in graph.users]

    for user_id in user_ids:
        paths_dict[user_id] = []
        for preds in predecessors_list:
            # precomputed shortest path for this user and this predecessor
            spath = get_shortest_path(preds, user_id)
            paths_dict[user_id].append(spath)
    return paths_dict

