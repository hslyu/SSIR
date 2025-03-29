from typing import Dict, List

import numpy as np

from ssir import basestations as bs
from ssir.pathfinder.astar import a_star, get_shortest_path


def get_solution_graph(
    graph: bs.IABRelayGraph, num_predecessors: int = 100, num_rounds: int = 10
):
    """
    Builds a solution graph by iteratively choosing the best paths for each user
    and updates them if a better throughput solution is found.
    """
    # Generate multiple predecessor lists for different metrics
    metrics = ["hop", "distance"] + ["random"] * num_predecessors
    predecessors_list = []
    for metric in metrics:
        _, preds = a_star(graph, metric=metric)
        predecessors_list.append(preds)

    all_shortest_paths = get_all_shortest_paths(graph, predecessors_list)

    # Initial assignment of paths
    result_graph = graph.copy()
    result_graph.reset()

    updated = True
    update_round = 0
    source = graph.nodes[0]
    # Sort the user IDs by distance from the source
    user_id_list = [user.get_id() for user in graph.users]
    distance_list = [source.get_distance(user) for user in graph.users]
    sorted_id_list = sorted(
        user_id_list, key=lambda x: distance_list[user_id_list.index(x)], reverse=False
    )
    while updated and update_round < num_rounds:
        updated = False

        throughput = result_graph.compute_network_throughput()
        for user_id in sorted_id_list:
            # Temporarily remove the user's path
            graph_wo_user = delete_user(result_graph, user_id)

            # Find the best reconnection for this user
            new_graph = get_best_candidate_graph(
                graph_wo_user, all_shortest_paths[user_id]
            )
            new_graph.compute_hops_for_one_user(user_id)

            # Check if the throughput has increased
            new_throughput = new_graph.compute_network_throughput()

            if update_round == 0 or new_throughput > throughput:
                updated = True
                result_graph = new_graph
                throughput = new_throughput

        update_round += 1
        # print(
        #     f"Round {update_round}: Throughput = {result_graph.compute_network_throughput()}"
        # )

    return result_graph


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


def get_best_candidate_graph(base_graph: bs.IABRelayGraph, path_list: List[List[int]]):
    """
    Returns the best candidate graph (highest throughput) for a given user,
    given a list of predecessor dictionaries.
    """
    candidate_graphs: List[bs.IABRelayGraph] = []
    for path in path_list:
        candidate_graph = get_aborescence_graph(base_graph, path)
        candidate_graphs.append(candidate_graph)

    throughput_list = [g.compute_network_throughput() for g in candidate_graphs]
    best_index = np.argmax(throughput_list)
    return candidate_graphs[best_index]


def get_aborescence_graph(graph: bs.IABRelayGraph, path: List[int]):
    """
    Get the aborescence graph from the path.

    args:
    - graph [IABRelayGraph]: the graph to which the path will be added
    - path [List[int]]: the path to be added
    """
    aborescence_graph = graph.copy()
    for i in range(len(path) - 1, 0, -1):  # path 리스트를 역순으로 탐색
        child = path[i]
        parent = path[i - 1]

        child_parent = graph.nodes[child].get_parent()
        if len(child_parent) == 0:
            aborescence_graph.add_edge(parent, child)
        elif len(child_parent) == 1:
            break
        else:
            raise ValueError("Multiple parents detected.")

    return aborescence_graph


def delete_user(source_graph: bs.IABRelayGraph, user_id: int):
    """
    Delete the user from the graph.

    args:
    - graph [IABRelayGraph]: the graph from which the user will be deleted
    - user [User]: the user to be deleted
    """
    graph = source_graph.copy()
    graph.compute_hops()
    user = graph.nodes[user_id]
    if not user.has_parent():
        return graph
    assert isinstance(user, bs.User)

    current = user
    while current.get_id() != 0:
        parent = current.get_parent()[0]
        if len(parent.connected_user) == 1:  # type: ignore
            graph.remove_edge(parent.get_id(), current.get_id())  # type: ignore
        parent.connected_user.remove(user)  # type: ignore
        current = parent

    user.hops = 0
    return graph
