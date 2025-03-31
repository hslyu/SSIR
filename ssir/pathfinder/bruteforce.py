import random
from typing import Dict, List

import numpy as np

from ssir import basestations as bs
from ssir.pathfinder.astar import a_star, get_shortest_path


def get_solution_graph(
    graph: bs.IABRelayGraph,
    n_trial: int = 5000,
    num_predecessors: int = 5,
    early_stop: int = 50,
    verbose: bool = False,
) -> bs.IABRelayGraph:
    """
    Performs a brute-force style solution search by generating multiple shortest-path candidates
    and assigning each user to the best candidate path.

    Args:
        graph (bs.IABRelayGraph): The base graph (including source/base stations and users).
        n_trial (int): The number of brute-force trials to perform.
        num_predecessors (int): How many additional 'random' predecessor lists to generate (plus 'hop' and 'distance').
        verbose (bool): If True, prints the progress of the search.

    Returns:
        bs.IABRelayGraph: The best graph configuration found during brute-force trials.
    """
    # Keep track of the best overall result
    best_graph: bs.IABRelayGraph = None
    best_throughput = -1

    count = 0
    for trial in range(1, n_trial + 1):
        # Generate multiple predecessor lists for different metrics (hop, distance, random)
        metrics = ["hop", "distance"] + ["random"] * num_predecessors
        predecessors_list = []
        for metric in metrics:
            _, preds = a_star(graph, metric=metric)
            predecessors_list.append(preds)

        # Gather all shortest-path candidates for each user
        all_shortest_paths = get_all_shortest_paths(graph, predecessors_list)

        # Copy the base graph once for this trial, then reset edges/hops
        trial_graph = graph.copy()
        trial_graph.reset()

        # Shuffle user order randomly
        user_list = trial_graph.users[:]
        random.shuffle(user_list)

        # For each user in random order, pick the best path among all candidates
        for user in user_list:
            user_id = user.get_id()
            # Remove this user's path in-place first
            deleted_edges = delete_user(trial_graph, user_id)

            # Find the best path for this user
            best_user_throughput = -1.0
            best_user_path: List[int] = []

            for path in all_shortest_paths[user_id]:
                # Temporarily add the candidate path
                added_edges = get_aborescence_graph(trial_graph, path)
                # Evaluate throughput
                throughput = trial_graph.compute_network_throughput()

                # Revert the candidate path
                remove_added_edges(trial_graph, user_id, added_edges)

                # Record the best candidate
                if throughput > best_user_throughput:
                    best_user_throughput = throughput
                    best_user_path = path

            # Permanently add the best path for this user
            if best_user_path:
                get_aborescence_graph(trial_graph, best_user_path)
            else:
                # If somehow no path was found, just continue
                # (though in most cases a path should exist)
                continue

        # Check final throughput of this trial's configuration
        trial_throughput = trial_graph.compute_network_throughput()
        if trial_throughput > best_throughput:
            best_throughput = trial_throughput
            best_graph = trial_graph
            count = 0
        else:
            count += 1
            if count >= early_stop:
                break

        if verbose:
            print(
                f"[{trial}/{n_trial}] Best throughput so far: {best_throughput:.2f}, Early stop count: {count}",
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
    added_edges = []
    # Add the path to the graph
    for i in reversed(range(1, len(path))):
        child = path[i]
        parent = path[i - 1]

        parent_node_list = graph.nodes[child].get_parent()
        if len(parent_node_list) == 0:
            graph.add_edge(parent, child)
            added_edges.append((parent, child))
        elif len(parent_node_list) == 1:
            break
        else:
            raise ValueError("Multiple parents detected.")

    graph.compute_hops_for_one_user(path[-1])
    return added_edges


def remove_added_edges(graph: bs.IABRelayGraph, user_id: int, added_edges: List[tuple]):
    """
    Removes the edges added to connect a user's path, effectively reverting the path assignment.

    Args:
        graph (bs.IABRelayGraph): The graph being modified.
        user_id (int): The user node ID whose edges are removed.
        added_edges (List[tuple]): The edges (parent, child) that were previously added.
    """
    user = graph.nodes[user_id]
    user.hops = 0

    # Remove each of the added edges from the graph
    for p, c in added_edges:
        # Remove the hop information
        graph.nodes[p].connected_user.remove(user)
        graph.remove_edge(p, c)


def delete_user(graph: bs.IABRelayGraph, user_id: int) -> List[tuple]:
    """
    Removes the existing path (edges) for a user from the graph, if any,
    returning the list of edges that were deleted.

    Args:
        graph (bs.IABRelayGraph): The graph being modified.
        user_id (int): The ID of the user to remove edges from.

    Returns:
        List[tuple]: A list of (parent, child) edges that were removed.
    """
    user = graph.nodes[user_id]
    if not user.has_parent():
        return []

    deleted_edges = []
    current = user
    while current.get_id() != 0:
        parents = current.get_parent()
        if not parents:
            break
        parent = parents[0]
        # Remove the edge if this user is the only one connected
        if parent.connected_user == [user]:
            graph.remove_edge(parent.get_id(), current.get_id())
            deleted_edges.append((parent.get_id(), current.get_id()))
        if user in parent.connected_user:
            parent.connected_user.remove(user)
        current = parent

    user.hops = 0
    return deleted_edges


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
