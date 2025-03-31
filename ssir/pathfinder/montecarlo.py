import time
from typing import Dict, List

from ssir import basestations as bs
from ssir.pathfinder.astar import a_star, get_shortest_path


def get_solution_graph(
    graph: bs.IABRelayGraph,
    num_predecessors: int = 300,
    num_rounds: int = 5,
    num_trials: int = 3,
    verbose=False,
):
    """
    Builds a solution graph by iteratively choosing the best paths for each user
    and updates them if a better throughput solution is found.
    """
    graph_list = []
    throughput_list = []
    for _ in range(num_trials):
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
            user_id_list,
            key=lambda x: distance_list[user_id_list.index(x)],
            reverse=False,
        )
        old_throughput = -1
        while updated and update_round < num_rounds:
            updated = False

            s = time.time()
            for user_id in sorted_id_list:
                # Temporarily remove the user's path
                deleted_edges = delete_user(result_graph, user_id)

                # Find the best reconnection for this user
                best_throughput, best_added_edges = get_best_candidate_graph(
                    result_graph, user_id, all_shortest_paths[user_id]
                )
                best_throughput = result_graph.compute_network_throughput()

                if update_round == 0 or best_throughput > old_throughput:
                    updated = True
                    old_throughput = best_throughput
                else:
                    # Revert the changes if not updated
                    remove_added_edges(result_graph, user_id, best_added_edges)
                    for p, c in deleted_edges:
                        result_graph.add_edge(p, c)
                    result_graph.compute_hops_for_one_user(user_id)

            update_round += 1
            if verbose:
                print(
                    f"Round {update_round}: Throughput = {result_graph.compute_network_throughput()}, Time = {time.time() - s}"
                )

        graph_list.append(result_graph)
        throughput_list.append(old_throughput)

    best_idx = throughput_list.index(max(throughput_list))
    best_graph = graph_list[best_idx]

    return best_graph


def get_best_candidate_graph(
    base_graph: bs.IABRelayGraph, user_id, path_list: List[List[int]]
):
    """
    Returns the best candidate graph (highest throughput) for a given user,
    given a list of predecessor dictionaries.
    """
    best_throughput = -1
    best_path = []

    # Try all paths and choose the one with the highest throughput
    for path in path_list:
        added_edges = get_aborescence_graph(base_graph, path)
        throughput = base_graph.compute_network_throughput(path[1:-1])

        if throughput > best_throughput:
            best_throughput = throughput
            best_path = path

        remove_added_edges(base_graph, user_id, added_edges)

    # Add the best path to the graph
    if best_path != []:
        best_added_edges = get_aborescence_graph(base_graph, best_path)
    else:
        raise ValueError("No path found.")

    return best_throughput, best_added_edges


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
