import time
from typing import List

from ssir import basestations as bs
from ssir.pathfinder import astar, utils


def get_solution_graph(
    graph: bs.IABRelayGraph,
    num_trials: int = 50,
    verbose=False,
):
    """
    Builds a solution graph by iteratively choosing the best paths for each user
    and updates them if a better throughput solution is found.
    """
    # Sort the user with the minimum hops to the source
    _, preds = astar.a_star(graph, metric="hop")
    ref_graph = astar.get_solution_graph(graph, preds)
    ref_graph.compute_hops()

    user_id_list = [user.get_id() for user in graph.users]
    hop_list = [user.hops for user in ref_graph.users]

    # Sort the user IDs by distance from the source
    sorted_id_list = sorted(
        user_id_list,
        key=lambda x: hop_list[user_id_list.index(x)],
        reverse=True,
    )

    # Initial assignment of paths
    best_graph = graph.copy()
    best_graph.reset()

    s = time.time()
    for i, user_id in enumerate(sorted_id_list):
        count = 0
        best_throughput = -1
        best_path = []
        while count < num_trials:
            # Create a random path
            _, preds = astar.a_star(graph, goal=user_id, metric="random")
            path = astar.get_shortest_path(preds, user_id)

            # Add the path to the graph
            added_edges = utils.get_aborescence_graph(best_graph, path)
            throughput = best_graph.compute_network_throughput(path[1:-1])

            if throughput > best_throughput:
                best_throughput = throughput
                best_path = path
                utils.remove_added_edges(best_graph, user_id, added_edges)
            else:
                # Revert the changes if not updated
                utils.remove_added_edges(best_graph, user_id, added_edges)
                count += 1
        utils.get_aborescence_graph(best_graph, best_path)

        if verbose:
            print(
                f"[{i+1}/{len(sorted_id_list)}] User {user_id}: Throughput = {best_graph.compute_network_throughput()}, Time = {time.time() - s}",
                end="\r",
                flush=True,
            )
    if verbose:
        print("")

    return best_graph
