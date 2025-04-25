import random
import time

from ssir import basestations as bs
from ssir.pathfinder import utils
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

    _, pred = a_star(graph, metric="hop")
    # Sort the user IDs by distance from the source
    uid_list = [user.get_id() for user in graph.users]
    hop_list = [len(get_shortest_path(pred, user.get_id())) for user in graph.users]
    sorted_id_list = sorted(
        uid_list,
        key=lambda x: hop_list[uid_list.index(x)],
        reverse=True,
    )

    for _ in range(num_trials):
        # Generate multiple predecessor lists for different metrics
        metrics = ["hop", "distance"] + ["random"] * num_predecessors
        predecessors_list = []
        for metric in metrics:
            _, preds = a_star(graph, metric=metric)
            predecessors_list.append(preds)

        all_shortest_paths = utils.get_all_shortest_paths(graph, predecessors_list)

        # Initial assignment of paths
        result_graph = graph.copy()
        result_graph.reset()

        updated = True
        update_round = 0
        source = graph.nodes[0]

        old_throughput = -1
        while updated and update_round < num_rounds:
            updated = False

            s = time.time()
            if update_round == 0:
                user_id_list = sorted_id_list
            else:
                random.shuffle(uid_list)
                user_id_list = uid_list

            for user_id in user_id_list:
                # Temporarily remove the user's path
                deleted_edges = utils.delete_user(result_graph, user_id)

                # Find the best reconnection for this user
                best_throughput, best_added_edges = utils.get_best_candidate_graph(
                    result_graph, user_id, all_shortest_paths[user_id]
                )
                best_throughput = result_graph.compute_network_throughput()

                if update_round == 0 or best_throughput > old_throughput:
                    updated = True
                    old_throughput = best_throughput
                else:
                    # Revert the changes if not updated
                    utils.remove_added_edges(result_graph, user_id, best_added_edges)
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
