import random
from typing import List

from ssir import basestations as bs
from ssir.pathfinder import utils
from ssir.pathfinder.astar import a_star


def get_solution_graph(
    graph: bs.IABRelayGraph,
    n_trial: int = 5000,
    num_predecessors: int = 3,
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
        all_shortest_paths = utils.get_all_shortest_paths(graph, predecessors_list)

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
            deleted_edges = utils.delete_user(trial_graph, user_id)

            # Find the best path for this user
            best_user_throughput = -1.0
            best_user_path: List[int] = []

            for path in all_shortest_paths[user_id]:
                # Temporarily add the candidate path
                added_edges = utils.get_aborescence_graph(trial_graph, path)
                # Evaluate throughput
                throughput = trial_graph.compute_network_throughput()

                # Revert the candidate path
                utils.remove_added_edges(trial_graph, user_id, added_edges)

                # Record the best candidate
                if throughput > best_user_throughput:
                    best_user_throughput = throughput
                    best_user_path = path

            # Permanently add the best path for this user
            if best_user_path:
                utils.get_aborescence_graph(trial_graph, best_user_path)
            else:
                # If somehow no path was found, just continue
                # (though in most cases a path should exist)
                continue

        # Check final throughput of this trial's configuration
        trial_throughput = trial_graph.compute_network_throughput()
        if trial_throughput > best_throughput:
            best_throughput = trial_throughput
            best_graph = trial_graph
            if best_throughput > 1e-4:
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
