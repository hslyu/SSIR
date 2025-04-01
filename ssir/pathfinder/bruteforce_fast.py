import random
from typing import List

from ssir import basestations as bs
from ssir.pathfinder import astar, utils


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
        # Copy the base graph once for this trial, then reset edges/hops
        trial_graph = graph.copy()
        trial_graph.reset()

        # Shuffle user order randomly
        user_list = trial_graph.users[:]
        random.shuffle(user_list)

        # For each user in random order, pick the best path among all candidates
        for user in user_list:
            user_id = user.get_id()
            _, predecessors = astar.a_star(graph, metric="random")
            path = astar.get_shortest_path(predecessors, user.get_id())
            utils.get_aborescence_graph(trial_graph, path)

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
