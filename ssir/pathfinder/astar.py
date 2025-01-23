import queue
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ssir import basestations as bs


def a_star(
    graph: bs.IABRelayGraph,
    heuristic: Dict[int, float] = defaultdict(lambda: 0),
    source: int = 0,
    goal: Optional[int] = None,
    metric: str = "distance",
) -> Tuple[Dict[int, float], Dict[int, int]]:
    """
    A* algorithm for finding the shortest path from a source node to all other nodes.

    :param graph: The graph represented as an adjacency list. Each key is a node, and the value is a list of tuples (neighbor, cost).
    :param heuristic: A dictionary containing heuristic values for each node.
                      If not provided, the algorithm operates as Dijkstra's algorithm.
    :param source: The sourcing node. Default is 0.
    :param goal: The goal node. If not provided, the algorithm will find the shortest path to all nodes.
    :param spanning_tree_topology: If True, the algorithm will remove edges that are not part of the shortest path.
    :param metric: The metric used to calculate the cost. Default is "distance".
                   There are another two options: "hop" and "random".
                   hop: The cost is the number of hops between two nodes.
                   random: The cost of edges is a random number between 0 and 1.
    :return: A tuple containing:
             - costs: The minimum cost to all nodes from the source node.
             - predecessors: The predecessor of each node in the shortest path.
    """
    costs = {node: float("infinity") for node in graph.nodes}
    predecessors = {
        node: -1 for node in graph.nodes
    }  # Dictionary to store the previous node for each node

    costs[source] = 0
    pq: queue.PriorityQueue[Tuple[float, int]] = queue.PriorityQueue()
    pq.put((0, source))

    while not pq.empty():
        current_priority, current_node_id = pq.get()

        # If we reach the goal, stop processing
        if current_node_id == goal:
            break

        # If the current distance is greater than the recorded distance, skip
        if current_priority > costs[current_node_id]:
            continue

        for neighbor in graph.get_neighbors(current_node_id):
            # Calculate the distance to the neighbor
            # TODO: current cost only supports distance. Need to support other metrics.
            if metric == "distance":
                edge_cost = graph.nodes[current_node_id].get_distance(
                    graph.nodes[neighbor]
                )
            elif metric == "hop":
                edge_cost = 1
            elif metric == "random":
                edge_cost = random.random()
            else:
                raise ValueError(f"Invalid metric: {metric}")
            new_cost = costs[current_node_id] + edge_cost

            # If the calculated distance is less than the recorded distance, update it
            if new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                predecessors[neighbor] = current_node_id  # Record the previous node
                priority = new_cost + heuristic[neighbor]  # f(n) = g(n) + h(n)
                pq.put((priority, neighbor))

    return costs, predecessors


def get_solution_graph(
    graph: bs.IABRelayGraph,
    predecessors: Dict[int, int],
) -> bs.IABRelayGraph:
    """
    Reconstructs the graph based on the predecessors dictionary.

    :param graph: The original graph
    :param predecessors: The predecessor dictionary returned by Dijkstra's algorithm
    :return: A new graph representing the shortest path
    """
    node_ids = list(predecessors.keys()) + list(predecessors.values())
    node_ids.remove(-1)
    new_graph = graph.copy_graph_with_selected_nodes(node_ids)
    new_graph.reset()

    for user in graph.users:
        path = get_shortest_path(predecessors, user.get_id())
        if path[0] == -1:
            continue
        for i in range(len(path) - 1):
            new_graph.add_edge(path[i], path[i + 1])
    return new_graph


def get_shortest_path(predecessors: Dict[int, int], target: int) -> List[int]:
    """
    Reconstructs the shortest path from the source node to the target node using the predecessors dictionary.

    :param predecessors: The predecessor dictionary returned by Dijkstra's algorithm
    :param target: The target node
    :return: A list representing the shortest path from the source to the target node
    """
    path = []
    current = target

    # Start from the target node and follow the predecessors to reconstruct the path
    while True:
        path.append(current)
        if current == 0 or current == -1:
            break
        current = predecessors[current]

    path.reverse()  # Reverse the path to start from the source node
    return path
