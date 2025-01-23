from typing import List, Optional

import basestations as bs
import pygad as ga

from .astar import a_star, get_shortest_path


def genetic_algorithm(graph: bs.IABRelayGraph):
    def on_generation(ga_instance):
        nonlocal last_best_fitness, no_improvement_counter

        current_best_fitness = ga_instance.best_solution()[1]

        # Print the best fitness value at every 5th generation
        if ga_instance.generations_completed % 1 == 0:
            print(
                f"Generation: {ga_instance.generations_completed}, Best Fitness: {current_best_fitness}"
            )

        # Check if the best fitness value is improved or not
        if last_best_fitness is None or current_best_fitness > last_best_fitness:
            last_best_fitness = current_best_fitness
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        # If the best fitness value is not improved in the last 500 generations, stop the execution
        if no_improvement_counter >= no_improve_limit:
            print(
                f"Stopped early at generation {ga_instance.generations_completed} due to no improvement."
            )
            return "stop"

    def fitness_func(ga_instance, solution, solution_idx):
        return compute_fitness(solution, graph, essential_nodes, optional_nodes)

    essential_nodes = _get_essential_nodes(graph)
    optional_nodes = [
        node for node in graph.nodes.values() if node not in essential_nodes
    ]

    num_optional_nodes = len(optional_nodes)

    # Genetic Algorithm Parameters
    ga_params = {
        "num_generations": 500,
        "sol_per_pop": 100,
        "num_parents_mating": 10,
        "fitness_func": fitness_func,
        "num_genes": num_optional_nodes,
        "gene_space": [0, 1],
        "keep_parents": 2,  # elite_percent
        "mutation_probability": 0.5,
        "crossover_probability": 0.5,
        "crossover_type": "single_point",
        "mutation_type": "random",
        "on_generation": on_generation,
    }

    # Callback function variables
    last_best_fitness: Optional[float] = None
    no_improvement_counter = 0
    no_improve_limit = 50

    # GA Instance
    ga_instance = ga.GA(**ga_params)

    ga_instance.run()

    solution, solution_fitness, _ = ga_instance.best_solution()

    # reconstruct the graph with the best solution
    max_throughput = 0
    solution_graph: Optional[bs.IABRelayGraph] = None

    for _ in range(50):
        test_graph = reconstruct_graph_from_solution(
            solution, graph, essential_nodes, optional_nodes
        )
        # run A* algorithm to find the shortest path
        costs, predecessors = a_star(test_graph, metric="random")
        test_graph.reset()

        wrong_graph_flag = False
        for user in graph.users:
            path = get_shortest_path(predecessors, user.get_id())

            if path[0] == -1:  # if the user is not connected to the graph
                wrong_graph_flag = True
                break

            for i in range(len(path) - 1):
                test_graph.add_edge(path[i], path[i + 1])
        if wrong_graph_flag:
            continue

        throughput = test_graph.compute_network_throughput()
        if max_throughput < throughput:
            solution_graph = test_graph

    return solution_graph, solution_fitness


def reconstruct_graph_from_solution(
    solution,
    graph: bs.IABRelayGraph,
    essential_nodes: List[bs.AbstractNode],
    optional_nodes: List[bs.AbstractNode],
):
    # select the optional nodes in the solution using enumeration
    selected_nodes = [optional_nodes[i] for i, gene in enumerate(solution) if gene == 1]
    nodes = essential_nodes + selected_nodes
    node_ids = [node.get_id() for node in nodes]

    # create a new graph with the selected nodes
    new_graph = graph.copy_graph_with_selected_nodes(node_ids)
    return new_graph


def compute_fitness(
    solution,
    graph: bs.IABRelayGraph,
    essential_nodes,
    optional_nodes,
):
    max_throughput = 0
    # repeat the random spanning tree topology 10 times to get the maximum throughput
    for _ in range(50):
        new_graph = reconstruct_graph_from_solution(
            solution, graph, essential_nodes, optional_nodes
        )
        # run A* algorithm to find the shortest path
        costs, predecessors = a_star(new_graph, metric="random")

        new_graph.reset()
        for user in graph.users:
            path = get_shortest_path(predecessors, user.get_id())
            if path[0] == -1:  # if the user is not connected to the graph
                return 0
            for i in range(len(path) - 1):
                new_graph.add_edge(path[i], path[i + 1])
        throughput = new_graph.compute_network_throughput()
        if max_throughput < throughput:
            max_throughput = throughput

    return max_throughput


def _get_essential_nodes(graph: bs.IABRelayGraph):
    essential_nodes = []
    essential_nodes.append(graph.nodes[0])
    for node in graph.users:
        essential_nodes.append(node)
        # If the connection is unique, add the parent node either.
        while True:
            parent = node.get_parent()
            if len(parent) == 1:
                essential_nodes.append(parent)
                node = parent[0]
            else:
                break
    return essential_nodes
