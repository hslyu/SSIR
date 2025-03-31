import random
from typing import List

import numpy as np
import pygad as ga

from ssir import basestations as bs
from ssir.pathfinder import astar


class GeneticAlgorithm:
    def __init__(self, graph: bs.IABRelayGraph, verbose=False):
        self.graph = graph
        self.verbose = verbose
        self.essential_nodes = _get_essential_nodes(graph)
        self.optional_nodes = [
            node for node in graph.nodes.values() if node not in self.essential_nodes
        ]
        self.num_optional_nodes = len(self.optional_nodes)

        self.ga_params = {
            "num_generations": 2000,
            "sol_per_pop": 50,
            "num_parents_mating": 10,
            "fitness_func": self.fitness_func,
            "num_genes": self.num_optional_nodes,
            "gene_space": [0, 1],
            "gene_type": int,
            "keep_parents": 4,  # elite_percent
            "mutation_probability": 0.15,
            "crossover_probability": 1.0,
            "crossover_type": "single_point",
            "mutation_type": "swap",
            "on_generation": self.on_generation,
        }
        self.ga_params["initial_population"] = self.set_initial_population()

        self.last_best_fitness = float("-inf")
        self.no_improve_counter = 0
        self.no_improve_limit = 200
        self.num_trial = 4

        # store the best solutions
        costs, predecessors = astar.a_star(graph, metric="hop")
        defualt_graph = astar.get_solution_graph(graph, predecessors)
        mask = self._get_astar_initial_mask(metric="hop")
        self.top_solutions_dict = {
            tuple(mask): (defualt_graph, defualt_graph.compute_network_throughput())
        }
        self.max_solution_length = 100
        self.min_fitness = float("-inf")

    def set_initial_population(self):
        mask_hop = self._get_astar_initial_mask(metric="hop")
        mask_distance = self._get_astar_initial_mask(metric="distance")
        mask_spectral_efficiency = self._get_astar_initial_mask(
            metric="spectral_efficiency"
        )

        # create the initial population with randomized binaries
        initial_population = np.random.randint(
            0, 2, (self.ga_params["sol_per_pop"], self.num_optional_nodes)
        )
        # Set the first 10 population to the mask
        initial_population[0] = np.array(mask_hop)
        initial_population[1] = np.array(mask_distance)
        initial_population[2] = np.array(mask_spectral_efficiency)
        initial_population[3] = np.ones(self.num_optional_nodes)

        return initial_population

    def run(self):
        ga_instance = ga.GA(**self.ga_params)
        ga_instance.run()

        # solution, solution_fitness, _ = ga_instance.best_solution()
        # solution_graph, _ = self.top_solutions_dict[tuple(solution)]
        #
        # return solution_graph, solution_fitness

        best_graph = None
        best_fitness = 0
        for graph, fitness in self.top_solutions_dict.values():
            if fitness > best_fitness:
                best_graph = graph
                best_fitness = fitness

        return best_graph, best_fitness

    def fitness_func(self, ga_instance, solution, solution_idx):
        return self.compute_fitness(
            solution,
            self.graph,
            self.essential_nodes,
            self.optional_nodes,
        )

    def compute_fitness(
        self,
        solution,
        graph: bs.IABRelayGraph,
        essential_nodes,
        optional_nodes,
    ):
        key = tuple(solution)

        best_graph = None
        max_fitness = float("-inf")
        if key in self.top_solutions_dict:
            max_fitness = self.top_solutions_dict[key][1]

        # repeat the random spanning tree topology 10 times to get the maximum throughput
        new_graph = reconstruct_graph_from_solution(
            solution, graph, essential_nodes, optional_nodes
        )

        # run A* algorithm to find the shortest path
        predecessors_list = []
        for _ in range(self.num_trial):
            costs, predecessors = astar.a_star(new_graph, metric="random")
            predecessors_list.append(predecessors)

        new_graph.reset()
        for user in graph.users:
            candidate = list(range(len(predecessors_list)))
            while predecessors_list:
                random_choice = random.choice(candidate)
                candidate.remove(random_choice)

                predecessors = predecessors_list[random_choice]
                path = astar.get_shortest_path(predecessors, user.get_id())
                if -1 in path:
                    return max_fitness

                # check aborescence condition
                is_aborescence = True
                for i in range(len(path) - 1):
                    parent = path[i]
                    child = path[i + 1]
                    if len(new_graph.nodes[child].get_parent()) == 0:
                        continue
                    else:
                        existing_parent_id = (
                            new_graph.nodes[child].get_parent()[0].get_id()
                        )
                        if parent != existing_parent_id:
                            is_aborescence = False

                if is_aborescence:
                    for i in range(len(path) - 1):
                        new_graph.add_edge(path[i], path[i + 1])
                    break
                else:
                    if len(candidate) == 0:
                        return max_fitness
                    continue

        new_fitness = new_graph.compute_network_throughput()
        if max_fitness < new_fitness:
            best_graph = new_graph
            max_fitness = new_fitness

        if best_graph is not None:
            self._update_top_solutions(key, best_graph, max_fitness)

        return max_fitness

    def on_generation(self, ga_instance):
        current_best_fitness = ga_instance.best_solution()[1]

        if self.verbose:
            print(
                f"[{ga_instance.generations_completed}/{self.ga_params['num_generations']}] "
                + f"Current fitness: {current_best_fitness:.4f} | Early-stopping count: {self.no_improve_counter}         ",
                end="\r",
                flush=True,
            )

        # Check if the best fitness value is improved or not
        if current_best_fitness > self.last_best_fitness:
            self.last_best_fitness = current_best_fitness
            self.no_improve_counter = 0
        else:
            self.no_improve_counter += 1

        # If the best fitness value is not improved in the last 500 generations, stop the execution
        if self.no_improve_counter >= self.no_improve_limit:
            if self.verbose:
                print(
                    f"\nStopped early at generation {ga_instance.generations_completed} due to no improvement."
                )
            return "stop"

    def _update_top_solutions(self, key, graph, fitness):
        self.top_solutions_dict[key] = (graph, fitness)

    def _get_astar_initial_mask(self, metric):
        costs, predecessors = astar.a_star(self.graph, metric=metric)
        # Append all the nodes in the path to the appeared_nodes set
        appeared_nodes = set()
        for user in self.graph.users:
            path = astar.get_shortest_path(predecessors, user.get_id())
            for node in path:
                appeared_nodes.add(node)

        # masking the optional nodes that are already appeared in the path
        mask = [
            1 if node.get_id() in appeared_nodes else 0 for node in self.optional_nodes
        ]

        return mask


def get_solution_graph(graph: bs.IABRelayGraph, verbose=False):
    genetic = GeneticAlgorithm(graph, verbose)
    return genetic.run()


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


def _get_essential_nodes(graph: bs.IABRelayGraph):
    essential_nodes = []
    essential_nodes.append(graph.nodes[0])
    for node in graph.users:
        essential_nodes.append(node)
        # If the connection is unique, add the parent node either.
        while True:
            parent = node.get_parent()
            if len(parent) == 1:
                essential_nodes.append(parent[0])
                node = parent[0]
            else:
                break
    return essential_nodes
