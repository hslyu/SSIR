from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import ssir.basestations as bs
from ssir.pathfinder import utils
from ssir.pathfinder.astar import a_star, get_shortest_path


@dataclass
class UserRouteCandidate:
    name: str
    user_id: int
    path: list[int]
    node_ids: list[int]
    edge_list: list[tuple[int, int]]
    route_hops: int


@dataclass
class UserRoutingProblem:
    user_order: list[int]
    candidates_by_user: dict[int, list[UserRouteCandidate]]


def _unique_paths(paths: list[list[int]]) -> list[list[int]]:
    seen = set()
    unique = []
    for path in paths:
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def build_user_order(master_graph: bs.IABRelayGraph) -> list[int]:
    _, pred = a_star(master_graph, metric="hop")
    user_ids = [user.get_id() for user in master_graph.users]
    hop_lengths = [len(get_shortest_path(pred, user_id)) for user_id in user_ids]
    return sorted(
        user_ids,
        key=lambda uid: hop_lengths[user_ids.index(uid)],
        reverse=True,
    )


def build_predecessor_path_candidates(
    master_graph: bs.IABRelayGraph,
    num_random_predecessors: int = 300,
    include_spectral_efficiency: bool = True,
) -> dict[int, list[list[int]]]:
    metrics = ["hop", "distance"]
    if include_spectral_efficiency:
        metrics.append("spectral_efficiency")
    metrics.extend(["random"] * num_random_predecessors)

    predecessors_list = []
    for metric in metrics:
        _, preds = a_star(master_graph, metric=metric)
        predecessors_list.append(preds)

    raw_paths = utils.get_all_shortest_paths(master_graph, predecessors_list)
    return {user_id: _unique_paths(path_list) for user_id, path_list in raw_paths.items()}


def build_user_routing_problem(
    master_graph: bs.IABRelayGraph,
    num_random_predecessors: int = 300,
    include_spectral_efficiency: bool = True,
) -> UserRoutingProblem:
    path_candidates = build_predecessor_path_candidates(
        master_graph,
        num_random_predecessors=num_random_predecessors,
        include_spectral_efficiency=include_spectral_efficiency,
    )
    candidates_by_user: dict[int, list[UserRouteCandidate]] = {}
    for user_id, path_list in path_candidates.items():
        candidates_by_user[user_id] = [
            UserRouteCandidate(
                name=f"user_{user_id:04d}_cand_{idx:03d}",
                user_id=user_id,
                path=path,
                node_ids=sorted(set(path)),
                edge_list=list(zip(path[:-1], path[1:])),
                route_hops=max(len(path) - 1, 0),
            )
            for idx, path in enumerate(path_list)
        ]

    return UserRoutingProblem(
        user_order=build_user_order(master_graph),
        candidates_by_user=candidates_by_user,
    )


def apply_candidate_route(
    current_graph: bs.IABRelayGraph,
    candidate: UserRouteCandidate,
) -> tuple[bs.IABRelayGraph, list[tuple[int, int]]]:
    updated_graph = current_graph.copy()
    added_edges = utils.get_aborescence_graph(updated_graph, list(candidate.path))
    return updated_graph, added_edges


def evaluate_candidate_throughput(
    current_graph: bs.IABRelayGraph,
    candidate: UserRouteCandidate,
) -> float:
    updated_graph, _ = apply_candidate_route(current_graph, candidate)
    return float(updated_graph.compute_network_throughput())


def save_routing_problem(problem: UserRoutingProblem, path: str | Path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as fp:
        pickle.dump(problem, fp)


def load_routing_problem(path: str | Path) -> UserRoutingProblem:
    with open(path, "rb") as fp:
        payload = pickle.load(fp)
    if not isinstance(payload, UserRoutingProblem):
        raise TypeError(f"Unexpected routing problem type: {type(payload)}")
    return payload
