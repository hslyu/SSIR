"""
Episode generation for throughput predictor dataset.

Generates episodes with ground-truth candidate evaluations.
"""

from __future__ import annotations

import random
from typing import List

import numpy as np

import ssir.basestations as bs
from ssir.pathfinder import utils
from ssir.pathfinder.astar import a_star, get_shortest_path
from ssir.pathfinder.rl.trajectory import UserRouteCandidate

from .data_schema import DataEntry, EpisodeDataset


def _generate_candidate_paths(
    master_graph: bs.IABRelayGraph,
    user_id: int,
    num_random_predecessors: int = 50,
) -> List[List[int]]:
    """
    Generate diverse candidate paths for a user.

    Uses multiple A* metrics ("hop", "distance") plus random variants.

    Args:
        master_graph: The full network topology
        user_id: User node ID
        num_random_predecessors: Number of random metric variants to generate

    Returns:
        List of unique paths (as lists of node IDs)
    """
    metrics = ["hop", "distance"] + ["random"] * num_random_predecessors
    predecessors_list = []

    for metric in metrics:
        _, preds = a_star(master_graph, metric=metric)
        predecessors_list.append(preds)

    # Get all paths for this user
    all_paths = []
    for preds in predecessors_list:
        path = get_shortest_path(preds, user_id)
        all_paths.append(path)

    # Remove duplicates while preserving order
    seen = set()
    unique_paths = []
    for path in all_paths:
        path_tuple = tuple(path)
        if path_tuple not in seen:
            seen.add(path_tuple)
            unique_paths.append(path)

    return unique_paths


def _paths_to_candidates(
    paths: List[List[int]],
    user_id: int,
) -> List[UserRouteCandidate]:
    """
    Convert paths to UserRouteCandidate objects.

    Args:
        paths: List of paths (node ID sequences)
        user_id: The user's node ID

    Returns:
        List of UserRouteCandidate objects
    """
    candidates = []
    for idx, path in enumerate(paths):
        candidate = UserRouteCandidate(
            name=f"user_{user_id:04d}_cand_{idx:03d}",
            user_id=user_id,
            path=path,
            node_ids=sorted(set(path)),
            edge_list=list(zip(path[:-1], path[1:])),
            route_hops=max(len(path) - 1, 0),
        )
        candidates.append(candidate)
    return candidates


def _evaluate_candidate(
    partial_graph: bs.IABRelayGraph,
    candidate: UserRouteCandidate,
) -> float:
    """
    Evaluate the throughput of a candidate route.

    Applies the candidate to a copy of the partial graph and computes throughput.

    Args:
        partial_graph: Current graph state (before this user)
        candidate: The route candidate to evaluate

    Returns:
        Throughput value (float). Returns 0.0 if NaN/inf detected.
    """
    test_graph = partial_graph.copy()
    utils.get_aborescence_graph(test_graph, list(candidate.path))

    throughput = test_graph.compute_network_throughput()

    # Handle NaN/inf
    if not np.isfinite(throughput):
        return 0.0

    return float(throughput)


def _select_route_epsilon_greedy(
    candidates: List[UserRouteCandidate],
    throughputs: List[float],
    epsilon: float = 0.1,
) -> int:
    """
    Select a candidate route using epsilon-greedy strategy.

    With probability epsilon: select highest-throughput route.
    Otherwise: select randomly from top 5% routes.

    Args:
        candidates: List of candidate routes
        throughputs: Throughput values for each candidate
        epsilon: Exploitation probability (default 0.1)

    Returns:
        Index of the selected candidate
    """
    if len(candidates) == 0:
        raise ValueError("No candidates provided")

    # Sort by throughput
    sorted_indices = sorted(
        range(len(throughputs)),
        key=lambda i: throughputs[i],
        reverse=True,
    )

    if random.random() < epsilon:
        # Exploit: pick highest throughput
        return sorted_indices[0]
    else:
        # Explore: pick randomly from top 5%
        top_5_percent = max(1, len(sorted_indices) // 20)  # at least 1
        top_indices = sorted_indices[:top_5_percent]
        return random.choice(top_indices)


def generate_episode(
    master_graph: bs.IABRelayGraph,
    spsc_threshold: float,
    eavesdropper_density: float,
    episode_id: int,
    num_candidates_per_user: int = 50,
    epsilon: float = 0.1,
    user_order: List[int] | None = None,
) -> EpisodeDataset:
    """
    Generate a single training episode with ground-truth evaluations.

    Iteratively assigns routes to users, evaluating all candidates for each user
    and selecting via epsilon-greedy strategy.

    Args:
        master_graph: The full network topology
        spsc_threshold: SPSC probability for this episode
        eavesdropper_density: Eavesdropper density for this episode
        episode_id: Unique identifier for this episode
        num_candidates_per_user: Number of candidate routes to generate per user
        epsilon: Epsilon-greedy exploitation probability
        user_order: Order to assign users (if None, uses distance-based default)

    Returns:
        EpisodeDataset containing all data entries for this episode
    """
    # Copy master graph to avoid modifications
    master_graph = master_graph.copy()
    for bs_node in master_graph.basestations:
        bs_node._set_transmission_and_jamming_power_density()

    # Determine user order (farthest first by default)
    if user_order is None:
        _, pred = a_star(master_graph, metric="hop")
        user_ids = [user.get_id() for user in master_graph.users]
        hop_lengths = [len(get_shortest_path(pred, uid)) for uid in user_ids]
        user_order = sorted(
            user_ids,
            key=lambda uid: hop_lengths[user_ids.index(uid)],
            reverse=True,
        )

    # Initialize partial graph (only source)
    partial_graph = master_graph.copy()
    partial_graph.reset()

    entries = []
    throughput_stats = {
        "min": float("inf"),
        "max": float("-inf"),
        "sum": 0.0,
        "count": 0,
    }

    # Process each user
    for user_index, user_id in enumerate(user_order):
        # Generate candidates
        candidate_paths = _generate_candidate_paths(
            master_graph,
            user_id,
            num_random_predecessors=num_candidates_per_user - 2,  # minus hop/distance
        )
        candidates = _paths_to_candidates(candidate_paths, user_id)

        # Evaluate all candidates
        true_throughputs = [
            _evaluate_candidate(partial_graph, cand) for cand in candidates
        ]

        # Select route via epsilon-greedy
        selected_idx = _select_route_epsilon_greedy(
            candidates, true_throughputs, epsilon=epsilon
        )

        # Record data entry
        entry = DataEntry(
            episode_id=episode_id,
            user_index=user_index,
            spsc_threshold=spsc_threshold,
            eavesdropper_density=eavesdropper_density,
            master_graph=master_graph,
            partial_graph=partial_graph.copy(),
            candidate_routes=candidates,
            true_throughputs=true_throughputs,
            selected_candidate_idx=selected_idx,
            metadata={
                "num_candidates": len(candidates),
                "max_throughput": max(true_throughputs) if true_throughputs else 0.0,
                "selected_throughput": true_throughputs[selected_idx],
            },
        )
        entries.append(entry)

        # Update stats
        for tp in true_throughputs:
            throughput_stats["min"] = min(throughput_stats["min"], tp)
            throughput_stats["max"] = max(throughput_stats["max"], tp)
            throughput_stats["sum"] += tp
            throughput_stats["count"] += 1

        # Apply selected route to partial graph for next user
        selected_candidate = candidates[selected_idx]
        utils.get_aborescence_graph(partial_graph, list(selected_candidate.path))

    # Finalize stats
    if throughput_stats["count"] > 0:
        throughput_stats["mean"] = (
            throughput_stats["sum"] / throughput_stats["count"]
        )
    else:
        throughput_stats["mean"] = 0.0

    if throughput_stats["min"] == float("inf"):
        throughput_stats["min"] = 0.0
    if throughput_stats["max"] == float("-inf"):
        throughput_stats["max"] = 0.0

    # Build episode dataset
    episode = EpisodeDataset(
        episode_id=episode_id,
        spsc_threshold=spsc_threshold,
        eavesdropper_density=eavesdropper_density,
        num_users=len(user_order),
        entries=entries,
        throughput_stats=throughput_stats,
    )

    return episode
