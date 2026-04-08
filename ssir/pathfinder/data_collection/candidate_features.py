"""
Candidate route feature encoding for neural network input.

Represents individual candidate routes as masks and load projections
over the graph structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Data

import ssir.basestations as bs
from ssir.pathfinder.rl.trajectory import UserRouteCandidate


@dataclass
class CandidateFeatures:
    """
    Features for a single candidate route within a graph context.

    Attributes:
        node_mask: Binary mask [num_nodes] indicating nodes in the route
        edge_mask: Binary mask [num_edges] indicating edges in the route
        load_projection: Load (number of downstream users) for each node [num_nodes]
        route_length: Number of hops in the route
        user_id: The user ID for this candidate
        candidate_idx: Index of this candidate (for reference)
    """

    node_mask: torch.Tensor  # [num_nodes]
    edge_mask: torch.Tensor  # [num_edges]
    load_projection: torch.Tensor  # [num_nodes]
    route_length: int
    user_id: int
    candidate_idx: int


def _build_node_mask(
    candidate: UserRouteCandidate,
    node_ids: list[int],
) -> torch.Tensor:
    """
    Create binary mask indicating nodes in candidate route.

    Args:
        candidate: The candidate route
        node_ids: List of all node IDs in order

    Returns:
        Binary mask tensor [num_nodes]
    """
    node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    mask = torch.zeros(len(node_ids), dtype=torch.float32)

    for node_id in candidate.node_ids:
        if node_id in node_id_to_idx:
            mask[node_id_to_idx[node_id]] = 1.0

    return mask


def _build_edge_mask(
    candidate: UserRouteCandidate,
    data: Data,
) -> torch.Tensor:
    """
    Create binary mask indicating edges in candidate route.

    Args:
        candidate: The candidate route
        data: PyTorch geometric Data object with edge_index

    Returns:
        Binary mask tensor [num_edges]
    """
    mask = torch.zeros(data.num_edges, dtype=torch.float32)

    if data.num_edges == 0:
        return mask

    # Build set of edges in candidate
    candidate_edges = set(candidate.edge_list)

    # Check each edge in data against candidate edges
    edge_index = data.edge_index
    for edge_idx in range(data.num_edges):
        from_node = edge_index[0, edge_idx].item()
        to_node = edge_index[1, edge_idx].item()

        # Map back to original node IDs
        from_node_id = data.node_ids[from_node].item()
        to_node_id = data.node_ids[to_node].item()

        if (from_node_id, to_node_id) in candidate_edges:
            mask[edge_idx] = 1.0

    return mask


def _compute_load_projection(
    candidate: UserRouteCandidate,
    user_id: int,
    graph: bs.IABRelayGraph,
    node_ids: list[int],
) -> torch.Tensor:
    """
    Compute the load (downstream users) for each node if candidate is applied.

    If a node is on the path to the user, it carries the user's load.
    More generally, "load" here is a count of how many users are downstream
    from each node if this candidate route is taken.

    Args:
        candidate: The candidate route
        user_id: The user being routed
        graph: The master graph (for reference)
        node_ids: List of all node IDs

    Returns:
        Load projection tensor [num_nodes] (1.0 for each node in path, 0 else)
    """
    node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    load = torch.zeros(len(node_ids), dtype=torch.float32)

    # Each node in the candidate path carries one user's load
    for node_id in candidate.node_ids:
        if node_id in node_id_to_idx:
            load[node_id_to_idx[node_id]] = 1.0

    return load


def encode_candidate_route(
    candidate: UserRouteCandidate,
    graph_data: Data,
    master_graph: bs.IABRelayGraph,
    candidate_idx: int = 0,
) -> CandidateFeatures:
    """
    Encode a candidate route into learnable features.

    Creates node/edge masks and load projections that indicate:
    - Which nodes are in the route
    - Which edges are in the route
    - Load impact on each node

    Args:
        candidate: The candidate route to encode
        graph_data: PyTorch geometric Data object from encode_graph_state()
        master_graph: The master graph (for reference)
        candidate_idx: Index of this candidate (for reference)

    Returns:
        CandidateFeatures object
    """
    # Extract node IDs from Data object
    node_ids = [nid.item() for nid in graph_data.node_ids]

    # Build masks
    node_mask = _build_node_mask(candidate, node_ids)
    edge_mask = _build_edge_mask(candidate, graph_data)

    # Compute load projection
    load_projection = _compute_load_projection(
        candidate,
        candidate.user_id,
        master_graph,
        node_ids,
    )

    # Route length
    route_length = candidate.route_hops

    return CandidateFeatures(
        node_mask=node_mask,
        edge_mask=edge_mask,
        load_projection=load_projection,
        route_length=route_length,
        user_id=candidate.user_id,
        candidate_idx=candidate_idx,
    )


def encode_candidates_batch(
    candidates: list[UserRouteCandidate],
    graph_data: Data,
    master_graph: bs.IABRelayGraph,
) -> list[CandidateFeatures]:
    """
    Encode multiple candidate routes for a single graph state.

    Args:
        candidates: List of candidate routes
        graph_data: PyTorch geometric Data object
        master_graph: The master graph

    Returns:
        List of CandidateFeatures objects
    """
    return [
        encode_candidate_route(candidate, graph_data, master_graph, idx)
        for idx, candidate in enumerate(candidates)
    ]


def stack_candidate_features(
    features_list: list[CandidateFeatures],
) -> dict[str, torch.Tensor]:
    """
    Stack multiple candidate features into batched tensors.

    Args:
        features_list: List of CandidateFeatures objects

    Returns:
        Dictionary with batched tensors:
        - node_masks: [batch_size, num_nodes]
        - edge_masks: [batch_size, num_edges]
        - load_projections: [batch_size, num_nodes]
        - route_lengths: [batch_size]
    """
    node_masks = torch.stack([f.node_mask for f in features_list])
    edge_masks = torch.stack([f.edge_mask for f in features_list])
    load_projections = torch.stack([f.load_projection for f in features_list])
    route_lengths = torch.tensor([f.route_length for f in features_list], dtype=torch.float32)

    return {
        "node_masks": node_masks,
        "edge_masks": edge_masks,
        "load_projections": load_projections,
        "route_lengths": route_lengths,
    }
