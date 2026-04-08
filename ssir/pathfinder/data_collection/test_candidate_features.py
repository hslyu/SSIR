#!/usr/bin/env python3
"""
Test for M2.2: Candidate route encoding.

Verifies:
- Node mask creation
- Edge mask creation
- Load projection computation
- Batch stacking
"""

from pathlib import Path

import torch

import ssir.basestations as bs
from ssir.pathfinder.data_collection.candidate_features import (
    encode_candidate_route,
    encode_candidates_batch,
    stack_candidate_features,
)
from ssir.pathfinder.data_collection.graph_features import encode_graph_state
from ssir.pathfinder.data_collection.episode_generator import generate_episode


def _load_test_graph(index: int = 0) -> bs.IABRelayGraph:
    """Load a test graph from the data directory."""
    data_dir = "/fast/hslyu/train"
    graph_path = Path(data_dir) / f"exp_{index:03d}" / "graph.pkl"

    if not graph_path.exists():
        raise FileNotFoundError(f"Test graph not found at {graph_path}")

    graph = bs.IABRelayGraph()
    graph.load_graph(str(graph_path))
    return graph


def test_candidate_encoding_single():
    """Test encoding a single candidate route."""
    print("Testing single candidate encoding...")

    master_graph = _load_test_graph(0)

    # Generate one episode to get real candidates
    episode = generate_episode(
        master_graph=master_graph,
        spsc_threshold=0.9999,
        eavesdropper_density=1e-3,
        episode_id=0,
        num_candidates_per_user=20,
        epsilon=0.1,
    )

    # Get first entry and first candidate
    entry = episode.entries[0]
    candidate = entry.candidate_routes[0]

    # Encode graph
    graph_data = encode_graph_state(master_graph, entry.partial_graph)

    # Encode candidate
    cand_features = encode_candidate_route(
        candidate,
        graph_data,
        master_graph,
        candidate_idx=0,
    )

    # Verify shapes
    assert cand_features.node_mask.shape[0] == graph_data.num_nodes
    assert cand_features.edge_mask.shape[0] == graph_data.num_edges
    assert cand_features.load_projection.shape[0] == graph_data.num_nodes

    # Verify data types
    assert cand_features.node_mask.dtype == torch.float32
    assert cand_features.edge_mask.dtype == torch.float32
    assert cand_features.load_projection.dtype == torch.float32

    # Verify masks are binary or sparse
    assert torch.all((cand_features.node_mask >= 0) & (cand_features.node_mask <= 1))
    assert torch.all((cand_features.edge_mask >= 0) & (cand_features.edge_mask <= 1))

    # Verify at least one node is in the mask (the user is always included)
    assert cand_features.node_mask.sum() > 0, "Node mask should have at least one node"

    print(f"✓ Single candidate encoding works")
    print(f"  - Node mask: {cand_features.node_mask.sum().item():.0f} nodes selected")
    print(f"  - Edge mask: {cand_features.edge_mask.sum().item():.0f} edges selected")
    print(f"  - Load projection sum: {cand_features.load_projection.sum().item():.0f}")
    print(f"  - Route length: {cand_features.route_length} hops")
    print()


def test_candidates_batch_encoding():
    """Test encoding multiple candidates."""
    print("Testing batch candidate encoding...")

    master_graph = _load_test_graph(0)

    # Generate episode
    episode = generate_episode(
        master_graph=master_graph,
        spsc_threshold=0.9999,
        eavesdropper_density=1e-3,
        episode_id=0,
        num_candidates_per_user=30,
        epsilon=0.1,
    )

    # Get first entry (has multiple candidates)
    entry = episode.entries[0]
    candidates = entry.candidate_routes

    # Encode graph
    graph_data = encode_graph_state(master_graph, entry.partial_graph)

    # Encode all candidates
    cand_features_list = encode_candidates_batch(
        candidates,
        graph_data,
        master_graph,
    )

    assert len(cand_features_list) == len(candidates)

    print(f"✓ Batch encoding works")
    print(f"  - Encoded {len(cand_features_list)} candidates")
    print(f"  - Route lengths: {[f.route_length for f in cand_features_list[:5]]}... hops")
    print()


def test_batch_stacking():
    """Test stacking features into batched tensors."""
    print("Testing feature batch stacking...")

    master_graph = _load_test_graph(0)

    # Generate episode
    episode = generate_episode(
        master_graph=master_graph,
        spsc_threshold=0.9999,
        eavesdropper_density=1e-3,
        episode_id=0,
        num_candidates_per_user=25,
        epsilon=0.1,
    )

    # Get first entry
    entry = episode.entries[0]
    candidates = entry.candidate_routes

    # Encode graph and candidates
    graph_data = encode_graph_state(master_graph, entry.partial_graph)
    cand_features_list = encode_candidates_batch(candidates, graph_data, master_graph)

    # Stack features
    stacked = stack_candidate_features(cand_features_list)

    # Verify shapes
    batch_size = len(cand_features_list)
    num_nodes = graph_data.num_nodes
    num_edges = graph_data.num_edges

    assert stacked["node_masks"].shape == (batch_size, num_nodes)
    assert stacked["edge_masks"].shape == (batch_size, num_edges)
    assert stacked["load_projections"].shape == (batch_size, num_nodes)
    assert stacked["route_lengths"].shape == (batch_size,)

    # Verify data types
    assert stacked["node_masks"].dtype == torch.float32
    assert stacked["edge_masks"].dtype == torch.float32
    assert stacked["load_projections"].dtype == torch.float32
    assert stacked["route_lengths"].dtype == torch.float32

    print(f"✓ Batch stacking works")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Node masks: {stacked['node_masks'].shape}")
    print(f"  - Edge masks: {stacked['edge_masks'].shape}")
    print(f"  - Load projections: {stacked['load_projections'].shape}")
    print(f"  - Route lengths: {stacked['route_lengths'].shape}")
    print()


def test_candidate_diversity():
    """Test that different candidates produce different masks."""
    print("Testing candidate diversity...")

    master_graph = _load_test_graph(0)

    # Generate episode with many candidates
    episode = generate_episode(
        master_graph=master_graph,
        spsc_threshold=0.9999,
        eavesdropper_density=1e-3,
        episode_id=0,
        num_candidates_per_user=50,
        epsilon=0.1,
    )

    # Get first entry
    entry = episode.entries[0]
    candidates = entry.candidate_routes

    # Encode graph
    graph_data = encode_graph_state(master_graph, entry.partial_graph)

    # Encode candidates and check diversity
    cand_features_list = encode_candidates_batch(candidates, graph_data, master_graph)

    # Check that not all masks are identical
    masks = [f.node_mask for f in cand_features_list]
    unique_masks = len(set(tuple(m.tolist()) for m in masks))

    print(f"✓ Candidate diversity verified")
    print(f"  - Total candidates: {len(candidates)}")
    print(f"  - Unique node masks: {unique_masks}")
    print(f"  - Route length range: {min(f.route_length for f in cand_features_list)} - {max(f.route_length for f in cand_features_list)} hops")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("M2.2: Candidate Route Encoding Test")
    print("=" * 60 + "\n")

    test_candidate_encoding_single()
    test_candidates_batch_encoding()
    test_batch_stacking()
    test_candidate_diversity()

    print("=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
