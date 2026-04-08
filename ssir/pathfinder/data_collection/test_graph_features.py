#!/usr/bin/env python3
"""
Test for M2.1: Graph encoding functions.

Verifies:
- Node feature extraction
- Edge feature extraction
- Global feature extraction
- Data object creation
- Feature dimensions
"""

from pathlib import Path

import torch
from torch_geometric.data import Data

import ssir.basestations as bs
from ssir.pathfinder.data_collection.graph_features import (
    encode_graph_state,
    get_edge_feature_dim,
    get_global_feature_dim,
    get_node_feature_dim,
)


def _load_test_graph(index: int = 0) -> bs.IABRelayGraph:
    """Load a test graph from the data directory."""
    data_dir = "/fast/hslyu/train"
    graph_path = Path(data_dir) / f"exp_{index:03d}" / "graph.pkl"

    if not graph_path.exists():
        raise FileNotFoundError(f"Test graph not found at {graph_path}")

    graph = bs.IABRelayGraph()
    graph.load_graph(str(graph_path))
    return graph


def test_node_features():
    """Test node feature extraction."""
    print("Testing node feature extraction...")

    master_graph = _load_test_graph(0)

    # Encode graph
    data = encode_graph_state(master_graph, master_graph)

    # Check dimensions
    num_nodes = len(master_graph.nodes)
    assert data.x.shape[0] == num_nodes, f"Expected {num_nodes} nodes, got {data.x.shape[0]}"

    expected_dim = get_node_feature_dim(master_graph)
    assert data.x.shape[1] == expected_dim, f"Expected feature dim {expected_dim}, got {data.x.shape[1]}"

    # Check data types
    assert data.x.dtype == torch.float32

    # Check feature ranges (should be reasonable)
    assert torch.isfinite(data.x).all(), "Non-finite values in node features"

    print(f"✓ Node features: {num_nodes} nodes × {expected_dim} features")
    print(f"  - Feature range: [{data.x.min():.4f}, {data.x.max():.4f}]")
    print()


def test_edge_features():
    """Test edge feature extraction."""
    print("Testing edge feature extraction...")

    master_graph = _load_test_graph(0)

    # Encode graph
    data = encode_graph_state(master_graph, master_graph)

    # Check dimensions
    num_edges = master_graph.adjacency_list
    total_edges = sum(len(neighbors) for neighbors in master_graph.adjacency_list.values())

    if total_edges > 0:
        assert data.edge_index.shape[0] == 2, "edge_index should have 2 rows"
        assert data.edge_index.shape[1] == total_edges, f"Expected {total_edges} edges"

        expected_edge_dim = get_edge_feature_dim()
        assert data.edge_attr.shape[1] == expected_edge_dim, f"Expected edge dim {expected_edge_dim}"

        # Check data types
        assert data.edge_index.dtype == torch.long
        assert data.edge_attr.dtype == torch.float32

        # Check edge attributes are reasonable (distances should be positive)
        assert (data.edge_attr >= 0).all(), "Distances should be non-negative"

        print(f"✓ Edge features: {total_edges} edges × {expected_edge_dim} features")
        print(f"  - Distance range: [{data.edge_attr.min():.4f}, {data.edge_attr.max():.4f}] km")
    else:
        print("✓ No edges in graph")

    print()


def test_global_features():
    """Test global feature extraction."""
    print("Testing global feature extraction...")

    master_graph = _load_test_graph(0)

    # Encode graph
    data = encode_graph_state(master_graph, master_graph)

    # Check global features
    expected_dim = get_global_feature_dim()
    assert data.global_features.shape[0] == expected_dim, f"Expected {expected_dim} global features"

    assert data.global_features.dtype == torch.float32
    assert torch.isfinite(data.global_features).all(), "Non-finite values in global features"

    # Check specific feature values
    spsc = data.global_features[0].item()
    noise = data.global_features[1].item()

    assert 0 < spsc < 1, f"SPSC should be in (0,1), got {spsc}"
    assert noise < 0, f"Noise power density should be negative dB, got {noise}"

    print(f"✓ Global features: {expected_dim} features")
    print(f"  - SPSC probability: {spsc:.4f}")
    print(f"  - Noise power density: {noise:.2f} dB")
    print()


def test_data_object_integrity():
    """Test that the returned Data object is valid PyTorch geometric object."""
    print("Testing Data object integrity...")

    master_graph = _load_test_graph(0)

    # Encode graph
    data = encode_graph_state(master_graph, master_graph)

    # Check that it's a valid Data object
    assert isinstance(data, Data), "Should return PyTorch geometric Data object"

    # Check required attributes
    assert hasattr(data, "x"), "Missing node features"
    assert hasattr(data, "edge_index"), "Missing edge index"
    assert hasattr(data, "edge_attr"), "Missing edge attributes"
    assert hasattr(data, "global_features"), "Missing global features"
    assert hasattr(data, "node_ids"), "Missing node IDs"

    # Check metadata
    num_nodes = len(master_graph.nodes)
    assert data.num_nodes == num_nodes, f"Expected {num_nodes} nodes"

    print(f"✓ Data object valid")
    print(f"  - num_nodes: {data.num_nodes}")
    print(f"  - num_edges: {data.num_edges}")
    print(f"  - node_feature_dim: {data.x.shape[1]}")
    if data.num_edges > 0:
        print(f"  - edge_feature_dim: {data.edge_attr.shape[1]}")
    print()


def test_partial_graph_state():
    """Test encoding with partial graph state (user routes applied)."""
    print("Testing partial graph state encoding...")

    master_graph = _load_test_graph(0)

    # Create a partial graph (only source connected to first basestation)
    partial_graph = master_graph.copy()
    partial_graph.reset()

    # Encode both states
    data_master = encode_graph_state(master_graph, master_graph)
    data_partial = encode_graph_state(master_graph, partial_graph)

    # Both should have same structure but different features
    assert data_master.x.shape == data_partial.x.shape, "Node dimensions should match"
    assert data_master.num_nodes == data_partial.num_nodes, "Number of nodes should match"

    # Partial graph should have fewer edges (or equal)
    assert data_partial.num_edges <= data_master.num_edges, "Partial graph should have fewer edges"

    print(f"✓ Partial graph encoding works")
    print(f"  - Master graph edges: {data_master.num_edges}")
    print(f"  - Partial graph edges: {data_partial.num_edges}")
    print()


def test_feature_dimensions():
    """Test feature dimension helper functions."""
    print("Testing feature dimension helpers...")

    master_graph = _load_test_graph(0)

    node_dim = get_node_feature_dim(master_graph)
    edge_dim = get_edge_feature_dim()
    global_dim = get_global_feature_dim()

    assert node_dim > 0, "Node feature dim should be positive"
    assert edge_dim > 0, "Edge feature dim should be positive"
    assert global_dim > 0, "Global feature dim should be positive"

    print(f"✓ Feature dimensions:")
    print(f"  - Node features: {node_dim}")
    print(f"  - Edge features: {edge_dim}")
    print(f"  - Global features: {global_dim}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("M2.1: Graph Encoding Functions Test")
    print("=" * 60 + "\n")

    test_node_features()
    test_edge_features()
    test_global_features()
    test_data_object_integrity()
    test_partial_graph_state()
    test_feature_dimensions()

    print("=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
