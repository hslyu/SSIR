"""
Graph feature extraction for neural network input.

Converts IABRelayGraph objects into PyTorch geometric Data objects with
node features, edge features, and global features.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

import ssir.basestations as bs


def _compute_hops(graph: bs.IABRelayGraph) -> dict[int, int]:
    """
    Compute hop count from source (node 0) to each node.

    Args:
        graph: The graph to compute hops for

    Returns:
        Dictionary mapping node_id to hop_count
    """
    hops = {0: 0}  # Source node is 0 hops
    queue = [0]
    visited = {0}

    while queue:
        current = queue.pop(0)
        for neighbor in graph.adjacency_list.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                hops[neighbor] = hops[current] + 1
                queue.append(neighbor)

    # Nodes not reachable have infinite hops
    for node_id in graph.nodes.keys():
        if node_id not in hops:
            hops[node_id] = float('inf')

    return hops


def _get_node_features(
    master_graph: bs.IABRelayGraph,
    current_graph: bs.IABRelayGraph,
) -> tuple[torch.Tensor, list[int]]:
    """
    Extract node features for all nodes in the graph.

    Features per node:
    - node_id (normalized by total nodes)
    - node_type (0=source, 1=user, 2=basestation)
    - position: lat, lon, alt
    - hops from source (in current graph state)
    - num_connected_users (for base stations)
    - config features (power_capacity, bandwidth, etc.) for base stations only

    Args:
        master_graph: The full network topology
        current_graph: The current state (may have partial user routes)

    Returns:
        (node_features tensor, node_id_list)
    """
    node_ids = sorted(master_graph.nodes.keys())
    num_nodes = len(node_ids)

    # Compute hops in current graph state
    hops = _compute_hops(current_graph)

    # Config feature keys that we know exist in BaseStationConfig
    config_feature_keys = [
        'power_capacity',
        'minimum_transit_power_ratio',
        'carrier_frequency',
        'bandwidth',
        'transmit_antenna_gain',
        'receive_antenna_gain',
        'antenna_gain_to_noise_temperature',
        'pathloss_exponent',
        'eavesdropper_density',
        'maximum_link_distance',
    ]

    features_list = []

    for node_id in node_ids:
        master_node = master_graph.nodes[node_id]
        current_node = current_graph.nodes[node_id]

        # Node type: 0=source, 1=user, 2=basestation
        if node_id == 0:
            node_type = 0.0
        elif isinstance(master_node, bs.User):
            node_type = 1.0
        elif isinstance(master_node, bs.BaseStation):
            node_type = 2.0
        else:
            node_type = -1.0  # Unknown type

        # Position features
        position = master_node.get_position()
        pos_lat, pos_lon, pos_alt = (
            float(position[0]),
            float(position[1]),
            float(position[2]),
        )

        # Hops from source
        hop_count = float(hops.get(node_id, 0))
        hop_count = 0.0 if np.isinf(hop_count) else hop_count

        # Connected users (for base stations)
        num_connected_users = 0.0
        if isinstance(master_node, bs.BaseStation):
            num_connected_users = float(len(current_node.connected_user))

        # Base features
        node_features = [
            float(node_id) / num_nodes,  # Normalized node ID
            node_type,
            pos_lat,
            pos_lon,
            pos_alt,
            hop_count,
            num_connected_users,
        ]

        # Config features (for base stations)
        if isinstance(master_node, bs.BaseStation):
            config = master_node.config
            config_features = [
                float(getattr(config, key, 0.0))
                for key in config_feature_keys
            ]
            node_features.extend(config_features)
        else:
            # Pad with zeros if not a base station
            node_features.extend([0.0] * len(config_feature_keys))

        features_list.append(node_features)

    node_features_tensor = torch.tensor(features_list, dtype=torch.float32)
    return node_features_tensor, node_ids


def _get_edge_features(
    master_graph: bs.IABRelayGraph,
    current_graph: bs.IABRelayGraph,
    node_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract edge features for all edges in the current graph.

    Features per edge:
    - distance (geo distance between nodes)
    - is_active (whether edge exists in current graph)

    Args:
        master_graph: The full network topology
        current_graph: The current state
        node_ids: List of node IDs (for index mapping)

    Returns:
        (edge_index tensor [2, num_edges], edge_attr tensor [num_edges, num_features])
    """
    node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

    edge_index_list = []
    edge_attr_list = []

    # Iterate over edges in current graph
    for from_node_id in node_ids:
        for to_node_id in current_graph.adjacency_list.get(from_node_id, []):
            from_idx = node_id_to_idx[from_node_id]
            to_idx = node_id_to_idx[to_node_id]

            edge_index_list.append([from_idx, to_idx])

            # Compute distance
            from_node = master_graph.nodes[from_node_id]
            to_node = master_graph.nodes[to_node_id]
            distance = from_node.get_distance(to_node)

            edge_attr_list.append([float(distance)])

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    else:
        # No edges: create empty tensors
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float32)

    return edge_index, edge_attr


def _get_global_features(
    current_graph: bs.IABRelayGraph,
) -> torch.Tensor:
    """
    Extract global graph features.

    Features:
    - SPSC_probability
    - noise_power_density

    Args:
        current_graph: The current graph state

    Returns:
        Global features tensor [num_global_features,]
    """
    env = current_graph.environmental_variables
    global_features = [
        float(env.SPSC_probability),
        float(env.noise_power_density),
    ]
    return torch.tensor(global_features, dtype=torch.float32)


def encode_graph_state(
    master_graph: bs.IABRelayGraph,
    current_graph: bs.IABRelayGraph,
) -> Data:
    """
    Encode a graph state into a PyTorch geometric Data object.

    This captures the current network state: which users have routes,
    what the partial graph looks like, all necessary for predicting
    throughput of candidate routes.

    Args:
        master_graph: The full network topology (static)
        current_graph: The current state (may have partial user routes)

    Returns:
        PyTorch geometric Data object with:
        - x: node features [num_nodes, num_node_features]
        - edge_index: edge connectivity [2, num_edges]
        - edge_attr: edge features [num_edges, num_edge_features]
        - global_features: global features [num_global_features]
    """
    # Extract features
    node_features, node_ids = _get_node_features(master_graph, current_graph)
    edge_index, edge_attr = _get_edge_features(master_graph, current_graph, node_ids)
    global_features = _get_global_features(current_graph)

    # Create Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    # Store metadata
    data.global_features = global_features
    data.node_ids = torch.tensor(node_ids, dtype=torch.long)

    return data


def get_node_feature_dim(master_graph: bs.IABRelayGraph | None = None) -> int:
    """
    Get the dimension of node features for a graph.

    Returns:
        Dimension of node feature vectors
    """
    # Base features: node_id, node_type, lat, lon, alt, hops, num_connected_users
    base_dim = 7
    # Config features: power_capacity, minimum_transit_power_ratio, carrier_frequency,
    #                  bandwidth, transmit_antenna_gain, receive_antenna_gain,
    #                  antenna_gain_to_noise_temperature, pathloss_exponent,
    #                  eavesdropper_density, maximum_link_distance
    config_dim = 10
    return base_dim + config_dim


def get_edge_feature_dim() -> int:
    """Get the dimension of edge features."""
    return 1  # distance


def get_global_feature_dim() -> int:
    """Get the dimension of global features."""
    return 2  # SPSC_probability, noise_power_density
