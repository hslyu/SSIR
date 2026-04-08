from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data

import ssir.basestations as bs
from ssir.pathfinder.rl.trajectory import (
    UserRouteCandidate,
    apply_candidate_route,
    evaluate_candidate_throughput,
)


@dataclass
class UserRouteBatch:
    graph_data: Data
    candidate_node_mask: torch.Tensor
    candidate_edge_mask: torch.Tensor
    candidate_node_aux: torch.Tensor
    candidate_edge_aux: torch.Tensor
    candidate_target: torch.Tensor
    candidate_name: list[str]
    user_id: int


def _safe_throughput(node: bs.BaseStation) -> float:
    throughput = node.compute_throughput()
    return 0.0 if not np.isfinite(throughput) else float(throughput)


def _node_feature_tensor(
    master_graph: bs.IABRelayGraph,
    current_graph: bs.IABRelayGraph,
) -> torch.Tensor:
    current_graph.compute_hops()
    node_features = []
    tau = current_graph.environmental_variables.SPSC_probability
    noise_density = current_graph.environmental_variables.noise_power_density

    for node_id in sorted(master_graph.nodes.keys()):
        master_node = master_graph.nodes[node_id]
        current_node = current_graph.nodes[node_id]
        node_type = 0.0 if node_id == 0 else (1.0 if isinstance(master_node, bs.User) else 2.0)
        base = [float(node_id), node_type, *map(float, master_node.get_position())]

        if isinstance(master_node, bs.BaseStation):
            master_node._set_transmission_and_jamming_power_density()
            current_node._set_transmission_and_jamming_power_density()
            cfg = master_node.config
            config_vec = [
                cfg.power_capacity,
                cfg.minimum_transit_power_ratio,
                cfg.carrier_frequency,
                cfg.bandwidth,
                cfg.transmit_antenna_gain,
                cfg.receive_antenna_gain,
                cfg.antenna_gain_to_noise_temperature,
                cfg.pathloss_exponent,
                cfg.eavesdropper_density,
                cfg.maximum_link_distance,
            ]
            assignment_vec = [
                float(len(current_node.get_children())),
                float(len(current_node.connected_user)),
                float(sum(user.hops for user in current_node.connected_user)),
                float(current_node._get_farthest_forward_link_distance()),
                float(current_node.transmission_power_density),
                float(current_node.jamming_power_density),
                float(_safe_throughput(current_node)),
                tau,
                noise_density,
            ]
        else:
            config_vec = [0.0] * 10
            assignment_vec = [
                float(len(current_node.get_parent())),
                0.0,
                float(current_node.hops),
                0.0,
                0.0,
                0.0,
                0.0,
                tau,
                noise_density,
            ]

        node_features.append(base + config_vec + assignment_vec)

    return torch.tensor(node_features, dtype=torch.float)


def _edge_feature_tensor(
    master_graph: bs.IABRelayGraph,
    current_graph: bs.IABRelayGraph,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_index = []
    edge_features = []
    current_edge_set = {
        (from_node_id, to_node_id)
        for from_node_id, neighbors in current_graph.adjacency_list.items()
        for to_node_id in neighbors
    }

    for from_node_id, neighbors in master_graph.adjacency_list.items():
        master_from = master_graph.nodes[from_node_id]
        for to_node_id in neighbors:
            master_to = master_graph.nodes[to_node_id]
            distance = float(master_from.get_distance(master_to))
            if isinstance(master_from, bs.BaseStation):
                master_from._set_transmission_and_jamming_power_density()
                snr = float(master_from._compute_snr(master_to))
                spectral_efficiency = float(np.log2(1.0 + snr))
                tx_density = float(master_from.transmission_power_density)
                jam_density = float(master_from.jamming_power_density)
                pathloss_exponent = float(master_from.config.pathloss_exponent)
                bandwidth = float(master_from.config.bandwidth)
            else:
                snr = 0.0
                spectral_efficiency = 0.0
                tx_density = 0.0
                jam_density = 0.0
                pathloss_exponent = 0.0
                bandwidth = 0.0

            assigned_flag = 1.0 if (from_node_id, to_node_id) in current_edge_set else 0.0
            edge_index.append([from_node_id, to_node_id])
            edge_features.append(
                [
                    distance,
                    snr,
                    spectral_efficiency,
                    tx_density,
                    jam_density,
                    pathloss_exponent,
                    bandwidth,
                    assigned_flag,
                ]
            )

    return (
        torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        torch.tensor(edge_features, dtype=torch.float),
    )


def build_user_route_batch(
    master_graph: bs.IABRelayGraph,
    current_graph: bs.IABRelayGraph,
    user_id: int,
    route_candidates: list[UserRouteCandidate],
) -> UserRouteBatch:
    graph_data = master_graph.to_torch_geometric()
    graph_data.x = _node_feature_tensor(master_graph, current_graph)
    graph_data.edge_index, graph_data.edge_attr = _edge_feature_tensor(
        master_graph, current_graph
    )

    edge_lookup = {
        (int(src), int(dst)): idx
        for idx, (src, dst) in enumerate(graph_data.edge_index.t().tolist())
    }
    current_edge_set = {
        (from_node_id, to_node_id)
        for from_node_id, neighbors in current_graph.adjacency_list.items()
        for to_node_id in neighbors
    }

    num_candidates = len(route_candidates)
    num_nodes = graph_data.x.shape[0]
    num_edges = graph_data.edge_attr.shape[0]

    candidate_node_mask = torch.zeros((num_candidates, num_nodes), dtype=torch.float)
    candidate_edge_mask = torch.zeros((num_candidates, num_edges), dtype=torch.float)
    candidate_node_aux = torch.zeros((num_candidates, num_nodes, 4), dtype=torch.float)
    candidate_edge_aux = torch.zeros((num_candidates, num_edges, 4), dtype=torch.float)
    candidate_target = torch.zeros((num_candidates, 1), dtype=torch.float)
    candidate_name: list[str] = []

    for candidate_idx, candidate in enumerate(route_candidates):
        candidate_name.append(candidate.name)
        candidate_target[candidate_idx, 0] = evaluate_candidate_throughput(
            current_graph, candidate
        )
        updated_graph, added_edges = apply_candidate_route(current_graph, candidate)
        updated_graph.compute_hops()
        target_user = updated_graph.nodes[user_id]
        route_hops = float(target_user.hops)
        added_edge_set = set(added_edges)

        for node_id in candidate.node_ids:
            candidate_node_mask[candidate_idx, int(node_id)] = 1.0
            if isinstance(updated_graph.nodes[node_id], bs.BaseStation):
                candidate_node_aux[candidate_idx, int(node_id), 0] = 1.0
                candidate_node_aux[candidate_idx, int(node_id), 1] = 1.0
                candidate_node_aux[candidate_idx, int(node_id), 2] = route_hops
            elif node_id == user_id:
                candidate_node_aux[candidate_idx, int(node_id), 0] = 1.0
                candidate_node_aux[candidate_idx, int(node_id), 3] = route_hops

        for edge in candidate.edge_list:
            edge_idx = edge_lookup.get(edge)
            if edge_idx is None:
                continue
            candidate_edge_mask[candidate_idx, edge_idx] = 1.0
            candidate_edge_aux[candidate_idx, edge_idx, 0] = 1.0
            candidate_edge_aux[candidate_idx, edge_idx, 1] = (
                0.0 if edge in current_edge_set else 1.0
            )
            candidate_edge_aux[candidate_idx, edge_idx, 2] = route_hops
            child_id = edge[1]
            candidate_edge_aux[candidate_idx, edge_idx, 3] = (
                1.0 if isinstance(updated_graph.nodes[child_id], bs.User) else 0.0
            )

        for parent_id, child_id in added_edge_set:
            if parent_id in updated_graph.nodes and isinstance(
                updated_graph.nodes[parent_id], bs.BaseStation
            ):
                candidate_node_aux[candidate_idx, int(parent_id), 1] = 1.0

    return UserRouteBatch(
        graph_data=graph_data,
        candidate_node_mask=candidate_node_mask,
        candidate_edge_mask=candidate_edge_mask,
        candidate_node_aux=candidate_node_aux,
        candidate_edge_aux=candidate_edge_aux,
        candidate_target=candidate_target,
        candidate_name=candidate_name,
        user_id=user_id,
    )
