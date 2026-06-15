import json
import os
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ssir import basestations as bs
from ssir.pathfinder import utils
from ssir.pathfinder.astar import a_star, get_shortest_path
from ssir.pathfinder.policy_candidate_utils import (
    prepare_candidates,
    set_seed,
    valid_candidate_paths,
)


_DENOM_EPS = 1e-8


@dataclass
class RLReducedCostConfig:
    hidden_dim: int = 128
    lr: float = 1e-3
    epochs: int = 20
    num_predecessors: int = 100
    policy_device: str = "cpu"
    seed: int = 0
    max_negatives: int = 96
    alpha: float = 20.0
    dual_kl_weight: float = 0.1
    temperature: float = 1.0


@dataclass
class DirectRLConfig:
    hidden_dim: int = 128
    lr: float = 3e-4
    epochs: int = 10
    policy_device: str = "cpu"
    seed: int = 0
    alpha: float = 20.0
    alpha_list: tuple[float, ...] = (20.0,)
    dual_kl_weight: float = 0.02
    temperature: float = 0.5
    reward_temperature: float = 0.15
    num_policy_paths: int = 16
    noise_scale: float = 0.25
    learned_mix: float = 0.5
    max_hop: int = 12
    hop_slack: int = 3
    user_order: str = "farthest"
    dual_mode: str = "smooth"
    lookahead_weight: float = 1.0
    barrier_weight: float = 1.0
    include_classic_paths: bool = True
    max_grad_norm: float = 5.0
    policy_arch: str = "mlp"
    gnn_layers: int = 3
    dual_anchor: str = "mix"


_FUTURE_FEATURE_DIM = 5


class DualPressureNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features: torch.Tensor):
        return self.net(node_features).squeeze(-1)


class GraphDualPressureNet(nn.Module):
    uses_graph = True

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        edge_dim: int = 3,
        num_layers: int = 3,
    ):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden_dim)
        self.message_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim + edge_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.update_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.output = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ):
        h = F.relu(self.input(node_features))
        if edge_index is None or edge_index.numel() == 0:
            return self.output(h).squeeze(-1)

        src, dst = edge_index
        for msg_layer, update_layer in zip(self.message_layers, self.update_layers):
            messages = msg_layer(torch.cat([h[src], edge_attr], dim=-1))
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, messages)
            deg = torch.zeros((h.size(0), 1), device=h.device, dtype=h.dtype)
            deg.index_add_(
                0,
                dst,
                torch.ones((dst.numel(), 1), device=h.device, dtype=h.dtype),
            )
            agg = agg / deg.clamp_min(1.0)
            h = F.relu(update_layer(torch.cat([h, agg], dim=-1)) + h)
        return self.output(h).squeeze(-1)


class FutureDemandGraphDualNet(nn.Module):
    uses_graph = True
    uses_future_features = True

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        edge_dim: int = 3,
        future_dim: int = _FUTURE_FEATURE_DIM,
        num_layers: int = 2,
    ):
        super().__init__()
        self.future_dim = future_dim
        self.future_input = nn.Linear(future_dim, hidden_dim)
        self.message_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim + edge_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.update_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.dual_head = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        future_features: torch.Tensor | None = None,
    ):
        if future_features is None:
            future_features = torch.zeros(
                (node_features.size(0), self.future_dim),
                device=node_features.device,
                dtype=node_features.dtype,
            )

        h = F.relu(self.future_input(future_features))
        if edge_index is not None and edge_index.numel() > 0:
            src, dst = edge_index
            for msg_layer, update_layer in zip(self.message_layers, self.update_layers):
                messages = msg_layer(torch.cat([h[src], edge_attr], dim=-1))
                agg = torch.zeros_like(h)
                agg.index_add_(0, dst, messages)
                deg = torch.zeros((h.size(0), 1), device=h.device, dtype=h.dtype)
                deg.index_add_(
                    0,
                    dst,
                    torch.ones((dst.numel(), 1), device=h.device, dtype=h.dtype),
                )
                agg = agg / deg.clamp_min(1.0)
                h = F.relu(update_layer(torch.cat([h, agg], dim=-1)) + h)

        return self.dual_head(torch.cat([node_features, h], dim=-1)).squeeze(-1)


def _transmission_power_density_for_dmax(parent: bs.BaseStation, max_distance: float):
    config = parent.basestation_type.config
    pathloss_exponent = config.pathloss_exponent
    power_capacity_density = (
        bs.dB_to_linear(config.power_capacity) / config.bandwidth / 1e6
    )
    noise_power_density = bs.dB_to_linear(bs.environmental_variables.noise_power_density)
    tau = bs.environmental_variables.SPSC_probability
    kappa = (
        np.pi
        * parent.basestation_type.config.eavesdropper_density
        / np.sin(2 * np.pi / pathloss_exponent)
    ) ** 0.806 / 0.11

    if tau != 0:
        jamming_power_density = (
            max(
                (-(kappa * max_distance**2) / np.log(tau))
                ** (pathloss_exponent / 2)
                - 1,
                0,
            )
            * noise_power_density
            * 3.1623
        )
    else:
        jamming_power_density = 0
    jamming_power_density = min(jamming_power_density, power_capacity_density)
    transmission_power_density = power_capacity_density - jamming_power_density
    return bs.linear_to_dB(transmission_power_density + 1e-16)


def _snr_with_distance(
    parent: bs.BaseStation,
    child,
    tx_power_density_dbm: float,
    distance_km: float,
):
    distance_m = distance_km * 1e3
    config = parent.basestation_type.config
    c = 3.0e8
    freq_hz = config.carrier_frequency * 1e9
    wavelength_m = c / freq_hz
    pathloss_1m = 20.0 * np.log10((4.0 * np.pi) / wavelength_m)
    pathloss_d = pathloss_1m + 10.0 * config.pathloss_exponent * np.log10(
        distance_m
    )

    tx_gain_db = config.transmit_antenna_gain
    rx_gain_db = config.receive_antenna_gain
    antenna_gain_to_noise_temperature = config.antenna_gain_to_noise_temperature
    if (
        parent.basestation_type.name != bs.BaseStationType.LEO.name
        and isinstance(child, bs.BaseStation)
        and child.basestation_type.name == bs.BaseStationType.LEO.name
    ):
        tx_gain_db = 43.2
        rx_gain_db = 39.7
        antenna_gain_to_noise_temperature = 16.2

    rx_power_dbm = tx_power_density_dbm + tx_gain_db + rx_gain_db - pathloss_d
    noise_power_density_dbm = (
        bs.environmental_variables.noise_power_density
        + antenna_gain_to_noise_temperature
    )
    return bs.dB_to_linear(rx_power_dbm - noise_power_density_dbm)


def _snr_with_tx_power(parent: bs.BaseStation, child, tx_power_density_dbm: float):
    return _snr_with_distance(
        parent,
        child,
        tx_power_density_dbm,
        parent.get_distance(child),
    )


def _bs_one_hot(node):
    values = [0.0, 0.0, 0.0, 0.0]
    if isinstance(node, bs.BaseStation):
        idx = bs._bs_type_to_id(node)
        if idx >= 0:
            values[idx] = 1.0
    return values


def _relay_ids(graph: bs.IABRelayGraph):
    return [
        node.get_id()
        for node in graph.basestations
        if node.get_id() != 0
    ]


def _child_hop_masses(parent: bs.BaseStation):
    masses = {}
    for child in parent.get_children():
        if isinstance(child, bs.User):
            masses[child.get_id()] = float(child.hops)
        elif isinstance(child, bs.BaseStation):
            masses[child.get_id()] = float(sum(user.hops for user in child.connected_user))
    return masses


def _denominator_for_children(
    graph: bs.IABRelayGraph,
    parent_id: int,
    child_ids,
    hop_masses,
):
    parent = graph.nodes[parent_id]
    if not isinstance(parent, bs.BaseStation) or not child_ids:
        return 0.0

    gammas = _spectral_efficiencies_for_children(graph, parent_id, child_ids)
    denominator = _DENOM_EPS
    for child_id in child_ids:
        spectral_efficiency = gammas.get(child_id, 0.0)
        if spectral_efficiency <= 0.0 or not np.isfinite(spectral_efficiency):
            return float("inf")
        denominator += float(hop_masses.get(child_id, 0.0)) / spectral_efficiency
    return float(denominator)


def _spectral_efficiencies_for_children(
    graph: bs.IABRelayGraph,
    parent_id: int,
    child_ids,
):
    parent = graph.nodes[parent_id]
    if not isinstance(parent, bs.BaseStation) or not child_ids:
        return {}

    distances = {
        child_id: parent.get_distance(graph.nodes[child_id])
        for child_id in child_ids
    }
    max_distance = max(distances.values())
    tx_power_density = _transmission_power_density_for_dmax(parent, max_distance)
    gammas = {}
    for child_id in child_ids:
        child = graph.nodes[child_id]
        snr = _snr_with_distance(parent, child, tx_power_density, distances[child_id])
        gamma = float(np.log2(1.0 + snr))
        gammas[child_id] = gamma if np.isfinite(gamma) and gamma > 0.0 else 0.0
    return gammas


class _ParentDeltaCache:
    def __init__(self, graph: bs.IABRelayGraph, parent_id: int):
        self.graph = graph
        self.parent_id = parent_id
        self.parent = graph.nodes[parent_id]
        self.valid = isinstance(self.parent, bs.BaseStation)
        if not self.valid:
            return

        self.active_child_ids = [
            child.get_id() for child in self.parent.get_children()
        ]
        self.active_set = set(self.active_child_ids)
        self.distance_cache = {}
        self.before_masses = _child_hop_masses(self.parent)
        self.before = _denominator_for_children(
            graph, parent_id, self.active_child_ids, self.before_masses
        )
        self.bandwidth_hz = self.parent.basestation_type.config.bandwidth * 1e6
        self.current_dmax = (
            max(
                self._distance_to_child(child_id)
                for child_id in self.active_child_ids
            )
            if self.active_child_ids
            else 0.0
        )
        self.current_gamma = _spectral_efficiencies_for_children(
            graph, parent_id, self.active_child_ids
        )
        self.child_cache = {}

    def _distance_to_child(self, child_id: int):
        distance = self.distance_cache.get(child_id)
        if distance is None:
            distance = self.parent.get_distance(self.graph.nodes[child_id])
            self.distance_cache[child_id] = distance
        return distance

    def _child_terms(self, child_id: int):
        cached = self.child_cache.get(child_id)
        if cached is not None:
            return cached

        if not self.valid or self.bandwidth_hz <= 0 or not np.isfinite(self.before):
            cached = (1e12, 0.0)
            self.child_cache[child_id] = cached
            return cached

        if child_id in self.active_set:
            gamma = self.current_gamma.get(child_id, 0.0)
            cached = (0.0, 1.0 / gamma if gamma > 0 else 1e12)
            self.child_cache[child_id] = cached
            return cached

        child_distance = self._distance_to_child(child_id)
        if child_distance <= self.current_dmax or not self.active_child_ids:
            child_ids = (
                self.active_child_ids
                if self.active_child_ids
                else [child_id]
            )
            gammas = _spectral_efficiencies_for_children(
                self.graph, self.parent_id, child_ids
            )
            gamma = gammas.get(child_id, 0.0)
            if gamma <= 0 and child_id not in child_ids:
                gammas = _spectral_efficiencies_for_children(
                    self.graph, self.parent_id, self.active_child_ids + [child_id]
                )
                gamma = gammas.get(child_id, 0.0)
            cached = (0.0, 1.0 / gamma if gamma > 0 else 1e12)
            self.child_cache[child_id] = cached
            return cached

        after_child_ids = self.active_child_ids + [child_id]
        after_masses = dict(self.before_masses)
        after_masses[child_id] = 0.0
        gammas = _spectral_efficiencies_for_children(
            self.graph, self.parent_id, after_child_ids
        )
        base_after = _DENOM_EPS
        for active_child_id in self.active_child_ids:
            gamma = gammas.get(active_child_id, 0.0)
            if gamma <= 0:
                cached = (1e12, 0.0)
                self.child_cache[child_id] = cached
                return cached
            base_after += (
                float(self.before_masses.get(active_child_id, 0.0)) / gamma
            )
        child_gamma = gammas.get(child_id, 0.0)
        if child_gamma <= 0:
            cached = (1e12, 0.0)
        else:
            cached = (base_after - self.before, 1.0 / child_gamma)
        self.child_cache[child_id] = cached
        return cached

    def load_increment(self, child_id: int, final_hop_count: int):
        base_delta, inv_child_gamma = self._child_terms(child_id)
        if base_delta >= 1e12:
            return 1e12
        delta = (base_delta + float(final_hop_count) * inv_child_gamma) / self.bandwidth_hz
        if not np.isfinite(delta):
            return 1e12
        return float(max(delta, 0.0))


def _local_load_increment(
    graph: bs.IABRelayGraph,
    parent_id: int,
    child_id: int,
    final_hop_count: int,
):
    parent = graph.nodes[parent_id]
    if not isinstance(parent, bs.BaseStation):
        return 0.0

    active_child_ids = [child.get_id() for child in parent.get_children()]
    before_masses = _child_hop_masses(parent)
    before = _denominator_for_children(
        graph, parent_id, active_child_ids, before_masses
    )

    after_child_ids = list(active_child_ids)
    if child_id not in after_child_ids:
        after_child_ids.append(child_id)
    after_masses = dict(before_masses)
    after_masses[child_id] = after_masses.get(child_id, 0.0) + float(final_hop_count)
    after = _denominator_for_children(graph, parent_id, after_child_ids, after_masses)

    bandwidth_hz = parent.basestation_type.config.bandwidth * 1e6
    if not np.isfinite(before) or not np.isfinite(after) or bandwidth_hz <= 0:
        return 1e12
    return float(max((after - before) / bandwidth_hz, 0.0))


def _state_loads(graph: bs.IABRelayGraph):
    graph.compute_hops()
    loads = {}
    denoms = {}
    throughputs = {}
    child_counts = {}
    dmax = {}
    sum_hops = {}

    for node_id, node in graph.nodes.items():
        child_counts[node_id] = float(len(node.get_children()))
        if not isinstance(node, bs.BaseStation):
            loads[node_id] = 0.0
            denoms[node_id] = 0.0
            throughputs[node_id] = float("inf")
            dmax[node_id] = 0.0
            sum_hops[node_id] = 0.0
            continue

        children = node.get_children()
        dmax[node_id] = (
            max(node.get_distance(child) for child in children) if children else 0.0
        )
        masses = _child_hop_masses(node)
        denom = _denominator_for_children(
            graph, node_id, [child.get_id() for child in children], masses
        )
        bandwidth_hz = node.basestation_type.config.bandwidth * 1e6
        denoms[node_id] = denom
        loads[node_id] = 0.0 if not children else float(denom / bandwidth_hz)
        throughputs[node_id] = (
            float("inf") if not children else float(bandwidth_hz / max(denom, 1e-30))
        )
        sum_hops[node_id] = float(sum(user.hops for user in node.connected_user))

    return loads, denoms, throughputs, child_counts, dmax, sum_hops


def _smooth_dual_for_relays(loads, relay_ids, alpha: float):
    if not relay_ids:
        return np.zeros(0, dtype=np.float32)
    values = np.array([loads.get(node_id, 0.0) for node_id in relay_ids], dtype=np.float64)
    std = float(values.std())
    if std > 1e-12:
        logits = alpha * ((values - float(values.mean())) / std)
    else:
        logits = alpha * values
    logits = logits - float(logits.max())
    probs = np.exp(logits)
    denom = float(probs.sum())
    if denom <= 0.0 or not np.isfinite(denom):
        return np.ones(len(relay_ids), dtype=np.float32) / max(len(relay_ids), 1)
    return (probs / denom).astype(np.float32)


def _topq_dual_for_relays(loads, relay_ids, q: int):
    if not relay_ids:
        return np.zeros(0, dtype=np.float32)
    values = np.array([loads.get(node_id, 0.0) for node_id in relay_ids], dtype=np.float64)
    q = max(1, min(int(q), len(relay_ids)))
    order = np.argsort(values)[::-1][:q]
    dual = np.zeros(len(relay_ids), dtype=np.float32)
    dual[order] = 1.0 / float(q)
    return dual


def _future_shortest_path_loads(
    feasible_graph: bs.IABRelayGraph,
    state_graph: bs.IABRelayGraph,
    static_context,
):
    future_loads = {node_id: 0.0 for node_id in state_graph.nodes}
    parent_delta_cache = {}

    for user in state_graph.users:
        user_id = user.get_id()
        if user.has_parent():
            continue
        path = get_shortest_path(static_context.hop_pred, user_id)
        if not path or path[0] == -1 or len(path) < 2:
            continue

        hop_count = max(len(path) - 1, 1)
        for parent_id, child_id in zip(path[:-1], path[1:]):
            if parent_id == 0:
                continue
            parent = state_graph.nodes[parent_id]
            if not isinstance(parent, bs.BaseStation):
                continue
            cache = parent_delta_cache.get(parent_id)
            if cache is None:
                cache = _ParentDeltaCache(state_graph, parent_id)
                parent_delta_cache[parent_id] = cache
            future_loads[parent_id] += cache.load_increment(child_id, hop_count)

    return future_loads


def _future_path_aggregate_features(
    feasible_graph: bs.IABRelayGraph,
    state_graph: bs.IABRelayGraph,
    static_context,
    relay_ids,
    max_hop: int,
):
    """Embed uncertain future demand by pooling over full expected paths.

    The full path signal is injected before message passing, so a shallow GNN
    smooths demand rather than having to discover long paths layer by layer.
    """
    if not relay_ids:
        return torch.empty((0, _FUTURE_FEATURE_DIM), dtype=torch.float)

    relay_set = set(relay_ids)
    stats = {
        node_id: {
            "count": 0.0,
            "hop_mass": 0.0,
            "delta_load": 0.0,
            "distance_mass": 0.0,
            "children": set(),
        }
        for node_id in relay_ids
    }
    parent_delta_cache = {}
    users_with_paths = 0

    for user in state_graph.users:
        user_id = user.get_id()
        if user.has_parent():
            continue

        path_candidates = []
        seen_paths = set()
        for pred in (static_context.hop_pred, static_context.distance_pred):
            path = get_shortest_path(pred, user_id)
            if not path or path[0] == -1 or len(path) < 2:
                continue
            if len(path) != len(set(path)):
                continue
            if len(path) - 1 > max_hop:
                continue
            key = tuple(path)
            if key not in seen_paths:
                seen_paths.add(key)
                path_candidates.append(path)

        if not path_candidates:
            continue

        users_with_paths += 1
        path_weight = 1.0 / float(len(path_candidates))
        for path in path_candidates:
            hop_count = max(len(path) - 1, 1)
            for parent_id, child_id in zip(path[:-1], path[1:]):
                if parent_id not in relay_set:
                    continue
                parent = state_graph.nodes[parent_id]
                child = state_graph.nodes[child_id]
                if not isinstance(parent, bs.BaseStation):
                    continue

                cache = parent_delta_cache.get(parent_id)
                if cache is None:
                    cache = _ParentDeltaCache(state_graph, parent_id)
                    parent_delta_cache[parent_id] = cache
                delta = cache.load_increment(child_id, hop_count)
                if not np.isfinite(delta) or delta >= 1e12:
                    delta = 1e12

                node_stats = stats[parent_id]
                node_stats["count"] += path_weight
                node_stats["hop_mass"] += path_weight * float(hop_count)
                node_stats["delta_load"] += path_weight * max(float(delta), 0.0)
                node_stats["distance_mass"] += path_weight * parent.get_distance(child)
                node_stats["children"].add(child_id)

    if users_with_paths == 0:
        return torch.zeros((len(relay_ids), _FUTURE_FEATURE_DIM), dtype=torch.float)

    denom_users = float(users_with_paths)
    denom_hops = max(denom_users * float(max(max_hop, 1)), 1.0)
    rows = []
    for node_id in relay_ids:
        node_stats = stats[node_id]
        count = float(node_stats["count"])
        avg_distance = (
            float(node_stats["distance_mass"]) / max(count, 1e-9)
            if count > 0.0
            else 0.0
        )
        rows.append(
            [
                count / denom_users,
                float(node_stats["hop_mass"]) / denom_hops,
                np.log1p(max(float(node_stats["delta_load"]), 0.0) * 1e12 / denom_users),
                np.log1p(max(avg_distance, 0.0)),
                np.log1p(float(len(node_stats["children"]))),
            ]
        )

    return torch.tensor(rows, dtype=torch.float)


def _unconnected_user_fraction(graph: bs.IABRelayGraph):
    if not graph.users:
        return 0.0
    unconnected = sum(1 for user in graph.users if not user.has_parent())
    return float(unconnected) / float(len(graph.users))


def _node_features(graph: bs.IABRelayGraph, alpha: float, state_stats=None):
    if state_stats is None:
        state_stats = _state_loads(graph)
    loads, denoms, throughputs, child_counts, dmax, sum_hops = state_stats
    relay_ids = _relay_ids(graph)
    base_dual = _smooth_dual_for_relays(loads, relay_ids, alpha)
    finite_tps = [
        1.0 / (throughputs[node_id] + 1e-9)
        for node_id in relay_ids
        if np.isfinite(throughputs[node_id])
    ]
    max_inv_tp = max(finite_tps) if finite_tps else 1.0

    rows = []
    for idx, node_id in enumerate(relay_ids):
        node = graph.nodes[node_id]
        assert isinstance(node, bs.BaseStation)
        cfg = node.basestation_type.config
        throughput = throughputs[node_id]
        inv_tp = 0.0 if np.isinf(throughput) else 1.0 / (throughput + 1e-9)
        rows.append(
            [
                *_bs_one_hot(node),
                np.log1p(max(loads[node_id], 0.0) * 1e12),
                np.log1p(max(denoms[node_id], 0.0)),
                inv_tp / max(max_inv_tp, 1e-30),
                float(base_dual[idx]),
                np.log1p(max(child_counts[node_id], 0.0)),
                np.log1p(max(dmax[node_id], 0.0)),
                np.log1p(max(sum_hops[node_id], 0.0)),
                np.log1p(max(cfg.bandwidth, 0.0)),
                np.log1p(max(cfg.power_capacity, 0.0)),
                np.log1p(max(cfg.pathloss_exponent, 0.0)),
            ]
        )

    if not rows:
        return relay_ids, torch.empty((0, 13), dtype=torch.float), torch.empty(0)
    return (
        relay_ids,
        torch.tensor(rows, dtype=torch.float),
        torch.tensor(base_dual, dtype=torch.float),
    )


def _make_dual_model(feature_dim: int, config):
    policy_arch = getattr(config, "policy_arch", "mlp")
    if policy_arch == "future_gnn_mlp":
        return FutureDemandGraphDualNet(
            feature_dim,
            hidden_dim=config.hidden_dim,
            num_layers=getattr(config, "gnn_layers", 2),
        )
    if policy_arch == "gnn":
        return GraphDualPressureNet(
            feature_dim,
            hidden_dim=config.hidden_dim,
            num_layers=getattr(config, "gnn_layers", 3),
        )
    if policy_arch != "mlp":
        raise ValueError("policy_arch must be one of: mlp, gnn, future_gnn_mlp")
    return DualPressureNet(feature_dim, config.hidden_dim)


def _dual_graph_inputs(
    feasible_graph: bs.IABRelayGraph,
    state_graph: bs.IABRelayGraph,
    relay_ids,
):
    relay_to_idx = {node_id: idx for idx, node_id in enumerate(relay_ids)}
    edges = []
    attrs = []

    for parent_id, child_id in feasible_graph.edges:
        if parent_id not in relay_to_idx or child_id not in relay_to_idx:
            continue
        parent = feasible_graph.nodes[parent_id]
        child = feasible_graph.nodes[child_id]
        if not isinstance(parent, bs.BaseStation):
            continue
        cfg = parent.basestation_type.config
        distance = parent.get_distance(child)
        max_link = max(float(cfg.maximum_link_distance), 1e-9)
        slack = np.clip((max_link - distance) / max_link, -10.0, 10.0)
        selected = float(child_id in state_graph.get_neighbors(parent_id))
        attr = [
            np.log1p(max(distance, 0.0)),
            slack,
            selected,
        ]
        src = relay_to_idx[parent_id]
        dst = relay_to_idx[child_id]
        edges.append((src, dst))
        attrs.append(attr)
        edges.append((dst, src))
        attrs.append(attr)

    if not edges:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 3), dtype=torch.float),
        )
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(attrs, dtype=torch.float)
    edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=30.0, neginf=-30.0)
    return edge_index, edge_attr


def _dual_model_logits(
    model,
    features: torch.Tensor,
    device: str,
    feasible_graph: bs.IABRelayGraph | None = None,
    state_graph: bs.IABRelayGraph | None = None,
    relay_ids=None,
    future_features: torch.Tensor | None = None,
):
    features = features.to(device)
    if getattr(model, "uses_graph", False):
        if feasible_graph is None or state_graph is None or relay_ids is None:
            raise ValueError("GNN dual model requires graph context.")
        edge_index, edge_attr = _dual_graph_inputs(feasible_graph, state_graph, relay_ids)
        if getattr(model, "uses_future_features", False):
            if future_features is None:
                future_features = torch.zeros(
                    (features.size(0), _FUTURE_FEATURE_DIM), dtype=torch.float
                )
            return model(
                features,
                edge_index.to(device),
                edge_attr.to(device),
                future_features.to(device),
            )
        return model(
            features,
            edge_index.to(device),
            edge_attr.to(device),
        )
    return model(features)


def _combine_dual_with_logits(
    base_dual: torch.Tensor,
    logits: torch.Tensor,
    learned_mix: float,
    dual_anchor: str = "mix",
):
    if dual_anchor == "logit_residual":
        residual = logits - logits.mean()
        residual_scale = residual.std(unbiased=False).clamp_min(1.0)
        residual = residual / residual_scale
        return F.softmax(
            torch.log(base_dual.clamp_min(1e-12)) + learned_mix * residual,
            dim=0,
        )
    if dual_anchor != "mix":
        raise ValueError("dual_anchor must be one of: mix, logit_residual")
    learned = F.softmax(logits, dim=0)
    if learned_mix <= 0:
        return base_dual
    dual = (1.0 - learned_mix) * base_dual + learned_mix * learned
    return dual / dual.sum().clamp_min(1e-12)


def _path_delta_vector(
    graph: bs.IABRelayGraph,
    path,
    relay_to_idx,
):
    vec = np.zeros(len(relay_to_idx), dtype=np.float32)
    final_hop_count = max(len(path) - 1, 1)
    for parent_id, child_id in zip(path[:-1], path[1:]):
        relay_idx = relay_to_idx.get(parent_id)
        if relay_idx is None:
            continue
        vec[relay_idx] += _local_load_increment(
            graph, parent_id, child_id, final_hop_count
        )
    return vec


def _exact_best_path(state_graph: bs.IABRelayGraph, user_id: int, path_list):
    best_score = -1.0
    best_path = None
    for path in path_list:
        added_edges = utils.get_aborescence_graph(state_graph, path)
        score = state_graph.compute_network_throughput(path[1:-1])
        if score > best_score:
            best_score = score
            best_path = path
        utils.remove_added_edges(state_graph, user_id, added_edges)
    if best_path is None:
        raise ValueError("No path found.")
    return best_path, best_score


def _exact_path_scores(state_graph: bs.IABRelayGraph, user_id: int, path_list):
    scores = []
    for path in path_list:
        added_edges = utils.get_aborescence_graph(state_graph, path)
        score = state_graph.compute_network_throughput(path[1:-1])
        scores.append(float(score))
        utils.remove_added_edges(state_graph, user_id, added_edges)
    return scores


def _path_delta_matrix(state_graph: bs.IABRelayGraph, path_list, relay_to_idx):
    rows = []
    for path in path_list:
        rows.append(_path_delta_vector(state_graph, path, relay_to_idx))
    return torch.tensor(np.stack(rows), dtype=torch.float)


def _build_training_samples(graphs, config: RLReducedCostConfig):
    samples = []
    for feasible_graph in graphs:
        user_order, all_shortest_paths = prepare_candidates(
            feasible_graph, config.num_predecessors
        )
        state_graph = feasible_graph.copy()
        state_graph.reset()

        for user_id in user_order:
            path_list = valid_candidate_paths(all_shortest_paths[user_id])
            path_list = [path for path in path_list if len(path) == len(set(path))]
            if len(path_list) < 2:
                continue

            utils.delete_user(state_graph, user_id)
            teacher_path, _ = _exact_best_path(state_graph, user_id, path_list)

            relay_ids, features, base_dual = _node_features(state_graph, config.alpha)
            relay_to_idx = {node_id: idx for idx, node_id in enumerate(relay_ids)}
            if not relay_to_idx:
                utils.get_aborescence_graph(state_graph, teacher_path)
                continue

            deltas = []
            teacher_idx = None
            for idx, path in enumerate(path_list):
                if path == teacher_path:
                    teacher_idx = idx
                deltas.append(_path_delta_vector(state_graph, path, relay_to_idx))

            if teacher_idx is not None:
                samples.append(
                    {
                        "features": features,
                        "base_dual": base_dual,
                        "path_deltas": torch.tensor(np.stack(deltas), dtype=torch.float),
                        "teacher_idx": teacher_idx,
                    }
                )

            utils.get_aborescence_graph(state_graph, teacher_path)
    return samples


def train_policy(
    graphs,
    save_path: str,
    config: RLReducedCostConfig | None = None,
    log_every_epoch: bool = True,
):
    config = config or RLReducedCostConfig()
    set_seed(config.seed)
    samples = _build_training_samples(graphs, config)
    if not samples:
        raise ValueError("No training samples were generated.")

    feature_dim = samples[0]["features"].shape[1]
    model = DualPressureNet(feature_dim, config.hidden_dim).to(config.policy_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    rng = random.Random(config.seed)

    history = []
    for epoch in range(config.epochs):
        rng.shuffle(samples)
        losses = []
        top1 = []
        for sample in samples:
            features = sample["features"].to(config.policy_device)
            base_dual = sample["base_dual"].to(config.policy_device)
            path_deltas = sample["path_deltas"].to(config.policy_device)
            teacher_idx = sample["teacher_idx"]

            if path_deltas.size(0) > config.max_negatives + 1:
                keep = {teacher_idx}
                while len(keep) < config.max_negatives + 1:
                    keep.add(rng.randrange(path_deltas.size(0)))
                order = sorted(keep)
                teacher_idx = order.index(sample["teacher_idx"])
                path_deltas = path_deltas[order]

            logits = model(features)
            dual = _combine_dual_with_logits(
                base_dual,
                logits,
                learned_mix=1.0,
                dual_anchor="mix",
            )
            path_costs = torch.matmul(path_deltas, dual)
            centered_costs = path_costs - path_costs.mean()
            cost_scale = centered_costs.std(unbiased=False).clamp_min(1e-12)
            path_logits = -(centered_costs / cost_scale) / max(
                config.temperature, 1e-6
            )
            target = torch.tensor([teacher_idx], device=config.policy_device)
            ce_loss = F.cross_entropy(path_logits.unsqueeze(0), target)
            kl_loss = F.kl_div(
                torch.log(dual + 1e-12),
                base_dual,
                reduction="sum",
            )
            loss = ce_loss + config.dual_kl_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))
            top1.append(float(int(torch.argmin(path_costs).item()) == teacher_idx))

        history.append(
            {
                "epoch": epoch,
                "avg_loss": float(np.mean(losses)) if losses else 0.0,
                "top1": float(np.mean(top1)) if top1 else 0.0,
            }
        )
        if log_every_epoch:
            last = history[-1]
            print(
                f"[epoch {last['epoch'] + 1}/{config.epochs}] "
                f"avg_loss={last['avg_loss']:.4f} top1={last['top1']:.3f}"
            )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(
        {
            "policy_type": "rl_reduced_cost_policy",
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "feature_dim": feature_dim,
        },
        save_path,
    )
    with open(f"{save_path}.history.json", "w") as f:
        json.dump(history, f, indent=2)
    return model, history


def train_policy_direct_rl(
    graphs,
    save_path: str,
    config: DirectRLConfig | None = None,
    log_every_epoch: bool = True,
):
    """Train the dual field directly from exact max-min routing rewards.

    This is not teacher/path-overlap distillation. The current policy proposes
    reduced-cost paths, those paths are exact-evaluated, and the dual network is
    updated to assign lower reduced cost to higher-reward paths.
    """
    config = config or DirectRLConfig()
    set_seed(config.seed)
    if not graphs:
        raise ValueError("No training graphs were provided.")

    probe_graph = graphs[0].copy()
    probe_graph.reset()
    feature_dim = _node_features(probe_graph, config.alpha)[1].shape[1]
    model = _make_dual_model(feature_dim, config).to(config.policy_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    history = []
    for epoch in range(config.epochs):
        losses = []
        ce_losses = []
        kl_losses = []
        top1 = []
        rewards = []
        sample_count = 0

        graph_order = list(graphs)
        random.shuffle(graph_order)
        for feasible_graph in graph_order:
            static_context = _StaticRoutingContext(feasible_graph)
            user_id_list = _initial_user_order(
                feasible_graph, static_context, config.user_order
            )
            state_graph = feasible_graph.copy()
            state_graph.reset()

            for user_id in user_id_list:
                utils.delete_user(state_graph, user_id)
                candidate_paths = _reduced_cost_candidate_paths(
                    feasible_graph,
                    state_graph,
                    user_id,
                    static_context=static_context,
                    model=model,
                    device=config.policy_device,
                    alpha_list=config.alpha_list,
                    num_policy_paths=config.num_policy_paths,
                    noise_scale=config.noise_scale,
                    learned_mix=config.learned_mix,
                    max_hop=config.max_hop,
                    hop_slack=config.hop_slack,
                    include_classic_paths=config.include_classic_paths,
                    dual_mode=config.dual_mode,
                    lookahead_weight=config.lookahead_weight,
                    barrier_weight=config.barrier_weight,
                    dual_anchor=config.dual_anchor,
                )
                if len(candidate_paths) < 2:
                    if candidate_paths:
                        utils.get_aborescence_graph(state_graph, candidate_paths[0])
                    continue

                state_stats = _state_loads(state_graph)
                relay_ids, features, base_dual = _node_features(
                    state_graph, config.alpha, state_stats=state_stats
                )
                future_features = None
                if getattr(model, "uses_future_features", False):
                    future_features = _future_path_aggregate_features(
                        feasible_graph,
                        state_graph,
                        static_context,
                        relay_ids,
                        max_hop=config.max_hop,
                    )
                relay_to_idx = {
                    node_id: idx for idx, node_id in enumerate(relay_ids)
                }
                if not relay_to_idx:
                    best_path = candidate_paths[0]
                    utils.get_aborescence_graph(state_graph, best_path)
                    continue

                scores = _exact_path_scores(state_graph, user_id, candidate_paths)
                finite_scores = np.array(
                    [
                        score if np.isfinite(score) and score > 0.0 else 1e-30
                        for score in scores
                    ],
                    dtype=np.float64,
                )
                if len(finite_scores) < 2:
                    best_path = candidate_paths[int(np.argmax(finite_scores))]
                    utils.get_aborescence_graph(state_graph, best_path)
                    continue

                path_deltas = _path_delta_matrix(
                    state_graph, candidate_paths, relay_to_idx
                )
                features = features.to(config.policy_device)
                base_dual = base_dual.to(config.policy_device)
                path_deltas = path_deltas.to(config.policy_device)

                logits = _dual_model_logits(
                    model,
                    features,
                    config.policy_device,
                    feasible_graph=feasible_graph,
                    state_graph=state_graph,
                    relay_ids=relay_ids,
                    future_features=future_features,
                )
                dual = _combine_dual_with_logits(
                    base_dual,
                    logits,
                    learned_mix=config.learned_mix,
                    dual_anchor=config.dual_anchor,
                )
                path_costs = torch.matmul(path_deltas, dual)
                centered_costs = path_costs - path_costs.mean()
                cost_scale = centered_costs.std(unbiased=False).clamp_min(1e-12)
                path_logits = -(centered_costs / cost_scale) / max(
                    config.temperature, 1e-6
                )

                reward_values = torch.tensor(
                    np.log(finite_scores), dtype=torch.float, device=config.policy_device
                )
                reward_logits = (
                    reward_values - reward_values.max()
                ) / max(config.reward_temperature, 1e-6)
                target_probs = F.softmax(reward_logits, dim=0).detach()
                log_probs = F.log_softmax(path_logits, dim=0)
                ce_loss = -(target_probs * log_probs).sum()
                kl_loss = F.kl_div(
                    torch.log(dual + 1e-12),
                    base_dual,
                    reduction="sum",
                )
                loss = ce_loss + config.dual_kl_weight * kl_loss

                optimizer.zero_grad()
                loss.backward()
                if config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.max_grad_norm
                    )
                optimizer.step()

                best_idx = int(np.argmax(finite_scores))
                best_path = candidate_paths[best_idx]
                utils.get_aborescence_graph(state_graph, best_path)

                losses.append(float(loss.item()))
                ce_losses.append(float(ce_loss.item()))
                kl_losses.append(float(kl_loss.item()))
                top1.append(float(int(torch.argmin(path_costs).item()) == best_idx))
                rewards.append(float(finite_scores[best_idx]))
                sample_count += 1

        history.append(
            {
                "epoch": epoch,
                "avg_loss": float(np.mean(losses)) if losses else 0.0,
                "ce_loss": float(np.mean(ce_losses)) if ce_losses else 0.0,
                "kl_loss": float(np.mean(kl_losses)) if kl_losses else 0.0,
                "top1_exact_reward": float(np.mean(top1)) if top1 else 0.0,
                "avg_best_reward": float(np.mean(rewards)) if rewards else 0.0,
                "samples": int(sample_count),
            }
        )
        if log_every_epoch:
            last = history[-1]
            print(
                f"[direct-rl epoch {last['epoch'] + 1}/{config.epochs}] "
                f"avg_loss={last['avg_loss']:.4f} "
                f"ce={last['ce_loss']:.4f} "
                f"kl={last['kl_loss']:.4f} "
                f"top1={last['top1_exact_reward']:.3f} "
                f"samples={last['samples']}"
            )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(
        {
            "policy_type": "rl_reduced_cost_direct_rl",
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "feature_dim": feature_dim,
        },
        save_path,
    )
    with open(f"{save_path}.history.json", "w") as f:
        json.dump(history, f, indent=2)
    return model, history


def load_policy(model_path: str, device: str = "cpu"):
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", {})
    if config.get("policy_arch", "mlp") == "future_gnn_mlp":
        model = FutureDemandGraphDualNet(
            checkpoint["feature_dim"],
            hidden_dim=config.get("hidden_dim", 128),
            num_layers=config.get("gnn_layers", 2),
        ).to(device)
    elif config.get("policy_arch", "mlp") == "gnn":
        model = GraphDualPressureNet(
            checkpoint["feature_dim"],
            hidden_dim=config.get("hidden_dim", 128),
            num_layers=config.get("gnn_layers", 3),
        ).to(device)
    else:
        model = DualPressureNet(
            checkpoint["feature_dim"],
            hidden_dim=config.get("hidden_dim", 128),
        ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config


def _mixed_dual(
    graph: bs.IABRelayGraph,
    alpha: float,
    model=None,
    device: str = "cpu",
    learned_mix: float = 0.25,
    noise_scale: float = 0.0,
    state_stats=None,
    lookahead_loads=None,
    lookahead_weight: float = 1.0,
    dual_mode: str = "smooth",
    feasible_graph: bs.IABRelayGraph | None = None,
    static_context=None,
    max_hop: int = 12,
    dual_anchor: str = "mix",
):
    if model is None:
        if state_stats is None:
            state_stats = _state_loads(graph)
        loads = state_stats[0]
        relay_ids = _relay_ids(graph)
        if lookahead_loads is not None and lookahead_weight != 0.0:
            terminal_loads = {
                node_id: float(loads.get(node_id, 0.0))
                + lookahead_weight * float(lookahead_loads.get(node_id, 0.0))
                for node_id in graph.nodes
            }
            terminal_dual = _smooth_dual_for_relays(
                terminal_loads, relay_ids, alpha
            ).astype(np.float64)
            if dual_mode == "lookahead_mix":
                current_dual = _smooth_dual_for_relays(
                    loads, relay_ids, alpha
                ).astype(np.float64)
                beta = _unconnected_user_fraction(graph)
                dual = (1.0 - beta) * current_dual + beta * terminal_dual
            else:
                dual = terminal_dual
        else:
            dual = _smooth_dual_for_relays(loads, relay_ids, alpha).astype(np.float64)
        if noise_scale > 0:
            eps = np.random.normal(0.0, noise_scale, size=dual.shape)
            dual = np.exp(np.log(np.maximum(dual, 1e-30)) + eps)
        dual_sum = float(dual.sum())
        if dual_sum <= 0 or not np.isfinite(dual_sum):
            dual = np.ones_like(dual) / max(len(dual), 1)
        else:
            dual = dual / dual_sum
        return relay_ids, dual.astype(np.float32)

    relay_ids, features, base_dual = _node_features(graph, alpha, state_stats)
    if len(relay_ids) == 0:
        return relay_ids, np.zeros(0, dtype=np.float32)

    future_features = None
    if model is not None and getattr(model, "uses_future_features", False):
        if static_context is None:
            static_context = _StaticRoutingContext(feasible_graph or graph)
        future_features = _future_path_aggregate_features(
            feasible_graph or graph,
            graph,
            static_context,
            relay_ids,
            max_hop=max_hop,
        )

    if lookahead_loads is not None and lookahead_weight != 0.0:
        loads = state_stats[0] if state_stats is not None else _state_loads(graph)[0]
        adjusted_loads = {
            node_id: float(loads.get(node_id, 0.0))
            + lookahead_weight * float(lookahead_loads.get(node_id, 0.0))
            for node_id in graph.nodes
        }
        terminal_dual = _smooth_dual_for_relays(adjusted_loads, relay_ids, alpha).astype(
            np.float64
        )
        if dual_mode == "lookahead_mix":
            current_dual = base_dual.numpy().astype(np.float64)
            beta = _unconnected_user_fraction(graph)
            dual = (1.0 - beta) * current_dual + beta * terminal_dual
        else:
            dual = terminal_dual
    else:
        dual = base_dual.numpy().astype(np.float64)
    if model is not None and learned_mix > 0:
        with torch.no_grad():
            logits = _dual_model_logits(
                model,
                features,
                device,
                feasible_graph=feasible_graph or graph,
                state_graph=graph,
                relay_ids=relay_ids,
                future_features=future_features,
            )
            base_dual_tensor = torch.tensor(
                dual, dtype=torch.float, device=device
            )
            learned_dual = _combine_dual_with_logits(
                base_dual_tensor,
                logits,
                learned_mix=learned_mix,
                dual_anchor=dual_anchor,
            )
            learned = learned_dual.detach().cpu().numpy()
        dual = learned

    if noise_scale > 0:
        eps = np.random.normal(0.0, noise_scale, size=dual.shape)
        dual = np.exp(np.log(np.maximum(dual, 1e-30)) + eps)

    dual_sum = float(dual.sum())
    if dual_sum <= 0 or not np.isfinite(dual_sum):
        dual = np.ones_like(dual) / max(len(dual), 1)
    else:
        dual = dual / dual_sum
    return relay_ids, dual.astype(np.float32)


def _deterministic_dual_variants(
    graph: bs.IABRelayGraph,
    state_stats,
    alpha: float,
    lookahead_loads=None,
    lookahead_weight: float = 1.0,
):
    loads = state_stats[0]
    relay_ids = _relay_ids(graph)
    if lookahead_loads is not None and lookahead_weight != 0.0:
        loads = {
            node_id: float(loads.get(node_id, 0.0))
            + lookahead_weight * float(lookahead_loads.get(node_id, 0.0))
            for node_id in graph.nodes
        }

    variants = [_smooth_dual_for_relays(loads, relay_ids, alpha)]
    for q in (1, 3, 5):
        variants.append(_topq_dual_for_relays(loads, relay_ids, q))

    deduped = []
    seen = set()
    for dual in variants:
        key = tuple(np.round(dual, 8).tolist())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dual)
    return relay_ids, deduped


def _edge_reduced_cost(
    parent_id: int,
    delta: float,
    relay_to_dual,
    loads,
    max_load: float,
    dual_mode: str,
    barrier_weight: float,
):
    cost = relay_to_dual.get(parent_id, 0.0) * delta
    if dual_mode == "barrier" and parent_id in relay_to_dual:
        violation = float(loads.get(parent_id, 0.0)) + float(delta) - max_load
        if violation > 0.0:
            cost += barrier_weight * violation
    return cost


class _StaticRoutingContext:
    def __init__(self, graph: bs.IABRelayGraph):
        self.edge_list = list(graph.edges)
        self.reverse_adj = {}
        for parent_id, child_id in self.edge_list:
            self.reverse_adj.setdefault(child_id, []).append(parent_id)

        _, self.hop_pred = a_star(graph, metric="hop")
        _, self.distance_pred = a_star(graph, metric="distance")
        self.shortest_hops = {}
        for user in graph.users:
            path = get_shortest_path(self.hop_pred, user.get_id())
            self.shortest_hops[user.get_id()] = (
                max(len(path) - 1, 1) if path and path[0] != -1 else 1
            )


def _initial_user_order(
    graph: bs.IABRelayGraph,
    static_context: _StaticRoutingContext,
    user_order: str,
):
    uid_list = [user.get_id() for user in graph.users]
    if user_order == "nearest":
        return sorted(uid_list, key=lambda x: static_context.shortest_hops.get(x, 1))
    if user_order == "random":
        shuffled = list(uid_list)
        random.shuffle(shuffled)
        return shuffled
    if user_order == "constrained":
        scores = {}
        for user_id in uid_list:
            shortest_hop = static_context.shortest_hops.get(user_id, 1)
            upper_hop = max(shortest_hop, min(12, shortest_hop + 3))
            scores[user_id] = len(
                _relevant_edges_for_user(
                    graph, user_id, upper_hop, static_context=static_context
                )
            )
        return sorted(
            uid_list,
            key=lambda x: (scores.get(x, 0), -static_context.shortest_hops.get(x, 1)),
        )
    if user_order != "farthest":
        raise ValueError(
            "user_order must be one of: farthest, nearest, random, constrained"
        )
    return sorted(
        uid_list,
        key=lambda x: static_context.shortest_hops.get(x, 1),
        reverse=True,
    )


def _edge_adjacency_indices(edge_list):
    by_parent = {}
    for idx, (parent_id, child_id) in enumerate(edge_list):
        by_parent.setdefault(parent_id, []).append((child_id, idx))
    return by_parent


def _layered_exact_h_path(adjacency, edge_costs, user_id: int, hop_count: int):
    dist = [{0: 0.0}]
    pred = [{}]
    for depth in range(1, hop_count + 1):
        current = {}
        current_pred = {}
        for parent_id, parent_cost in dist[-1].items():
            for child_id, edge_idx in adjacency.get(parent_id, []):
                new_cost = parent_cost + edge_costs[edge_idx]
                if new_cost < current.get(child_id, float("inf")):
                    current[child_id] = new_cost
                    current_pred[child_id] = parent_id
        dist.append(current)
        pred.append(current_pred)

    if user_id not in dist[hop_count]:
        return None, float("inf")

    path = [user_id]
    current = user_id
    for depth in range(hop_count, 0, -1):
        current = pred[depth][current]
        path.append(current)
    path.reverse()
    return path, float(dist[hop_count][user_id])


def _relevant_edges_for_user(
    feasible_graph: bs.IABRelayGraph,
    user_id: int,
    max_hop: int,
    static_context: _StaticRoutingContext | None = None,
):
    forward = {0}
    frontier = {0}
    for _ in range(max_hop):
        next_frontier = set()
        for node_id in frontier:
            next_frontier.update(feasible_graph.get_neighbors(node_id))
        frontier = next_frontier - forward
        forward.update(next_frontier)

    edge_list = static_context.edge_list if static_context is not None else list(feasible_graph.edges)
    reverse_adj = static_context.reverse_adj if static_context is not None else {}
    if static_context is None:
        for parent_id, child_id in edge_list:
            reverse_adj.setdefault(child_id, []).append(parent_id)
    backward = {user_id}
    frontier = {user_id}
    for _ in range(max_hop):
        next_frontier = set()
        for node_id in frontier:
            next_frontier.update(reverse_adj.get(node_id, []))
        frontier = next_frontier - backward
        backward.update(next_frontier)

    return [
        (parent_id, child_id)
        for parent_id, child_id in edge_list
        if parent_id in forward and child_id in backward
    ]


def _reduced_cost_candidate_paths(
    feasible_graph: bs.IABRelayGraph,
    state_graph: bs.IABRelayGraph,
    user_id: int,
    static_context: _StaticRoutingContext | None = None,
    model=None,
    device: str = "cpu",
    alpha_list=(20.0,),
    num_policy_paths: int = 16,
    noise_scale: float = 0.15,
    learned_mix: float = 0.25,
    max_hop: int = 12,
    hop_slack: int = 3,
    include_classic_paths: bool = True,
    dual_mode: str = "smooth",
    lookahead_weight: float = 1.0,
    barrier_weight: float = 1.0,
    dual_anchor: str = "mix",
):
    if static_context is not None:
        shortest_hop = static_context.shortest_hops.get(user_id, 1)
    else:
        try:
            _, hop_pred = a_star(feasible_graph, goal=user_id, metric="hop")
            shortest_hop = max(len(get_shortest_path(hop_pred, user_id)) - 1, 1)
        except Exception:
            shortest_hop = 1
    upper_hop = max(shortest_hop, min(max_hop, shortest_hop + max(hop_slack, 0)))
    hop_budgets = list(range(shortest_hop, upper_hop + 1))
    edge_list = _relevant_edges_for_user(
        feasible_graph, user_id, upper_hop, static_context=static_context
    )
    if not edge_list:
        return []
    edge_adjacency = _edge_adjacency_indices(edge_list)

    state_stats = _state_loads(state_graph)
    lookahead_loads = None
    if dual_mode in {"lookahead", "lookahead_mix", "active_set"}:
        if static_context is None:
            static_context = _StaticRoutingContext(feasible_graph)
        lookahead_loads = _future_shortest_path_loads(
            feasible_graph, state_graph, static_context
        )
    elif dual_mode not in {"smooth", "barrier"}:
        raise ValueError(
            "dual_mode must be one of: smooth, barrier, lookahead, lookahead_mix, active_set"
        )
    delta_by_hop = {}
    parent_delta_cache = {}
    base_terms = np.zeros(len(edge_list), dtype=np.float64)
    inv_gamma_terms = np.zeros(len(edge_list), dtype=np.float64)
    bandwidth_terms = np.ones(len(edge_list), dtype=np.float64)
    invalid_terms = np.zeros(len(edge_list), dtype=bool)

    for edge_idx, (parent_id, child_id) in enumerate(edge_list):
        if parent_id == 0:
            continue
        parent = state_graph.nodes[parent_id]
        if not isinstance(parent, bs.BaseStation):
            continue
        cache = parent_delta_cache.get(parent_id)
        if cache is None:
            cache = _ParentDeltaCache(state_graph, parent_id)
            parent_delta_cache[parent_id] = cache
        base_delta, inv_child_gamma = cache._child_terms(child_id)
        base_terms[edge_idx] = base_delta
        inv_gamma_terms[edge_idx] = inv_child_gamma
        bandwidth_terms[edge_idx] = cache.bandwidth_hz
        invalid_terms[edge_idx] = base_delta >= 1e12

    for hop_count in hop_budgets:
        deltas = (base_terms + float(hop_count) * inv_gamma_terms) / bandwidth_terms
        deltas = np.maximum(deltas, 0.0)
        deltas[invalid_terms | ~np.isfinite(deltas)] = 1e12
        delta_by_hop[hop_count] = deltas.tolist()

    candidate_with_cost = []
    safety_paths = []
    loads = state_stats[0]
    relay_ids_for_max = _relay_ids(state_graph)
    max_load = (
        max(float(loads.get(node_id, 0.0)) for node_id in relay_ids_for_max)
        if relay_ids_for_max
        else 0.0
    )

    paths_per_variant = max(len(alpha_list) * len(hop_budgets), 1)
    noise_variants = max(1, int(np.ceil(num_policy_paths / paths_per_variant)))
    for alpha in alpha_list:
        if dual_mode == "active_set":
            relay_ids, dual_variants = _deterministic_dual_variants(
                state_graph,
                state_stats,
                alpha,
                lookahead_loads=lookahead_loads,
                lookahead_weight=lookahead_weight,
            )
            for dual in dual_variants:
                relay_to_dual = {
                    node_id: float(dual[idx]) for idx, node_id in enumerate(relay_ids)
                }
                for hop_count in hop_budgets:
                    edge_costs = []
                    for (parent_id, _), delta in zip(edge_list, delta_by_hop[hop_count]):
                        edge_costs.append(
                            _edge_reduced_cost(
                                parent_id,
                                delta,
                                relay_to_dual,
                                loads,
                                max_load,
                                dual_mode,
                                barrier_weight,
                            )
                        )
                    path, cost = _layered_exact_h_path(
                        edge_adjacency, edge_costs, user_id, hop_count
                    )
                    if path is not None and len(path) == len(set(path)):
                        candidate_with_cost.append((cost, path))
            continue

        for noise_idx in range(noise_variants):
            relay_ids, dual = _mixed_dual(
                state_graph,
                alpha=alpha,
                model=model,
                device=device,
                learned_mix=learned_mix,
                noise_scale=0.0 if noise_idx == 0 else noise_scale,
                state_stats=state_stats,
                lookahead_loads=lookahead_loads,
                lookahead_weight=lookahead_weight,
                dual_mode=dual_mode,
                feasible_graph=feasible_graph,
                static_context=static_context,
                max_hop=max_hop,
                dual_anchor=dual_anchor,
            )
            relay_to_dual = {node_id: float(dual[idx]) for idx, node_id in enumerate(relay_ids)}
            for hop_count in hop_budgets:
                edge_costs = []
                for (parent_id, _), delta in zip(edge_list, delta_by_hop[hop_count]):
                    edge_costs.append(
                        _edge_reduced_cost(
                            parent_id,
                            delta,
                            relay_to_dual,
                            loads,
                            max_load,
                            dual_mode,
                            barrier_weight,
                        )
                    )
                path, cost = _layered_exact_h_path(
                    edge_adjacency, edge_costs, user_id, hop_count
                )
                if path is not None and len(path) == len(set(path)):
                    candidate_with_cost.append((cost, path))

    if model is not None and getattr(model, "uses_future_features", False):
        # Guard against learned duals assigning near-zero mass to a relay whose
        # large edge delta would catastrophically collapse throughput.
        for alpha in alpha_list:
            relay_ids, smooth_dual = _mixed_dual(
                state_graph,
                alpha=alpha,
                model=None,
                device=device,
                learned_mix=0.0,
                noise_scale=0.0,
                state_stats=state_stats,
                lookahead_loads=lookahead_loads,
                lookahead_weight=lookahead_weight,
                dual_mode="smooth",
                feasible_graph=feasible_graph,
            )
            relay_to_dual = {
                node_id: float(smooth_dual[idx])
                for idx, node_id in enumerate(relay_ids)
            }
            for safety_mode in ("smooth", "barrier"):
                for hop_count in hop_budgets:
                    edge_costs = [
                        _edge_reduced_cost(
                            parent_id,
                            delta,
                            relay_to_dual,
                            loads,
                            max_load,
                            safety_mode,
                            barrier_weight,
                        )
                        for (parent_id, _), delta in zip(
                            edge_list, delta_by_hop[hop_count]
                        )
                    ]
                    path, _ = _layered_exact_h_path(
                        edge_adjacency, edge_costs, user_id, hop_count
                    )
                    if path is not None and len(path) == len(set(path)):
                        safety_paths.append(path)

    classic_paths = []
    if include_classic_paths:
        if static_context is not None:
            for pred in [static_context.hop_pred, static_context.distance_pred]:
                path = get_shortest_path(pred, user_id)
                if path and path[0] != -1:
                    classic_paths.append(path)
            try:
                _, pred = a_star(
                    feasible_graph, goal=user_id, metric="spectral_efficiency"
                )
                path = get_shortest_path(pred, user_id)
                if path and path[0] != -1:
                    classic_paths.append(path)
            except Exception:
                pass
        else:
            for metric in ["hop", "distance", "spectral_efficiency"]:
                try:
                    _, pred = a_star(feasible_graph, goal=user_id, metric=metric)
                    path = get_shortest_path(pred, user_id)
                    if path and path[0] != -1:
                        classic_paths.append(path)
                except Exception:
                    continue

    candidate_with_cost.sort(key=lambda item: item[0])
    deduped = []
    seen = set()
    for _, path in candidate_with_cost:
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
        if len(deduped) >= num_policy_paths:
            break
    for path in safety_paths + classic_paths:
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return valid_candidate_paths(deduped)


@torch.no_grad()
def get_solution_graph(
    graph: bs.IABRelayGraph,
    model_path: str | None = None,
    num_rounds: int = 1,
    num_trials: int = 1,
    verbose: bool = False,
    policy_device: str = "cpu",
    alpha_list=(20.0,),
    num_policy_paths: int = 16,
    noise_scale: float = 0.15,
    learned_mix: float = 0.25,
    max_hop: int = 12,
    hop_slack: int = 3,
    user_order: str = "farthest",
    include_classic_paths: bool = True,
    exact_commit: bool = True,
    dual_mode: str = "smooth",
    lookahead_weight: float = 1.0,
    barrier_weight: float = 1.0,
    catastrophe_threshold: float = 0.0,
    dual_anchor: str = "mix",
):
    model = None
    if model_path:
        model, _ = load_policy(model_path, device=policy_device)

    graph_list = []
    throughput_list = []
    static_context = _StaticRoutingContext(graph)
    uid_list = [user.get_id() for user in graph.users]
    sorted_id_list = _initial_user_order(graph, static_context, user_order)

    for _ in range(num_trials):
        result_graph = graph.copy()
        result_graph.reset()
        updated = True
        update_round = 0
        old_throughput = -1.0

        while updated and update_round < num_rounds:
            updated = False
            start = time.time()
            if update_round == 0:
                user_id_list = sorted_id_list
            else:
                random.shuffle(uid_list)
                user_id_list = uid_list

            for user_id in user_id_list:
                deleted_edges = utils.delete_user(result_graph, user_id)
                candidate_paths = _reduced_cost_candidate_paths(
                    graph,
                    result_graph,
                    user_id,
                    static_context=static_context,
                    model=model,
                    device=policy_device,
                    alpha_list=alpha_list,
                    num_policy_paths=num_policy_paths,
                    noise_scale=noise_scale,
                    learned_mix=learned_mix,
                    max_hop=max_hop,
                    hop_slack=hop_slack,
                    include_classic_paths=include_classic_paths,
                    dual_mode=dual_mode,
                    lookahead_weight=lookahead_weight,
                    barrier_weight=barrier_weight,
                    dual_anchor=dual_anchor,
                )
                if not candidate_paths:
                    for parent_id, child_id in deleted_edges:
                        result_graph.add_edge(parent_id, child_id)
                    result_graph.compute_hops_for_one_user(user_id)
                    continue

                if exact_commit:
                    _, added_edges = utils.get_best_candidate_graph(
                        result_graph, user_id, candidate_paths
                    )
                else:
                    added_edges = utils.get_aborescence_graph(
                        result_graph, candidate_paths[0]
                    )

                current_throughput = result_graph.compute_network_throughput()
                if update_round == 0 or current_throughput > old_throughput:
                    updated = True
                    old_throughput = current_throughput
                else:
                    utils.remove_added_edges(result_graph, user_id, added_edges)
                    for parent_id, child_id in deleted_edges:
                        result_graph.add_edge(parent_id, child_id)
                    result_graph.compute_hops_for_one_user(user_id)

            update_round += 1
            if verbose:
                print(
                    f"Round {update_round}: Throughput = "
                    f"{result_graph.compute_network_throughput()}, "
                    f"Time = {time.time() - start}"
                )

        graph_list.append(result_graph)
        throughput_list.append(old_throughput)

    best_idx = throughput_list.index(max(throughput_list))
    best_graph = graph_list[best_idx]
    best_throughput = float(throughput_list[best_idx])

    if (
        model is not None
        and getattr(model, "uses_future_features", False)
        and catastrophe_threshold > 0.0
        and best_throughput < catastrophe_threshold
    ):
        fallback_graph = get_solution_graph(
            graph,
            model_path=None,
            num_rounds=num_rounds,
            num_trials=num_trials,
            verbose=verbose,
            policy_device=policy_device,
            alpha_list=alpha_list,
            num_policy_paths=num_policy_paths,
            noise_scale=noise_scale,
            learned_mix=0.0,
            max_hop=max_hop,
            hop_slack=hop_slack,
            user_order=user_order,
            include_classic_paths=include_classic_paths,
            exact_commit=exact_commit,
            dual_mode="smooth",
            lookahead_weight=lookahead_weight,
            barrier_weight=barrier_weight,
            catastrophe_threshold=0.0,
        )
        fallback_throughput = float(fallback_graph.compute_network_throughput())
        if fallback_throughput > best_throughput:
            return fallback_graph

    return best_graph
