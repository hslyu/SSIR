import heapq
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


@dataclass
class RLEdgeCostConfig:
    hidden_dim: int = 128
    lr: float = 1e-3
    epochs: int = 20
    num_predecessors: int = 100
    policy_device: str = "cpu"
    seed: int = 0
    max_negatives: int = 96
    margin: float = 1.0
    policy_arch: str = "mlp"
    gnn_layers: int = 2


class EdgeCostNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor):
        return F.softplus(self.net(features).squeeze(-1)) + 1e-6


class GraphEdgeCostNet(nn.Module):
    uses_graph = True

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.node_input = nn.Linear(node_dim, hidden_dim)
        self.edge_input = nn.Linear(edge_dim, hidden_dim)
        self.message_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
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
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ):
        h = F.relu(self.node_input(node_features))
        e = F.relu(self.edge_input(edge_features))

        if edge_index.numel() > 0:
            src, dst = edge_index
            for msg_layer, update_layer in zip(self.message_layers, self.update_layers):
                messages = msg_layer(torch.cat([h[src], e], dim=-1))
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

        src, dst = edge_index
        edge_repr = torch.cat([h[src], h[dst], e], dim=-1)
        return F.softplus(self.output(edge_repr).squeeze(-1)) + 1e-6


def _bs_one_hot(node):
    values = [0.0, 0.0, 0.0, 0.0]
    if isinstance(node, bs.BaseStation):
        idx = bs._bs_type_to_id(node)
        if idx >= 0:
            values[idx] = 1.0
    return values


def _state_stats(state_graph: bs.IABRelayGraph):
    state_graph.compute_hops()
    load = {}
    pressure = {}
    child_count = {}
    throughput = {}
    for node_id, node in state_graph.nodes.items():
        child_count[node_id] = float(len(node.get_children()))
        if isinstance(node, bs.BaseStation):
            node_load = float(sum(user.hops for user in node.connected_user))
            load[node_id] = node_load
            tp = float(node.compute_throughput())
            throughput[node_id] = tp
            pressure[node_id] = 0.0 if np.isinf(tp) else 1.0 / (tp + 1e-9)
        else:
            load[node_id] = float(node.hops)
            throughput[node_id] = float("inf")
            pressure[node_id] = 0.0
    max_pressure = max(pressure.values()) if pressure else 0.0
    if max_pressure > 0:
        pressure = {node_id: val / max_pressure for node_id, val in pressure.items()}
    return load, pressure, child_count, throughput


def _edge_features(
    feasible_graph: bs.IABRelayGraph,
    state_graph: bs.IABRelayGraph,
    user_id: int,
):
    target = feasible_graph.nodes[user_id]
    load, pressure, child_count, throughput = _state_stats(state_graph)
    edge_list = list(feasible_graph.edges)
    rows = []

    for parent_id, child_id in edge_list:
        parent = feasible_graph.nodes[parent_id]
        child = feasible_graph.nodes[child_id]
        distance = parent.get_distance(child)
        parent_target_distance = parent.get_distance(target)
        child_target_distance = child.get_distance(target)

        cfg = parent.basestation_type.config if isinstance(parent, bs.BaseStation) else None
        pathloss = float(cfg.pathloss_exponent) if cfg is not None else 0.0
        bandwidth = float(cfg.bandwidth) if cfg is not None else 0.0
        max_link = float(cfg.maximum_link_distance) if cfg is not None else 1.0
        slack = (max_link - distance) / max(max_link, 1e-9)

        state_child = state_graph.nodes[child_id]
        state_parent = state_graph.nodes[parent_id]
        selected = float(child_id in state_graph.get_neighbors(parent_id))
        child_has_parent = float(state_child.has_parent())
        parent_has_parent = float(state_parent.has_parent())

        row = [
            float(parent_id == 0),
            float(isinstance(parent, bs.User)),
            *_bs_one_hot(parent),
            float(child_id == user_id),
            float(isinstance(child, bs.User)),
            *_bs_one_hot(child),
            np.log1p(distance),
            np.log1p(parent_target_distance),
            np.log1p(child_target_distance),
            np.log1p(max(load.get(parent_id, 0.0), 0.0)),
            np.log1p(max(load.get(child_id, 0.0), 0.0)),
            pressure.get(parent_id, 0.0),
            pressure.get(child_id, 0.0),
            np.log1p(child_count.get(parent_id, 0.0)),
            np.log1p(child_count.get(child_id, 0.0)),
            selected,
            child_has_parent,
            parent_has_parent,
            np.log1p(max(pathloss, 0.0)),
            np.log1p(max(bandwidth, 0.0)),
            slack,
        ]
        rows.append(row)

    return edge_list, torch.tensor(rows, dtype=torch.float)


def _graph_edge_features(
    feasible_graph: bs.IABRelayGraph,
    state_graph: bs.IABRelayGraph,
    user_id: int,
):
    edge_list, edge_features = _edge_features(feasible_graph, state_graph, user_id)
    target = feasible_graph.nodes[user_id]
    load, pressure, child_count, throughput = _state_stats(state_graph)
    node_ids = sorted(feasible_graph.nodes)
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

    rows = []
    for node_id in node_ids:
        node = feasible_graph.nodes[node_id]
        state_node = state_graph.nodes[node_id]
        tp = throughput.get(node_id, float("inf"))
        rows.append(
            [
                float(node_id == 0),
                float(node_id == user_id),
                float(isinstance(node, bs.User)),
                *_bs_one_hot(node),
                np.log1p(max(node.get_distance(target), 0.0)),
                np.log1p(max(load.get(node_id, 0.0), 0.0)),
                pressure.get(node_id, 0.0),
                np.log1p(max(child_count.get(node_id, 0.0), 0.0)),
                float(state_node.has_parent()),
                0.0 if np.isinf(tp) else np.log1p(max(tp, 0.0)),
            ]
        )

    edge_index = torch.tensor(
        [
            [node_to_idx[parent_id], node_to_idx[child_id]]
            for parent_id, child_id in edge_list
        ],
        dtype=torch.long,
    ).t().contiguous()
    return (
        edge_list,
        edge_features,
        torch.tensor(rows, dtype=torch.float),
        edge_index,
    )


def _make_model(config: RLEdgeCostConfig, feature_dim: int, sample=None):
    if config.policy_arch == "gnn":
        if sample is None:
            raise ValueError("A graph sample is required to build the GNN policy.")
        return GraphEdgeCostNet(
            sample["node_features"].shape[1],
            feature_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.gnn_layers,
        )
    if config.policy_arch != "mlp":
        raise ValueError("policy_arch must be one of: mlp, gnn")
    return EdgeCostNet(feature_dim, config.hidden_dim)


def _path_edge_indices(path, edge_to_idx):
    indices = []
    for parent, child in zip(path[:-1], path[1:]):
        idx = edge_to_idx.get((parent, child))
        if idx is None:
            return None
        indices.append(idx)
    return indices


def _path_cost(edge_costs, edge_indices):
    return edge_costs[torch.tensor(edge_indices, device=edge_costs.device)].sum()


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


def _build_training_samples(graphs, config: RLEdgeCostConfig):
    samples = []
    for feasible_graph in graphs:
        user_order, all_shortest_paths = prepare_candidates(
            feasible_graph, config.num_predecessors
        )
        state_graph = feasible_graph.copy()
        state_graph.reset()

        for user_id in user_order:
            path_list = valid_candidate_paths(all_shortest_paths[user_id])
            if len(path_list) < 2:
                continue

            utils.delete_user(state_graph, user_id)
            teacher_path, _ = _exact_best_path(state_graph, user_id, path_list)

            if config.policy_arch == "gnn":
                edge_list, features, node_features, edge_index = _graph_edge_features(
                    feasible_graph, state_graph, user_id
                )
            else:
                edge_list, features = _edge_features(feasible_graph, state_graph, user_id)
                node_features = None
                edge_index = None
            edge_to_idx = {edge: idx for idx, edge in enumerate(edge_list)}
            candidate_indices = []
            teacher_idx = None
            for idx, path in enumerate(path_list):
                edge_indices = _path_edge_indices(path, edge_to_idx)
                if edge_indices is None:
                    continue
                if path == teacher_path:
                    teacher_idx = len(candidate_indices)
                candidate_indices.append(edge_indices)

            if teacher_idx is not None and len(candidate_indices) > 1:
                samples.append(
                    {
                        "features": features,
                        "node_features": node_features,
                        "edge_index": edge_index,
                        "candidate_indices": candidate_indices,
                        "teacher_idx": teacher_idx,
                    }
                )

            utils.get_aborescence_graph(state_graph, teacher_path)
    return samples


def _model_edge_costs(model, sample, device: str):
    features = sample["features"].to(device)
    if getattr(model, "uses_graph", False):
        return model(
            sample["node_features"].to(device),
            sample["edge_index"].to(device),
            features,
        )
    return model(features)


def train_policy(
    graphs,
    save_path: str,
    config: RLEdgeCostConfig | None = None,
    log_every_epoch: bool = True,
):
    config = config or RLEdgeCostConfig()
    set_seed(config.seed)
    samples = _build_training_samples(graphs, config)
    if not samples:
        raise ValueError("No training samples were generated.")

    feature_dim = samples[0]["features"].shape[1]
    model = _make_model(config, feature_dim, sample=samples[0]).to(config.policy_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    rng = random.Random(config.seed)

    history = []
    for epoch in range(config.epochs):
        rng.shuffle(samples)
        losses = []
        top1 = []
        for sample in samples:
            edge_costs = _model_edge_costs(model, sample, config.policy_device)
            candidate_indices = sample["candidate_indices"]
            teacher_idx = sample["teacher_idx"]

            if len(candidate_indices) > config.max_negatives + 1:
                keep = {teacher_idx}
                while len(keep) < config.max_negatives + 1:
                    keep.add(rng.randrange(len(candidate_indices)))
                order = sorted(keep)
                teacher_idx = order.index(sample["teacher_idx"])
                candidate_indices = [candidate_indices[i] for i in order]

            path_costs = torch.stack(
                [_path_cost(edge_costs, indices) for indices in candidate_indices]
            )
            logits = -path_costs
            target = torch.tensor([teacher_idx], device=config.policy_device)
            ce_loss = F.cross_entropy(logits.unsqueeze(0), target)

            teacher_cost = path_costs[teacher_idx]
            negative_mask = torch.ones(
                len(candidate_indices), dtype=torch.bool, device=config.policy_device
            )
            negative_mask[teacher_idx] = False
            margin_loss = F.relu(
                teacher_cost - path_costs[negative_mask] + config.margin
            ).mean()
            loss = ce_loss + 0.25 * margin_loss

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
            "policy_type": "rl_edge_cost_policy",
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "feature_dim": feature_dim,
            "node_feature_dim": (
                None
                if samples[0].get("node_features") is None
                else samples[0]["node_features"].shape[1]
            ),
        },
        save_path,
    )
    with open(f"{save_path}.history.json", "w") as f:
        json.dump(history, f, indent=2)
    return model, history


def load_policy(model_path: str, device: str = "cpu"):
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", {})
    if config.get("policy_arch", "mlp") == "gnn":
        model = GraphEdgeCostNet(
            checkpoint["node_feature_dim"],
            checkpoint["feature_dim"],
            hidden_dim=config.get("hidden_dim", 128),
            num_layers=config.get("gnn_layers", 2),
        ).to(device)
    else:
        model = EdgeCostNet(
            checkpoint["feature_dim"],
            hidden_dim=config.get("hidden_dim", 128),
        ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config


@torch.no_grad()
def _shortest_path_from_edge_costs(edge_list, costs, user_id: int):
    adjacency = {}
    for (parent, child), cost in zip(edge_list, costs):
        adjacency.setdefault(parent, []).append((child, float(cost)))

    pq = [(0.0, 0)]
    dist = {0: 0.0}
    pred = {0: -1}
    while pq:
        current_cost, node_id = heapq.heappop(pq)
        if node_id == user_id:
            break
        if current_cost > dist.get(node_id, float("inf")):
            continue
        for neighbor, edge_cost in adjacency.get(node_id, []):
            new_cost = current_cost + edge_cost
            if new_cost < dist.get(neighbor, float("inf")):
                dist[neighbor] = new_cost
                pred[neighbor] = node_id
                heapq.heappush(pq, (new_cost, neighbor))

    if user_id not in pred:
        return [-1, user_id]
    return get_shortest_path(pred, user_id)


@torch.no_grad()
def _learned_candidate_paths(
    model,
    feasible_graph: bs.IABRelayGraph,
    state_graph: bs.IABRelayGraph,
    user_id: int,
    device: str,
    num_policy_paths: int,
    noise_scale: float,
    include_classic_paths: bool,
):
    if getattr(model, "uses_graph", False):
        edge_list, features, node_features, edge_index = _graph_edge_features(
            feasible_graph, state_graph, user_id
        )
        base_costs = model(
            node_features.to(device),
            edge_index.to(device),
            features.to(device),
        ).detach().cpu().numpy()
    else:
        edge_list, features = _edge_features(feasible_graph, state_graph, user_id)
        base_costs = model(features.to(device)).detach().cpu().numpy()
    paths = []

    for idx in range(max(num_policy_paths, 1)):
        if idx == 0 or noise_scale <= 0:
            costs = base_costs
        else:
            noise = np.random.normal(0.0, noise_scale, size=base_costs.shape)
            costs = base_costs * np.exp(noise)
        paths.append(_shortest_path_from_edge_costs(edge_list, costs, user_id))

    if include_classic_paths:
        for metric in ["hop", "distance", "spectral_efficiency"]:
            try:
                _, pred = a_star(feasible_graph, goal=user_id, metric=metric)
                paths.append(get_shortest_path(pred, user_id))
            except Exception:
                continue

    return valid_candidate_paths(paths)


@torch.no_grad()
def get_solution_graph(
    graph: bs.IABRelayGraph,
    model_path: str,
    num_rounds: int = 1,
    num_trials: int = 1,
    verbose: bool = False,
    policy_device: str = "cpu",
    num_policy_paths: int = 16,
    noise_scale: float = 0.35,
    include_classic_paths: bool = True,
    exact_commit: bool = True,
):
    model, _ = load_policy(model_path, device=policy_device)

    graph_list = []
    throughput_list = []
    _, pred = a_star(graph, metric="hop")
    uid_list = [user.get_id() for user in graph.users]
    hop_list = [len(get_shortest_path(pred, user.get_id())) for user in graph.users]
    sorted_id_list = sorted(
        uid_list,
        key=lambda x: hop_list[uid_list.index(x)],
        reverse=True,
    )

    for _ in range(num_trials):
        result_graph = graph.copy()
        result_graph.reset()
        updated = True
        update_round = 0
        old_throughput = -1.0

        while updated and update_round < num_rounds:
            updated = False
            s = time.time()
            if update_round == 0:
                user_id_list = sorted_id_list
            else:
                random.shuffle(uid_list)
                user_id_list = uid_list

            for user_id in user_id_list:
                deleted_edges = utils.delete_user(result_graph, user_id)
                candidate_paths = _learned_candidate_paths(
                    model,
                    graph,
                    result_graph,
                    user_id,
                    policy_device,
                    num_policy_paths=num_policy_paths,
                    noise_scale=noise_scale,
                    include_classic_paths=include_classic_paths,
                )
                if not candidate_paths:
                    for p, c in deleted_edges:
                        result_graph.add_edge(p, c)
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
                    for p, c in deleted_edges:
                        result_graph.add_edge(p, c)
                    result_graph.compute_hops_for_one_user(user_id)

            update_round += 1
            if verbose:
                print(
                    f"Round {update_round}: Throughput = "
                    f"{result_graph.compute_network_throughput()}, "
                    f"Time = {time.time() - s}"
                )

        graph_list.append(result_graph)
        throughput_list.append(old_throughput)

    best_idx = throughput_list.index(max(throughput_list))
    return graph_list[best_idx]
