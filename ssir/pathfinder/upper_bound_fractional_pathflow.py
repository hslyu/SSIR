import math
import time
from dataclasses import dataclass

import numpy as np

from ssir import basestations as bs
from ssir.pathfinder.rl_reduced_cost_policy import (
    _relay_ids,
    _snr_with_distance,
    _transmission_power_density_for_dmax,
)


@dataclass
class ArborescenceRestrictedBoundConfig:
    hop_limit: int | None = 16
    gamma_floor: float = 1e-30
    time_limit_seconds: float | None = 60.0
    mip_rel_gap: float = 1e-4
    load_scale_quantile: float = 50.0
    max_scaled_load_coeff: float = 1e6
    max_non_anchor_paths: int | None = None
    max_non_anchor_fraction: float | None = None


def _load_counted_relays(graph: bs.IABRelayGraph):
    return _relay_ids(graph)


def _optimistic_gamma_bar(graph: bs.IABRelayGraph, parent_id: int, child_id: int):
    parent = graph.nodes[parent_id]
    if not isinstance(parent, bs.BaseStation):
        return 0.0
    child = graph.nodes[child_id]
    distance = parent.get_distance(child)
    tx_power_density = _transmission_power_density_for_dmax(parent, distance)
    snr = _snr_with_distance(parent, child, tx_power_density, distance)
    gamma = float(np.log2(1.0 + snr))
    return gamma if np.isfinite(gamma) and gamma > 0.0 else 0.0


class _BoundProblem:
    def __init__(
        self,
        graph: bs.IABRelayGraph,
        config: ArborescenceRestrictedBoundConfig,
    ):
        self.graph = graph
        self.config = config
        self.node_ids = sorted(graph.nodes)
        self.node_to_idx = {node_id: idx for idx, node_id in enumerate(self.node_ids)}
        self.source_idx = self.node_to_idx.get(0)
        if self.source_idx is None:
            raise ValueError("Source node 0 is required.")

        self.user_ids = [user.get_id() for user in graph.users]
        self.user_indices = [
            self.node_to_idx[user_id]
            for user_id in self.user_ids
            if user_id in self.node_to_idx
        ]
        self.relay_ids = _load_counted_relays(graph)
        self.relay_to_idx = {
            node_id: idx for idx, node_id in enumerate(self.relay_ids)
        }
        if not self.relay_ids:
            raise ValueError("No load-counted relay nodes found.")

        self.hop_limit = (
            len(self.node_ids) - 1
            if config.hop_limit is None or config.hop_limit <= 0
            else int(config.hop_limit)
        )
        self._build_edges()

    def _build_edges(self):
        src = []
        dst = []
        relay_idx = []
        inv_load_coeff = []
        gamma_bar = {}

        for parent_id, child_id in self.graph.edges:
            if parent_id not in self.node_to_idx or child_id not in self.node_to_idx:
                continue
            src.append(self.node_to_idx[parent_id])
            dst.append(self.node_to_idx[child_id])
            r_idx = self.relay_to_idx.get(parent_id, -1)
            relay_idx.append(r_idx)

            if r_idx < 0:
                inv_load_coeff.append(0.0)
                continue

            gamma = _optimistic_gamma_bar(self.graph, parent_id, child_id)
            gamma = max(gamma, self.config.gamma_floor)
            gamma_bar[(parent_id, child_id)] = gamma
            parent = self.graph.nodes[parent_id]
            bandwidth_hz = parent.basestation_type.config.bandwidth * 1e6
            inv_load_coeff.append(1.0 / (bandwidth_hz * gamma))

        self.src = np.array(src, dtype=np.int64)
        self.dst = np.array(dst, dtype=np.int64)
        self.edge_relay_idx = np.array(relay_idx, dtype=np.int64)
        self.inv_load_coeff = np.array(inv_load_coeff, dtype=np.float64)
        self.gamma_bar = gamma_bar
        self.num_nodes = len(self.node_ids)
        self.num_edges = len(self.src)
        if self.num_edges == 0:
            raise ValueError("No feasible edges found.")

        self.out_edges = [[] for _ in range(self.num_nodes)]
        for edge_idx, src_idx in enumerate(self.src):
            self.out_edges[int(src_idx)].append(edge_idx)

def compute_gap(eta_solution: float, eta_upper_bound: float):
    if eta_solution <= 0.0 or not np.isfinite(eta_solution):
        return float("inf")
    return eta_upper_bound / eta_solution - 1.0


def _dedup_valid_candidate_paths(problem: _BoundProblem, candidate_paths_by_user):
    edge_to_idx = {
        (
            int(problem.node_ids[int(src_idx)]),
            int(problem.node_ids[int(dst_idx)]),
        ): edge_idx
        for edge_idx, (src_idx, dst_idx) in enumerate(zip(problem.src, problem.dst))
    }
    max_hop = problem.hop_limit
    valid_by_user = {}

    for user_id in problem.user_ids:
        seen = set()
        valid_paths = []
        for path in candidate_paths_by_user.get(user_id, []):
            if not path:
                continue
            path = [int(node_id) for node_id in path]
            key = tuple(path)
            if key in seen:
                continue
            seen.add(key)
            if not path or path[0] != 0 or path[-1] != user_id:
                continue
            if len(path) != len(set(path)):
                continue
            if len(path) - 1 > max_hop:
                continue
            if all(
                (parent, child) in edge_to_idx
                for parent, child in zip(path[:-1], path[1:])
            ):
                valid_paths.append(path)
        valid_by_user[user_id] = valid_paths
    return valid_by_user, edge_to_idx


def compute_arborescence_restricted_upper_bound(
    graph: bs.IABRelayGraph,
    candidate_paths_by_user,
    config: ArborescenceRestrictedBoundConfig | None = None,
):
    """Solve a tree-aware restricted candidate bound.

    This is a diagnostic bound over the provided candidate paths, not a certified
    global upper bound. Binary edge variables enforce the single-parent
    arborescence condition on the selected candidate path family.
    """
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError as exc:
        raise RuntimeError("scipy.optimize.milp is required for arborescence bound.") from exc

    config = config or ArborescenceRestrictedBoundConfig()
    start_time = time.perf_counter()
    problem = _BoundProblem(graph, config)
    valid_by_user, edge_to_idx = _dedup_valid_candidate_paths(
        problem, candidate_paths_by_user
    )

    missing_users = [
        user_id for user_id, paths in valid_by_user.items() if not paths
    ]
    if missing_users:
        elapsed = time.perf_counter() - start_time
        return {
            "eta_upper_bound": float("inf"),
            "load_lower_bound": 0.0,
            "restricted_load": float("inf"),
            "bound_type": "arborescence_restricted_candidate",
            "status": "missing_candidates",
            "success": False,
            "missing_users": missing_users,
            "hop_limit": problem.hop_limit,
            "num_edges": problem.num_edges,
            "num_nodes": problem.num_nodes,
            "num_users": len(problem.user_indices),
            "elapsed_seconds": elapsed,
            "candidate_stats": {
                "num_candidate_paths": sum(len(paths) for paths in valid_by_user.values()),
                "min_paths_per_user": 0,
                "max_paths_per_user": max(
                    [len(paths) for paths in valid_by_user.values()] or [0]
                ),
            },
        }

    path_records = []
    user_to_path_indices = {}
    used_edge_indices = set()
    for user_id, paths in valid_by_user.items():
        indices = []
        for path in paths:
            edge_indices = [
                edge_to_idx[(parent, child)]
                for parent, child in zip(path[:-1], path[1:])
            ]
            path_idx = len(path_records)
            indices.append(path_idx)
            used_edge_indices.update(edge_indices)
            path_records.append(
                {
                    "user_id": user_id,
                    "path": path,
                    "edge_indices": edge_indices,
                    "hop_count": len(path) - 1,
                }
            )
        user_to_path_indices[user_id] = indices

    path_load_coeffs = []
    for record in path_records:
        coeffs = np.zeros(len(problem.relay_ids), dtype=np.float64)
        for edge_idx in record["edge_indices"]:
            relay_idx = int(problem.edge_relay_idx[edge_idx])
            if relay_idx >= 0:
                coeffs[relay_idx] += (
                    float(record["hop_count"])
                    * float(problem.inv_load_coeff[edge_idx])
                )
        path_load_coeffs.append(coeffs)

    anchor_loads = np.zeros(len(problem.relay_ids), dtype=np.float64)
    for indices in user_to_path_indices.values():
        if indices:
            anchor_loads += path_load_coeffs[indices[0]]
    anchor_load = float(np.max(anchor_loads)) if anchor_loads.size else 0.0
    anchor_eta_upper_bound = (
        float("inf") if anchor_load <= 0.0 else 1.0 / anchor_load
    )

    finite_positive_coeffs = [
        float(coeff)
        for coeffs in path_load_coeffs
        for coeff in coeffs
        if np.isfinite(coeff) and coeff > 0.0
    ]
    if finite_positive_coeffs:
        quantile = min(max(float(config.load_scale_quantile), 0.0), 100.0)
        load_scale = float(np.percentile(finite_positive_coeffs, quantile))
    else:
        load_scale = 1.0
    if load_scale <= 0.0 or not np.isfinite(load_scale):
        load_scale = 1.0
    max_scaled_load_coeff = float(config.max_scaled_load_coeff)
    if max_scaled_load_coeff <= 0.0 or not np.isfinite(max_scaled_load_coeff):
        max_scaled_load_coeff = float("inf")

    used_edge_indices = sorted(used_edge_indices)
    edge_idx_to_var = {
        edge_idx: idx for idx, edge_idx in enumerate(used_edge_indices)
    }
    num_x = len(path_records)
    num_y = len(used_edge_indices)
    t_idx = num_x + num_y
    num_vars = t_idx + 1
    path_counts = [len(paths) for paths in valid_by_user.values()]

    def _solve_milp(
        current_load_scale: float,
        current_max_scaled_load_coeff: float,
        solve_mode: str,
        fallback_reason: str | None = None,
    ):
        current_load_scale = float(current_load_scale)
        if current_load_scale <= 0.0 or not np.isfinite(current_load_scale):
            current_load_scale = 1.0
        current_max_scaled_load_coeff = float(current_max_scaled_load_coeff)
        if (
            current_max_scaled_load_coeff <= 0.0
            or not np.isfinite(current_max_scaled_load_coeff)
        ):
            current_max_scaled_load_coeff = float("inf")

        rows = []
        lower = []
        upper = []

        # Each user chooses one full path. x is continuous, but the binary parent
        # constraints below make distinct paths to the same user mutually exclusive.
        for user_id in problem.user_ids:
            row_idx = len(rows)
            rows.append([])
            lower.append(1.0)
            upper.append(1.0)
            for path_idx in user_to_path_indices[user_id]:
                rows[row_idx].append((path_idx, 1.0))

        effective_switch_budget = None
        switch_budget = config.max_non_anchor_paths
        if switch_budget is None and config.max_non_anchor_fraction is not None:
            switch_budget = math.floor(
                float(config.max_non_anchor_fraction) * len(problem.user_ids)
            )
        if switch_budget is not None:
            switch_budget = max(0, int(switch_budget))
            effective_switch_budget = switch_budget
            row_idx = len(rows)
            rows.append([])
            lower.append(-np.inf)
            upper.append(float(switch_budget))
            for indices in user_to_path_indices.values():
                for path_idx in indices[1:]:
                    rows[row_idx].append((path_idx, 1.0))

        # Relay load upper envelope: sum path loads <= t.
        clipped_load_coefficients = 0
        max_scaled_load_coeff_seen = 0.0
        for relay_idx in range(len(problem.relay_ids)):
            row_idx = len(rows)
            rows.append([(t_idx, -1.0)])
            lower.append(-np.inf)
            upper.append(0.0)
            for path_idx, coeffs in enumerate(path_load_coeffs):
                coeff = float(coeffs[relay_idx])
                if coeff:
                    if not np.isfinite(coeff):
                        coeff = (
                            current_load_scale
                            * current_max_scaled_load_coeff
                        )
                    scaled_coeff = coeff / current_load_scale
                    if scaled_coeff > current_max_scaled_load_coeff:
                        scaled_coeff = current_max_scaled_load_coeff
                        clipped_load_coefficients += 1
                    max_scaled_load_coeff_seen = max(
                        max_scaled_load_coeff_seen, scaled_coeff
                    )
                    rows[row_idx].append((path_idx, scaled_coeff))

        # Link active path variables to selected tree edges.
        for path_idx, record in enumerate(path_records):
            for edge_idx in record["edge_indices"]:
                row_idx = len(rows)
                rows.append(
                    [
                        (path_idx, 1.0),
                        (num_x + edge_idx_to_var[edge_idx], -1.0),
                    ]
                )
                lower.append(-np.inf)
                upper.append(0.0)

        # Arborescence single-parent condition: each non-source node has at most one
        # selected incoming edge across all chosen candidate paths.
        incoming_by_child = {}
        for edge_idx in used_edge_indices:
            child_id = int(problem.node_ids[int(problem.dst[edge_idx])])
            if child_id == 0:
                continue
            incoming_by_child.setdefault(child_id, []).append(edge_idx)
        for edge_indices in incoming_by_child.values():
            row_idx = len(rows)
            rows.append(
                [
                    (num_x + edge_idx_to_var[edge_idx], 1.0)
                    for edge_idx in edge_indices
                ]
            )
            lower.append(-np.inf)
            upper.append(1.0)

        matrix = lil_matrix((len(rows), num_vars), dtype=np.float64)
        for row_idx, entries in enumerate(rows):
            for col_idx, value in entries:
                matrix[row_idx, col_idx] = value

        objective = np.zeros(num_vars, dtype=np.float64)
        objective[t_idx] = 1.0
        lb = np.zeros(num_vars, dtype=np.float64)
        ub = np.ones(num_vars, dtype=np.float64)
        ub[t_idx] = np.inf
        integrality = np.zeros(num_vars, dtype=np.int8)
        integrality[num_x : num_x + num_y] = 1

        options = {"mip_rel_gap": config.mip_rel_gap}
        if config.time_limit_seconds is not None and config.time_limit_seconds > 0:
            options["time_limit"] = float(config.time_limit_seconds)

        result = milp(
            c=objective,
            integrality=integrality,
            bounds=Bounds(lb, ub),
            constraints=LinearConstraint(
                matrix.tocsr(), np.array(lower), np.array(upper)
            ),
            options=options,
        )

        elapsed = time.perf_counter() - start_time
        success = bool(result.success and np.isfinite(result.x[t_idx]))
        scaled_load = float(result.x[t_idx]) if success else float("inf")
        restricted_load = (
            scaled_load * current_load_scale if success else float("inf")
        )
        eta_upper_bound = (
            float("inf")
            if not success or restricted_load <= 0.0
            else 1.0 / restricted_load
        )
        x_values = result.x[:num_x] if result.x is not None else np.zeros(num_x)
        active_paths = int(np.sum(x_values > 1e-7))
        split_users = 0
        switched_users = 0
        non_anchor_mass = 0.0
        for indices in user_to_path_indices.values():
            if sum(float(x_values[idx]) > 1e-7 for idx in indices) > 1:
                split_users += 1
            active_non_anchor = [
                float(x_values[idx])
                for idx in indices[1:]
                if float(x_values[idx]) > 1e-7
            ]
            if active_non_anchor:
                switched_users += 1
                non_anchor_mass += sum(active_non_anchor)
        mip_gap = getattr(result, "mip_gap", None)
        max_scaled_stat = (
            None
            if not np.isfinite(current_max_scaled_load_coeff)
            else float(current_max_scaled_load_coeff)
        )

        return {
            "eta_upper_bound": eta_upper_bound,
            "load_lower_bound": restricted_load,
            "restricted_load": restricted_load,
            "bound_type": "arborescence_restricted_candidate",
            "status": int(result.status),
            "message": result.message,
            "success": success,
            "objective": None if result.fun is None else float(result.fun),
            "solve_mode": solve_mode,
            "fallback_reason": fallback_reason,
            "load_scale": float(current_load_scale),
            "scaled_restricted_load": scaled_load,
            "anchor_load": anchor_load,
            "anchor_eta_upper_bound": anchor_eta_upper_bound,
            "mip_gap": None if mip_gap is None else float(mip_gap),
            "hop_limit": problem.hop_limit,
            "num_edges": problem.num_edges,
            "num_nodes": problem.num_nodes,
            "num_users": len(problem.user_indices),
            "elapsed_seconds": elapsed,
            "candidate_stats": {
                "num_candidate_paths": num_x,
                "num_candidate_edges": num_y,
                "min_paths_per_user": int(min(path_counts)),
                "max_paths_per_user": int(max(path_counts)),
                "mean_paths_per_user": float(np.mean(path_counts)),
                "active_paths": active_paths,
                "split_users": split_users,
                "switched_users": switched_users,
                "non_anchor_mass": float(non_anchor_mass),
                "max_non_anchor_paths": (
                    None
                    if config.max_non_anchor_paths is None
                    else int(config.max_non_anchor_paths)
                ),
                "max_non_anchor_fraction": config.max_non_anchor_fraction,
                "effective_max_non_anchor_paths": effective_switch_budget,
                "anchor_load": anchor_load,
                "anchor_eta_upper_bound": anchor_eta_upper_bound,
                "load_scale": float(current_load_scale),
                "load_scale_quantile": float(config.load_scale_quantile),
                "max_scaled_load_coeff": max_scaled_stat,
                "max_scaled_load_coeff_seen": float(max_scaled_load_coeff_seen),
                "clipped_load_coefficients": clipped_load_coefficients,
            },
        }

    exact = _solve_milp(1.0, float("inf"), "exact")
    if exact["success"] and np.isfinite(exact["restricted_load"]):
        return exact

    return _solve_milp(
        load_scale,
        max_scaled_load_coeff,
        "scaled_capped_fallback",
        fallback_reason=exact.get("message"),
    )
