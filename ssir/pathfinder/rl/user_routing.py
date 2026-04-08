from __future__ import annotations

import torch

import ssir.basestations as bs
from ssir.pathfinder.rl.candidate_data import UserRouteBatch, build_user_route_batch
from ssir.pathfinder.rl.candidate_network import CandidateThroughputNetwork
from ssir.pathfinder.rl.trajectory import (
    UserRouteCandidate,
    UserRoutingProblem,
    apply_candidate_route,
)


@torch.no_grad()
def select_best_route_for_user(
    model: CandidateThroughputNetwork,
    master_graph: bs.IABRelayGraph,
    current_graph: bs.IABRelayGraph,
    user_id: int,
    route_candidates: list[UserRouteCandidate],
    device: str | torch.device = "cpu",
) -> tuple[bs.IABRelayGraph, UserRouteBatch, torch.Tensor, UserRouteCandidate]:
    batch = build_user_route_batch(master_graph, current_graph, user_id, route_candidates)
    graph_data = batch.graph_data
    scores = model(
        x=graph_data.x.to(device),
        edge_index=graph_data.edge_index.to(device),
        edge_attr=graph_data.edge_attr.to(device),
        candidate_node_mask=batch.candidate_node_mask.to(device),
        candidate_edge_mask=batch.candidate_edge_mask.to(device),
        candidate_node_aux=batch.candidate_node_aux.to(device),
        candidate_edge_aux=batch.candidate_edge_aux.to(device),
    ).squeeze(-1)
    best_index = int(torch.argmax(scores).item())
    best_candidate = route_candidates[best_index]
    best_graph, _ = apply_candidate_route(current_graph, best_candidate)
    return best_graph, batch, scores.cpu(), best_candidate


def rollout_routing_problem(
    model: CandidateThroughputNetwork,
    master_graph: bs.IABRelayGraph,
    problem: UserRoutingProblem,
    device: str | torch.device = "cpu",
) -> bs.IABRelayGraph:
    current_graph = master_graph.copy()
    current_graph.reset()
    for user_id in problem.user_order:
        route_candidates = problem.candidates_by_user[user_id]
        current_graph, _, _, _ = select_best_route_for_user(
            model=model,
            master_graph=master_graph,
            current_graph=current_graph,
            user_id=user_id,
            route_candidates=route_candidates,
            device=device,
        )
    return current_graph
