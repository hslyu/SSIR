from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv


class GraphEmbeddingModule(nn.Module):
    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        heads: int = 2,
    ):
        super().__init__()
        self.node_proj = nn.Linear(node_input_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_input_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [
                GATConv(
                    hidden_dim,
                    hidden_dim,
                    heads=heads,
                    concat=False,
                    edge_dim=hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_emb = self.node_proj(x)
        edge_emb = self.edge_proj(edge_attr)
        for conv in self.convs:
            residual = node_emb
            node_emb = conv(node_emb, edge_index, edge_emb)
            node_emb = F.gelu(node_emb + residual)
        return node_emb, edge_emb


class CandidateRoutePooling(nn.Module):
    def __init__(self, hidden_dim: int, node_aux_dim: int, edge_aux_dim: int):
        super().__init__()
        self.node_aux_proj = nn.Linear(node_aux_dim, hidden_dim)
        self.edge_aux_proj = nn.Linear(edge_aux_dim, hidden_dim)
        self.node_gate = nn.Linear(hidden_dim, 1)
        self.edge_gate = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        node_emb: torch.Tensor,
        edge_emb: torch.Tensor,
        candidate_node_mask: torch.Tensor,
        candidate_edge_mask: torch.Tensor,
        candidate_node_aux: torch.Tensor,
        candidate_edge_aux: torch.Tensor,
    ) -> torch.Tensor:
        candidate_node_emb = node_emb.unsqueeze(0) + self.node_aux_proj(candidate_node_aux)
        candidate_edge_emb = edge_emb.unsqueeze(0) + self.edge_aux_proj(candidate_edge_aux)

        node_gate = torch.sigmoid(self.node_gate(candidate_node_emb))
        edge_gate = torch.sigmoid(self.edge_gate(candidate_edge_emb))

        node_mask = candidate_node_mask.unsqueeze(-1)
        edge_mask = candidate_edge_mask.unsqueeze(-1)

        masked_node_emb = candidate_node_emb * node_gate * node_mask
        masked_edge_emb = candidate_edge_emb * edge_gate * edge_mask

        node_count = node_mask.sum(dim=1).clamp_min(1.0)
        edge_count = edge_mask.sum(dim=1).clamp_min(1.0)

        pooled_node_mean = masked_node_emb.sum(dim=1) / node_count
        pooled_edge_mean = masked_edge_emb.sum(dim=1) / edge_count
        pooled_node_max = masked_node_emb.max(dim=1).values
        pooled_edge_max = masked_edge_emb.max(dim=1).values

        return torch.cat(
            [
                pooled_node_mean,
                pooled_edge_mean,
                pooled_node_max,
                pooled_edge_max,
                node_count,
                edge_count,
            ],
            dim=1,
        )


class CandidateThroughputNetwork(nn.Module):
    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        node_aux_dim: int = 4,
        edge_aux_dim: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.encoder = GraphEmbeddingModule(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
        )
        self.pool = CandidateRoutePooling(
            hidden_dim=hidden_dim,
            node_aux_dim=node_aux_dim,
            edge_aux_dim=edge_aux_dim,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        candidate_node_mask: torch.Tensor,
        candidate_edge_mask: torch.Tensor,
        candidate_node_aux: torch.Tensor,
        candidate_edge_aux: torch.Tensor,
    ) -> torch.Tensor:
        node_emb, edge_emb = self.encoder(x, edge_index, edge_attr)
        candidate_repr = self.pool(
            node_emb=node_emb,
            edge_emb=edge_emb,
            candidate_node_mask=candidate_node_mask,
            candidate_edge_mask=candidate_edge_mask,
            candidate_node_aux=candidate_node_aux,
            candidate_edge_aux=candidate_edge_aux,
        )
        return self.head(candidate_repr)
