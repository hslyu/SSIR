"""
Neural network model for throughput prediction.

Adapts CandidateThroughputNetwork from rl module for our specific feature dimensions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv


class GraphEmbeddingModule(nn.Module):
    """
    Graph embedding via Graph Attention Networks.

    Encodes node and edge features into learned embeddings.
    """

    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize graph embedding module.

        Args:
            node_input_dim: Dimension of input node features
            edge_input_dim: Dimension of input edge features
            hidden_dim: Hidden dimension for embeddings
            num_layers: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.node_proj = nn.Linear(node_input_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_input_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATConv(
                    hidden_dim,
                    hidden_dim,
                    heads=heads,
                    concat=False,
                    edge_dim=hidden_dim,
                    dropout=dropout,
                )
            )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, node_input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_input_dim]

        Returns:
            (node_emb, edge_emb): Node and edge embeddings
        """
        node_emb = self.node_proj(x)
        edge_emb = self.edge_proj(edge_attr)

        for conv in self.convs:
            residual = node_emb
            node_emb = conv(node_emb, edge_index, edge_emb)
            node_emb = F.gelu(node_emb + residual)
            node_emb = self.dropout(node_emb)

        return node_emb, edge_emb


class CandidateRoutePooling(nn.Module):
    """
    Pool graph embeddings along candidate routes.

    Uses masks and gating to aggregate features relevant to each candidate.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_route_features: int = 2,
    ):
        """
        Initialize candidate pooling module.

        Args:
            hidden_dim: Dimension of embeddings
            num_route_features: Number of auxiliary route features (route length + global)
        """
        super().__init__()

        self.node_gate = nn.Linear(hidden_dim, 1)
        self.edge_gate = nn.Linear(hidden_dim, 1)
        self.num_route_features = num_route_features

    def forward(
        self,
        node_emb: torch.Tensor,
        edge_emb: torch.Tensor,
        candidate_node_mask: torch.Tensor,
        candidate_edge_mask: torch.Tensor,
        candidate_node_aux: torch.Tensor,
        route_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool embeddings along candidate routes.

        Args:
            node_emb: Node embeddings [num_nodes, hidden_dim]
            edge_emb: Edge embeddings [num_edges, hidden_dim]
            candidate_node_mask: Node mask [num_candidates, num_nodes]
            candidate_edge_mask: Edge mask [num_candidates, num_edges]
            candidate_node_aux: Auxiliary node features [num_candidates, num_nodes, aux_dim]
            route_features: Route-level features [num_candidates, num_route_features]

        Returns:
            Pooled representation [num_candidates, pooled_dim]
        """
        # Gate node and edge embeddings
        node_gate = torch.sigmoid(self.node_gate(node_emb))  # [num_nodes, 1]
        edge_gate = torch.sigmoid(self.edge_gate(edge_emb))  # [num_edges, 1]

        # Apply masks
        node_mask = candidate_node_mask.unsqueeze(-1)  # [num_candidates, num_nodes, 1]
        edge_mask = candidate_edge_mask.unsqueeze(-1)  # [num_candidates, num_edges, 1]

        # Expand embeddings for batching
        # node_emb: [1, num_nodes, hidden_dim] -> [num_candidates, num_nodes, hidden_dim]
        expanded_node_emb = node_emb.unsqueeze(0).expand(
            candidate_node_mask.shape[0], -1, -1
        )
        expanded_edge_emb = edge_emb.unsqueeze(0).expand(
            candidate_edge_mask.shape[0], -1, -1
        )

        # Combine with auxiliary features
        candidate_node_emb = expanded_node_emb + candidate_node_aux

        # Apply gating and masking
        masked_node_emb = candidate_node_emb * node_gate * node_mask
        masked_edge_emb = expanded_edge_emb * edge_gate * edge_mask

        # Aggregate along nodes and edges
        node_count = node_mask.sum(dim=1).clamp_min(1.0)
        edge_count = edge_mask.sum(dim=1).clamp_min(1.0)

        pooled_node_mean = masked_node_emb.sum(dim=1) / node_count
        pooled_node_max = (masked_node_emb + (1 - node_mask) * -1e9).max(dim=1).values

        # Handle edge features (may be empty)
        if masked_edge_emb.shape[1] > 0:
            pooled_edge_mean = masked_edge_emb.sum(dim=1) / edge_count
            pooled_edge_max = (masked_edge_emb + (1 - edge_mask) * -1e9).max(dim=1).values
        else:
            # No edges: use zero vectors
            pooled_edge_mean = torch.zeros(
                masked_edge_emb.shape[0], masked_edge_emb.shape[2],
                device=masked_edge_emb.device,
                dtype=masked_edge_emb.dtype,
            )
            pooled_edge_max = torch.zeros_like(pooled_edge_mean)

        # Concatenate all pooled features
        pooled = torch.cat(
            [
                pooled_node_mean,
                pooled_edge_mean,
                pooled_node_max,
                pooled_edge_max,
                route_features,
            ],
            dim=1,
        )

        return pooled


class ThroughputPredictorModel(nn.Module):
    """
    Full throughput prediction model.

    Encodes graph, pools candidate routes, and predicts normalized throughput.
    """

    def __init__(
        self,
        node_input_dim: int = 17,
        edge_input_dim: int = 1,
        global_input_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize throughput predictor.

        Args:
            node_input_dim: Dimension of node features (default 17)
            edge_input_dim: Dimension of edge features (default 1)
            global_input_dim: Dimension of global features (default 2)
            hidden_dim: Hidden dimension for embeddings
            num_layers: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.encoder = GraphEmbeddingModule(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )

        # Route features: route_length (1) + global features (2)
        num_route_features = 1 + global_input_dim

        self.pooler = CandidateRoutePooling(
            hidden_dim=hidden_dim,
            num_route_features=num_route_features,
        )

        # Pooled dimension: 4 * hidden_dim (mean/max for nodes/edges) + route_features
        pooled_dim = 4 * hidden_dim + num_route_features

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        graph_data,
        candidate_node_masks: torch.Tensor,
        candidate_edge_masks: torch.Tensor,
        candidate_load_projections: torch.Tensor,
        candidate_route_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            graph_data: PyTorch geometric Data object
            candidate_node_masks: [num_candidates, num_nodes] binary masks
            candidate_edge_masks: [num_candidates, num_edges] binary masks
            candidate_load_projections: [num_candidates, num_nodes] load projection
            candidate_route_lengths: [num_candidates] route lengths

        Returns:
            Predicted normalized throughputs [num_candidates]
        """
        # Extract graph features
        x = graph_data.x
        edge_index = graph_data.edge_index
        edge_attr = graph_data.edge_attr
        global_features = graph_data.global_features

        # Encode graph
        node_emb, edge_emb = self.encoder(x, edge_index, edge_attr)

        # Prepare route features: concatenate route length with global features
        num_candidates = candidate_node_masks.shape[0]
        route_lengths = candidate_route_lengths.unsqueeze(1)  # [num_candidates, 1]
        global_feat_expanded = global_features.unsqueeze(0).expand(
            num_candidates, -1
        )  # [num_candidates, global_dim]
        route_features = torch.cat(
            [route_lengths, global_feat_expanded],
            dim=1,
        )  # [num_candidates, 1 + global_dim]

        # Pool along candidate routes
        pooled = self.pooler(
            node_emb=node_emb,
            edge_emb=edge_emb,
            candidate_node_mask=candidate_node_masks,
            candidate_edge_mask=candidate_edge_masks,
            candidate_node_aux=candidate_load_projections.unsqueeze(-1),
            route_features=route_features,
        )

        # Predict throughput
        predictions = self.head(pooled).squeeze(-1)

        return predictions
