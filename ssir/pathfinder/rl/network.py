import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, global_max_pool


class GraphQNetwork(nn.Module):
    def __init__(
        self,
        input_channels=15,
        embedding_channels=64,
        num_conv_layers=2,
        num_fc_layers=2,
        use_residual=True,
        heads=4,
    ):
        """
        Integrated graph Q-network using GATConv with edge weights.

        Args:
            input_channels (int): Dimension of input node features.
            embedding_channels (int): Hidden dimension for attention layers.
            num_conv_layers (int): Number of GATConv layers.
            use_residual (bool): Whether to apply residual connections.
            heads (int): Number of attention heads.
        """
        super(GraphQNetwork, self).__init__()
        self.use_residual = use_residual

        # Linear mapping for input features to desired embedding dimension.
        self.linear_in = nn.Linear(input_channels, embedding_channels)

        # Build GATConv layers with edge_dim=1 (for se_inverse as scalar weight).
        self.conv_layers = nn.ModuleList()
        for _ in range(num_conv_layers):
            self.conv_layers.append(
                GATConv(
                    embedding_channels,
                    embedding_channels,
                    heads=heads,
                    edge_dim=3,
                    concat=False,
                )
            )

        # Fully connected layer to map each node's feature to a scalar Q-value.
        self.fc1 = nn.Linear(embedding_channels, embedding_channels)
        self.fc2 = nn.Linear(embedding_channels, embedding_channels)
        self.fc3 = nn.Linear(embedding_channels, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Preprocess edge attributes:
        spectral_efficiency_inverse = 1e6 / edge_attr[:, 2]
        edge_attr[:, 2] = spectral_efficiency_inverse
        # edge_weight = spectral_efficiency_inverse.unsqueeze(1)

        batch = data.batch if hasattr(data, "batch") else None

        x = F.gelu(self.linear_in(x))
        # Apply stacked GATConv layers with optional residual connections.
        for idx, conv in enumerate(self.conv_layers):
            identity = x
            x = conv(x, edge_index, edge_attr)
            if self.use_residual and idx > 0:
                if identity.shape == x.shape:
                    x = F.gelu(x + identity)
                else:
                    x = F.gelu(x)
            else:
                x = F.gelu(x)

        # Map each node's feature to a scalar Q-value.
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        q_node = self.fc3(x)

        # Global min pooling over nodes.
        if batch is not None:
            q = -global_max_pool(-q_node, batch)  # trick to compute min pooling
        else:
            q, _ = torch.min(q_node, dim=0, keepdim=True)
            q = q.squeeze(0)

        return q
