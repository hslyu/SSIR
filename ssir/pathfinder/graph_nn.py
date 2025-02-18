import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.serialization import add_safe_globals
from torch.utils.data import Dataset
from torch_geometric.data import Batch, storage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.nn import GATConv, GCNConv, NNConv
from tqdm import tqdm

add_safe_globals([DataEdgeAttr, DataTensorAttr, storage.GlobalStorage])


######################
# 1. GraphDataset & Collate Function
######################


class GraphDataset(Dataset):
    def __init__(self, root_dir, total_files=500, preload=True):
        """
        Args:
            root_dir (str): Directory path where the .pt files are stored.
            total_files (int): Total number of files to process.
            preload (bool): If True, all data will be preloaded into memory.
        """
        self.root_dir = root_dir
        self.file_list = []
        # Iterate from 0 to total_files - 1 and add only the files that exist to the list.
        for i in range(total_files):
            file_path = os.path.join(root_dir, f"{i}.pt")
            if os.path.isfile(file_path):
                self.file_list.append(file_path)
        print(f"Found {len(self.file_list)} files out of {total_files} total files.")

        self.preload = preload
        if self.preload:
            print("Preloading files into memory...")
            self.data = []
            for fp in self.file_list:
                # Load data and label from file
                data, label = torch.load(fp)
                # Store label edge_index as an attribute in data
                # Compute edge_label attribute and store in data
                data.edge_label = self.compute_edge_label(data, label)
                self.data.append(data)
            print("File loading complete.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.preload:
            return self.data[idx]
        else:
            file_path = self.file_list[idx]
            data, label = torch.load(file_path)
            data.file_path = file_path
            data.edge_label = self.compute_edge_label(data, label)
            return data

    def compute_edge_label(self, data, label):
        """
        For each edge in the data graph, compute whether it exists in the label graph.
        If an edge in data.edge_index is present in label.edge_index, its label is 1.0; otherwise 0.0.
        """
        device = data.x.device
        num_nodes = data.x.size(0)
        multiplier = (
            num_nodes + 1
        )  # Multiplier greater than the maximum number of nodes

        # Compute keys for data edges using local node indices
        data_u = data.edge_index[0]
        data_v = data.edge_index[1]
        data_keys = data_u * multiplier + data_v

        # Compute keys for label edges using local node indices
        label_u = label.edge_index[0]
        label_v = label.edge_index[1]
        label_keys = label_u * multiplier + label_v

        # Check whether each data edge key exists in the label edge keys
        isin = torch.isin(data_keys, label_keys)
        edge_label = isin.float().view(-1, 1)
        return edge_label


######################
# 2. GAT Model (Edge Classification)
######################
class GATEdgeClassifier(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        heads=16,
        num_attention_layers=4,
        num_linear_layers=3,
        embedding_channels=32,
        use_residual=True,
    ):
        """
        Args:
            in_channels (int): Input node feature dimension.
            hidden_channels (int): Hidden dimension for GATConv and MLP.
            heads (int): Number of attention heads.
            num_attention_layers (int): Number of GATConv layers.
            num_linear_layers (int): Number of linear layers in the edge classifier.
            embedding_channels (int): Output channels for NNConv edge embedding.
            use_residual (bool): Whether to use residual connections.
        """
        super(GATEdgeClassifier, self).__init__()
        self.use_residual = use_residual

        # Build NNConv layer for edge embeddings.
        edge_nn = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, in_channels * embedding_channels),
        )
        self.nn_conv = NNConv(in_channels, embedding_channels, edge_nn, aggr="add")

        # Build attention layers.
        self.att_layers = nn.ModuleList()
        # First attention layer: embedding_channels -> hidden_channels.
        self.att_layers.append(
            GATConv(embedding_channels, hidden_channels, heads=heads, concat=False)
        )
        # Subsequent attention layers: hidden_channels -> hidden_channels.
        for _ in range(num_attention_layers - 1):
            self.att_layers.append(
                GATConv(hidden_channels, hidden_channels, heads=heads, concat=False)
            )

        # Build MLP layers for edge classification.
        self.mlp_layers = nn.ModuleList()
        if num_linear_layers == 1:
            self.mlp_layers.append(nn.Linear(2 * hidden_channels, 1))
        else:
            # First linear layer: 2*hidden_channels -> hidden_channels.
            self.mlp_layers.append(nn.Linear(2 * hidden_channels, hidden_channels))
            # Intermediate linear layers.
            for _ in range(num_linear_layers - 2):
                self.mlp_layers.append(nn.Linear(hidden_channels, hidden_channels))
            # Final linear layer: hidden_channels -> 1.
            self.mlp_layers.append(nn.Linear(hidden_channels, 1))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Compute edge embeddings.
        x = self.nn_conv(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.att_layers[0](x, edge_index)
        x = F.elu(x)

        # Apply attention layers with residual connections.
        for idx, layer in enumerate(self.att_layers[1:]):
            identity = x
            x = layer(x, edge_index)
            if self.use_residual and identity.shape == x.shape:
                x = x + identity
            x = F.elu(x)

        # Concatenate node embeddings for each edge.
        row, col = edge_index
        edge_repr = torch.cat([x[row], x[col]], dim=1)

        # Apply MLP layers with residual connections on intermediate layers.
        for idx, layer in enumerate(self.mlp_layers):
            out = layer(edge_repr)
            if idx < len(self.mlp_layers) - 1:
                if self.use_residual and out.shape == edge_repr.shape:
                    edge_repr = F.relu(out + edge_repr)
                else:
                    edge_repr = F.relu(out)
            else:
                edge_repr = out

        return edge_repr


class GCNEdgeClassifier(nn.Module):
    def __init__(
        self,
        in_channels=13,
        hidden_channels=128,
        num_conv_layers=8,
        num_fc_layers=3,
        embedding_channels=32,
        use_residual=True,
    ):
        """
        Args:
            in_channels (int): Input node feature dimension.
            hidden_channels (int): Hidden dimension for GCNConv and MLP.
            num_conv_layers (int): Total number of convolutional layers.
            num_fc_layers (int): Total number of fully connected layers for edge classification.
            embedding_channels (int): Output channels for NNConv edge embedding.
            use_residual (bool): Whether to use residual connections.
        """
        super(GCNEdgeClassifier, self).__init__()
        self.use_residual = use_residual

        # Build NNConv layer for edge embeddings.
        edge_nn = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, in_channels * embedding_channels),
        )
        self.nn_conv = NNConv(in_channels, embedding_channels, edge_nn, aggr="add")

        # Build GCNConv layers.
        self.conv_layers = nn.ModuleList()
        # First conv layer: embedding_channels -> hidden_channels.
        self.conv_layers.append(GCNConv(embedding_channels, hidden_channels))
        # Subsequent conv layers: hidden_channels -> hidden_channels.
        for _ in range(num_conv_layers - 1):
            self.conv_layers.append(GCNConv(hidden_channels, hidden_channels))

        # Build MLP layers for edge classification.
        self.mlp_layers = nn.ModuleList()
        if num_fc_layers == 1:
            self.mlp_layers.append(nn.Linear(2 * hidden_channels, 1))
        else:
            # First FC layer: 2*hidden_channels -> hidden_channels.
            self.mlp_layers.append(nn.Linear(2 * hidden_channels, hidden_channels))
            # Intermediate FC layers.
            for _ in range(num_fc_layers - 2):
                self.mlp_layers.append(nn.Linear(hidden_channels, hidden_channels))
            # Final FC layer: hidden_channels -> 1.
            self.mlp_layers.append(nn.Linear(hidden_channels, 1))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Compute edge embeddings.
        x = self.nn_conv(x, edge_index, edge_attr)
        x = F.relu(x)

        # Apply GCNConv layers with residual connections.
        for idx, layer in enumerate(self.conv_layers):
            identity = x
            x = layer(x, edge_index)
            if self.use_residual and idx > 0:
                if identity.shape == x.shape:
                    x = F.relu(x + identity)
                else:
                    x = F.relu(x)
            else:
                x = F.relu(x)

        # Concatenate node embeddings for each edge.
        row, col = edge_index
        edge_repr = torch.cat([x[row], x[col]], dim=1)

        # Apply MLP layers with residual connections on intermediate layers.
        for idx, layer in enumerate(self.mlp_layers):
            out = layer(edge_repr)
            if idx < len(self.mlp_layers) - 1:
                if self.use_residual and out.shape == edge_repr.shape:
                    edge_repr = F.relu(out + edge_repr)
                else:
                    edge_repr = F.relu(out)
            else:
                edge_repr = out

        return edge_repr


######################
# 4. Training & Evaluation Functions
######################


def train(model, dataloader, criterion, optimizer, device):
    """
    Train the model using DataBatch directly.
    The model is applied to the entire batch, which contains a concatenated edge_label attribute.
    """
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Train", leave=False)

    for batch in progress_bar:
        # Move the entire batch to the device
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass on the complete DataBatch
        logits = model(batch)  # Expected shape: [total_edges_in_batch, 1]
        targets = batch.edge_label  # Expected shape: [total_edges_in_batch, 1]

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, threshold=0.5):
    """
    Evaluate the model using the entire DataBatch directly.
    The model processes the concatenated graph batch and uses the pre-computed
    'edge_label' attribute from the batch for loss and metric computation.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    progress_bar = tqdm(dataloader, desc="Evaluate", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            # Move the entire batch to the device
            batch = batch.to(device)

            # Forward pass on the complete DataBatch
            logits = model(batch)  # Expected shape: [total_edges_in_batch, 1]
            targets = batch.edge_label  # Expected shape: [total_edges_in_batch, 1]

            loss = criterion(logits, targets)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) > threshold).float()
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    # Concatenate all predictions and targets along the edge dimension
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute overall accuracy
    total_edges = all_targets.numel()
    total_correct = (all_preds == all_targets).sum().item()
    total_accuracy = total_correct / total_edges if total_edges > 0 else 0

    # Compute per-class accuracy
    mask_0 = all_targets == 0
    mask_1 = all_targets == 1
    correct_0 = (all_preds[mask_0] == all_targets[mask_0]).sum().item()
    correct_1 = (all_preds[mask_1] == all_targets[mask_1]).sum().item()
    total_0 = mask_0.sum().item()
    total_1 = mask_1.sum().item()
    accuracy_0 = correct_0 / total_0 if total_0 > 0 else 0
    accuracy_1 = correct_1 / total_1 if total_1 > 0 else 0

    # Compute F1 score for the positive class (1)
    TP = ((all_preds == 1) & (all_targets == 1)).sum().item()
    FP = ((all_preds == 1) & (all_targets == 0)).sum().item()
    FN = ((all_preds == 0) & (all_targets == 1)).sum().item()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    avg_loss = total_loss / len(dataloader)
    return avg_loss, total_accuracy, accuracy_0, accuracy_1, f1
