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
        # Iterate from 1 to total_files and add only the files that exist to the list.
        for i in range(1, total_files + 1):
            file_path = os.path.join(root_dir, f"{i}.pt")
            if os.path.isfile(file_path):
                self.file_list.append(file_path)
        print(f"Found {len(self.file_list)} files out of {total_files} total files.")

        self.preload = preload
        if self.preload:
            print("Preloading files into memory...")
            self.data = [torch.load(fp) for fp in self.file_list]
            print("File loading complete.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.preload:
            return self.data[idx]
        else:
            file_path = self.file_list[idx]
            data, label = torch.load(file_path)
            return data, label


def graph_collate_fn(batch):
    """
    각 샘플이 (data, label) 형태로 주어질 때,
    data에 label.edge_index를 복사한 뒤, Batch.from_data_list로 병합한다.
    단, 각 샘플별 노드 재매핑(offset)을 고려하여 label_edge_index도 합친다.
    """
    data_list, label_list = zip(*batch)
    # 각 data에 label.edge_index 필드를 임시로 저장
    for data, label in zip(data_list, label_list):
        data.label_edge_index = label.edge_index
    # data 객체를 Batch로 합침 (여기서 x, edge_index 등은 자동 re-indexing됨)
    batched_data = Batch.from_data_list(data_list)

    # 각 샘플별로 label_edge_index에 재매핑(offset)을 직접 적용
    label_edge_index_list = []
    offset = 0
    for data in data_list:
        # data.x의 노드 수 만큼 offset 적용
        # data.label_edge_index: [2, num_edges] (원래 sample 내 인덱스)
        label_edge_index_list.append(data.label_edge_index + offset)
        offset += data.x.size(0)
    batched_data.label_edge_index = torch.cat(label_edge_index_list, dim=1)

    return batched_data


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
# 3. Functions to Compute Edge Targets
######################
def compute_edge_targets(data_batch):
    """
    batched_data에 대해, data.edge_index와 data_batch.label_edge_index를 이용하여
    각 edge가 label에도 존재하는지 판단하는 타겟 벡터를 생성한다.
    """
    device = data_batch.x.device
    # 전체 노드 수보다 큰 multiplier 선택
    max_nodes = data_batch.x.size(0)
    multiplier = max_nodes + 1

    # data edge key 생성
    data_u = data_batch.edge_index[0]
    data_v = data_batch.edge_index[1]
    data_keys = data_u * multiplier + data_v

    # label edge key 생성 (label_edge_index는 이미 re-indexed 되어 있음)
    label_u = data_batch.label_edge_index[0]
    label_v = data_batch.label_edge_index[1]
    label_keys = label_u * multiplier + label_v

    isin = torch.isin(data_keys, label_keys)
    targets = isin.float().view(-1, 1)
    return targets


######################
# 4. Training & Evaluation Functions
######################


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Train", leave=False)
    for data_batch in progress_bar:
        data_batch = data_batch.to(device)
        optimizer.zero_grad()
        logits = model(data_batch)  # [num_edges, 1]
        targets = compute_edge_targets(data_batch)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    progress_bar = tqdm(dataloader, desc="Evaluate", leave=False)
    with torch.no_grad():
        for data_batch in progress_bar:
            data_batch = data_batch.to(device)
            logits = model(data_batch)
            targets = compute_edge_targets(data_batch)
            loss = criterion(logits, targets)
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > threshold).float()
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # 전체 accuracy 계산
    total_edges = all_targets.numel()
    total_correct = (all_preds == all_targets).sum().item()
    total_accuracy = total_correct / total_edges if total_edges > 0 else 0

    # 클래스별 accuracy 계산
    mask_0 = all_targets == 0
    mask_1 = all_targets == 1
    correct_0 = (all_preds[mask_0] == all_targets[mask_0]).sum().item()
    correct_1 = (all_preds[mask_1] == all_targets[mask_1]).sum().item()
    total_0 = mask_0.sum().item()
    total_1 = mask_1.sum().item()
    accuracy_0 = correct_0 / total_0 if total_0 > 0 else 0
    accuracy_1 = correct_1 / total_1 if total_1 > 0 else 0

    # F1 score (positive class: 1) 계산
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
