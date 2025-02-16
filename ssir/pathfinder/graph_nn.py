import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.serialization import add_safe_globals
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, storage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.nn import GATConv, GCNConv
from tqdm import tqdm

add_safe_globals([DataEdgeAttr, DataTensorAttr, storage.GlobalStorage])


class FocalLoss(nn.Module):
    """
    Implementation of standard Focal Loss for binary classification.
    For samples with label 1 (positive class), the loss is weighted by alpha,
    and for samples with label 0 (negative class), the loss is weighted by (1 - alpha).
    """

    def __init__(self, alpha=0.25, gamma=3.0, reduction="mean"):
        """
        Args:
            alpha (float): Weighting factor for the positive class.
            gamma (float): Focusing parameter to down-weight easy examples.
            reduction (str): Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: Predicted logits from the model (before applying sigmoid).
            targets: Ground truth binary labels (0 or 1).
        Returns:
            Computed focal loss.
        """
        # Compute binary cross entropy loss without reduction
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Compute probability for the true class: p if label=1, (1-p) if label=0
        pt = torch.where(targets == 1, probs, 1 - probs)

        # Compute focal weight factor
        focal_weight = (1 - pt) ** self.gamma

        # Compute alpha factor: alpha for positive class and (1 - alpha) for negative class
        alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Combine factors with BCE loss
        loss = alpha_factor * focal_weight * bce_loss

        # Apply reduction method
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


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
        use_residual=True,
    ):
        """
        Args:
            in_channels (int): Input node feature dimension.
            hidden_channels (int): Hidden dimension for GATConv and MLP.
            heads (int): Number of attention heads.
            num_attention_layers (int): Number of GATConv layers.
            num_linear_layers (int): Number of linear layers in the edge classifier.
            use_residual (bool): Whether to add residual connections.
        """
        super(GATEdgeClassifier, self).__init__()
        self.use_residual = use_residual

        # Build attention (GATConv) layers.
        self.att_layers = nn.ModuleList()
        # First layer: in_channels -> hidden_channels.
        self.att_layers.append(
            GATConv(in_channels, hidden_channels, heads=heads, concat=False)
        )
        # Subsequent layers: hidden_channels -> hidden_channels.
        for _ in range(num_attention_layers - 1):
            self.att_layers.append(
                GATConv(hidden_channels, hidden_channels, heads=heads, concat=False)
            )

        # For residual connection in the first layer, if dimensions differ.
        if use_residual and in_channels != hidden_channels:
            self.att_residual_proj = nn.Linear(in_channels, hidden_channels)
        else:
            self.att_residual_proj = None

        # Build linear layers for edge classification.
        self.linear_layers = nn.ModuleList()
        if num_linear_layers == 1:
            self.linear_layers.append(nn.Linear(2 * hidden_channels, 1))
        else:
            # First linear layer: 2*hidden_channels -> hidden_channels.
            self.linear_layers.append(nn.Linear(2 * hidden_channels, hidden_channels))
            # Intermediate linear layers: hidden_channels -> hidden_channels.
            for _ in range(num_linear_layers - 2):
                self.linear_layers.append(nn.Linear(hidden_channels, hidden_channels))
            # Final linear layer: hidden_channels -> 1.
            self.linear_layers.append(nn.Linear(hidden_channels, 1))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Process through attention layers with optional residuals.
        for idx, att in enumerate(self.att_layers):
            x_prev = x
            x = att(x, edge_index)
            if self.use_residual:
                if idx == 0 and self.att_residual_proj is not None:
                    x = x + self.att_residual_proj(x_prev)
                elif idx > 0:
                    x = x + x_prev
            x = F.elu(x)

        # For each edge, concatenate the source and target node embeddings.
        row, col = edge_index
        edge_repr = torch.cat([x[row], x[col]], dim=1)

        # Process through linear layers with residuals on intermediate layers.
        for idx, linear in enumerate(self.linear_layers):
            if idx < len(self.linear_layers) - 1:
                out = linear(edge_repr)
                # Apply residual connection if dimensions match.
                if self.use_residual and out.shape == edge_repr.shape:
                    edge_repr = F.relu(out + edge_repr)
                else:
                    edge_repr = F.relu(out)
            else:
                edge_repr = linear(edge_repr)
        logits = edge_repr
        return logits


class GCNEdgeClassifier(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, num_conv_layers=8, num_fc_layers=3
    ):
        """
        Args:
            in_channels (int): Dimension of input node features.
            hidden_channels (int): Hidden dimension used in GCNConv and MLP layers.
            num_conv_layers (int): Total number of convolutional layers.
            num_fc_layers (int): Total number of fully connected layers for edge classification.
        """
        super(GCNEdgeClassifier, self).__init__()

        # Create a list of GCNConv layers.
        # The first conv layer maps from in_channels to hidden_channels without a residual connection.
        # Subsequent conv layers map from hidden_channels to hidden_channels and use residual connections.
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_conv_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Build fully connected layers for edge classification.
        # Input dimension for the first FC layer is 2 * hidden_channels because we concatenate
        # the embeddings of the two nodes forming an edge.
        self.fcs = nn.ModuleList()
        if num_fc_layers == 1:
            # Single layer directly mapping to the output logit.
            self.fcs.append(nn.Linear(2 * hidden_channels, 1))
        else:
            # First FC layer.
            self.fcs.append(nn.Linear(2 * hidden_channels, hidden_channels))
            # Intermediate FC layers.
            for _ in range(num_fc_layers - 2):
                self.fcs.append(nn.Linear(hidden_channels, hidden_channels))
            # Final FC layer mapping to a single logit.
            self.fcs.append(nn.Linear(hidden_channels, 1))

    def forward(self, data):
        """
        Args:
            data (torch_geometric.data.Data or Batch):
                x: [num_nodes, in_channels] - Node features.
                edge_index: [2, num_edges] - Graph connectivity.
        Returns:
            logits: [num_edges, 1] – Logit value for each edge.
        """
        x, edge_index = data.x, data.edge_index

        # Apply the first conv layer (without residual connection) followed by ReLU.
        x = self.convs[0](x, edge_index)
        x = F.relu(x)

        # Apply the remaining conv layers with residual connections.
        for conv in self.convs[1:]:
            identity = x  # Save the current features for the residual connection.
            out = conv(x, edge_index)
            x = F.relu(out + identity)

        # For each edge, concatenate the source and target node embeddings.
        row, col = edge_index
        edge_repr = torch.cat([x[row], x[col]], dim=1)

        # Pass the concatenated edge features through the fully connected layers.
        for i, fc in enumerate(self.fcs):
            edge_repr = fc(edge_repr)
            # Apply ReLU activation for all layers except the final layer.
            if i < len(self.fcs) - 1:
                edge_repr = F.relu(edge_repr)

        logits = edge_repr
        return logits


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


def compute_class_balance(targets):
    # targets: [num_edges, 1]
    positives = targets.sum().item()
    negatives = targets.numel() - positives
    if positives == 0:
        return 1.0
    return negatives / positives


def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Train", leave=False)
    for data_batch in progress_bar:
        data_batch = data_batch.to(device)
        optimizer.zero_grad()
        logits = model(data_batch)  # [num_edges, 1]
        targets = compute_edge_targets(data_batch)

        pos_weight_value = compute_class_balance(targets)
        # pos_weight는 tensor 형태로 전달, device에 맞춰야 함.
        pos_weight_tensor = torch.tensor(pos_weight_value, device=device)
        weighted_criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight_tensor, reduction="mean"
        )

        loss = weighted_criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


# def train(model, dataloader, optimizer, criterion, device):
#     model.train()
#     total_loss = 0
#     progress_bar = tqdm(dataloader, desc="Train", leave=False)
#     for data_batch in progress_bar:
#         data_batch = data_batch.to(device)
#         optimizer.zero_grad()
#         logits = model(data_batch)  # [num_edges, 1]
#         targets = compute_edge_targets(data_batch)
#
#         loss = criterion(logits, targets)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#         progress_bar.set_postfix(loss=f"{loss.item():.4f}")
#
#     return total_loss / len(dataloader)


def evaluate(model, dataloader, device, threshold=0.5):
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
            pos_weight_value = compute_class_balance(targets)
            # pos_weight는 tensor 형태로 전달, device에 맞춰야 함.
            pos_weight_tensor = torch.tensor(pos_weight_value, device=device)
            weighted_criterion = nn.BCEWithLogitsLoss(
                pos_weight=pos_weight_tensor, reduction="mean"
            )

            loss = weighted_criterion(logits, targets)
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


######################
# 5. Main Execution Block: Prepare Dataset, Initialize Model, and Run Training Loop
######################
if __name__ == "__main__":
    # Dataset directory (modify to the actual path)
    root_dir = "../../scripts/results_pt"
    dataset = GraphDataset(root_dir, total_files=2000)

    # Split the dataset into train and test sets (e.g., 80% training, 20% test)
    train_size = 10
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        collate_fn=graph_collate_fn,
        num_workers=16,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        collate_fn=graph_collate_fn,
        num_workers=16,
    )

    # Initialize model:
    # - in_channels: Dimension of node features (e.g., 13)
    # - hidden_channels: Hidden dimension (e.g., 32)
    in_channels = 13  # Modify according to your data
    hidden_channels = 32  # Modify according to your data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = GCNEdgeClassifier(in_channels, hidden_channels).to(device)
    model = GATEdgeClassifier(in_channels, hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc, acc_0, acc_1, f1 = evaluate(
            model, test_loader, device, threshold=0.5
        )
        print(
            f"Train Loss: {train_loss:.3f} | Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f} | Acc 0: {acc_0:.3f} | Acc 1: {acc_1:.3f} | F1: {f1:.3f}"
        )
        # test_loss, test_acc = evaluate(
        #     model, test_loader, criterion, device, threshold=0.5
        # )
        # print(
        #     f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        # )
