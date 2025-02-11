import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

data_path = "/home/hslyu/research/SSIR/scripts/results_pkl/10/master_graph.pkl"
label_path = "/home/hslyu/research/SSIR/scripts/results_pkl/10/graph_genetic.pkl"
with open(data_path, "rb") as f:
    graph = pickle.load(f)
with open(label_path, "rb") as f:
    graph_label = pickle.load(f)

data = graph.to_torch_geometric()
label = graph_label.to_torch_geometric()


# Edge 분류를 위한 GCN 기반 네트워크 정의
class GCNEdgeClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, embedding_dim):
        super(GCNEdgeClassifier, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, embedding_dim)
        # 두 노드 임베딩을 입력받아 edge 존재 여부(logit)를 예측하는 MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),  # 출력: logit 하나 (BCEWithLogitsLoss 사용)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 노드 feature가 없으면 1로 채워진 feature 사용
        if x is None:
            x = torch.ones((data.num_nodes, 1), device=edge_index.device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        # 각 edge의 양 끝 노드 임베딩을 이어붙임
        src, dst = edge_index
        edge_features = torch.cat([x[src], x[dst]], dim=1)
        edge_logits = self.edge_mlp(edge_features).squeeze()  # shape: [num_edges]
        return edge_logits


# label 그래프의 edge와 data 그래프의 edge를 비교하여 target label 생성 함수
def get_edge_labels(data_edge_index, label_edge_index):
    # undirected graph인 경우, (min, max) 튜플로 정규화
    label_edges = set()
    for a, b in zip(label_edge_index[0].tolist(), label_edge_index[1].tolist()):
        label_edges.add((min(a, b), max(a, b)))

    labels = []
    for a, b in zip(data_edge_index[0].tolist(), data_edge_index[1].tolist()):
        if (min(a, b), max(a, b)) in label_edges:
            labels.append(1.0)
        else:
            labels.append(0.0)
    return torch.tensor(labels, dtype=torch.float32)


# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)  # data 그래프: master_graph.json 기반
label = label.to(device)  # label 그래프: graph_genetic.json 기반

# 노드 feature 차원이 없으면 1로 대체되도록 in_channels 결정
in_channels = data.num_node_features if data.x is not None else 1

# 모델 초기화
model = GCNEdgeClassifier(
    in_channels=in_channels, hidden_channels=16, embedding_dim=16
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

# data 그래프의 edge와 label 그래프의 edge를 비교하여 target 생성
edge_labels = get_edge_labels(data.edge_index, label.edge_index).to(device)

# overfitting을 위해 단일 데이터에 대해 다수의 epoch 동안 학습
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    logits = model(data)  # 각 edge에 대한 logit 출력
    loss = criterion(logits, edge_labels)
    loss.backward()
    optimizer.step()

    if epoch % 1 == 0:
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            accuracy = (preds == edge_labels).sum().item() / edge_labels.size(0)
            print(
                f"Epoch {epoch:04d} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.4f}"
            )
