import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

import ssir.pathfinder.graph_nn as gnn


def init_xavier_normal(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def compute_global_pos_weight(dataloader):
    total_edges = 0
    total_positives = 0
    # Compute the global pos_weight for the BCEWithLogitsLoss
    for batch in dataloader:
        data = batch
        total_positives += data.edge_label.sum().item()
        total_edges += data.x.numel()
    negatives = total_edges - total_positives
    return negatives / total_positives if total_positives > 0 else 1.0


if __name__ == "__main__":
    # Load dataset
    train_dir = "/fast/hslyu/results_pt/train"
    test_dir = "/fast/hslyu/results_pt/test"
    train_dataset = gnn.GraphDataset(train_dir, total_files=40000, preload=False)
    test_dataset = gnn.GraphDataset(train_dir, total_files=10000, preload=False)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )

    # Initialize model:
    # - in_channels: Dimension of node features (e.g., 13)
    # - hidden_channels: Hidden dimension (e.g., 32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = gnn.GCNEdgeClassifier(
        in_channels=14, hidden_channels=128, num_conv_layers=12, num_fc_layers=3
    ).to(device)
    # model = gnn.GATEdgeClassifier(
    #     in_channels=13,
    #     hidden_channels=16,
    #     heads=3,
    #     num_attention_layers=8,
    #     num_linear_layers=3,
    # ).to(device)

    # initialize weights
    model.apply(init_xavier_normal)

    # Compute global pos_weight and define the loss function
    print("Computing global pos_weight... ", end="")
    pos_weight = compute_global_pos_weight(train_loader)
    # criterion = nn.BCEWithLogitsLoss(
    #     pos_weight=torch.tensor(pos_weight, device=device), reduction="mean"
    # )
    criterion = nn.CrossEntropyLoss(torch.Tensor([1 / pos_weight, 1]).to(device))
    print(f"{pos_weight:.4f}")

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    num_epochs = 500
    min_loss = float("inf")
    no_improve_counter = 0
    no_improve_threshold = 15

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        train_loss = gnn.train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, acc_0, acc_1, f1 = gnn.evaluate(
            model,
            test_loader,
            criterion,
            device,
        )
        print(
            f"Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}, Test Acc: {test_acc:.2f}, "
            f"Acc 0: {acc_0:.3f}, Acc 1: {acc_1:.3f}, F1: {f1:.2f}, lr: {optimizer.param_groups[0]['lr']:.1e}"
        )

        # Save the model
        if test_loss < min_loss:
            min_loss = test_loss
            torch.save(model.state_dict(), "model.pt")
            no_improve_counter = 0
        else:
            no_improve_counter += 1

        if no_improve_counter >= no_improve_threshold:
            print("Early stopping triggered.")
            break

        lr_scheduler.step(test_loss)
