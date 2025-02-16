import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import ssir.pathfinder.graph_nn as gnn


def init_xavier_normal(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


if __name__ == "__main__":
    root_dir = "results_pt"
    dataset = gnn.GraphDataset(root_dir, total_files=50000, preload=False)

    # Split the dataset into train and test sets (e.g., 80% training, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=True,
        collate_fn=gnn.graph_collate_fn,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        collate_fn=gnn.graph_collate_fn,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )

    # Initialize model:
    # - in_channels: Dimension of node features (e.g., 13)
    # - hidden_channels: Hidden dimension (e.g., 32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = gnn.GCNEdgeClassifier(
        in_channels=13, hidden_channels=64, num_conv_layers=32, num_fc_layers=8
    ).to(device)
    # model = GATEdgeClassifier(
    #     in_channels=13,
    #     hidden_channels=4,
    #     heads=8,
    #     num_attention_layers=16,
    #     num_linear_layers=2,
    # ).to(device)

    # initialize weights
    model.apply(init_xavier_normal)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = gnn.DynamicBCEWithLogitsLoss()
    # on plateau scheduler
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
            model, test_loader, criterion, device, threshold=0.5
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
