#!/usr/bin/env python3
"""
Test for M3.1-M3.3: NN model, loss, metrics, and training loop.

Verifies:
- Model forward pass
- Loss computation
- Metrics calculation
- Training loop execution
"""

import logging
import tempfile
from pathlib import Path

import torch

import ssir.basestations as bs
from ssir.pathfinder.data_collection import CollectionConfig, EpisodeCollector
from ssir.pathfinder.data_collection.dataset import ThroughputDataLoader, ThroughputDataset
from ssir.pathfinder.data_collection.normalization import (
    NormalizationStats,
    ThroughputNormalizer,
    compute_normalization_stats,
)
from ssir.pathfinder.train_nn.loss_metrics import ThroughputLoss, ThroughputMetrics
from ssir.pathfinder.train_nn.model import ThroughputPredictorModel
from ssir.pathfinder.train_nn.trainer import ThroughputTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_test_graph(index: int = 0) -> bs.IABRelayGraph:
    """Load a test graph from the data directory."""
    data_dir = "/fast/hslyu/train"
    graph_path = Path(data_dir) / f"exp_{index:03d}" / "graph.pkl"

    if not graph_path.exists():
        raise FileNotFoundError(f"Test graph not found at {graph_path}")

    graph = bs.IABRelayGraph()
    graph.load_graph(str(graph_path))
    return graph


def _create_test_dataset(tmpdir: str, num_episodes: int = 2) -> tuple[Path, NormalizationStats]:
    """Create test episodes and return paths and normalization stats."""
    graph = _load_test_graph(0)

    config = CollectionConfig(
        num_episodes=num_episodes,
        candidates_per_user=15,
        epsilon=0.1,
        output_dir=tmpdir,
        verbose=False,
    )

    collector = EpisodeCollector(config)
    stats = collector.collect(graph)

    # Compute normalization stats
    episode_files = sorted(Path(tmpdir).glob("episode_*.pkl"))
    all_throughputs = []

    from ssir.pathfinder.data_collection.data_schema import load_episode

    for episode_file in episode_files:
        episode = load_episode(episode_file)
        for entry in episode.entries:
            all_throughputs.extend(entry.true_throughputs)

    norm_stats = compute_normalization_stats(all_throughputs)

    return episode_files, norm_stats


def test_model_forward():
    """Test model forward pass."""
    print("Testing model forward pass...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ThroughputPredictorModel(
        node_input_dim=17,
        edge_input_dim=1,
        global_input_dim=2,
        hidden_dim=64,
        num_layers=2,
        heads=2,
    ).to(device)

    # Create synthetic batch
    num_nodes = 364
    num_edges = 100
    num_candidates = 10

    # Create synthetic graph data
    from torch_geometric.data import Data

    x = torch.randn(num_nodes, 17).to(device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges)).to(device)
    edge_attr = torch.randn(num_edges, 1).to(device)
    global_features = torch.randn(2).to(device)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        global_features=global_features,
    )

    # Create synthetic masks
    node_masks = torch.randint(0, 2, (num_candidates, num_nodes)).float().to(device)
    edge_masks = torch.randint(0, 2, (num_candidates, num_edges)).float().to(device)
    load_proj = torch.randn(num_candidates, num_nodes).to(device)
    route_lengths = torch.randint(5, 20, (num_candidates,)).float().to(device)

    # Forward pass
    predictions = model(data, node_masks, edge_masks, load_proj, route_lengths)

    assert predictions.shape == (num_candidates,), f"Expected shape {(num_candidates,)}, got {predictions.shape}"
    assert torch.isfinite(predictions).all(), "Model output contains NaN/inf"

    print(f"✓ Model forward pass works")
    print(f"  - Input: {num_nodes} nodes, {num_edges} edges")
    print(f"  - Output shape: {predictions.shape}")
    print(f"  - Output range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print()


def test_loss_metrics():
    """Test loss computation and metrics."""
    print("Testing loss and metrics...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create loss and metrics
    criterion = ThroughputLoss(loss_type="mse")
    metrics = ThroughputMetrics()

    # Create synthetic predictions and targets
    predictions = torch.randn(32).to(device)
    targets = torch.randn(32).to(device)

    # Compute loss
    loss = criterion(predictions, targets)
    assert loss.item() > 0, "Loss should be positive"

    # Update metrics
    metrics.update_batch(predictions, targets)

    metric_dict = metrics.get_metrics()
    assert "mae" in metric_dict
    assert "rmse" in metric_dict

    print(f"✓ Loss and metrics work")
    print(f"  - Loss: {loss.item():.6f}")
    print(f"  - Metrics: MAE={metric_dict['mae']:.6f}, RMSE={metric_dict['rmse']:.6f}")
    print()


def test_trainer_setup():
    """Test trainer initialization."""
    print("Testing trainer setup...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model and normalizer
    model = ThroughputPredictorModel(hidden_dim=64).to(device)

    norm_stats = NormalizationStats(
        log_throughput_min=6.9,
        log_throughput_max=9.7,
        log_throughput_mean=8.3,
        log_throughput_std=1.0,
        throughput_min=1000.0,
        throughput_max=15000.0,
        throughput_mean=5000.0,
        num_samples=1000,
    )
    normalizer = ThroughputNormalizer(norm_stats)

    # Create trainer
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = ThroughputTrainer(
            model=model,
            normalizer=normalizer,
            device=device,
            checkpoint_dir=tmpdir,
            lr=1e-4,
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None

    print(f"✓ Trainer setup works")
    print(f"  - Device: {device}")
    print()


def test_training_loop():
    """Test a full training loop."""
    print("Testing training loop...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create small dataset
        episode_files, norm_stats = _create_test_dataset(tmpdir, num_episodes=2)

        # Create dataset and loaders
        dataset_train = ThroughputDataset(episode_files, split="train", split_ratio=0.8)
        dataset_val = ThroughputDataset(episode_files, split="val", split_ratio=0.8)

        train_loader = ThroughputDataLoader(dataset_train, batch_size=8, shuffle=False)
        val_loader = ThroughputDataLoader(dataset_val, batch_size=8, shuffle=False)

        logger.info(f"Train samples: {len(dataset_train)}, Val samples: {len(dataset_val)}")

        # Create model
        model = ThroughputPredictorModel(hidden_dim=64).to(device)
        normalizer = ThroughputNormalizer(norm_stats)

        # Create trainer
        trainer = ThroughputTrainer(
            model=model,
            normalizer=normalizer,
            device=device,
            checkpoint_dir=tmpdir,
            lr=1e-3,
        )

        # Training loop (just a few iterations for testing)
        train_results = trainer.train_epoch(train_loader)
        logger.info(f"Train results: {train_results}")

        # Validation loop
        val_results = trainer.validate(val_loader)
        logger.info(f"Val results: {val_results}")

        assert "loss" in train_results
        assert "loss" in val_results
        assert "metrics" in train_results
        assert "metrics" in val_results

        print(f"✓ Training loop works")
        print(f"  - Train loss: {train_results['loss']:.6f}")
        print(f"  - Val loss: {val_results['loss']:.6f}")
        print()


if __name__ == "__main__":
    print("=" * 60)
    print("M3.1-M3.3: NN Model, Loss, and Training Test")
    print("=" * 60 + "\n")

    test_model_forward()
    test_loss_metrics()
    test_trainer_setup()
    test_training_loop()

    print("=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
