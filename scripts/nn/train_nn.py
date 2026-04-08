#!/usr/bin/env python3
"""
Standalone script for training throughput predictor neural network.

Usage:
    # Quick test (5 epochs, small batch)
    python scripts/train_nn.py --data-dir data/train --epochs 5

    # Production training
    python scripts/train_nn.py \
        --data-dir data/train \
        --output-dir models/throughput_predictor \
        --epochs 100 \
        --batch-size 32 \
        --lr 1e-4 \
        --hidden-dim 128 \
        --early-stop-patience 10

    # With specific GPU
    python scripts/train_nn.py \
        --data-dir data/train \
        --device cuda:0 \
        --batch-size 64
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from ssir.pathfinder.data_collection.data_schema import load_episode
from ssir.pathfinder.data_collection.dataset import (
    ThroughputDataLoader,
    ThroughputDataset,
)
from ssir.pathfinder.data_collection.normalization import (
    NormalizationStats,
    ThroughputNormalizer,
    compute_normalization_stats,
)
from ssir.pathfinder.train_nn.evaluate import ModelEvaluator
from ssir.pathfinder.train_nn.model import ThroughputPredictorModel
from ssir.pathfinder.train_nn.trainer import ThroughputTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train throughput predictor neural network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/train",
        help="Directory containing episode pickle files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/throughput_predictor",
        help="Directory to save models and results",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for data loader (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for optimizer (default: 1e-5)",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help="Patience for early stopping (default: 10)",
    )

    # Model arguments
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension for model (default: 128)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of GAT layers (default: 3)",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=4,
        help="Number of attention heads (default: 4)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability (default: 0.1)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse", "huber"],
        help="Loss function (default: mse)",
    )

    # Data split arguments
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle training data",
    )

    # Device argument
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to train on (cuda, cpu, cuda:0, etc.) (default: cuda)",
    )

    args = parser.parse_args()

    # Setup
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Validate data directory
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    episode_files = sorted(data_dir.rglob("episode_*.pkl"))
    if not episode_files:
        logger.error(f"No episode files found in {data_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check device
    device = args.device
    if "cuda" in device and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"

    logger.info("=" * 70)
    logger.info("Throughput Predictor - Neural Network Training")
    logger.info("=" * 70)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Episodes: {len(episode_files)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(
        f"Model: {args.hidden_dim}D hidden, {args.num_layers} layers, "
        f"{args.heads} heads"
    )
    logger.info("=" * 70 + "\n")

    # Load or compute normalization stats
    norm_stats_path = data_dir / "norm_stats.json"
    if norm_stats_path.exists():
        logger.info(f"Loading normalization stats from {norm_stats_path}")
        norm_stats = NormalizationStats.load(norm_stats_path)
    else:
        logger.info("Computing normalization stats from episodes...")
        all_throughputs = []

        for episode_file in episode_files:
            episode = load_episode(episode_file)
            for entry in episode.entries:
                all_throughputs.extend(entry.true_throughputs)

        norm_stats = compute_normalization_stats(all_throughputs)
        norm_stats.save(norm_stats_path)
        logger.info(f"Saved to {norm_stats_path}")

    normalizer = ThroughputNormalizer(norm_stats)

    # Create datasets
    logger.info("\nCreating datasets...")
    dataset_train = ThroughputDataset(
        episode_files,
        cache_in_memory=False,
        split="train",
        split_ratio=args.train_split,
    )
    dataset_val = ThroughputDataset(
        episode_files,
        cache_in_memory=False,
        split="val",
        split_ratio=args.train_split,
    )

    logger.info(f"Train samples: {len(dataset_train)}")
    logger.info(f"Val samples: {len(dataset_val)}")

    # Create data loaders
    train_loader = ThroughputDataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle,
    )
    val_loader = ThroughputDataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Create model
    logger.info("\nInitializing model...")
    model = ThroughputPredictorModel(
        node_input_dim=17,
        edge_input_dim=1,
        global_input_dim=2,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
    )
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Create trainer
    logger.info("\nInitializing trainer...")
    trainer = ThroughputTrainer(
        model=model,
        normalizer=normalizer,
        device=device,
        checkpoint_dir=output_dir,
        loss_type=args.loss,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Train
    logger.info("\nStarting training...\n")
    history = trainer.fit(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        early_stop_patience=args.early_stop_patience,
    )

    # Evaluate
    logger.info("\nEvaluating on validation set...")
    evaluator = ModelEvaluator(model, normalizer, device=device)
    results = evaluator.evaluate(val_loader)
    evaluator.generate_report(results, output_dir=output_dir / "evaluation")

    # Save config
    config = {
        "model": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "heads": args.heads,
            "dropout": args.dropout,
            "num_parameters": num_params,
        },
        "training": {
            "optimizer": "Adam",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "scheduler": "CosineAnnealing",
            "loss_fn": args.loss,
            "epochs_completed": len(history["train_loss"]),
            "total_epochs": args.epochs,
            "early_stop_patience": args.early_stop_patience,
            "batch_size": args.batch_size,
        },
        "data": {
            "data_dir": str(data_dir),
            "num_episodes": len(episode_files),
            "train_samples": len(dataset_train),
            "val_samples": len(dataset_val),
            "train_split": args.train_split,
        },
        "normalization": {
            "throughput_min": float(norm_stats.throughput_min),
            "throughput_max": float(norm_stats.throughput_max),
            "throughput_mean": float(norm_stats.throughput_mean),
            "log_throughput_min": float(norm_stats.log_throughput_min),
            "log_throughput_max": float(norm_stats.log_throughput_max),
            "log_throughput_mean": float(norm_stats.log_throughput_mean),
            "log_throughput_std": float(norm_stats.log_throughput_std),
        },
        "results": {
            "best_val_loss": float(min(history["val_loss"])),
            "best_epoch": history["best_epoch"],
        },
    }

    config_path = output_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to {config_path}")

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("Training Complete!")
    logger.info("=" * 70)
    logger.info(f"Models saved to: {output_dir}")
    logger.info(f"Best model: {output_dir / 'best_model_*.pth'}")
    logger.info(f"Evaluation results: {output_dir / 'evaluation'}")
    logger.info(f"Best validation loss: {config['results']['best_val_loss']:.6f}")
    logger.info(f"Training metrics saved to: {output_dir / 'training_history.json'}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
