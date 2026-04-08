"""
Training loop for throughput predictor.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ssir.pathfinder.data_collection.dataset import ThroughputDataLoader
from ssir.pathfinder.data_collection.normalization import ThroughputNormalizer
from ssir.pathfinder.train_nn.loss_metrics import ThroughputLoss, ThroughputMetrics
from ssir.pathfinder.train_nn.model import ThroughputPredictorModel

logger = logging.getLogger(__name__)


class ThroughputTrainer:
    """
    Trainer for throughput predictor model.

    Handles training, validation, checkpointing, and early stopping.
    """

    def __init__(
        self,
        model: ThroughputPredictorModel,
        normalizer: ThroughputNormalizer,
        device: str | torch.device = "cuda",
        checkpoint_dir: str | Path = "models",
        loss_type: str = "mse",
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        """
        Initialize trainer.

        Args:
            model: ThroughputPredictorModel instance
            normalizer: ThroughputNormalizer instance
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            loss_type: "mse" or "huber"
            lr: Learning rate
            weight_decay: Weight decay for optimizer
        """
        self.model = model
        self.normalizer = normalizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Loss and metrics
        self.criterion = ThroughputLoss(loss_type=loss_type)
        self.train_metrics = ThroughputMetrics()
        self.val_metrics = ThroughputMetrics()

        # Optimizer and scheduler
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)

        # Training state
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.early_stop_counter = 0

    def train_epoch(self, train_loader: ThroughputDataLoader, epoch: int, num_epochs: int) -> Dict:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader with training samples

        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        self.train_metrics.reset()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Train {epoch}/{num_epochs}",
            leave=True,
            dynamic_ncols=True,
            mininterval=0.5,
            unit="batch",
        )
        for batch_dict in pbar:
            samples = batch_dict["samples"]
            batch_losses = []
            batch_metric_tuples = []

            for sample in samples:
                # Move to device
                graph_data = sample.graph_data.to(self.device)
                node_masks = sample.node_masks.to(self.device)
                edge_masks = sample.edge_masks.to(self.device)
                load_proj = sample.load_projections.to(self.device)
                route_lengths = sample.route_lengths.to(self.device)
                targets = sample.true_throughputs.to(self.device)

                # Normalize targets
                targets_np = targets.cpu().numpy()
                targets_normalized = torch.from_numpy(
                    self.normalizer.normalize_array(targets_np)
                ).float().to(self.device)
                if not torch.isfinite(targets_normalized).all():
                    continue

                # Forward pass
                predictions = self.model(
                    graph_data,
                    node_masks,
                    edge_masks,
                    load_proj,
                    route_lengths,
                )
                if not torch.isfinite(predictions).all():
                    continue

                # Loss
                loss = self.criterion(predictions, targets_normalized)
                if not torch.isfinite(loss):
                    continue

                batch_losses.append(loss)
                batch_metric_tuples.append(
                    (
                        predictions.detach(),
                        targets_normalized.detach(),
                        targets.detach(),
                    )
                )

            if not batch_losses:
                continue

            batch_loss = torch.stack(batch_losses).mean()
            self.optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1

            for predictions, targets_normalized, targets in batch_metric_tuples:
                self.train_metrics.update_batch(
                    predictions,
                    targets_normalized,
                    denormalizer=self.normalizer,
                    original_throughputs=targets,
                )

            pbar.set_postfix({"train_loss": f"{total_loss / max(num_batches, 1):.6f}"})

        return {
            "loss": total_loss / max(num_batches, 1),
            "metrics": self.train_metrics.get_metrics(),
        }

    @torch.no_grad()
    def validate(self, val_loader: ThroughputDataLoader, epoch: int, num_epochs: int) -> Dict:
        """
        Validate on validation set.

        Args:
            val_loader: DataLoader with validation samples

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        self.val_metrics.reset()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            val_loader,
            desc=f"Val {epoch}/{num_epochs}",
            leave=True,
            dynamic_ncols=True,
            mininterval=0.5,
            unit="batch",
        )
        for batch_dict in pbar:
            samples = batch_dict["samples"]

            for sample in samples:
                # Move to device
                graph_data = sample.graph_data.to(self.device)
                node_masks = sample.node_masks.to(self.device)
                edge_masks = sample.edge_masks.to(self.device)
                load_proj = sample.load_projections.to(self.device)
                route_lengths = sample.route_lengths.to(self.device)
                targets = sample.true_throughputs.to(self.device)

                # Normalize targets
                targets_np = targets.cpu().numpy()
                targets_normalized = torch.from_numpy(
                    self.normalizer.normalize_array(targets_np)
                ).float().to(self.device)
                if not torch.isfinite(targets_normalized).all():
                    continue

                # Forward pass
                predictions = self.model(
                    graph_data,
                    node_masks,
                    edge_masks,
                    load_proj,
                    route_lengths,
                )
                if not torch.isfinite(predictions).all():
                    continue

                # Loss
                loss = self.criterion(predictions, targets_normalized)
                if not torch.isfinite(loss):
                    continue
                total_loss += loss.item()
                num_batches += 1

                # Update metrics
                self.val_metrics.update_batch(
                    predictions,
                    targets_normalized,
                    denormalizer=self.normalizer,
                    original_throughputs=targets,
                )

            pbar.set_postfix({"val_loss": f"{total_loss / max(num_batches, 1):.6f}"})

        return {
            "loss": total_loss / max(num_batches, 1),
            "metrics": self.val_metrics.get_metrics(),
        }

    def fit(
        self,
        train_loader: ThroughputDataLoader,
        val_loader: ThroughputDataLoader,
        num_epochs: int = 100,
        early_stop_patience: int = 10,
    ) -> Dict:
        """
        Train model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            early_stop_patience: Patience for early stopping

        Returns:
            Dictionary with training history
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
            "best_epoch": 0,
            "best_val_loss": float("inf"),
        }

        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_results = self.train_epoch(train_loader, epoch + 1, num_epochs)
            logger.info(f"Train Loss: {train_results['loss']:.6f}")
            logger.info(f"Train Metrics: {ThroughputMetrics.__str__(self.train_metrics)}")

            # Validate
            val_results = self.validate(val_loader, epoch + 1, num_epochs)
            logger.info(f"Val Loss: {val_results['loss']:.6f}")
            logger.info(f"Val Metrics: {ThroughputMetrics.__str__(self.val_metrics)}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs} complete")

            # Update history
            history["train_loss"].append(train_results["loss"])
            history["val_loss"].append(val_results["loss"])
            history["train_metrics"].append(train_results["metrics"])
            history["val_metrics"].append(val_results["metrics"])

            # Learning rate scheduling
            self.scheduler.step()

            # Early stopping
            if val_results["loss"] < self.best_val_loss:
                self.best_val_loss = val_results["loss"]
                self.best_epoch = epoch
                self.early_stop_counter = 0

                # Save checkpoint
                self.save_checkpoint(f"best_model_epoch_{epoch:04d}.pth")
                history["best_epoch"] = epoch
                history["best_val_loss"] = self.best_val_loss

                logger.info(f"✓ Best model saved (val_loss: {self.best_val_loss:.6f})")
            else:
                self.early_stop_counter += 1
                logger.info(f"No improvement ({self.early_stop_counter}/{early_stop_patience})")

                if self.early_stop_counter >= early_stop_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # Save final state
        self.save_checkpoint("final_model.pth")
        self.save_history(history)

        return history

    def save_checkpoint(self, filename: str) -> None:
        """
        Save model checkpoint.

        Args:
            filename: Filename to save to
        """
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            },
            path,
        )
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, filename: str) -> None:
        """
        Load model checkpoint.

        Args:
            filename: Filename to load from
        """
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        logger.info(f"Checkpoint loaded from {path}")

    def save_history(self, history: Dict) -> None:
        """
        Save training history to JSON.

        Args:
            history: Training history dictionary
        """
        def convert_to_native(obj):
            """Recursively convert numpy/torch types to native Python types."""
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(v) for v in obj]
            elif hasattr(obj, 'item'):  # torch tensor or numpy scalar
                return float(obj.item())
            elif isinstance(obj, (float, int, str, bool, type(None))):
                return obj
            else:
                return float(obj)  # Fallback for other numeric types

        # Convert non-serializable objects
        history_json = {
            "train_loss": convert_to_native(history["train_loss"]),
            "val_loss": convert_to_native(history["val_loss"]),
            "best_epoch": history["best_epoch"],
            "best_val_loss": float(history["best_val_loss"]),
            "train_metrics": convert_to_native(history["train_metrics"]),
            "val_metrics": convert_to_native(history["val_metrics"]),
        }

        path = self.checkpoint_dir / "training_history.json"
        with open(path, "w") as f:
            json.dump(history_json, f, indent=2)
        logger.info(f"History saved to {path}")
