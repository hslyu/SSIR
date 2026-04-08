"""
Loss functions and metrics for throughput prediction.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ThroughputLoss(nn.Module):
    """
    Loss function for throughput prediction.

    Uses MSE on normalized throughput predictions.
    """

    def __init__(self, loss_type: str = "mse", reduction: str = "mean"):
        """
        Initialize loss function.

        Args:
            loss_type: "mse" (mean squared error) or "huber"
            reduction: "mean", "sum", or "none"
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction

        if loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction=reduction)
        elif loss_type == "huber":
            self.loss_fn = nn.HuberLoss(reduction=reduction, delta=1.0)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss.

        Args:
            predictions: Predicted normalized throughputs [batch_size]
            targets: Target normalized throughputs [batch_size]

        Returns:
            Loss scalar (or vector if reduction='none')
        """
        return self.loss_fn(predictions, targets)


class ThroughputMetrics:
    """
    Metrics for throughput prediction evaluation.

    Tracks MAE, ranking accuracy, and regime-specific errors.
    """

    def __init__(self, throughput_threshold: float = 5000.0):
        """
        Initialize metrics.

        Args:
            throughput_threshold: Threshold to split high/low throughput regimes
        """
        self.threshold = throughput_threshold
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.total_mae = 0.0
        self.total_mape = 0.0  # Mean absolute percentage error
        self.total_rmse = 0.0
        self.num_samples = 0

        # Per-regime metrics
        self.high_regime_mae = 0.0
        self.high_regime_count = 0
        self.low_regime_mae = 0.0
        self.low_regime_count = 0

        # Ranking metrics
        self.top1_correct = 0
        self.top5_correct = 0
        self.num_batches = 0

    def update_batch(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        denormalizer=None,
        original_throughputs: torch.Tensor | None = None,
    ) -> None:
        """
        Update metrics with a batch of predictions.

        Args:
            predictions: Predicted normalized throughputs [batch_size]
            targets: Target normalized throughputs [batch_size]
            denormalizer: Denormalizer function (for converting back to original scale)
            original_throughputs: Original throughput values for regime classification
        """
        batch_size = predictions.shape[0]
        self.num_samples += batch_size

        # Detach and convert to numpy
        pred_np = predictions.detach().cpu().numpy()
        targ_np = targets.detach().cpu().numpy()

        # Compute basic metrics (in normalized space)
        mae = np.abs(pred_np - targ_np).mean()
        rmse = np.sqrt(((pred_np - targ_np) ** 2).mean())
        mape = np.abs((targ_np - pred_np) / (np.abs(targ_np) + 1e-8)).mean()

        self.total_mae += mae * batch_size
        self.total_rmse += rmse * batch_size
        self.total_mape += mape * batch_size

        # Denormalize for regime-specific metrics
        if denormalizer is not None:
            pred_denorm = denormalizer.denormalize_array(pred_np)
            targ_denorm = denormalizer.denormalize_array(targ_np)

            # Regime-specific MAE
            if original_throughputs is not None:
                orig_np = original_throughputs.detach().cpu().numpy()
                high_mask = orig_np >= self.threshold
                low_mask = ~high_mask

                if high_mask.any():
                    high_mae = np.abs(
                        pred_denorm[high_mask] - targ_denorm[high_mask]
                    ).mean()
                    self.high_regime_mae += high_mae * high_mask.sum()
                    self.high_regime_count += high_mask.sum()

                if low_mask.any():
                    low_mae = np.abs(
                        pred_denorm[low_mask] - targ_denorm[low_mask]
                    ).mean()
                    self.low_regime_mae += low_mae * low_mask.sum()
                    self.low_regime_count += low_mask.sum()

        self.num_batches += 1

    def update_batch_ranking(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """
        Update ranking metrics (for multi-candidate batches).

        Args:
            predictions: Predicted throughputs [num_candidates]
            targets: Target throughputs [num_candidates]
        """
        # Ranking: check if top-1 and top-5 predictions match targets
        if predictions.shape[0] >= 1:
            pred_top1_idx = predictions.argmax().item()
            targ_top1_idx = targets.argmax().item()

            if pred_top1_idx == targ_top1_idx:
                self.top1_correct += 1

            if predictions.shape[0] >= 5:
                pred_top5_idx = torch.topk(predictions, min(5, len(predictions)))[1]
                targ_top1_idx = targets.argmax().item()

                if targ_top1_idx in pred_top5_idx:
                    self.top5_correct += 1

        self.num_batches += 1

    def get_metrics(self) -> dict:
        """
        Get aggregated metrics.

        Returns:
            Dictionary with all metrics
        """
        if self.num_samples == 0:
            return {
                "mae": 0.0,
                "rmse": 0.0,
                "mape": 0.0,
                "high_regime_mae": 0.0,
                "low_regime_mae": 0.0,
                "top1_accuracy": 0.0,
                "top5_accuracy": 0.0,
            }

        metrics = {
            "mae": self.total_mae / self.num_samples,
            "rmse": self.total_rmse / self.num_samples,
            "mape": self.total_mape / self.num_samples,
        }

        if self.high_regime_count > 0:
            metrics["high_regime_mae"] = self.high_regime_mae / self.high_regime_count
        else:
            metrics["high_regime_mae"] = 0.0

        if self.low_regime_count > 0:
            metrics["low_regime_mae"] = self.low_regime_mae / self.low_regime_count
        else:
            metrics["low_regime_mae"] = 0.0

        if self.num_batches > 0:
            metrics["top1_accuracy"] = self.top1_correct / self.num_batches
            metrics["top5_accuracy"] = self.top5_correct / self.num_batches
        else:
            metrics["top1_accuracy"] = 0.0
            metrics["top5_accuracy"] = 0.0

        return metrics

    def __str__(self) -> str:
        """String representation of metrics."""
        m = self.get_metrics()
        return (
            f"MAE: {m['mae']:.4f}, RMSE: {m['rmse']:.4f}, MAPE: {m['mape']:.4f}, "
            f"High-regime MAE: {m['high_regime_mae']:.4f}, "
            f"Low-regime MAE: {m['low_regime_mae']:.4f}, "
            f"Top-1 Acc: {m['top1_accuracy']:.2%}, "
            f"Top-5 Acc: {m['top5_accuracy']:.2%}"
        )
