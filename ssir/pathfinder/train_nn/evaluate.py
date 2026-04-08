"""
Evaluation and analysis for trained throughput predictor model.

M3.4: Validates model performance and generates analysis reports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from ssir.pathfinder.data_collection.dataset import ThroughputDataLoader
from ssir.pathfinder.data_collection.normalization import ThroughputNormalizer
from ssir.pathfinder.train_nn.loss_metrics import ThroughputMetrics
from ssir.pathfinder.train_nn.model import ThroughputPredictorModel

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluate trained throughput predictor model.

    Computes metrics, generates visualizations, and analyzes failure modes.
    """

    def __init__(
        self,
        model: ThroughputPredictorModel,
        normalizer: ThroughputNormalizer,
        device: str | torch.device = "cuda",
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained ThroughputPredictorModel
            normalizer: ThroughputNormalizer instance
            device: Device to run on
        """
        self.model = model
        self.normalizer = normalizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, val_loader: ThroughputDataLoader) -> Dict:
        """
        Evaluate model on validation set.

        Args:
            val_loader: DataLoader with validation samples

        Returns:
            Dictionary with evaluation results
        """
        metrics = ThroughputMetrics()
        predictions_all = []
        targets_all = []
        original_throughputs_all = []

        for batch_dict in val_loader:
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

                # Forward pass
                predictions = self.model(
                    graph_data,
                    node_masks,
                    edge_masks,
                    load_proj,
                    route_lengths,
                )

                # Update metrics
                metrics.update_batch(
                    predictions,
                    targets_normalized,
                    denormalizer=self.normalizer,
                    original_throughputs=targets,
                )

                # Collect for visualization
                predictions_all.append(predictions.cpu().numpy())
                targets_all.append(targets_normalized.cpu().numpy())
                original_throughputs_all.append(targets_np)

        # Concatenate all
        predictions_all = np.concatenate(predictions_all)
        targets_all = np.concatenate(targets_all)
        original_throughputs_all = np.concatenate(original_throughputs_all)

        # Denormalize
        predictions_denorm = self.normalizer.denormalize_array(predictions_all)
        targets_denorm = self.normalizer.denormalize_array(targets_all)

        return {
            "metrics": metrics.get_metrics(),
            "predictions_normalized": predictions_all,
            "targets_normalized": targets_all,
            "predictions_denorm": predictions_denorm,
            "targets_denorm": targets_denorm,
            "original_throughputs": original_throughputs_all,
        }

    def generate_report(
        self,
        eval_results: Dict,
        output_dir: str | Path = "results",
    ) -> None:
        """
        Generate evaluation report with visualizations.

        Args:
            eval_results: Results from evaluate()
            output_dir: Directory to save report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics = eval_results["metrics"]
        predictions_denorm = eval_results["predictions_denorm"]
        targets_denorm = eval_results["targets_denorm"]

        # Save metrics
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(
                {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                 for k, v in metrics.items()},
                f,
                indent=2,
            )
        logger.info(f"Metrics saved to {metrics_path}")

        # Plot 1: Actual vs Predicted (scatter)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(targets_denorm, predictions_denorm, alpha=0.5, s=10)
        ax.plot([targets_denorm.min(), targets_denorm.max()],
               [targets_denorm.min(), targets_denorm.max()],
               'r--', label='Perfect prediction')
        ax.set_xlabel('Target Throughput (bps)')
        ax.set_ylabel('Predicted Throughput (bps)')
        ax.set_title('Actual vs Predicted Throughput')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot1_path = output_dir / "actual_vs_predicted.png"
        fig.savefig(plot1_path, dpi=100)
        plt.close()
        logger.info(f"Plot saved to {plot1_path}")

        # Plot 2: Log-scale scatter
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(targets_denorm, predictions_denorm, alpha=0.5, s=10)
        ax.plot([targets_denorm.min(), targets_denorm.max()],
               [targets_denorm.min(), targets_denorm.max()],
               'r--', label='Perfect prediction')
        ax.set_xlabel('Target Throughput (bps)', fontsize=12)
        ax.set_ylabel('Predicted Throughput (bps)', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('Actual vs Predicted Throughput (log scale)')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')

        plot2_path = output_dir / "actual_vs_predicted_logscale.png"
        fig.savefig(plot2_path, dpi=100)
        plt.close()
        logger.info(f"Plot saved to {plot2_path}")

        # Plot 3: Error distribution
        absolute_errors = np.abs(predictions_denorm - targets_denorm)
        relative_errors = absolute_errors / (np.abs(targets_denorm) + 1e-8)

        # Filter out infinite and NaN values for plotting
        valid_abs_mask = np.isfinite(absolute_errors)
        valid_rel_mask = np.isfinite(relative_errors)
        abs_errors_valid = absolute_errors[valid_abs_mask]
        rel_errors_valid = relative_errors[valid_rel_mask]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        if len(abs_errors_valid) > 0:
            axes[0].hist(abs_errors_valid, bins=50, edgecolor='black')
        axes[0].set_xlabel('Absolute Error (bps)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Absolute Error Distribution')
        axes[0].grid(True, alpha=0.3, axis='y')

        if len(rel_errors_valid) > 0:
            axes[1].hist(rel_errors_valid, bins=50, edgecolor='black')
        axes[1].set_xlabel('Relative Error (fraction)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Relative Error Distribution')
        axes[1].grid(True, alpha=0.3, axis='y')

        plot3_path = output_dir / "error_distributions.png"
        fig.savefig(plot3_path, dpi=100)
        plt.close()
        logger.info(f"Plot saved to {plot3_path}")

        # Save summary
        summary = {
            "num_samples": len(targets_denorm),
            "metrics": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                       for k, v in metrics.items()},
            "error_stats": {
                "absolute_error_mean": float(abs_errors_valid.mean()) if len(abs_errors_valid) > 0 else float('nan'),
                "absolute_error_std": float(abs_errors_valid.std()) if len(abs_errors_valid) > 0 else float('nan'),
                "absolute_error_min": float(abs_errors_valid.min()) if len(abs_errors_valid) > 0 else float('nan'),
                "absolute_error_max": float(abs_errors_valid.max()) if len(abs_errors_valid) > 0 else float('nan'),
                "relative_error_mean": float(rel_errors_valid.mean()) if len(rel_errors_valid) > 0 else float('nan'),
                "relative_error_std": float(rel_errors_valid.std()) if len(rel_errors_valid) > 0 else float('nan'),
                "relative_error_p95": float(np.percentile(rel_errors_valid, 95)) if len(rel_errors_valid) > 0 else float('nan'),
            },
        }

        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {summary_path}")

        logger.info("\n" + "=" * 60)
        logger.info("Evaluation Summary")
        logger.info("=" * 60)
        logger.info(f"Samples: {summary['num_samples']}")
        logger.info(f"MAE: {metrics['mae']:.4f}")
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
        logger.info(f"MAPE: {metrics['mape']:.4%}")
        logger.info(f"Top-1 Accuracy: {metrics['top1_accuracy']:.2%}")
        logger.info(f"Top-5 Accuracy: {metrics['top5_accuracy']:.2%}")
        logger.info("=" * 60)
