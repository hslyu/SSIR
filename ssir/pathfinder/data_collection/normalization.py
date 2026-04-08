"""
Throughput normalization for handling wide dynamic range.

The throughput varies from ~1 Kbps (near-source, many users)
to ~15 Kbps (leaf nodes, few users). Log-scale normalization
handles this range effectively.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NormalizationStats:
    """
    Statistics for throughput normalization.

    Uses log-scale: targets are log(throughput).
    """

    log_throughput_min: float  # log(min_throughput)
    log_throughput_max: float  # log(max_throughput)
    log_throughput_mean: float  # mean of log-throughputs
    log_throughput_std: float  # std of log-throughputs
    throughput_min: float  # Original min (for reference)
    throughput_max: float  # Original max (for reference)
    throughput_mean: float  # Original mean (for reference)
    num_samples: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Save stats to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Normalization stats saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> NormalizationStats:
        """Load stats from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def compute_normalization_stats(
    throughputs: List[float],
) -> NormalizationStats:
    """
    Compute normalization statistics from raw throughput values.

    Args:
        throughputs: List of throughput values (bps)

    Returns:
        NormalizationStats object
    """
    if not throughputs:
        raise ValueError("No throughput values provided")

    # Convert to numpy array
    throughputs_arr = np.array(throughputs, dtype=np.float32)

    # Filter out invalid values
    valid_mask = np.isfinite(throughputs_arr) & (throughputs_arr > 0)
    valid_throughputs = throughputs_arr[valid_mask]

    if len(valid_throughputs) == 0:
        raise ValueError("No valid throughput values")

    logger.info(
        f"Filtering throughputs: {len(throughputs)} → {len(valid_throughputs)} valid"
    )

    # Compute log-scale stats
    log_throughputs = np.log(valid_throughputs)

    stats = NormalizationStats(
        log_throughput_min=float(np.min(log_throughputs)),
        log_throughput_max=float(np.max(log_throughputs)),
        log_throughput_mean=float(np.mean(log_throughputs)),
        log_throughput_std=float(np.std(log_throughputs)),
        throughput_min=float(np.min(valid_throughputs)),
        throughput_max=float(np.max(valid_throughputs)),
        throughput_mean=float(np.mean(valid_throughputs)),
        num_samples=len(valid_throughputs),
    )

    logger.info(f"Normalization stats computed:")
    logger.info(f"  - Throughput range: {stats.throughput_min:.2e} - {stats.throughput_max:.2e} bps")
    logger.info(f"  - Log-throughput range: {stats.log_throughput_min:.4f} - {stats.log_throughput_max:.4f}")
    logger.info(f"  - Mean: {stats.throughput_mean:.2e} bps")
    logger.info(f"  - Log-std: {stats.log_throughput_std:.4f}")

    return stats


class ThroughputNormalizer:
    """
    Normalize and denormalize throughput values using log-scale.

    The neural network predicts normalized log-throughputs, which are
    then converted back to actual throughput values.
    """

    def __init__(self, stats: NormalizationStats):
        """
        Initialize normalizer with statistics.

        Args:
            stats: NormalizationStats object
        """
        self.stats = stats

    def normalize(self, throughput: float) -> float:
        """
        Normalize a throughput value to log-scale.

        Args:
            throughput: Throughput in bps (positive)

        Returns:
            Normalized log-throughput (zero-centered)
        """
        if throughput <= 0:
            raise ValueError(f"Throughput must be positive, got {throughput}")

        log_throughput = np.log(throughput)

        # Normalize to zero mean, unit variance
        normalized = (log_throughput - self.stats.log_throughput_mean) / (
            self.stats.log_throughput_std + 1e-8
        )

        return float(normalized)

    def denormalize(self, normalized: float) -> float:
        """
        Denormalize a log-throughput value back to actual throughput.

        Args:
            normalized: Normalized log-throughput

        Returns:
            Throughput in bps
        """
        log_throughput = (
            normalized * self.stats.log_throughput_std + self.stats.log_throughput_mean
        )
        throughput = np.exp(log_throughput)
        return float(throughput)

    def normalize_batch(self, throughputs: list[float]) -> list[float]:
        """
        Normalize multiple throughput values.

        Args:
            throughputs: List of throughput values

        Returns:
            List of normalized values
        """
        return [self.normalize(tp) for tp in throughputs]

    def denormalize_batch(self, normalized: list[float]) -> list[float]:
        """
        Denormalize multiple values.

        Args:
            normalized: List of normalized values

        Returns:
            List of throughput values
        """
        return [self.denormalize(n) for n in normalized]

    def normalize_array(self, throughputs: np.ndarray) -> np.ndarray:
        """
        Normalize a numpy array of throughput values.

        Args:
            throughputs: Array of throughput values

        Returns:
            Array of normalized values
        """
        throughputs = np.asarray(throughputs, dtype=np.float32)
        min_tp = max(float(self.stats.throughput_min), 1e-8)
        throughputs = np.where(np.isfinite(throughputs), throughputs, min_tp)
        throughputs = np.clip(throughputs, min_tp, None)
        log_throughputs = np.log(throughputs)
        normalized = (log_throughputs - self.stats.log_throughput_mean) / (
            self.stats.log_throughput_std + 1e-8
        )
        return normalized

    def denormalize_array(self, normalized: np.ndarray) -> np.ndarray:
        """
        Denormalize a numpy array.

        Args:
            normalized: Array of normalized values

        Returns:
            Array of throughput values
        """
        log_throughputs = (
            normalized * self.stats.log_throughput_std + self.stats.log_throughput_mean
        )
        throughputs = np.exp(log_throughputs)
        return throughputs


def validate_normalization(
    stats: NormalizationStats,
    normalizer: ThroughputNormalizer,
    test_values: List[float] | None = None,
) -> dict:
    """
    Validate that normalization works correctly and handles full range.

    Args:
        stats: NormalizationStats object
        normalizer: ThroughputNormalizer object
        test_values: Optional list of values to test (default: min/max/mean)

    Returns:
        Dictionary with validation results
    """
    if test_values is None:
        test_values = [
            stats.throughput_min,
            stats.throughput_max,
            stats.throughput_mean,
        ]

    results = {
        "test_values": {},
        "passed": True,
        "issues": [],
    }

    for tp in test_values:
        # Normalize then denormalize
        normalized = normalizer.normalize(tp)
        denormalized = normalizer.denormalize(normalized)

        # Check reconstruction error
        rel_error = abs(denormalized - tp) / tp

        results["test_values"][tp] = {
            "normalized": normalized,
            "denormalized": denormalized,
            "relative_error": float(rel_error),
        }

        if rel_error > 0.01:  # 1% threshold
            results["passed"] = False
            results["issues"].append(
                f"High reconstruction error for {tp:.2e}: {rel_error:.4%}"
            )

    # Check that std dev is computed
    if stats.log_throughput_std < 0.01:
        results["passed"] = False
        results["issues"].append(f"Very low log-std: {stats.log_throughput_std:.6f}")

    # Check range
    if stats.throughput_max / stats.throughput_min < 10:
        logger.warning(
            f"Narrow throughput range: {stats.throughput_max / stats.throughput_min:.1f}x"
        )

    if results["passed"]:
        logger.info("✓ Normalization validation passed")
    else:
        logger.error(f"✗ Normalization validation failed: {results['issues']}")

    return results
