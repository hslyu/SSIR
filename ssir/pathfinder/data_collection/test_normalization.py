#!/usr/bin/env python3
"""
Test for M2.4: Throughput normalization.

Verifies:
- Normalization stats computation
- Log-scale normalization and denormalization
- Handling of wide dynamic range (1e3-1.6e4)
- Reconstruction error validation
"""

import json
import tempfile
from pathlib import Path

import numpy as np

from ssir.pathfinder.data_collection.normalization import (
    NormalizationStats,
    ThroughputNormalizer,
    compute_normalization_stats,
    validate_normalization,
)


def test_stats_computation():
    """Test computing normalization statistics."""
    print("Testing stats computation...")

    # Realistic throughput range from actual data (1e3-1.6e4 bps)
    throughputs = [
        1030.4,  # Low (near-source, many users)
        2000.0,
        5000.0,
        10000.0,
        15909.1,  # High (leaf, few users)
    ] * 100  # Replicate for more samples

    stats = compute_normalization_stats(throughputs)

    # Verify stats
    assert stats.throughput_min > 0
    assert stats.throughput_max > stats.throughput_min
    assert stats.throughput_min < stats.throughput_mean < stats.throughput_max
    assert stats.log_throughput_std > 0

    print(f"✓ Stats computed successfully")
    print(f"  - Throughput range: {stats.throughput_min:.2e} - {stats.throughput_max:.2e} bps")
    print(f"  - Log-throughput range: {stats.log_throughput_min:.4f} - {stats.log_throughput_max:.4f}")
    print(f"  - Log-std: {stats.log_throughput_std:.4f}")
    print(f"  - Samples: {stats.num_samples}")
    print()

    return stats


def test_normalization_denormalization(stats):
    """Test normalize/denormalize round-trip."""
    print("Testing normalization/denormalization...")

    normalizer = ThroughputNormalizer(stats)

    # Test specific values
    test_values = [
        stats.throughput_min,
        stats.throughput_mean,
        stats.throughput_max,
        3000.0,  # Intermediate value
    ]

    for tp in test_values:
        normalized = normalizer.normalize(tp)
        denormalized = normalizer.denormalize(normalized)
        rel_error = abs(denormalized - tp) / tp

        assert rel_error < 0.01, f"Reconstruction error too high: {rel_error:.4%}"

    print(f"✓ Normalization/denormalization works")
    print(f"  - Max reconstruction error: <1% over range")
    print()


def test_batch_operations(stats):
    """Test batch normalization."""
    print("Testing batch operations...")

    normalizer = ThroughputNormalizer(stats)

    # Test batch normalize
    throughputs = [1000.0, 2000.0, 5000.0, 10000.0, 15000.0]
    normalized = normalizer.normalize_batch(throughputs)

    assert len(normalized) == len(throughputs)
    assert all(isinstance(n, float) for n in normalized)

    # Test batch denormalize
    denormalized = normalizer.denormalize_batch(normalized)
    assert len(denormalized) == len(throughputs)

    # Check reconstruction
    for orig, recon in zip(throughputs, denormalized):
        rel_error = abs(recon - orig) / orig
        assert rel_error < 0.01, f"Batch reconstruction error too high: {rel_error:.4%}"

    print(f"✓ Batch operations work correctly")
    print()


def test_numpy_operations(stats):
    """Test numpy array operations."""
    print("Testing numpy array operations...")

    normalizer = ThroughputNormalizer(stats)

    # Create array
    throughputs = np.array([1000.0, 2000.0, 5000.0, 10000.0, 15000.0])

    # Normalize
    normalized = normalizer.normalize_array(throughputs)
    assert normalized.shape == throughputs.shape
    assert np.isfinite(normalized).all()

    # Denormalize
    denormalized = normalizer.denormalize_array(normalized)
    assert denormalized.shape == throughputs.shape
    assert np.isfinite(denormalized).all()

    # Check reconstruction
    rel_errors = np.abs(denormalized - throughputs) / throughputs
    assert np.all(rel_errors < 0.01), f"Max rel error: {rel_errors.max():.4%}"

    print(f"✓ Numpy operations work correctly")
    print(f"  - Max reconstruction error: {rel_errors.max():.4%}")
    print()


def test_wide_range_handling():
    """Test that normalization handles the full throughput range."""
    print("Testing wide range handling (1e3-1.6e4 bps)...")

    # Create samples across the full range
    log_min = np.log(1000.0)
    log_max = np.log(15000.0)

    # Sample uniformly in log-space
    log_samples = np.linspace(log_min, log_max, 50)
    throughputs = list(np.exp(log_samples)) * 2  # Replicate

    stats = compute_normalization_stats(throughputs)
    normalizer = ThroughputNormalizer(stats)

    # Test across full range
    test_percentiles = [0, 25, 50, 75, 100]
    for percentile in test_percentiles:
        tp = np.percentile(throughputs, percentile)
        normalized = normalizer.normalize(tp)
        denormalized = normalizer.denormalize(normalized)
        rel_error = abs(denormalized - tp) / tp

        assert rel_error < 0.01, f"Failed at {percentile}%ile: {rel_error:.4%}"

    print(f"✓ Wide range (1e3-1.6e4 bps) handled correctly")
    print(f"  - Dynamic range: {max(throughputs) / min(throughputs):.1f}x")
    print()


def test_save_load_stats():
    """Test saving and loading normalization stats."""
    print("Testing save/load stats...")

    # Create initial stats
    throughputs = [1030.4, 2000.0, 5000.0, 10000.0, 15909.1] * 100
    stats = compute_normalization_stats(throughputs)

    with tempfile.TemporaryDirectory() as tmpdir:
        stats_path = Path(tmpdir) / "norm_stats.json"

        # Save
        stats.save(stats_path)
        assert stats_path.exists()

        # Load
        loaded_stats = NormalizationStats.load(stats_path)

        # Verify
        assert loaded_stats.throughput_min == stats.throughput_min
        assert loaded_stats.throughput_max == stats.throughput_max
        assert loaded_stats.log_throughput_mean == stats.log_throughput_mean
        assert loaded_stats.log_throughput_std == stats.log_throughput_std

        # Verify normalizer still works
        original_normalizer = ThroughputNormalizer(stats)
        loaded_normalizer = ThroughputNormalizer(loaded_stats)

        test_tp = 5000.0
        orig_norm = original_normalizer.normalize(test_tp)
        loaded_norm = loaded_normalizer.normalize(test_tp)

        assert abs(orig_norm - loaded_norm) < 1e-6

    print(f"✓ Save/load stats works correctly")
    print()


def test_validation():
    """Test normalization validation."""
    print("Testing normalization validation...")

    throughputs = [1030.4, 2000.0, 5000.0, 10000.0, 15909.1] * 100
    stats = compute_normalization_stats(throughputs)
    normalizer = ThroughputNormalizer(stats)

    results = validate_normalization(stats, normalizer)

    assert results["passed"], f"Validation failed: {results['issues']}"
    assert len(results["test_values"]) > 0

    # Check error rates
    for tp, result in results["test_values"].items():
        error = result["relative_error"]
        assert error < 0.01, f"High error for {tp}: {error:.4%}"

    print(f"✓ Validation passed")
    print(f"  - Test values: {list(results['test_values'].keys())}")
    print()


def test_extreme_values():
    """Test handling of extreme values."""
    print("Testing extreme value handling...")

    stats = NormalizationStats(
        log_throughput_min=np.log(1000.0),
        log_throughput_max=np.log(15000.0),
        log_throughput_mean=np.log(5000.0),
        log_throughput_std=1.0,
        throughput_min=1000.0,
        throughput_max=15000.0,
        throughput_mean=5000.0,
        num_samples=1000,
    )

    normalizer = ThroughputNormalizer(stats)

    # Test minimum
    norm_min = normalizer.normalize(1000.0)
    assert np.isfinite(norm_min)

    # Test maximum
    norm_max = normalizer.normalize(15000.0)
    assert np.isfinite(norm_max)

    # Test denormalization
    denorm_min = normalizer.denormalize(norm_min)
    denorm_max = normalizer.denormalize(norm_max)

    assert np.isfinite(denorm_min) and denorm_min > 0
    assert np.isfinite(denorm_max) and denorm_max > 0

    print(f"✓ Extreme values handled correctly")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("M2.4: Throughput Normalization Test")
    print("=" * 60 + "\n")

    stats = test_stats_computation()
    test_normalization_denormalization(stats)
    test_batch_operations(stats)
    test_numpy_operations(stats)
    test_wide_range_handling()
    test_save_load_stats()
    test_validation()
    test_extreme_values()

    print("=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
