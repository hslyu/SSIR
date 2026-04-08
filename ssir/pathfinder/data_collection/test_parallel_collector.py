#!/usr/bin/env python3
"""
Test for M1.5: Multi-process episode collector.

Verifies:
- Parallel episode generation with multiple workers
- Correct episode count
- Statistics aggregation across workers
- File I/O
"""

import json
import tempfile
from pathlib import Path

import ssir.basestations as bs
from ssir.pathfinder.data_collection import (
    CollectionConfig,
    ParallelEpisodeCollector,
    load_episode,
)


def _load_test_graph(index: int = 0) -> bs.IABRelayGraph:
    """Load a test graph from the data directory."""
    data_dir = "/fast/hslyu/train"
    graph_path = Path(data_dir) / f"exp_{index:03d}" / "graph.pkl"

    if not graph_path.exists():
        raise FileNotFoundError(f"Test graph not found at {graph_path}")

    graph = bs.IABRelayGraph()
    graph.load_graph(str(graph_path))
    return graph


def test_parallel_collection_2_workers():
    """Test parallel collection with 2 workers."""
    print("Testing parallel collection with 2 workers...")

    graph = _load_test_graph(0)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = CollectionConfig(
            num_episodes=4,
            candidates_per_user=25,
            epsilon=0.1,
            output_dir=tmpdir,
            spsc_thresholds=[0.9999],
            eavesdropper_densities=[1e-3],
            verbose=False,
        )

        collector = ParallelEpisodeCollector(config, num_workers=2)
        stats = collector.collect(graph)

        # Verify stats
        assert stats.num_episodes == 4
        assert stats.total_users > 0
        assert stats.total_candidates > 0
        assert stats.throughput_global_min > 0

        print(f"✓ Collected {stats.num_episodes} episodes with 2 workers")
        print(f"  - Total users: {stats.total_users}")
        print(f"  - Total candidates: {stats.total_candidates}")
        print(f"  - Throughput range: {stats.throughput_global_min:.2e} - {stats.throughput_global_max:.2e}")
        print(f"  - Mean throughput: {stats.throughput_global_mean:.2e}")
        print(f"  - Runtime: {stats.runtime_seconds:.2f}s")

        # Verify files
        output_path = Path(tmpdir)
        episode_files = list(output_path.glob("episode_*.pkl"))
        assert len(episode_files) == 4, f"Expected 4 episode files, got {len(episode_files)}"

        stats_file = output_path / "parallel_collection_stats.json"
        assert stats_file.exists(), "Stats file not created"

        print("✓ Files and stats verified\n")


def test_parallel_collection_4_workers():
    """Test parallel collection with 4 workers."""
    print("Testing parallel collection with 4 workers...")

    graph = _load_test_graph(0)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = CollectionConfig(
            num_episodes=8,
            candidates_per_user=20,
            epsilon=0.15,
            output_dir=tmpdir,
            spsc_thresholds=[0.99, 0.9999],
            eavesdropper_densities=[1e-3, 1e-4],
            verbose=False,
        )

        collector = ParallelEpisodeCollector(config, num_workers=4)
        stats = collector.collect(graph)

        print(f"✓ Collected {stats.num_episodes} episodes with 4 workers")
        print(f"  - Total users: {stats.total_users}")
        print(f"  - Configurations sampled:")
        for config_key, count in stats.episodes_by_config.items():
            print(f"    - {config_key}: {count} episodes")
        print(f"  - Runtime: {stats.runtime_seconds:.2f}s\n")


def test_episode_file_integrity_parallel():
    """Test that episodes from parallel collection are valid."""
    print("Testing parallel episode file integrity...")

    graph = _load_test_graph(0)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = CollectionConfig(
            num_episodes=4,
            candidates_per_user=20,
            epsilon=0.1,
            output_dir=tmpdir,
            verbose=False,
        )

        collector = ParallelEpisodeCollector(config, num_workers=2)
        stats = collector.collect(graph)

        # Load and verify episodes
        output_path = Path(tmpdir)
        episode_files = sorted(output_path.glob("episode_*.pkl"))

        for episode_file in episode_files:
            episode = load_episode(episode_file)
            assert episode.num_users > 0
            assert len(episode.entries) == episode.num_users
            assert episode.throughput_stats["min"] > 0

        print(f"✓ All {len(episode_files)} episodes verified\n")


def test_scaling_comparison():
    """Compare single-threaded vs parallel collection."""
    print("Comparing single-threaded vs parallel performance...")

    graph = _load_test_graph(0)

    # Single-threaded
    from ssir.pathfinder.data_collection import EpisodeCollector

    with tempfile.TemporaryDirectory() as tmpdir:
        config = CollectionConfig(
            num_episodes=6,
            candidates_per_user=25,
            epsilon=0.1,
            output_dir=tmpdir,
            verbose=False,
        )

        collector = EpisodeCollector(config)
        stats_single = collector.collect(graph)
        time_single = stats_single.runtime_seconds

    # Parallel (2 workers)
    with tempfile.TemporaryDirectory() as tmpdir:
        config = CollectionConfig(
            num_episodes=6,
            candidates_per_user=25,
            epsilon=0.1,
            output_dir=tmpdir,
            verbose=False,
        )

        collector = ParallelEpisodeCollector(config, num_workers=2)
        stats_parallel = collector.collect(graph)
        time_parallel = stats_parallel.runtime_seconds

    speedup = time_single / time_parallel

    print(f"✓ Single-threaded: {time_single:.2f}s")
    print(f"✓ Parallel (2 workers): {time_parallel:.2f}s")
    print(f"✓ Speedup: {speedup:.2f}x\n")


if __name__ == "__main__":
    print("=" * 60)
    print("M1.5: Multi-Process Episode Collection Test")
    print("=" * 60 + "\n")

    test_parallel_collection_2_workers()
    test_parallel_collection_4_workers()
    test_episode_file_integrity_parallel()
    test_scaling_comparison()

    print("=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
