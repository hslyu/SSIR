#!/usr/bin/env python3
"""
Test for M1.4: Single-threaded episode collector.

Verifies:
- Episode collection with multiple configurations
- Statistics aggregation
- File I/O
- Progress tracking
"""

import json
import tempfile
from pathlib import Path

import ssir.basestations as bs
from ssir.pathfinder.data_collection import (
    CollectionConfig,
    EpisodeCollector,
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


def test_collection_single_config():
    """Test collecting episodes with a single configuration."""
    print("Testing collection with single configuration...")

    graph = _load_test_graph(0)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = CollectionConfig(
            num_episodes=5,
            candidates_per_user=30,
            epsilon=0.1,
            output_dir=tmpdir,
            spsc_thresholds=[0.9999],
            eavesdropper_densities=[1e-3],
            verbose=False,
        )

        collector = EpisodeCollector(config)
        stats = collector.collect(graph)

        # Verify stats
        assert stats.num_episodes == 5
        assert stats.total_users > 0
        assert stats.total_candidates > 0
        assert stats.throughput_global_min > 0
        assert stats.throughput_global_max >= stats.throughput_global_min
        assert stats.throughput_global_mean > 0

        print(f"✓ Collected {stats.num_episodes} episodes")
        print(f"  - Total users: {stats.total_users}")
        print(f"  - Total candidates: {stats.total_candidates}")
        print(f"  - Throughput range: {stats.throughput_global_min:.2e} - {stats.throughput_global_max:.2e}")
        print(f"  - Mean throughput: {stats.throughput_global_mean:.2e}")
        print(f"  - Runtime: {stats.runtime_seconds:.2f}s")

        # Verify files
        output_path = Path(tmpdir)
        episode_files = list(output_path.glob("episode_*.pkl"))
        assert len(episode_files) == 5, f"Expected 5 episode files, got {len(episode_files)}"

        stats_file = output_path / "collection_stats.json"
        assert stats_file.exists(), "Stats file not created"

        # Verify stats file content
        with open(stats_file) as f:
            saved_stats = json.load(f)
        assert saved_stats["num_episodes"] == 5

        print("✓ Files and stats verified\n")


def test_collection_multiple_configs():
    """Test collecting episodes with multiple configurations."""
    print("Testing collection with multiple configurations...")

    graph = _load_test_graph(0)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = CollectionConfig(
            num_episodes=6,
            candidates_per_user=25,
            epsilon=0.15,
            output_dir=tmpdir,
            spsc_thresholds=[0.99, 0.9999],
            eavesdropper_densities=[1e-3, 1e-4],
            verbose=False,
            seed=42,
        )

        collector = EpisodeCollector(config)
        stats = collector.collect(graph)

        print(f"✓ Collected {stats.num_episodes} episodes with mixed configs")
        print(f"  - Configurations sampled:")
        for config_key, count in stats.episodes_by_config.items():
            print(f"    - {config_key}: {count} episodes")
        print(f"  - Total users: {stats.total_users}")
        print(f"  - Runtime: {stats.runtime_seconds:.2f}s\n")


def test_episode_file_integrity():
    """Test that saved episodes can be loaded correctly."""
    print("Testing episode file integrity...")

    graph = _load_test_graph(0)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = CollectionConfig(
            num_episodes=2,
            candidates_per_user=20,
            epsilon=0.1,
            output_dir=tmpdir,
            verbose=False,
        )

        collector = EpisodeCollector(config)
        stats = collector.collect(graph)

        # Load and verify episodes
        output_path = Path(tmpdir)
        episode_files = sorted(output_path.glob("episode_*.pkl"))

        for episode_file in episode_files:
            episode = load_episode(episode_file)
            assert episode.num_users > 0
            assert len(episode.entries) == episode.num_users
            assert episode.throughput_stats["min"] > 0
            assert episode.throughput_stats["max"] >= episode.throughput_stats["min"]

        print(f"✓ All {len(episode_files)} episodes verified\n")


if __name__ == "__main__":
    print("=" * 60)
    print("M1.4: Single-Threaded Episode Collection Test")
    print("=" * 60 + "\n")

    test_collection_single_config()
    test_collection_multiple_configs()
    test_episode_file_integrity()

    print("=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
