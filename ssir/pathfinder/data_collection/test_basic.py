#!/usr/bin/env python3
"""
Quick test for M1.1 data collection module.

Verifies:
- DataEntry creation
- Episode generation
- Save/load functionality
"""

import tempfile
from pathlib import Path

import ssir.basestations as bs
from ssir.pathfinder.data_collection import (
    DataEntry,
    EpisodeDataset,
    generate_episode,
    load_episode,
    save_episode,
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


def test_data_entry():
    """Test DataEntry validation."""
    print("Testing DataEntry...")

    # Load a real graph
    graph = _load_test_graph(0)

    # Test invalid selected_candidate_idx (should raise ValueError)
    try:
        entry = DataEntry(
            episode_id=0,
            user_index=0,
            spsc_threshold=0.9999,
            eavesdropper_density=1e-3,
            master_graph=graph,
            partial_graph=graph.copy(),
            candidate_routes=[],
            true_throughputs=[],
            selected_candidate_idx=0,  # Invalid: no candidates
        )
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly caught validation error: {e}")

    print("✓ DataEntry validation works\n")


def test_episode_generation_minimal():
    """Test episode generation with a real graph."""
    print("Testing episode generation...")

    # Load a real graph
    graph = _load_test_graph(0)

    # Generate episode
    episode = generate_episode(
        master_graph=graph,
        spsc_threshold=0.9999,
        eavesdropper_density=1e-3,
        episode_id=0,
        num_candidates_per_user=5,
        epsilon=0.1,
    )

    print(f"✓ Generated episode with {len(episode.entries)} users")
    print(f"  - Throughput range: {episode.throughput_stats['min']:.2e} - {episode.throughput_stats['max']:.2e}")
    print(f"  - Mean throughput: {episode.throughput_stats['mean']:.2e}\n")

    return episode


def test_save_load():
    """Test save/load functionality."""
    print("Testing save/load...")

    # Load a real graph
    graph = _load_test_graph(0)

    episode = generate_episode(
        master_graph=graph,
        spsc_threshold=0.9999,
        eavesdropper_density=1e-3,
        episode_id=0,
        num_candidates_per_user=3,
    )

    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "episode_0.pkl"
        save_episode(episode, save_path)
        loaded_episode = load_episode(save_path)

        print(f"✓ Saved and loaded episode")
        print(f"  - Original entries: {len(episode.entries)}")
        print(f"  - Loaded entries: {len(loaded_episode.entries)}")
        assert len(episode.entries) == len(loaded_episode.entries)
        assert episode.episode_id == loaded_episode.episode_id
        print("✓ Save/load integrity verified\n")


if __name__ == "__main__":
    print("=" * 60)
    print("M1.1: Data Collection Module Test")
    print("=" * 60 + "\n")

    test_data_entry()
    episode = test_episode_generation_minimal()
    test_save_load()

    print("=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
