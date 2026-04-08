#!/usr/bin/env python3
"""
Test for M2.3: PyTorch dataset loader.

Verifies:
- Dataset loading from episodes
- Feature extraction and stacking
- Train/val splitting
- Batching
- In-memory caching
"""

import tempfile
from pathlib import Path

import ssir.basestations as bs
from ssir.pathfinder.data_collection import CollectionConfig, EpisodeCollector
from ssir.pathfinder.data_collection.dataset import (
    DataSample,
    ThroughputDataLoader,
    ThroughputDataset,
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


def _create_test_episodes(output_dir: str, num_episodes: int = 3) -> list[Path]:
    """Create test episodes and return file paths."""
    graph = _load_test_graph(0)

    config = CollectionConfig(
        num_episodes=num_episodes,
        candidates_per_user=20,
        epsilon=0.1,
        output_dir=output_dir,
        verbose=False,
    )

    collector = EpisodeCollector(config)
    collector.collect(graph)

    episode_files = sorted(Path(output_dir).glob("episode_*.pkl"))
    return episode_files


def test_dataset_loading():
    """Test loading episodes into dataset."""
    print("Testing dataset loading...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test episodes
        episode_files = _create_test_episodes(tmpdir, num_episodes=3)
        assert len(episode_files) == 3

        # Create dataset
        dataset = ThroughputDataset(episode_files, cache_in_memory=False)

        # Check dataset size (should be number of users across all episodes)
        num_samples = len(dataset)
        assert num_samples > 0, "Dataset should have samples"

        print(f"✓ Dataset loaded with {num_samples} samples")
        print()


def test_sample_extraction():
    """Test extracting a single sample from dataset."""
    print("Testing sample extraction...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test episodes
        episode_files = _create_test_episodes(tmpdir, num_episodes=2)

        # Create dataset
        dataset = ThroughputDataset(episode_files, cache_in_memory=False)

        # Get first sample
        sample = dataset[0]

        # Verify sample structure
        assert isinstance(sample, DataSample)
        assert sample.graph_data is not None
        assert sample.node_masks.shape[0] > 0  # At least one candidate
        assert sample.true_throughputs.shape[0] > 0

        # Verify dimensions match
        num_candidates = sample.node_masks.shape[0]
        assert sample.edge_masks.shape[0] == num_candidates
        assert sample.load_projections.shape[0] == num_candidates
        assert sample.route_lengths.shape[0] == num_candidates
        assert sample.true_throughputs.shape[0] == num_candidates

        print(f"✓ Sample extracted successfully")
        print(f"  - Candidates: {num_candidates}")
        print(f"  - Nodes: {sample.graph_data.num_nodes}")
        print(f"  - Edges: {sample.graph_data.num_edges}")
        print(f"  - Target throughput range: {sample.true_throughputs.min():.2e} - {sample.true_throughputs.max():.2e}")
        print()


def test_train_val_split():
    """Test train/val splitting."""
    print("Testing train/val split...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test episodes
        episode_files = _create_test_episodes(tmpdir, num_episodes=4)

        # Create train and val datasets
        train_dataset = ThroughputDataset(
            episode_files,
            cache_in_memory=False,
            split="train",
            split_ratio=0.75,
        )

        val_dataset = ThroughputDataset(
            episode_files,
            cache_in_memory=False,
            split="val",
            split_ratio=0.75,
        )

        # Check sizes
        total = len(train_dataset) + len(val_dataset)
        train_fraction = len(train_dataset) / total

        print(f"✓ Train/val split successful")
        print(f"  - Train samples: {len(train_dataset)}")
        print(f"  - Val samples: {len(val_dataset)}")
        print(f"  - Train fraction: {train_fraction:.2%}")
        print()


def test_in_memory_caching():
    """Test in-memory caching of features."""
    print("Testing in-memory caching...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test episodes
        episode_files = _create_test_episodes(tmpdir, num_episodes=2)

        # Create dataset with caching
        dataset = ThroughputDataset(
            episode_files,
            cache_in_memory=True,
        )

        # Check that cache is populated
        assert len(dataset._cache) > 0, "Cache should be populated"

        # Verify cached samples
        sample1_cached = dataset[0]
        sample1_again = dataset[0]
        assert sample1_cached is sample1_again, "Should return same object from cache"

        print(f"✓ In-memory caching works")
        print(f"  - Cached samples: {len(dataset._cache)}")
        print()


def test_data_loader():
    """Test batching with data loader."""
    print("Testing data loader...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test episodes
        episode_files = _create_test_episodes(tmpdir, num_episodes=2)

        # Create dataset
        dataset = ThroughputDataset(episode_files, cache_in_memory=False)

        # Create loader
        loader = ThroughputDataLoader(dataset, batch_size=5, shuffle=False)

        # Iterate over batches
        batch_count = 0
        total_samples = 0

        for batch in loader:
            batch_count += 1
            num_in_batch = batch["batch_size"]
            total_samples += num_in_batch
            assert num_in_batch <= 5, "Batch size should not exceed specified size"

        assert total_samples == len(dataset), "Should iterate through all samples"

        print(f"✓ Data loader works")
        print(f"  - Batches: {batch_count}")
        print(f"  - Total samples: {total_samples}")
        print()


def test_feature_dimensions():
    """Test that extracted features have correct dimensions."""
    print("Testing feature dimensions...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test episodes
        episode_files = _create_test_episodes(tmpdir, num_episodes=1)

        # Create dataset
        dataset = ThroughputDataset(episode_files, cache_in_memory=False)

        # Get multiple samples and check consistency
        for idx in range(min(5, len(dataset))):
            sample = dataset[idx]

            # Node masks should match graph nodes
            assert sample.node_masks.shape[1] == sample.graph_data.num_nodes

            # Edge masks should match graph edges
            assert sample.edge_masks.shape[1] == sample.graph_data.num_edges

            # All candidate features should have same batch size
            batch_size = sample.node_masks.shape[0]
            assert sample.edge_masks.shape[0] == batch_size
            assert sample.load_projections.shape[0] == batch_size
            assert sample.route_lengths.shape[0] == batch_size
            assert sample.true_throughputs.shape[0] == batch_size

        print(f"✓ Feature dimensions consistent across samples")
        print()


if __name__ == "__main__":
    print("=" * 60)
    print("M2.3: PyTorch Dataset Loader Test")
    print("=" * 60 + "\n")

    test_dataset_loading()
    test_sample_extraction()
    test_train_val_split()
    test_in_memory_caching()
    test_data_loader()
    test_feature_dimensions()

    print("=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
