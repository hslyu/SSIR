"""
PyTorch dataset for throughput predictor training.

Loads episodes from disk, extracts features, and provides batched training data.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

import ssir.basestations as bs
from ssir.pathfinder.data_collection.candidate_features import (
    encode_candidates_batch,
    stack_candidate_features,
)
from ssir.pathfinder.data_collection.data_schema import load_episode
from ssir.pathfinder.data_collection.graph_features import encode_graph_state

logger = logging.getLogger(__name__)


@dataclass
class DataSample:
    """
    A single training sample: graph state + multiple candidates + targets.

    Attributes:
        graph_data: PyTorch geometric Data object for the graph
        node_masks: [num_candidates, num_nodes]
        edge_masks: [num_candidates, num_edges]
        load_projections: [num_candidates, num_nodes]
        route_lengths: [num_candidates]
        true_throughputs: [num_candidates]
        user_id: ID of the user being routed
        episode_id: Episode ID
        candidate_indices: [num_candidates] - original candidate indices
    """

    graph_data: Data
    node_masks: torch.Tensor
    edge_masks: torch.Tensor
    load_projections: torch.Tensor
    route_lengths: torch.Tensor
    true_throughputs: torch.Tensor
    user_id: int
    episode_id: int
    candidate_indices: torch.Tensor


class ThroughputDataset(Dataset):
    """
    PyTorch dataset for throughput predictor training.

    Loads episodes from pickle files and provides training samples.
    Each sample consists of a graph state and multiple candidate routes
    with their ground-truth throughputs.
    """

    def __init__(
        self,
        episode_files: List[str | Path],
        cache_in_memory: bool = False,
        split: Optional[str] = None,
        split_ratio: float = 0.8,
        episode_cache_size: int = 16,
    ):
        """
        Initialize dataset.

        Args:
            episode_files: List of paths to episode pickle files
            cache_in_memory: Whether to cache extracted features in memory
            split: "train" (first split_ratio), "val" (rest), or None (all)
            split_ratio: Fraction to use for training (default 0.8)
        """
        self.episode_files = [Path(f) for f in episode_files]
        self.cache_in_memory = cache_in_memory
        self.split = split
        self.split_ratio = split_ratio
        self.episode_cache_size = episode_cache_size

        # Filter files by split
        if split is not None:
            split_point = int(len(self.episode_files) * split_ratio)
            if split == "train":
                self.episode_files = self.episode_files[:split_point]
            elif split == "val":
                self.episode_files = self.episode_files[split_point:]
            else:
                raise ValueError(f"Unknown split: {split}")

        # Verify files exist
        missing = [f for f in self.episode_files if not f.exists()]
        if missing:
            raise FileNotFoundError(f"Missing episode files: {missing[:5]}...")

        logger.info(f"Initialized dataset with {len(self.episode_files)} episodes ({split})")

        # In-memory cache
        self._cache: Dict[int, DataSample] = {}
        self._episode_cache: OrderedDict[int, object] = OrderedDict()
        self._metadata: List[Tuple[int, int]] = []  # (episode_idx, entry_idx)

        # Build metadata: list of all (episode_idx, entry_idx) pairs
        for episode_idx, episode_file in enumerate(
            tqdm(
                self.episode_files,
                desc=f"Indexing dataset ({split or 'all'})",
                leave=True,
                dynamic_ncols=True,
                mininterval=0.5,
            )
        ):
            episode = self._load_episode_cached(episode_idx)
            num_entries = len(episode.entries)
            for entry_idx in range(num_entries):
                self._metadata.append((episode_idx, entry_idx))

        logger.info(f"Dataset has {len(self._metadata)} samples")

        # Pre-cache if requested
        if cache_in_memory:
            logger.info("Pre-caching dataset in memory...")
            for idx in range(len(self)):
                _ = self[idx]
            logger.info(f"Cached {len(self._cache)} samples")

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self._metadata)

    def __getitem__(self, idx: int) -> DataSample:
        """
        Get a single training sample.

        Args:
            idx: Index of sample to retrieve

        Returns:
            DataSample with graph features, candidate features, and targets
        """
        # Check cache
        if idx in self._cache:
            return self._cache[idx]

        # Load from disk
        episode_idx, entry_idx = self._metadata[idx]
        episode = self._load_episode_cached(episode_idx)
        entry = episode.entries[entry_idx]

        # Extract features
        sample = self._extract_sample(entry, episode_idx)

        # Cache if enabled
        if self.cache_in_memory:
            self._cache[idx] = sample

        return sample

    def _load_episode_cached(self, episode_idx: int):
        if episode_idx in self._episode_cache:
            episode = self._episode_cache.pop(episode_idx)
            self._episode_cache[episode_idx] = episode
            return episode

        episode = load_episode(self.episode_files[episode_idx])
        self._episode_cache[episode_idx] = episode
        if len(self._episode_cache) > self.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return episode

    def _extract_sample(self, entry, episode_idx: int) -> DataSample:
        """Extract features from a data entry."""
        # Encode graph state
        graph_data = encode_graph_state(entry.master_graph, entry.partial_graph)

        # Encode candidates
        cand_features = encode_candidates_batch(
            entry.candidate_routes,
            graph_data,
            entry.master_graph,
        )

        # Stack candidate features
        stacked = stack_candidate_features(cand_features)

        # Target throughputs
        true_throughputs = torch.tensor(
            entry.true_throughputs,
            dtype=torch.float32,
        )

        # Candidate indices
        candidate_indices = torch.arange(len(entry.candidate_routes), dtype=torch.long)

        return DataSample(
            graph_data=graph_data,
            node_masks=stacked["node_masks"],
            edge_masks=stacked["edge_masks"],
            load_projections=stacked["load_projections"],
            route_lengths=stacked["route_lengths"],
            true_throughputs=true_throughputs,
            user_id=entry.candidate_routes[0].user_id,
            episode_id=episode_idx,
            candidate_indices=candidate_indices,
        )


class ThroughputDataLoader:
    """
    Utility class for loading and batching throughput dataset.

    Handles batching of variable-size graphs and candidate sets.
    """

    def __init__(
        self,
        dataset: ThroughputDataset,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        """
        Initialize data loader.

        Args:
            dataset: ThroughputDataset instance
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle samples
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Create indices
        self.indices = list(range(len(dataset)))
        if shuffle:
            import random
            random.shuffle(self.indices)

    def __len__(self) -> int:
        """Return number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """Iterate over batches."""
        for batch_idx in range(len(self)):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(self.dataset))
            batch_indices = self.indices[start:end]

            # Load samples
            samples = [self.dataset[i] for i in batch_indices]

            # Batch samples (simple stacking for now)
            batch = self._batch_samples(samples)
            yield batch

    def _batch_samples(self, samples: List[DataSample]) -> Dict:
        """Batch multiple samples together."""
        # For now, just return list of samples
        # A more sophisticated batcher would handle variable-size graphs
        return {
            "samples": samples,
            "batch_size": len(samples),
        }
