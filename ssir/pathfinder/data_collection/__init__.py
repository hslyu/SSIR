"""
Data collection module for throughput predictor training.

Generates episodes with ground-truth candidate evaluations for supervised learning.
"""

from .collector import CollectionConfig, CollectionStats, EpisodeCollector
from .data_schema import (
    DataEntry,
    EpisodeDataset,
    load_data_entry,
    load_episode,
    save_data_entry,
    save_episode,
)
from .episode_generator import generate_episode
from .parallel_collector import ParallelEpisodeCollector
from .normalization import (
    NormalizationStats,
    ThroughputNormalizer,
    compute_normalization_stats,
)

__all__ = [
    "DataEntry",
    "EpisodeDataset",
    "generate_episode",
    "save_episode",
    "load_episode",
    "save_data_entry",
    "load_data_entry",
    "EpisodeCollector",
    "ParallelEpisodeCollector",
    "CollectionConfig",
    "CollectionStats",
    "NormalizationStats",
    "ThroughputNormalizer",
    "compute_normalization_stats",
]
