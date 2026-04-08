"""
Data schema for throughput predictor training dataset.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import ssir.basestations as bs
from ssir.pathfinder.rl.trajectory import UserRouteCandidate


@dataclass
class DataEntry:
    """
    A single data point for training the throughput predictor.

    Represents one user's route selection in an episode, with:
    - The current state of the network (partial_graph)
    - Candidate routes to choose from
    - Ground-truth throughputs for each candidate
    - The selected candidate (via epsilon-greedy)
    """

    # Episode metadata
    episode_id: int
    user_index: int  # which user in the episode (0-indexed)

    # Environment configuration
    spsc_threshold: float
    eavesdropper_density: float

    # Graph state
    master_graph: bs.IABRelayGraph  # full topology (unchanged)
    partial_graph: bs.IABRelayGraph  # state before this user's route assignment

    # Candidate routes and their evaluations
    candidate_routes: List[UserRouteCandidate]
    true_throughputs: List[float]  # throughput[i] for candidate_routes[i]

    # Selected route information
    selected_candidate_idx: int  # index of route picked by epsilon-greedy

    # Metadata for analysis
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate consistency of candidates and throughputs."""
        if len(self.candidate_routes) != len(self.true_throughputs):
            raise ValueError(
                f"Mismatch: {len(self.candidate_routes)} candidates but "
                f"{len(self.true_throughputs)} throughputs"
            )
        if not (0 <= self.selected_candidate_idx < len(self.candidate_routes)):
            raise ValueError(
                f"Invalid selected_candidate_idx {self.selected_candidate_idx}, "
                f"only {len(self.candidate_routes)} candidates available"
            )


@dataclass
class EpisodeDataset:
    """
    A complete episode dataset for training.

    Contains metadata about the episode and list of data entries.
    """

    episode_id: int
    spsc_threshold: float
    eavesdropper_density: float
    num_users: int
    entries: List[DataEntry]

    # Optional: summary statistics
    throughput_stats: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate consistency."""
        if len(self.entries) != self.num_users:
            raise ValueError(
                f"Mismatch: {self.num_users} users but {len(self.entries)} entries"
            )
        for i, entry in enumerate(self.entries):
            if entry.user_index != i:
                raise ValueError(
                    f"Entry {i} has mismatched user_index {entry.user_index}"
                )
            if entry.episode_id != self.episode_id:
                raise ValueError(
                    f"Entry {i} has mismatched episode_id {entry.episode_id}"
                )


def save_episode(episode: EpisodeDataset, output_path: str | Path) -> None:
    """
    Save an episode dataset to disk (pickle format).

    Args:
        episode: The episode dataset to save
        output_path: Path to write the pickle file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(episode, f)


def load_episode(input_path: str | Path) -> EpisodeDataset:
    """
    Load an episode dataset from disk.

    Args:
        input_path: Path to the pickle file

    Returns:
        The loaded EpisodeDataset
    """
    with open(input_path, "rb") as f:
        episode = pickle.load(f)

    if not isinstance(episode, EpisodeDataset):
        raise TypeError(f"Expected EpisodeDataset, got {type(episode)}")

    return episode


def save_data_entry(entry: DataEntry, output_path: str | Path) -> None:
    """
    Save a single data entry (for debugging/analysis).

    Args:
        entry: The data entry to save
        output_path: Path to write the pickle file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(entry, f)


def load_data_entry(input_path: str | Path) -> DataEntry:
    """Load a single data entry from disk."""
    with open(input_path, "rb") as f:
        entry = pickle.load(f)

    if not isinstance(entry, DataEntry):
        raise TypeError(f"Expected DataEntry, got {type(entry)}")

    return entry
