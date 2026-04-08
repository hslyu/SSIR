"""
Single-threaded episode collector for throughput predictor dataset.

Generates multiple episodes and saves them to disk with aggregated statistics.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import ssir.basestations as bs
from ssir.pathfinder.data_collection.data_schema import EpisodeDataset, save_episode
from ssir.pathfinder.data_collection.episode_generator import generate_episode

logger = logging.getLogger(__name__)


@dataclass
class CollectionConfig:
    """Configuration for episode collection."""

    num_episodes: int  # Total episodes to generate
    candidates_per_user: int = 50  # Candidate routes per user
    epsilon: float = 0.1  # Epsilon-greedy exploitation probability
    output_dir: str = "data/train"  # Directory to save episodes
    spsc_thresholds: List[float] | None = None  # List of SPSC values (None = default 0.9999)
    eavesdropper_densities: List[float] | None = None  # List of densities (None = default 1e-3)
    episode_mode: str = "mixed"  # mixed, spsc, or density
    seed: int = 42  # Random seed for reproducibility
    verbose: bool = True  # Print progress

    def __post_init__(self):
        """Set defaults."""
        if self.spsc_thresholds is None:
            self.spsc_thresholds = [0.9999]
        if self.eavesdropper_densities is None:
            self.eavesdropper_densities = [1e-3]
        if self.episode_mode not in {"mixed", "spsc", "density"}:
            raise ValueError(
                "episode_mode must be one of: mixed, spsc, density"
            )


@dataclass
class CollectionStats:
    """Aggregated statistics from collected episodes."""

    num_episodes: int
    total_users: int
    total_candidates: int
    throughput_global_min: float
    throughput_global_max: float
    throughput_global_mean: float
    candidates_per_user_mean: float
    candidates_per_user_min: int
    candidates_per_user_max: int
    episodes_by_config: Dict[str, int]  # {(spsc, density): count}
    runtime_seconds: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class EpisodeCollector:
    """
    Single-threaded episode collector.

    Generates multiple episodes with various configurations and saves them to disk.
    """

    def __init__(self, config: CollectionConfig):
        """
        Initialize collector.

        Args:
            config: Collection configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        log_path = self.output_dir / "collection.log"
        handler = logging.FileHandler(log_path)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        self._handler = handler

        # Stats tracking
        self.episode_counter = 0
        self.total_users = 0
        self.total_candidates = 0
        self.throughputs: List[float] = []
        self.candidates_per_user_list: List[int] = []
        self.episodes_by_config: Dict[str, int] = {}

    def collect(self, graph: bs.IABRelayGraph) -> CollectionStats:
        """
        Generate and collect episodes.

        Args:
            graph: The master graph to use for all episodes

        Returns:
            CollectionStats with aggregated statistics
        """
        import random
        import time

        random.seed(self.config.seed)

        start_time = time.time()
        logger.info(
            f"Starting collection: {self.config.num_episodes} episodes, "
            f"SPSC={self.config.spsc_thresholds}, "
            f"densities={self.config.eavesdropper_densities}, "
            f"episode_mode={self.config.episode_mode}"
        )

        for episode_id in range(self.config.num_episodes):
            episode_kind = self._sample_episode_kind()
            episode_graph = graph.copy()
            original_tau = bs.environmental_variables.SPSC_probability

            if episode_kind == "spsc":
                spsc = random.choice(self.config.spsc_thresholds)
                density = self._default_graph_density(episode_graph)
                bs.environmental_variables.SPSC_probability = spsc
                config_key = f"spsc:{spsc:.4f}"
            else:
                spsc = bs.environmental_variables.SPSC_probability
                density = random.choice(self.config.eavesdropper_densities)
                self._apply_density_override(episode_graph, density)
                bs.environmental_variables.SPSC_probability = original_tau
                config_key = f"density:{density:.2e}"

            try:
                # Generate episode
                episode = generate_episode(
                    master_graph=episode_graph,
                    spsc_threshold=spsc,
                    eavesdropper_density=density,
                    episode_id=episode_id,
                    num_candidates_per_user=self.config.candidates_per_user,
                    epsilon=self.config.epsilon,
                )

                # Save episode
                episode_path = self.output_dir / f"episode_{episode_id:06d}.pkl"
                save_episode(episode, episode_path)

                # Update stats
                self._update_stats(episode, config_key)

                logger.info(
                    f"Collected {episode_id + 1}/{self.config.num_episodes} episodes, "
                    f"avg throughput: {self._compute_mean_throughput():.2e}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to generate episode {episode_id} with config {config_key}: {e}"
                )
                raise
            finally:
                bs.environmental_variables.SPSC_probability = original_tau

        runtime = time.time() - start_time

        # Finalize stats
        stats = self._finalize_stats(runtime)
        logger.info(f"Collection complete in {runtime:.2f}s")
        logger.info(f"Stats: {asdict(stats)}")

        return stats

    def close(self) -> None:
        if getattr(self, "_handler", None) is not None:
            logger.removeHandler(self._handler)
            self._handler.close()
            self._handler = None

    def _sample_episode_kind(self) -> str:
        if self.config.episode_mode == "mixed":
            return random.choice(["spsc", "density"])
        return self.config.episode_mode

    @staticmethod
    def _default_graph_density(graph: bs.IABRelayGraph) -> float:
        densities = [
            float(node.basestation_type.config.eavesdropper_density)
            for node in graph.basestations
        ]
        return sum(densities) / len(densities) if densities else 0.0

    @staticmethod
    def _apply_density_override(graph: bs.IABRelayGraph, density: float) -> None:
        for bs_node in graph.basestations:
            bs_node.basestation_type.config.eavesdropper_density = density

    def _update_stats(self, episode: EpisodeDataset, config_key: str) -> None:
        """Update running statistics from an episode."""
        self.episode_counter += 1
        self.total_users += len(episode.entries)

        # Throughput stats
        for entry in episode.entries:
            self.throughputs.extend(entry.true_throughputs)
            self.candidates_per_user_list.append(len(entry.candidate_routes))
            self.total_candidates += len(entry.candidate_routes)

        # Config tracking
        self.episodes_by_config[config_key] = (
            self.episodes_by_config.get(config_key, 0) + 1
        )

    def _compute_mean_throughput(self) -> float:
        """Compute current mean throughput."""
        return (
            sum(self.throughputs) / len(self.throughputs)
            if self.throughputs
            else 0.0
        )

    def _finalize_stats(self, runtime: float) -> CollectionStats:
        """Finalize and compute aggregated statistics."""
        if not self.throughputs:
            raise ValueError("No throughputs collected")

        stats = CollectionStats(
            num_episodes=self.episode_counter,
            total_users=self.total_users,
            total_candidates=self.total_candidates,
            throughput_global_min=min(self.throughputs),
            throughput_global_max=max(self.throughputs),
            throughput_global_mean=sum(self.throughputs) / len(self.throughputs),
            candidates_per_user_mean=(
                sum(self.candidates_per_user_list) / len(self.candidates_per_user_list)
            ),
            candidates_per_user_min=min(self.candidates_per_user_list),
            candidates_per_user_max=max(self.candidates_per_user_list),
            episodes_by_config=self.episodes_by_config,
            runtime_seconds=runtime,
        )

        # Save stats to file
        stats_path = self.output_dir / "collection_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats.to_dict(), f, indent=2)

        return stats
