"""
Multi-process episode collector for throughput predictor dataset.

Generates episodes in parallel using a worker pool for faster data collection.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import pickle
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import ssir.basestations as bs
from ssir.pathfinder.data_collection.collector import CollectionConfig, CollectionStats
from ssir.pathfinder.data_collection.data_schema import EpisodeDataset, save_episode
from ssir.pathfinder.data_collection.episode_generator import generate_episode
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _generate_single_episode(
    args: tuple,
) -> Dict:
    """
    Generate a single episode in a worker process.

    This function is called once per episode by the worker pool.

    Args:
        args: Tuple of (episode_id, graph_path, config, output_dir)

    Returns:
        Dictionary with episode statistics
    """
    import time

    episode_id, graph_path, config, output_dir = args

    # Load graph in worker process (pickle for efficiency)
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    for bs_node in graph.basestations:
        bs_node._set_transmission_and_jamming_power_density()

    # Setup local random state based on episode ID
    rng = random.Random(config.seed + episode_id)

    start_time = time.time()
    stats = {
        "episode_id": episode_id,
        "num_users": 0,
        "total_candidates": 0,
        "throughputs": [],
        "candidates_per_user": [],
        "config_key": "",
        "error": None,
    }

    try:
        # Determine episode kind
        episode_kind = (
            rng.choice(["spsc", "density"])
            if config.episode_mode == "mixed"
            else config.episode_mode
        )

        # Make a copy for this episode
        episode_graph = graph.copy()
        original_tau = bs.environmental_variables.SPSC_probability

        # Configure SPSC or density
        if episode_kind == "spsc":
            spsc = rng.choice(config.spsc_thresholds)
            density = sum(
                float(node.basestation_type.config.eavesdropper_density)
                for node in episode_graph.basestations
            ) / len(episode_graph.basestations)
            bs.environmental_variables.SPSC_probability = spsc
            config_key = f"spsc:{spsc:.4f}"
        else:
            spsc = bs.environmental_variables.SPSC_probability
            density = rng.choice(config.eavesdropper_densities)
            for bs_node in episode_graph.basestations:
                bs_node.basestation_type.config.eavesdropper_density = density
            config_key = f"density:{density:.2e}"

        # Generate episode
        episode = generate_episode(
            master_graph=episode_graph,
            spsc_threshold=spsc,
            eavesdropper_density=density,
            episode_id=episode_id,
            num_candidates_per_user=config.candidates_per_user,
            epsilon=config.epsilon,
        )

        # Save episode
        episode_path = Path(output_dir) / f"episode_{episode_id:06d}.pkl"
        save_episode(episode, episode_path)

        # Collect stats
        stats["num_users"] = len(episode.entries)
        stats["config_key"] = config_key

        for entry in episode.entries:
            stats["throughputs"].extend(entry.true_throughputs)
            stats["candidates_per_user"].append(len(entry.candidate_routes))
            stats["total_candidates"] += len(entry.candidate_routes)

    except Exception as e:
        logger.error(f"Episode {episode_id} failed: {e}")
        stats["error"] = str(e)
        raise
    finally:
        bs.environmental_variables.SPSC_probability = original_tau

    stats["runtime_seconds"] = time.time() - start_time
    return stats


class ParallelEpisodeCollector:
    """
    Multi-process episode collector.

    Generates episodes in parallel using a worker pool.
    """

    def __init__(self, config: CollectionConfig, num_workers: int = 4):
        """
        Initialize parallel collector.

        Args:
            config: Collection configuration
            num_workers: Number of worker processes
        """
        self.config = config
        self.num_workers = num_workers
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        log_path = self.output_dir / "parallel_collection.log"
        handler = logging.FileHandler(log_path)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        self._handler = handler

    def collect(self, graph: bs.IABRelayGraph) -> CollectionStats:
        """
        Generate and collect episodes in parallel.

        Each episode is generated as an independent task. Workers dynamically pick up
        the next available episode as they finish, enabling true episode-level parallelism.

        Args:
            graph: The master graph to use for all episodes

        Returns:
            CollectionStats with aggregated statistics
        """
        import time

        # Save graph to a temporary pickle for workers to load
        graph_path = self.output_dir / "_master_graph.pkl"
        with open(graph_path, "wb") as f:
            pickle.dump(graph, f)

        logger.info(
            f"Starting parallel collection: {self.config.num_episodes} episodes "
            f"using {min(self.num_workers, self.config.num_episodes)} worker processes"
        )

        start_time = time.time()

        # Create one task per episode
        episode_tasks = [
            (episode_id, str(graph_path), self.config, str(self.output_dir))
            for episode_id in range(self.config.num_episodes)
        ]

        # Only spawn as many workers as needed (avoid idle processes)
        actual_workers = min(self.num_workers, self.config.num_episodes)

        # Run workers in parallel with dynamic task distribution
        episode_results = []
        with mp.Pool(processes=actual_workers) as pool:
            for result in tqdm(
                pool.imap_unordered(
                    _generate_single_episode,
                    episode_tasks,
                ),
                total=self.config.num_episodes,
                desc="Episodes",
            ):
                episode_results.append(result)

        runtime = time.time() - start_time

        # Aggregate statistics from all episodes
        stats = self._aggregate_stats(episode_results, runtime)

        # Cleanup temporary graph file
        graph_path.unlink()

        logger.info(f"Parallel collection complete in {runtime:.2f}s")
        logger.info(f"Stats: {asdict(stats)}")

        return stats

    def close(self) -> None:
        if getattr(self, "_handler", None) is not None:
            logger.removeHandler(self._handler)
            self._handler.close()
            self._handler = None

    def _aggregate_stats(self, episode_results: List[Dict], runtime: float) -> CollectionStats:
        """
        Aggregate statistics from all episodes.

        Args:
            episode_results: List of per-episode result dictionaries
            runtime: Total collection runtime in seconds

        Returns:
            Aggregated CollectionStats
        """
        num_episodes = len(episode_results)
        total_users = 0
        total_candidates = 0
        all_throughputs = []
        all_candidates_per_user = []
        all_errors = 0
        episodes_by_config: Dict[str, int] = {}

        for episode_result in episode_results:
            if episode_result.get("error"):
                all_errors += 1
                continue

            total_users += episode_result["num_users"]
            total_candidates += episode_result["total_candidates"]
            all_throughputs.extend(episode_result["throughputs"])
            all_candidates_per_user.extend(episode_result["candidates_per_user"])

            # Count episodes by configuration
            config_key = episode_result.get("config_key", "unknown")
            episodes_by_config[config_key] = (
                episodes_by_config.get(config_key, 0) + 1
            )

        if all_errors > 0:
            logger.error(f"Total errors during collection: {all_errors}")

        if not all_throughputs:
            raise ValueError("No throughputs collected")

        stats = CollectionStats(
            num_episodes=num_episodes - all_errors,
            total_users=total_users,
            total_candidates=total_candidates,
            throughput_global_min=min(all_throughputs),
            throughput_global_max=max(all_throughputs),
            throughput_global_mean=sum(all_throughputs) / len(all_throughputs),
            candidates_per_user_mean=(
                sum(all_candidates_per_user) / len(all_candidates_per_user)
            ),
            candidates_per_user_min=min(all_candidates_per_user),
            candidates_per_user_max=max(all_candidates_per_user),
            episodes_by_config=episodes_by_config,
            runtime_seconds=runtime,
        )

        # Save stats to file
        stats_path = self.output_dir / "parallel_collection_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats.to_dict(), f, indent=2)

        return stats
