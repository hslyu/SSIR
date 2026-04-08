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


def _worker_generate_episodes(
    worker_id: int,
    num_episodes: int,
    graph_path: str,
    config: CollectionConfig,
    output_dir: str,
) -> Dict:
    """
    Worker function for generating episodes in a separate process.

    Args:
        worker_id: Unique worker ID
        num_episodes: Number of episodes this worker should generate
        graph_path: Path to master graph pickle file
        config: Collection configuration
        output_dir: Directory to save episodes

    Returns:
        Dictionary with worker statistics
    """
    import random
    import time

    # Load graph in worker process
    graph = bs.IABRelayGraph()
    graph.load_graph(graph_path)
    for bs_node in graph.basestations:
        bs_node._set_transmission_and_jamming_power_density()

    # Setup local random state
    random.seed(config.seed + worker_id)

    # Track stats
    stats = {
        "worker_id": worker_id,
        "episodes_generated": 0,
        "total_users": 0,
        "total_candidates": 0,
        "throughputs": [],
        "candidates_per_user": [],
        "episodes_by_config": {},
        "errors": 0,
    }

    start_time = time.time()

    for local_episode_id in range(num_episodes):
        try:
            episode_kind = random.choice(["spsc", "density"]) if config.episode_mode == "mixed" else config.episode_mode
            episode_graph = graph.copy()
            original_tau = bs.environmental_variables.SPSC_probability

            if episode_kind == "spsc":
                spsc = random.choice(config.spsc_thresholds)
                density = sum(
                    float(node.basestation_type.config.eavesdropper_density)
                    for node in episode_graph.basestations
                ) / len(episode_graph.basestations)
                bs.environmental_variables.SPSC_probability = spsc
                config_key = f"spsc:{spsc:.4f}"
            else:
                spsc = bs.environmental_variables.SPSC_probability
                density = random.choice(config.eavesdropper_densities)
                for bs_node in episode_graph.basestations:
                    bs_node.basestation_type.config.eavesdropper_density = density
                bs.environmental_variables.SPSC_probability = original_tau
                config_key = f"density:{density:.2e}"

            # Generate episode
            episode = generate_episode(
                master_graph=episode_graph,
                spsc_threshold=spsc,
                eavesdropper_density=density,
                episode_id=-1,  # Will be reassigned later
                num_candidates_per_user=config.candidates_per_user,
                epsilon=config.epsilon,
            )

            # Save episode (temporary location with worker ID)
            global_episode_id = worker_id * num_episodes + local_episode_id
            episode_path = Path(output_dir) / f"episode_{global_episode_id:06d}.pkl"
            save_episode(episode, episode_path)

            # Update stats
            stats["episodes_generated"] += 1
            stats["total_users"] += len(episode.entries)

            for entry in episode.entries:
                stats["throughputs"].extend(entry.true_throughputs)
                stats["candidates_per_user"].append(len(entry.candidate_routes))
                stats["total_candidates"] += len(entry.candidate_routes)

            stats["episodes_by_config"][config_key] = (
                stats["episodes_by_config"].get(config_key, 0) + 1
            )

        except Exception as e:
            logger.error(
                f"Worker {worker_id} failed on local episode {local_episode_id}: {e}"
            )
            stats["errors"] += 1
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

    def collect(self, graph: bs.IABRelayGraph) -> CollectionStats:
        """
        Generate and collect episodes in parallel.

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
            f"using {self.num_workers} workers"
        )

        # Calculate episodes per worker
        episodes_per_worker = self.config.num_episodes // self.num_workers
        remainder = self.config.num_episodes % self.num_workers

        start_time = time.time()

        # Create worker tasks
        worker_tasks = []
        for worker_id in range(self.num_workers):
            num_episodes_for_worker = episodes_per_worker + (
                1 if worker_id < remainder else 0
            )
            worker_tasks.append(
                (
                    worker_id,
                    num_episodes_for_worker,
                    str(graph_path),
                    self.config,
                    str(self.output_dir),
                )
            )

        # Run workers in parallel
        logger.info(f"Spawning {self.num_workers} workers...")
        with mp.Pool(processes=self.num_workers) as pool:
            worker_results = list(
                tqdm(
                    pool.starmap(
                        _worker_generate_episodes,
                        worker_tasks,
                    ),
                    total=len(worker_tasks),
                    desc="Worker progress",
                )
            )

        runtime = time.time() - start_time

        # Aggregate statistics from workers
        stats = self._aggregate_stats(worker_results, runtime)

        # Cleanup temporary graph file
        graph_path.unlink()

        logger.info(f"Parallel collection complete in {runtime:.2f}s")
        logger.info(f"Stats: {asdict(stats)}")

        return stats

    def _aggregate_stats(self, worker_results: List[Dict], runtime: float) -> CollectionStats:
        """Aggregate statistics from all workers."""
        num_episodes = 0
        total_users = 0
        total_candidates = 0
        all_throughputs = []
        all_candidates_per_user = []
        all_errors = 0
        episodes_by_config: Dict[str, int] = {}

        for worker_result in worker_results:
            num_episodes += worker_result["episodes_generated"]
            total_users += worker_result["total_users"]
            total_candidates += worker_result["total_candidates"]
            all_throughputs.extend(worker_result["throughputs"])
            all_candidates_per_user.extend(worker_result["candidates_per_user"])
            all_errors += worker_result["errors"]

            # Merge configuration counts
            for config_key, count in worker_result["episodes_by_config"].items():
                episodes_by_config[config_key] = (
                    episodes_by_config.get(config_key, 0) + count
                )

        if all_errors > 0:
            logger.error(f"Total errors during collection: {all_errors}")

        if not all_throughputs:
            raise ValueError("No throughputs collected")

        stats = CollectionStats(
            num_episodes=num_episodes,
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
