#!/usr/bin/env python3
"""
Standalone script for collecting training data for throughput predictor.

Usage:
    # Quick test (100 episodes, single-threaded)
    python scripts/collect_data.py --num-episodes 100

    # Production (10K episodes, 4 workers)
    python scripts/collect_data.py \
        --num-episodes 10000 \
        --num-workers 4 \
        --output-dir data/train_large \
        --candidates-per-user 50 \
        --verbose

    # With multiple SPSC/density configurations
    python scripts/collect_data.py \
        --num-episodes 5000 \
        --num-workers 8 \
        --spsc-thresholds 0.99 0.999 0.9999 \
        --eavesdropper-densities 1e-4 1e-3 1e-2

    # Use all graphs under /fast/hslyu/train/exp_*/graph.pkl
    python scripts/collect_data.py --graph-dir /fast/hslyu/train
"""

import argparse
import json
import logging
import multiprocessing as mp
import pickle
import random
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

import ssir.basestations as bs
from ssir.pathfinder.data_collection import (
    CollectionConfig,
    CollectionStats,
)
from ssir.pathfinder.data_collection.data_schema import load_episode, save_episode
from ssir.pathfinder.data_collection.episode_generator import generate_episode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def discover_graph_files(graph_dir: Path) -> list[Path]:
    if graph_dir.is_file():
        return [graph_dir]
    if not graph_dir.is_dir():
        raise FileNotFoundError(f"Graph directory not found: {graph_dir}")
    graph_files = sorted(graph_dir.glob("exp_*/graph.pkl"))
    if not graph_files:
        raise FileNotFoundError(
            f"No graph.pkl files found under {graph_dir}/exp_*/graph.pkl"
        )
    return graph_files


def split_episode_budget(total_episodes: int, num_graphs: int) -> list[int]:
    if num_graphs <= 0:
        return []
    base = total_episodes // num_graphs
    remainder = total_episodes % num_graphs
    return [base + (1 if idx < remainder else 0) for idx in range(num_graphs)]


def _generate_episode_task(args: tuple) -> dict:
    """
    Generate a single episode in a worker process.

    The task loads its graph, samples the episode configuration, generates the
    episode, and writes it to the graph-specific output directory.
    """
    episode_id, graph_path, config, output_dir = args

    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    for bs_node in graph.basestations:
        bs_node._set_transmission_and_jamming_power_density()

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
        episode_kind = (
            rng.choice(["spsc", "density"])
            if config.episode_mode == "mixed"
            else config.episode_mode
        )

        episode_graph = graph.copy()
        original_tau = bs.environmental_variables.SPSC_probability

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

        episode = generate_episode(
            master_graph=episode_graph,
            spsc_threshold=spsc,
            eavesdropper_density=density,
            episode_id=episode_id,
            num_candidates_per_user=config.candidates_per_user,
            epsilon=config.epsilon,
        )

        episode_path = Path(output_dir) / f"episode_{episode_id:06d}.pkl"
        save_episode(episode, episode_path)

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


def _aggregate_episode_results(results: list[dict], runtime: float) -> CollectionStats:
    """
    Aggregate per-episode results into dataset-level statistics.
    """
    total_users = 0
    total_candidates = 0
    all_throughputs = []
    all_candidates_per_user = []
    episodes_by_config: dict[str, int] = {}

    for result in results:
        if result.get("error"):
            continue

        total_users += result["num_users"]
        total_candidates += result["total_candidates"]
        all_throughputs.extend(result["throughputs"])
        all_candidates_per_user.extend(result["candidates_per_user"])

        config_key = result.get("config_key", "unknown")
        episodes_by_config[config_key] = episodes_by_config.get(config_key, 0) + 1

    if not all_throughputs:
        raise ValueError("No throughputs collected")

    return CollectionStats(
        num_episodes=len(results),
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


def main():
    parser = argparse.ArgumentParser(
        description="Collect training data for throughput predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    default_spsc_thresholds = [
        float(x)
        for x in np.concatenate(
            (
                np.logspace(-5, -4, 7, base=10)[:-1],
                np.logspace(-4, 0, 13, base=10),
            )
        )
    ]
    default_eavesdropper_densities = [
        float(x) for x in np.logspace(-5, -1, 13, base=10)
    ]

    # Data collection arguments
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of episodes to generate (default: 100)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=60,
        help="Number of worker processes",
    )
    parser.add_argument(
        "--graph-dir",
        type=str,
        default="/fast/hslyu/train",
        help="Directory containing exp_xxx/graph.pkl files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/train",
        help="Directory to save episodes (default: data/train)",
    )

    # Candidate generation arguments
    parser.add_argument(
        "--candidates-per-user",
        type=int,
        default=300,
        help="Number of candidate routes per user",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Epsilon for epsilon-greedy selection (default: 0.1)",
    )

    # Environment configuration arguments
    parser.add_argument(
        "--spsc-thresholds",
        type=float,
        nargs="+",
        default=default_spsc_thresholds,
        help="SPSC probability thresholds to sample (default: full mmf_vs_spsc range)",
    )
    parser.add_argument(
        "--eavesdropper-densities",
        type=float,
        nargs="+",
        default=default_eavesdropper_densities,
        help="Eavesdropper densities to sample (default: full mmf_vs_density range)",
    )

    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress updates",
    )
    parser.add_argument(
        "--compute-norm-stats",
        action="store_true",
        default=True,
        help="Compute and save normalization stats (default: True)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    graph_dir = Path(args.graph_dir)
    graph_files = discover_graph_files(graph_dir)
    episode_counts = split_episode_budget(args.num_episodes, len(graph_files))
    config = CollectionConfig(
        num_episodes=args.num_episodes,
        candidates_per_user=args.candidates_per_user,
        epsilon=args.epsilon,
        output_dir=str(output_dir),
        spsc_thresholds=args.spsc_thresholds,
        eavesdropper_densities=args.eavesdropper_densities,
        seed=args.seed,
        verbose=args.verbose,
    )

    logger.info("=" * 70)
    logger.info("Throughput Predictor - Data Collection")
    logger.info("=" * 70)
    logger.info(f"Graph dir: {graph_dir}")
    logger.info(f"Graphs found: {len(graph_files)}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Episodes total: {args.num_episodes}")
    logger.info(f"Workers: {args.num_workers}")
    logger.info(f"Candidates per user: {args.candidates_per_user}")
    logger.info(f"SPSC thresholds: {args.spsc_thresholds}")
    logger.info(f"Eavesdropper densities: {args.eavesdropper_densities}")
    logger.info("=" * 70 + "\n")

    total_assigned = sum(episode_counts)
    logger.info(
        f"Distributing {total_assigned} episodes across {len(graph_files)} graphs"
    )
    logger.info(
        "Per-graph episode counts: "
        + ", ".join(
            f"{graph_path.parent.name}={count}"
            for graph_path, count in zip(graph_files, episode_counts)
            if count > 0
        )
    )

    episode_tasks = []
    episode_id = 0
    for graph_path, num_graph_episodes in zip(graph_files, episode_counts):
        if num_graph_episodes <= 0:
            continue

        graph_output_dir = output_dir / graph_path.parent.name
        graph_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Assigning {num_graph_episodes} episodes to {graph_path.parent.name}"
        )
        for _ in range(num_graph_episodes):
            episode_tasks.append(
                (episode_id, str(graph_path), config, str(graph_output_dir))
            )
            episode_id += 1

    logger.info(f"Total episode tasks: {len(episode_tasks)}")

    start_time = time.time()
    if len(episode_tasks) == 0:
        episode_results = []
    elif args.num_workers <= 1 or len(episode_tasks) == 1:
        episode_results = [_generate_episode_task(task) for task in episode_tasks]
    else:
        actual_workers = min(args.num_workers, len(episode_tasks))
        logger.info(f"Using {actual_workers} worker processes")
        with mp.Pool(processes=actual_workers) as pool:
            episode_results = list(
                tqdm(
                    pool.imap_unordered(
                        _generate_episode_task, episode_tasks, chunksize=1
                    ),
                    total=len(episode_tasks),
                    desc="Episodes",
                )
            )

    runtime = time.time() - start_time
    if episode_results:
        stats = _aggregate_episode_results(episode_results, runtime)
    else:
        stats = CollectionStats(
            num_episodes=0,
            total_users=0,
            total_candidates=0,
            throughput_global_min=0.0,
            throughput_global_max=0.0,
            throughput_global_mean=0.0,
            candidates_per_user_mean=0.0,
            candidates_per_user_min=0,
            candidates_per_user_max=0,
            episodes_by_config={},
            runtime_seconds=runtime,
        )

    stats_path = output_dir / "collection_stats.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats.to_dict(), f, indent=2)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Collection Complete!")
    logger.info("=" * 70)
    logger.info(f"Episodes collected: {stats.num_episodes}")
    logger.info(f"Total users: {stats.total_users}")
    logger.info(f"Total candidates: {stats.total_candidates}")
    logger.info(
        f"Throughput range: {stats.throughput_global_min:.2e} - "
        f"{stats.throughput_global_max:.2e} bps"
    )
    logger.info(f"Mean throughput: {stats.throughput_global_mean:.2e} bps")
    logger.info(f"Runtime: {stats.runtime_seconds:.2f}s")
    logger.info(f"Stats saved to: {stats_path}")
    logger.info("=" * 70)

    # Compute normalization stats
    if args.compute_norm_stats:
        logger.info("\nComputing normalization statistics...")

        episode_files = sorted(output_dir.rglob("episode_*.pkl"))
        all_throughputs = []

        for episode_file in episode_files:
            episode = load_episode(episode_file)
            for entry in episode.entries:
                all_throughputs.extend(entry.true_throughputs)

        if not all_throughputs:
            logger.info("No episodes found for normalization stats; skipping.")
            logger.info(f"\nEpisodes saved to: {output_dir}")
            logger.info("\nNext step: Train the model with:")
            logger.info(f"  python scripts/train_nn.py --data-dir {output_dir}")
            return

        from ssir.pathfinder.data_collection.normalization import (
            compute_normalization_stats,
        )

        norm_stats = compute_normalization_stats(all_throughputs)
        norm_stats.save(output_dir / "norm_stats.json")

        logger.info(f"Normalization stats saved to {output_dir / 'norm_stats.json'}")
        logger.info(
            f"Log-throughput range: {norm_stats.log_throughput_min:.4f} - "
            f"{norm_stats.log_throughput_max:.4f}"
        )
        logger.info(f"Log-std: {norm_stats.log_throughput_std:.4f}")

    logger.info(f"\nEpisodes saved to: {output_dir}")
    logger.info("\nNext step: Train the model with:")
    logger.info(f"  python scripts/train_nn.py --data-dir {output_dir}")


if __name__ == "__main__":
    main()
