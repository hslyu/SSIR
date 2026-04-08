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
import logging
from pathlib import Path

import numpy as np

import ssir.basestations as bs
from ssir.pathfinder.data_collection import (
    CollectionConfig,
    EpisodeCollector,
    ParallelEpisodeCollector,
    compute_normalization_stats,
)
from ssir.pathfinder.data_collection.data_schema import load_episode

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
        default=13,
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

    all_stats = []
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

    for graph_path, num_graph_episodes in zip(graph_files, episode_counts):
        if num_graph_episodes <= 0:
            continue

        graph = bs.IABRelayGraph()
        graph.load_graph(str(graph_path))

        graph_output_dir = output_dir / graph_path.parent.name
        config = CollectionConfig(
            num_episodes=num_graph_episodes,
            candidates_per_user=args.candidates_per_user,
            epsilon=args.epsilon,
            output_dir=str(graph_output_dir),
            spsc_thresholds=args.spsc_thresholds,
            eavesdropper_densities=args.eavesdropper_densities,
            seed=args.seed,
            verbose=args.verbose,
        )

        logger.info(
            f"Collecting {num_graph_episodes} episodes from {graph_path.parent.name}..."
        )
        if args.num_workers > 1:
            collector = ParallelEpisodeCollector(config, num_workers=args.num_workers)
        else:
            collector = EpisodeCollector(config)

        try:
            stats = collector.collect(graph)
            all_stats.append(stats)
        finally:
            collector.close()

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Collection Complete!")
    logger.info("=" * 70)
    if all_stats:
        total_episodes = sum(stat.num_episodes for stat in all_stats)
        total_users = sum(stat.total_users for stat in all_stats)
        total_candidates = sum(stat.total_candidates for stat in all_stats)
        throughput_min = min(stat.throughput_global_min for stat in all_stats)
        throughput_max = max(stat.throughput_global_max for stat in all_stats)
        throughput_mean = sum(
            stat.throughput_global_mean * stat.total_candidates for stat in all_stats
        ) / max(sum(stat.total_candidates for stat in all_stats), 1)
        logger.info(f"Episodes collected: {total_episodes}")
        logger.info(f"Total users: {total_users}")
        logger.info(f"Total candidates: {total_candidates}")
        logger.info(
            f"Throughput range: {throughput_min:.2e} - {throughput_max:.2e} bps"
        )
        logger.info(f"Mean throughput: {throughput_mean:.2e} bps")
    else:
        logger.info("No episodes were collected.")
    logger.info(f"Runtime: {sum(stat.runtime_seconds for stat in all_stats):.2f}s")
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
