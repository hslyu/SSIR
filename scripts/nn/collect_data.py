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
"""

import argparse
import logging
import sys
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
        default=2000,
        help="Number of episodes to generate (default: 100)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Number of worker processes",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default="/fast/hslyu/train/exp_000/graph.pkl",
        help="Path to master graph pickle file",
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

    # Validate arguments
    graph_path = Path(args.graph)
    if not graph_path.exists():
        logger.error(f"Graph file not found: {graph_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)

    logger.info("=" * 70)
    logger.info("Throughput Predictor - Data Collection")
    logger.info("=" * 70)
    logger.info(f"Graph: {graph_path}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Episodes: {args.num_episodes}")
    logger.info(f"Workers: {args.num_workers}")
    logger.info(f"Candidates per user: {args.candidates_per_user}")
    logger.info(f"SPSC thresholds: {args.spsc_thresholds}")
    logger.info(f"Eavesdropper densities: {args.eavesdropper_densities}")
    logger.info("=" * 70 + "\n")

    # Load graph
    logger.info("Loading graph...")
    graph = bs.IABRelayGraph()
    graph.load_graph(str(graph_path))
    logger.info(f"Graph loaded: {graph}")

    # Create config
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

    # Collect data
    logger.info(f"Starting collection with {args.num_workers} worker(s)...\n")

    if args.num_workers > 1:
        collector = ParallelEpisodeCollector(config, num_workers=args.num_workers)
    else:
        collector = EpisodeCollector(config)

    stats = collector.collect(graph)

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
    logger.info(
        f"Candidates per user: {stats.candidates_per_user_min} - "
        f"{stats.candidates_per_user_max} (mean: {stats.candidates_per_user_mean:.1f})"
    )
    logger.info(f"Runtime: {stats.runtime_seconds:.2f}s")
    logger.info("=" * 70)

    # Compute normalization stats
    if args.compute_norm_stats:
        logger.info("\nComputing normalization statistics...")

        episode_files = sorted(output_dir.glob("episode_*.pkl"))
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
