#!/usr/bin/env python3
"""
CLI for data collection: generating training episodes for throughput predictor.

Usage (single-threaded):
    python -m ssir.pathfinder.data_collection \
        --graph /path/to/graph.pkl \
        --num-episodes 1000 \
        --output-dir data/train \
        --candidates-per-user 50 \
        --epsilon 0.1

Usage (parallel with 4 workers):
    python -m ssir.pathfinder.data_collection \
        --graph /path/to/graph.pkl \
        --num-episodes 10000 \
        --output-dir data/train \
        --candidates-per-user 50 \
        --epsilon 0.1 \
        --num-workers 4 \
        --verbose
"""

import argparse
import json
import logging
from pathlib import Path

import ssir.basestations as bs
from ssir.pathfinder.data_collection import (
    CollectionConfig,
    EpisodeCollector,
    ParallelEpisodeCollector,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate training episodes for throughput predictor"
    )
    parser.add_argument(
        "--graph",
        type=str,
        required=True,
        help="Path to master graph pickle file",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of episodes to generate (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/train",
        help="Directory to save episodes (default: data/train)",
    )
    parser.add_argument(
        "--candidates-per-user",
        type=int,
        default=50,
        help="Number of candidate routes per user (default: 50)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Epsilon for epsilon-greedy selection (default: 0.1)",
    )
    parser.add_argument(
        "--spsc-thresholds",
        type=float,
        nargs="+",
        default=[0.9999],
        help="SPSC probability thresholds to sample (default: 0.9999)",
    )
    parser.add_argument(
        "--eavesdropper-densities",
        type=float,
        nargs="+",
        default=[1e-3],
        help="Eavesdropper densities to sample (default: 1e-3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel collection (default: 1)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress updates",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Verify graph file exists
    graph_path = Path(args.graph)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    logger.info(f"Loading graph from {graph_path}")
    graph = bs.IABRelayGraph()
    graph.load_graph(str(graph_path))
    logger.info(f"Loaded graph: {graph}")

    # Create config
    config = CollectionConfig(
        num_episodes=args.num_episodes,
        candidates_per_user=args.candidates_per_user,
        epsilon=args.epsilon,
        output_dir=args.output_dir,
        spsc_thresholds=args.spsc_thresholds,
        eavesdropper_densities=args.eavesdropper_densities,
        seed=args.seed,
        verbose=args.verbose,
    )

    logger.info(f"Collection config: {config}")

    # Create and run collector (parallel or single-threaded)
    if args.num_workers > 1:
        logger.info(f"Using parallel collection with {args.num_workers} workers")
        collector = ParallelEpisodeCollector(config, num_workers=args.num_workers)
    else:
        logger.info("Using single-threaded collection")
        collector = EpisodeCollector(config)

    stats = collector.collect(graph)

    # Print summary
    logger.info("=" * 60)
    logger.info("Collection Summary")
    logger.info("=" * 60)
    logger.info(f"Episodes collected: {stats.num_episodes}")
    logger.info(f"Total users: {stats.total_users}")
    logger.info(f"Total candidates: {stats.total_candidates}")
    logger.info(f"Throughput range: {stats.throughput_global_min:.2e} - {stats.throughput_global_max:.2e} bps")
    logger.info(f"Mean throughput: {stats.throughput_global_mean:.2e} bps")
    logger.info(f"Candidates per user: {stats.candidates_per_user_min} - {stats.candidates_per_user_max}")
    logger.info(f"Runtime: {stats.runtime_seconds:.2f}s")
    logger.info("=" * 60)

    logger.info(f"Episodes saved to: {config.output_dir}")
    logger.info(f"Stats saved to: {Path(config.output_dir) / 'collection_stats.json'}")


if __name__ == "__main__":
    main()
