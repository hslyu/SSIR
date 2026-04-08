#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from ssir import basestations as bs
from ssir.pathfinder.rl.train import DEFAULT_CONFIG, train_with_config

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SPSC_THRESHOLD = bs.environmental_variables.SPSC_probability


def spsc_thresholds():
    raw_logspace = np.concatenate(
        (np.logspace(-5, -4, 7, base=10)[:-1], np.logspace(-4, 0, 13, base=10))
    )
    return 1 - raw_logspace


def density_values():
    return np.logspace(-5, -1, 13, base=10)


def build_episode_conditions(args):
    conditions = []

    if args.mode in {"all", "spsc"}:
        for threshold in spsc_thresholds():
            conditions.append(
                {
                    "kind": "spsc",
                    "label": f"spsc_{threshold:.6f}",
                    "spsc_threshold": float(threshold),
                    "eavesdropper_density": None,
                    "reference_results_dir": str(args.spsc_reference_results_dir),
                    "reference_scheme": args.reference_scheme,
                    "min_reference_throughput": args.min_reference_throughput,
                    "expert_guidance_prob": args.expert_guidance_prob,
                }
            )

    if args.mode in {"all", "density"}:
        for density in density_values():
            conditions.append(
                {
                    "kind": "density",
                    "label": f"density_{density:.2e}",
                    "spsc_threshold": float(args.fixed_spsc_threshold),
                    "eavesdropper_density": float(density),
                    "reference_results_dir": str(args.density_reference_results_dir),
                    "reference_scheme": args.reference_scheme,
                    "min_reference_throughput": args.min_reference_throughput,
                    "expert_guidance_prob": args.expert_guidance_prob,
                }
            )

    return conditions


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train one RL model across mixed SPSC-threshold and eavesdropper-density "
            "episodes. Each episode samples one pre-defined condition from a shuffled cycle."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["all", "spsc", "density"],
        default="all",
        help="Which condition families to include in the mixed episode schedule.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/fast/hslyu/mmf_vs_spsc/mmf_result_1_3k/env"),
        help="Base environment directory shared across mixed-condition episodes.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=DEFAULT_CONFIG["num_episodes"],
        help="Total number of mixed-condition episodes for the single RL model.",
    )
    parser.add_argument(
        "--total-files",
        type=int,
        default=DEFAULT_CONFIG["total_files"],
        help="Maximum indexed samples to scan when the env uses <index>/ layout.",
    )
    parser.add_argument(
        "--model-name",
        default="best_model.pth",
        help="Checkpoint filename for the single mixed-condition model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results_rl_training" / "mixed",
        help="Directory where the mixed-condition checkpoint and summary are written.",
    )
    parser.add_argument(
        "--fixed-spsc-threshold",
        type=float,
        default=DEFAULT_SPSC_THRESHOLD,
        help="Fixed SPSC threshold used while density is being swept.",
    )
    parser.add_argument(
        "--condition-schedule-seed",
        type=int,
        default=0,
        help="Seed used to shuffle the per-episode condition schedule.",
    )
    parser.add_argument(
        "--reference-scheme",
        default="montecarlo",
        help="Reference scheme name used from each result.json.",
    )
    parser.add_argument(
        "--min-reference-throughput",
        type=float,
        default=1.0,
        help="Skip percent comparison for episodes whose montecarlo throughput is below this value.",
    )
    parser.add_argument(
        "--expert-guidance-prob",
        type=float,
        default=0.35,
        help="Probability of following the montecarlo reference action when available.",
    )
    parser.add_argument(
        "--reward-mode",
        choices=["min_throughput", "best_closeness", "best_advantage", "potential"],
        default="best_closeness",
        help="Per-step reward definition for RL training.",
    )
    parser.add_argument(
        "--num-candidate-workers",
        type=int,
        default=15,
        help="Number of worker processes used to evaluate candidate routes per user step.",
    )
    parser.add_argument(
        "--spsc-reference-results-dir",
        type=Path,
        default=Path("/fast/hslyu/mmf_vs_spsc/mmf_result_1_3k"),
        help="Reference result directory for SPSC episodes.",
    )
    parser.add_argument(
        "--density-reference-results-dir",
        type=Path,
        default=Path("/fast/hslyu/results_mmf_vs_density"),
        help="Reference result directory for density episodes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    episode_conditions = build_episode_conditions(args)
    config = DEFAULT_CONFIG.copy()
    config["data_dir"] = str(args.data_dir)
    config["total_files"] = args.total_files
    config["num_episodes"] = args.num_episodes
    config["model_dir"] = str(args.output_dir)
    config["model_name"] = args.model_name
    config["reference_scheme"] = args.reference_scheme
    config["condition_schedule_seed"] = args.condition_schedule_seed
    config["expert_guidance_prob"] = args.expert_guidance_prob
    config["reward_mode"] = args.reward_mode
    config["num_candidate_workers"] = args.num_candidate_workers
    config["episode_conditions"] = episode_conditions

    print(
        f"[RL] Training one model for {args.num_episodes} episodes with "
        + f"{len(episode_conditions)} mixed conditions, seed={args.condition_schedule_seed}."
    )
    metrics = train_with_config(config)

    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as fp:
        json.dump(
            {
                "scheme": "rl",
                "training_mode": "mixed_conditions_single_model",
                "num_episode_conditions": len(episode_conditions),
                "condition_schedule_seed": args.condition_schedule_seed,
                "fixed_spsc_threshold_for_density": args.fixed_spsc_threshold,
                "episode_conditions": episode_conditions,
                "metrics": metrics,
            },
            fp,
            indent=4,
        )


if __name__ == "__main__":
    main()
