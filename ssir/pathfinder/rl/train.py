import argparse
import copy
import json
import os
import random
from collections import Counter, deque
from pathlib import Path

import torch

from ssir import basestations as bs
from ssir.pathfinder.rl import agent, environment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CONFIG = {
    "state_channels": 17,
    "action_dim": 20,
    "window_size": 30,
    "num_episodes": 10000,
    "warmup_episodes": 0,
    "training_updates_per_episode": 10,
    "data_dir": os.environ.get("SSIR_RL_DATA_DIR", "/fast/hslyu/train"),
    "total_files": int(os.environ.get("SSIR_RL_TOTAL_FILES", "50000")),
    "embedding_channels": 128,
    "lr": 1e-4,
    "gamma": 0.99,
    "tau": 1e-3,
    "epsilon": 1.00,
    "epsilon_decay": 0.995,
    "buffer_size": 10000,
    "batch_size": 256,
    "scheduler": "cosine",
    "spsc_threshold": None,
    "eavesdropper_density": None,
    "target_bs_types": [bs.BaseStationType.MARITIME.name],
    "model_dir": "./models",
    "model_name": "best_model.pth",
    "reference_results_dir": None,
    "reference_scheme": "montecarlo",
    "min_reference_throughput": 1.0,
    "expert_guidance_prob": 0.35,
    "reward_mode": "best_closeness",
    "episode_conditions": None,
    "condition_schedule_seed": 0,
    "num_candidate_workers": 0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train the SSIR RL pathfinder.")
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_CONFIG["data_dir"],
        help=(
            "Directory containing RL training samples laid out as "
            "<data-dir>/<index>/master_graph.pkl or <data-dir>/exp_###/graph.pkl."
        ),
    )
    parser.add_argument(
        "--total-files",
        type=int,
        default=DEFAULT_CONFIG["total_files"],
        help="Maximum dataset indices to scan for indexed layouts.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=DEFAULT_CONFIG["num_episodes"],
        help="Number of RL episodes to train.",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_CONFIG["model_dir"],
        help="Directory where the best checkpoint will be stored.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_CONFIG["model_name"],
        help="Checkpoint filename for the best model.",
    )
    parser.add_argument(
        "--spsc-threshold",
        type=float,
        default=DEFAULT_CONFIG["spsc_threshold"],
        help="Override global SPSC threshold during training.",
    )
    parser.add_argument(
        "--eavesdropper-density",
        type=float,
        default=DEFAULT_CONFIG["eavesdropper_density"],
        help="Override the eavesdropper density for target base-station types.",
    )
    parser.add_argument(
        "--target-bs-type",
        action="append",
        choices=[member.name for member in bs.BaseStationType],
        help="Base-station type whose eavesdropper density will be overridden.",
    )
    parser.add_argument(
        "--reference-results-dir",
        default=DEFAULT_CONFIG["reference_results_dir"],
        help="Optional experiment result directory used for online comparison.",
    )
    parser.add_argument(
        "--reference-scheme",
        default=DEFAULT_CONFIG["reference_scheme"],
        help="Scheme key in result.json used as the training-time baseline.",
    )
    parser.add_argument(
        "--min-reference-throughput",
        type=float,
        default=DEFAULT_CONFIG["min_reference_throughput"],
        help="Ignore ratio logging/scoring when the reference throughput is below this value.",
    )
    parser.add_argument(
        "--expert-guidance-prob",
        type=float,
        default=DEFAULT_CONFIG["expert_guidance_prob"],
        help="Probability of following the reference scheme trajectory when available.",
    )
    parser.add_argument(
        "--reward-mode",
        choices=["min_throughput", "best_closeness", "best_advantage", "potential"],
        default=DEFAULT_CONFIG["reward_mode"],
        help="Per-step reward definition.",
    )
    parser.add_argument(
        "--condition-schedule-seed",
        type=int,
        default=DEFAULT_CONFIG["condition_schedule_seed"],
        help="Seed for shuffling mixed training conditions between episodes.",
    )
    parser.add_argument(
        "--num-candidate-workers",
        type=int,
        default=DEFAULT_CONFIG["num_candidate_workers"],
        help="Number of worker processes used to evaluate candidate routes per user step.",
    )
    return parser.parse_args()


def build_config(args):
    config = DEFAULT_CONFIG.copy()
    config["data_dir"] = args.data_dir
    config["total_files"] = args.total_files
    config["num_episodes"] = args.num_episodes
    config["model_dir"] = args.model_dir
    config["model_name"] = args.model_name
    config["spsc_threshold"] = args.spsc_threshold
    config["eavesdropper_density"] = args.eavesdropper_density
    if args.target_bs_type:
        config["target_bs_types"] = args.target_bs_type
    config["reference_results_dir"] = args.reference_results_dir
    config["reference_scheme"] = args.reference_scheme
    config["min_reference_throughput"] = args.min_reference_throughput
    config["expert_guidance_prob"] = args.expert_guidance_prob
    config["reward_mode"] = args.reward_mode
    config["condition_schedule_seed"] = args.condition_schedule_seed
    config["num_candidate_workers"] = args.num_candidate_workers
    return config


def validate_data_dir(data_dir):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"RL dataset directory does not exist: {data_dir}. "
            "Pass --data-dir or set SSIR_RL_DATA_DIR."
        )


def _format_reference_dirname(condition):
    if condition.get("reference_dirname"):
        return condition["reference_dirname"]

    kind = condition.get("kind", "fixed")
    if kind == "density":
        density = condition.get("eavesdropper_density")
        if density is None:
            return None
        return f"density_{density:.2e}"

    threshold = condition.get("spsc_threshold")
    if threshold is None:
        return None
    return f"spsc_{threshold:.6f}"


def _load_reference_throughput(condition, env):
    base_dir = condition.get("reference_results_dir")
    scheme = condition.get("reference_scheme", DEFAULT_CONFIG["reference_scheme"])
    min_reference_throughput = condition.get(
        "min_reference_throughput", DEFAULT_CONFIG["min_reference_throughput"]
    )
    sample_name = getattr(env, "current_sample_name", None)
    dirname = _format_reference_dirname(condition)
    if not base_dir or not dirname or not sample_name:
        return None

    result_path = Path(base_dir) / dirname / sample_name / "result.json"
    if not result_path.is_file():
        return None

    with open(result_path, "r") as fp:
        payload = json.load(fp)
    value = payload.get(scheme)
    if value is None or value < min_reference_throughput:
        return None
    return float(value)


def _base_condition_from_config(config):
    return {
        "kind": "fixed",
        "spsc_threshold": config.get("spsc_threshold"),
        "eavesdropper_density": config.get("eavesdropper_density"),
        "target_bs_types": config.get("target_bs_types"),
        "reference_results_dir": config.get("reference_results_dir"),
        "reference_scheme": config.get("reference_scheme"),
        "label": "fixed",
    }


def _build_episode_schedule(config):
    num_episodes = config["num_episodes"]
    base_conditions = config.get("episode_conditions")
    if not base_conditions:
        return [_base_condition_from_config(config) for _ in range(num_episodes)]

    normalized_conditions = []
    for condition in base_conditions:
        merged = {
            "kind": condition.get("kind", "fixed"),
            "spsc_threshold": condition.get("spsc_threshold"),
            "eavesdropper_density": condition.get("eavesdropper_density"),
            "target_bs_types": condition.get(
                "target_bs_types", config.get("target_bs_types")
            ),
            "reference_results_dir": condition.get("reference_results_dir"),
            "reference_scheme": condition.get(
                "reference_scheme", config.get("reference_scheme")
            ),
            "min_reference_throughput": condition.get(
                "min_reference_throughput", config.get("min_reference_throughput")
            ),
            "expert_guidance_prob": condition.get(
                "expert_guidance_prob", config.get("expert_guidance_prob")
            ),
            "reference_dirname": condition.get("reference_dirname"),
            "label": condition.get("label"),
        }
        normalized_conditions.append(merged)

    rng = random.Random(config.get("condition_schedule_seed", 0))
    schedule = []
    while len(schedule) < num_episodes:
        cycle = copy.deepcopy(normalized_conditions)
        rng.shuffle(cycle)
        schedule.extend(cycle)
    return schedule[:num_episodes]


def train_with_config(config):
    validate_data_dir(config["data_dir"])

    env = environment.IABRelayEnvironment(
        config["state_channels"],
        config["action_dim"],
        config["data_dir"],
        total_files=config["total_files"],
        spsc_threshold=config.get("spsc_threshold"),
        eavesdropper_density=config.get("eavesdropper_density"),
        target_bs_types=config.get("target_bs_types"),
        reward_mode=config.get("reward_mode", DEFAULT_CONFIG["reward_mode"]),
    )

    state_channels = config["state_channels"]
    action_dim = config["action_dim"]
    window_size = config["window_size"]
    num_episodes = config["num_episodes"]
    warmup_episodes = config["warmup_episodes"]
    training_updates_per_episode = config["training_updates_per_episode"]
    episode_schedule = _build_episode_schedule(config)

    IABagent = agent.Agent(
        input_channels=state_channels,
        num_action=action_dim,
        embedding_channels=config["embedding_channels"],
        criterion=torch.nn.HuberLoss(reduction="none"),
        lr=config["lr"],
        gamma=config["gamma"],
        tau=config["tau"],
        epsilon=config["epsilon"],
        epsilon_decay=config["epsilon_decay"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        expert_guidance_prob=config["expert_guidance_prob"],
        num_candidate_workers=config.get("num_candidate_workers", 0),
        device=device,
        deterministic=False,
    )

    os.makedirs(config["model_dir"], exist_ok=True)

    throughput_window = deque(maxlen=window_size)
    ratio_window = deque(maxlen=window_size)
    reward_window = deque(maxlen=window_size)
    best_score = float("-inf")
    last_metrics = {}
    last_reference_ratio = None
    condition_counter = Counter()

    try:
        for i, condition in enumerate(episode_schedule):
            state = env.reset(condition=condition)
            IABagent.set_master_graph(
                env.master_graph, expert_graph=env.reference_solution_graph
            )

            done = False
            episode_reward = 0.0
            info = {}
            while not done:
                action, action_info = IABagent.select_action(state)
                next_state, reward, done, info = env.step(
                    action, action_info=action_info
                )
                IABagent.step(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state

            if i >= warmup_episodes:
                for _ in range(training_updates_per_episode):
                    if len(IABagent.memory) > IABagent.batch_size:
                        experiences = IABagent.memory.sample()
                        IABagent.learn(experiences)

            metrics = info.get("metrics", env.prev_metrics or {})
            last_metrics = metrics
            throughput = metrics.get(
                "min_throughput", env.state.compute_network_throughput()
            )
            throughput_window.append(throughput)
            reward_window.append(episode_reward)
            avg_throughput = sum(throughput_window) / len(throughput_window)
            avg_reward = sum(reward_window) / len(reward_window)

            reference_throughput = _load_reference_throughput(condition, env)
            if reference_throughput is not None:
                last_reference_ratio = throughput / reference_throughput * 100.0
                ratio_window.append(last_reference_ratio)
                avg_reference_ratio = sum(ratio_window) / len(ratio_window)
                reference_summary = (
                    f", {condition['reference_scheme']}: {last_reference_ratio:.1f}%, "
                    + f"moving average {condition['reference_scheme']}: {avg_reference_ratio:.1f}%"
                )
            else:
                reference_summary = ""

            selection_score = (
                sum(ratio_window) / len(ratio_window) if ratio_window else avg_reward
            )
            condition_label = condition.get("label") or condition.get("kind", "fixed")
            condition_counter[condition.get("kind", "fixed")] += 1

            if reference_throughput is not None:
                mc_perf_ratio = f"{last_reference_ratio:.1f}%"
            else:
                mc_perf_ratio = "N/A"
            print(
                f"[{i+1}/{num_episodes}, {condition_label}] "
                + f"Loss: {IABagent.latest_loss:.2f}, "
                + f"Reward: {episode_reward:.2f}, "
                + f"Tput: {throughput:.2f}, "
                + f"Eps: {IABagent.epsilon:.2f}, "
                + f"MC_perf_ratio: {mc_perf_ratio}"
            )
            if selection_score > best_score:
                best_score = selection_score
                IABagent.save_network(config["model_dir"], config["model_name"])
    finally:
        IABagent.close()

    return {
        "best_score": best_score,
        "last_metrics": last_metrics,
        "num_episodes": num_episodes,
        "reference_scheme": config.get("reference_scheme"),
        "reward_mode": config.get("reward_mode"),
        "last_reference_ratio": last_reference_ratio,
        "average_reference_ratio": (
            sum(ratio_window) / len(ratio_window) if ratio_window else None
        ),
        "average_reward": sum(reward_window) / len(reward_window),
        "average_throughput": sum(throughput_window) / len(throughput_window),
        "condition_counts": dict(condition_counter),
        "model_path": os.path.join(config["model_dir"], config["model_name"]),
    }


def train(args):
    return train_with_config(build_config(args))


if __name__ == "__main__":
    train(parse_args())
