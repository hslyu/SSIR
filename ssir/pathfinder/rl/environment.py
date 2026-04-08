import os
import random

import numpy as np

import ssir.basestations as bs

DEFAULT_SPSC_PROBABILITY = bs.environmental_variables.SPSC_probability
DEFAULT_EAVESDROPPER_DENSITIES = {
    member.name: member.config.eavesdropper_density for member in bs.BaseStationType
}


class IABRelayEnvironment:
    def __init__(
        self,
        state_dim,
        action_dim,
        env_dir,
        total_files=40000,
        spsc_threshold=None,
        eavesdropper_density=None,
        target_bs_types=None,
        reward_mode="closeness_to_best",
    ):
        self.env_dir = env_dir
        self.target_bs_types = (
            set(target_bs_types)
            if target_bs_types is not None
            else {bs.BaseStationType.MARITIME.name}
        )

        self.sample_dirs = self._discover_sample_dirs(total_files)
        print(f"Found {len(self.sample_dirs)} files out of {total_files} total files.")
        if not self.sample_dirs:
            raise ValueError(
                "No RL dataset files were found. "
                f"Expected directories like '{env_dir}/<index>/master_graph.pkl' "
                f"or '{env_dir}/exp_###/graph.pkl'."
            )
        random.shuffle(self.sample_dirs)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state = bs.IABRelayGraph()
        self.reward_mode = reward_mode
        self.current_sample_dir = None
        self.current_sample_name = None
        self.current_condition = {}
        self.prev_metrics = None
        self.prev_reward_potential = 0.0
        self.reference_solution_graph = None

        self.set_condition(
            {
                "kind": "fixed",
                "spsc_threshold": spsc_threshold,
                "eavesdropper_density": eavesdropper_density,
                "target_bs_types": list(self.target_bs_types),
            }
        )

    def set_condition(self, condition):
        merged_condition = {
            "kind": condition.get("kind", "fixed"),
            "spsc_threshold": condition.get("spsc_threshold"),
            "eavesdropper_density": condition.get("eavesdropper_density"),
            "target_bs_types": condition.get(
                "target_bs_types", list(self.target_bs_types)
            ),
            "reference_results_dir": condition.get("reference_results_dir"),
            "reference_scheme": condition.get("reference_scheme", "montecarlo"),
            "reference_dirname": condition.get("reference_dirname"),
            "label": condition.get("label"),
        }
        self.current_condition = merged_condition
        self.target_bs_types = set(merged_condition["target_bs_types"])

    def reset(self, condition=None):
        if condition is not None:
            self.set_condition(condition)

        sample_dir = random.choice(self.sample_dirs)
        self.current_sample_dir = sample_dir
        self.current_sample_name = os.path.basename(sample_dir)
        file_path = self._resolve_sample_path(sample_dir)

        self.master_graph = bs.IABRelayGraph()
        self.master_graph.load_graph(file_path)
        self._apply_environment_overrides(self.master_graph)
        self.reference_solution_graph = self._load_reference_solution_graph()
        for bs_node in self.master_graph.basestations:
            bs_node._set_transmission_and_jamming_power_density()

        self.state = self.master_graph.copy()
        self.state.reset()
        self.state.is_hop_computed = True
        self.count = 0
        self.prev_metrics = self._compute_state_metrics(self.state)
        self.prev_reward_potential = self._compute_reward_potential(self.prev_metrics)
        return self.state.copy()

    def step(self, state: bs.IABRelayGraph, action_info=None):
        self.count += 1
        self.state = state
        self.state.is_hop_computed = False
        for user in self.state.users:
            user.hops = 0
        for base in self.state.basestations:
            base.connected_user = []

        metrics = self._compute_state_metrics(self.state)
        reward_potential = self._compute_reward_potential(metrics)
        reward = self._compute_reward(metrics, action_info)
        self.prev_reward_potential = reward_potential
        self.prev_metrics = metrics

        done = metrics["connected_users"] == metrics["total_users"]
        if done:
            # Preserve the original objective at episode completion while keeping
            # intermediate rewards dense enough to differentiate actions.
            reward += 0.5 * np.log1p(metrics["min_throughput"])
            reward += 0.25 * metrics["low_log_throughput_mean"]

        info = {
            "metrics": metrics,
            "condition": dict(self.current_condition),
            "reward_mode": self.reward_mode,
        }
        if action_info is not None:
            info["candidate_reward"] = self._compute_candidate_reward_summary(action_info)
        return self.state.copy(), reward, done, info

    def _discover_sample_dirs(self, total_files):
        sample_dirs = []
        for i in range(total_files):
            indexed_dir = os.path.join(self.env_dir, str(i))
            if os.path.isdir(indexed_dir):
                sample_dirs.append(indexed_dir)

        if sample_dirs:
            return sample_dirs

        for entry in sorted(os.listdir(self.env_dir)):
            sample_dir = os.path.join(self.env_dir, entry)
            if entry.startswith("exp_") and os.path.isdir(sample_dir):
                sample_dirs.append(sample_dir)
        return sample_dirs

    def _resolve_sample_path(self, sample_dir):
        master_graph_path = os.path.join(sample_dir, "master_graph.pkl")
        legacy_graph_path = os.path.join(sample_dir, "graph.pkl")

        if os.path.isfile(master_graph_path):
            return master_graph_path
        if os.path.isfile(legacy_graph_path):
            return legacy_graph_path
        raise FileNotFoundError(f"No master graph found in sample directory: {sample_dir}")

    def _apply_environment_overrides(self, graph):
        bs.environmental_variables.SPSC_probability = DEFAULT_SPSC_PROBABILITY
        for member in bs.BaseStationType:
            member.config.eavesdropper_density = DEFAULT_EAVESDROPPER_DENSITIES[
                member.name
            ]

        threshold = self.current_condition.get("spsc_threshold")
        if threshold is not None:
            bs.environmental_variables.SPSC_probability = threshold

        density = self.current_condition.get("eavesdropper_density")
        if density is None:
            return

        for bs_node in graph.basestations:
            if bs_node.basestation_type.name in self.target_bs_types:
                bs_node.basestation_type.config.eavesdropper_density = density

    def _compute_state_metrics(self, graph):
        graph.compute_hops()

        total_users = len(graph.users)
        connected_users = sum(1 for user in graph.users if user.has_parent())
        connected_ratio = connected_users / total_users if total_users else 0.0
        connected_hops = [user.hops for user in graph.users if user.has_parent()]
        avg_connected_hops = (
            float(np.mean(connected_hops)) if connected_hops else float(total_users)
        )

        throughputs = []
        for node in graph.basestations[1:]:
            throughput = node.compute_throughput()
            if np.isfinite(throughput):
                throughputs.append(float(throughput))

        if throughputs:
            sorted_tp = sorted(throughputs)
            low_count = max(1, min(3, len(sorted_tp)))
            low_tp = sorted_tp[:low_count]
            min_throughput = sorted_tp[0]
            low_throughput_mean = float(np.mean(low_tp))
            mean_log_throughput = float(np.mean(np.log1p(sorted_tp)))
            low_log_throughput_mean = float(np.mean(np.log1p(low_tp)))
        else:
            min_throughput = 0.0
            low_throughput_mean = 0.0
            mean_log_throughput = 0.0
            low_log_throughput_mean = 0.0

        return {
            "min_throughput": min_throughput,
            "low_throughput_mean": low_throughput_mean,
            "mean_log_throughput": mean_log_throughput,
            "low_log_throughput_mean": low_log_throughput_mean,
            "connected_users": connected_users,
            "total_users": total_users,
            "connected_ratio": connected_ratio,
            "avg_connected_hops": avg_connected_hops,
        }

    def _compute_reward_potential(self, metrics):
        return (
            float(metrics["connected_users"])
            + 0.5 * float(metrics["low_log_throughput_mean"])
            + 0.2 * float(metrics["mean_log_throughput"])
            - 0.05 * float(metrics["avg_connected_hops"])
        )

    def _compute_reward(self, metrics, action_info):
        if action_info is None:
            reward_potential = self._compute_reward_potential(metrics)
            return reward_potential - self.prev_reward_potential

        candidate_summary = self._compute_candidate_reward_summary(action_info)
        if self.reward_mode == "min_throughput":
            return float(metrics["min_throughput"])

        if self.reward_mode == "best_closeness":
            return float(candidate_summary["closeness_reward"])

        if self.reward_mode == "best_advantage":
            return float(candidate_summary["advantage_reward"])

        reward_potential = self._compute_reward_potential(metrics)
        return reward_potential - self.prev_reward_potential

    def _compute_candidate_reward_summary(self, action_info):
        candidate_min_throughputs = action_info.get("candidate_min_throughputs")
        selected_index = action_info["selected_index"]
        if candidate_min_throughputs is None:
            candidate_state_list = action_info["candidate_state_list"]
            candidate_min_throughputs = [
                self._compute_state_metrics(candidate_state)["min_throughput"]
                for candidate_state in candidate_state_list
            ]
        best_value = max(candidate_min_throughputs)
        worst_value = min(candidate_min_throughputs)
        selected_value = candidate_min_throughputs[selected_index]
        if best_value <= 0:
            closeness_reward = 0.0
        else:
            closeness_reward = selected_value / best_value

        spread = best_value - worst_value
        if spread <= 1e-8:
            rank_reward = 1.0
        else:
            rank_reward = (selected_value - worst_value) / spread

        mean_value = float(np.mean(candidate_min_throughputs))
        std_value = float(np.std(candidate_min_throughputs))
        if std_value <= 1e-8:
            advantage_reward = 0.0
        else:
            advantage_reward = (selected_value - mean_value) / std_value

        return {
            "candidate_min_throughputs": candidate_min_throughputs,
            "selected_index": selected_index,
            "selected_min_throughput": selected_value,
            "best_min_throughput": best_value,
            "worst_min_throughput": worst_value,
            "closeness_reward": closeness_reward,
            "rank_reward": rank_reward,
            "advantage_reward": advantage_reward,
        }

    def _load_reference_solution_graph(self):
        base_dir = self.current_condition.get("reference_results_dir")
        scheme = self.current_condition.get("reference_scheme", "montecarlo")
        sample_name = self.current_sample_name
        dirname = self.current_condition.get("reference_dirname")
        if dirname is None:
            kind = self.current_condition.get("kind", "fixed")
            if kind == "density":
                density = self.current_condition.get("eavesdropper_density")
                if density is None:
                    return None
                dirname = f"density_{density:.2e}"
            else:
                threshold = self.current_condition.get("spsc_threshold")
                if threshold is None:
                    return None
                dirname = f"spsc_{threshold:.6f}"

        if not base_dir or not sample_name:
            return None

        solution_path = (
            os.path.join(base_dir, dirname, sample_name, f"solution_{scheme}.pkl")
        )
        if not os.path.isfile(solution_path):
            return None

        graph = bs.IABRelayGraph()
        graph.load_graph(solution_path, pkl=True)
        return graph


if __name__ == "__main__":
    env = IABRelayEnvironment(1, 1, "/fast/hslyu/train", 100)
    state = env.reset()
