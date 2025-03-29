import os
import random

import ssir.basestations as bs


class IABRelayEnvironment:
    def __init__(
        self,
        state_dim,
        action_dim,
        env_dir,
        total_files=40000,
    ):
        self.env_dir = env_dir

        # Load the environment files
        self.file_index_list = []
        for i in range(total_files):
            file_path = os.path.join(env_dir, f"{i}/master_graph.pkl")
            check_path = os.path.join(env_dir, f"{i}/graph_genetic.pkl")
            if os.path.isfile(check_path):
                self.file_index_list.append(i)
        print(
            f"Found {len(self.file_index_list)} files out of {total_files} total files."
        )
        # Randomly shuffle the file list
        random.shuffle(self.file_index_list)

        # Initialize environment parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state = bs.IABRelayGraph()
        self.prev_throughput = 0

    def reset(self):
        file_index = random.choice(self.file_index_list)
        file_path = os.path.join(self.env_dir, f"{file_index}/master_graph.pkl")
        check_path = os.path.join(self.env_dir, f"{file_index}/graph_genetic.pkl")
        check_path2 = os.path.join(self.env_dir, f"{file_index}/graph_astar_hop.pkl")

        self.master_graph = bs.IABRelayGraph()
        self.master_graph.load_graph(file_path)
        self.genetic_graph = bs.IABRelayGraph()
        self.genetic_graph.load_graph(check_path)
        self.astar_graph = bs.IABRelayGraph()
        self.astar_graph.load_graph(check_path2)
        for bs_node in self.master_graph.basestations:
            bs_node._set_transmission_and_jamming_power_density()

        self.state = self.master_graph.copy()
        self.state.reset()
        # Avoid to compute the hop in compute_network_throughput
        self.state.is_hop_computed = True
        self.count = 0
        return self.state.copy()

    def step(self, state: bs.IABRelayGraph):
        self.count += 1
        # Set the action as new state. There's no action in this environment
        self.state = state
        self.state.is_hop_computed = False
        for user in self.state.users:
            user.hops = 0
        for base in self.state.basestations:
            base.connected_user = []

        # The maximum throughput is configured to be 40
        throughput = min(self.state.compute_network_throughput(), 80)
        # Reward is throughput * num_connected_users.
        # This is because the max-min reward does not count the number of connected users
        # Which exponentially descreases the reward as the number of users increases.
        num_connected_users = sum([1 for user in self.state.users if user.has_parent()])
        throughput = throughput * num_connected_users / len(self.state.users)
        reward = throughput - self.prev_throughput
        self.prev_throughput = throughput

        done = True
        for users in self.state.users:
            if not users.has_parent():
                done = False
                break

        # print(
        #     f"reward: {reward}, throughput: {throughput}, num_connected_users: {num_connected_users}"
        # )
        # Return the new state, reward, done flag, and an empty info dictionary
        return self.state.copy(), reward, done, {}


if __name__ == "__main__":
    env = IABRelayEnvironment(1, 1, "/fast/hslyu/train", 100)
    state = env.reset()
