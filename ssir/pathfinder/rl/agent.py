import multiprocessing as mp
import os
import random
from collections import deque, namedtuple
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch

import ssir.basestations as bs
from ssir.pathfinder import astar, rl, utils

# Experience tuple for storing transitions
Experience = namedtuple(
    "Experience", field_names=["state", "action", "reward", "next_state", "done"]
)


def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def _candidate_eval_targets_from_path(graph, path):
    return [
        node_id
        for node_id in path[1:-1]
        if isinstance(graph.nodes[node_id], bs.BaseStation)
    ]


def _evaluate_candidate_worker(payload):
    state, path = payload
    candidate_state = state.copy()
    utils.get_aborescence_graph(candidate_state, list(path))
    eval_targets = _candidate_eval_targets_from_path(candidate_state, path)
    if eval_targets:
        value = float(candidate_state.compute_network_throughput(eval_targets))
        if not np.isfinite(value):
            value = float(candidate_state.compute_network_throughput())
    else:
        value = float(candidate_state.compute_network_throughput())
    if not np.isfinite(value):
        value = 0.0
    edge_signature = tuple(
        sorted(
            (from_node_id, to_node_id)
            for from_node_id, neighbors in candidate_state.adjacency_list.items()
            for to_node_id in neighbors
        )
    )
    return candidate_state, eval_targets, value, edge_signature


class Agent:
    def __init__(
        self,
        input_channels,
        num_action,
        embedding_channels,
        criterion,
        lr,
        gamma,
        tau,
        epsilon,
        epsilon_decay,
        min_epsilon=0.01,
        soft_update_period=1,
        buffer_size=10000,
        batch_size=512,
        n_step=5,
        deterministic=False,
        expert_guidance_prob=0.35,
        num_candidate_workers=0,
        device: str | torch.device = "",
    ):
        if device == "":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Graph variables
        self.master_graph: bs.IABRelayGraph | None = None
        self.expert_graph: bs.IABRelayGraph | None = None
        self.predecessors_list = []
        self.candidate_path_cache: Dict[int, List[List[int]]] = {}
        self.expert_path_cache: Dict[int, List[int]] = {}

        # Agent variables
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        # Network variables
        self.input_channels = input_channels
        self.num_action = num_action
        self.target_q_network = rl.network.GraphQNetwork(
            input_channels, embedding_channels
        ).to(device)
        self.target_q_network.apply(xavier_init)
        self.local_q_network = rl.network.GraphQNetwork(
            input_channels, embedding_channels
        ).to(device)
        self.local_q_network.apply(xavier_init)
        self.update_count = 0
        self.soft_update_period = soft_update_period

        # Learning variables
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.step_count = 0
        self.optimizer = torch.optim.Adam(self.local_q_network.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=400, eta_min=5e-4
        )
        self.criterion = criterion

        self.latest_loss = -1
        self.memory = PrioritizedNStepReplayBuffer(
            buffer_size, batch_size, n_step, gamma, device=device
        )
        self.batch_size = batch_size
        self.deterministic = deterministic
        self.expert_guidance_prob = expert_guidance_prob
        self.num_candidate_workers = num_candidate_workers
        self._candidate_pool = None
        self.device = device

    def set_master_graph(
        self,
        master_graph: bs.IABRelayGraph,
        expert_graph: bs.IABRelayGraph | None = None,
    ):
        self.master_graph = master_graph
        self.expert_graph = expert_graph
        self.predecessors_list = []
        _, predecessors = astar.a_star(self.master_graph, metric="hop")
        self.predecessors_list.append(predecessors)
        _, predecessors = astar.a_star(self.master_graph, metric="distance")
        self.predecessors_list.append(predecessors)
        _, predecessors = astar.a_star(self.master_graph, metric="spectral_efficiency")
        self.predecessors_list.append(predecessors)
        for _ in range(self.num_action - 2):
            _, predecessors = astar.a_star(self.master_graph, metric="random")
            self.predecessors_list.append(predecessors)
        self.candidate_path_cache = self._build_candidate_path_cache()
        self.expert_path_cache = self._build_expert_path_cache()

    def load_network(self, path):
        self.target_q_network.load_state_dict(torch.load(path))
        self.local_q_network.load_state_dict(torch.load(path))

    def save_network(self, path="./", filename="model.pth"):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.target_q_network.state_dict(), os.path.join(path, filename))

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def step(
        self,
        state: bs.IABRelayGraph,
        action,
        reward: float,
        next_state: bs.IABRelayGraph,
        done: bool,
    ):
        self.memory.add(
            self._convert_state_to_data(state),
            action,
            reward,
            self._convert_state_to_data(next_state),
            done,
        )

    def act(self, state: bs.IABRelayGraph):
        next_state, _ = self.select_action(state)
        return next_state

    def select_action(self, state: bs.IABRelayGraph):
        """
        Returns actions for given state as per current policy.

        Params
        ======
            state (bs.IABRelayGraph): current state graph
        """
        # Identify the target user: choose a user that has no parent (i.e., not yet connected)
        target_user: bs.User | None = None
        for user in state.users:
            if not user.has_parent():
                target_user = user
                break

        if target_user is None:
            raise ValueError("All users are connected to a base station.")

        (
            candidate_state_list,
            candidate_eval_targets,
            candidate_min_throughputs,
            expert_index,
        ) = self._build_candidate_states(state, target_user)

        if (
            not self.deterministic
            and expert_index is not None
            and random.random() < self.expert_guidance_prob
        ):
            return candidate_state_list[expert_index], {
                "target_user_id": target_user.get_id(),
                "candidate_state_list": candidate_state_list,
                "candidate_eval_targets": candidate_eval_targets,
                "candidate_min_throughputs": candidate_min_throughputs,
                "selected_index": expert_index,
                "expert_index": expert_index,
                "selection_mode": "expert",
            }

        if random.random() < self.epsilon:
            index = random.randint(0, len(candidate_state_list) - 1)
            return candidate_state_list[index], {
                "target_user_id": target_user.get_id(),
                "candidate_state_list": candidate_state_list,
                "candidate_eval_targets": candidate_eval_targets,
                "candidate_min_throughputs": candidate_min_throughputs,
                "selected_index": index,
                "expert_index": expert_index,
                "selection_mode": "epsilon",
            }

        if self.deterministic:
            best_index = int(np.argmax(candidate_min_throughputs))
        else:
            candidate_data_list = [
                self._convert_state_to_data(s) for s in candidate_state_list
            ]
            candidate_batch = Batch.from_data_list(candidate_data_list).to(  # type: ignore
                self.device
            )
            self.local_q_network.eval()
            with torch.no_grad():
                q_values = self.local_q_network(candidate_batch)
            self.local_q_network.train()
            best_index = int(torch.argmax(q_values, dim=0).item())

        best_state = candidate_state_list[best_index]
        return best_state, {
            "target_user_id": target_user.get_id(),
            "candidate_state_list": candidate_state_list,
            "candidate_eval_targets": candidate_eval_targets,
            "candidate_min_throughputs": candidate_min_throughputs,
            "selected_index": best_index,
            "expert_index": expert_index,
            "selection_mode": "policy",
        }

    def _build_candidate_states(
        self, state: bs.IABRelayGraph, target_user: bs.User
    ) -> tuple[List[bs.IABRelayGraph], List[List[int]], List[float], int | None]:
        candidate_state_list: List[bs.IABRelayGraph] = []
        candidate_eval_targets: List[List[int]] = []
        candidate_min_throughputs: List[float] = []
        seen_edge_sets = set()
        expert_index = None

        cached_paths = self.candidate_path_cache.get(target_user.get_id(), [])
        for candidate_state, eval_targets, value, edge_signature in self._evaluate_candidates(
            state, cached_paths
        ):
            if edge_signature in seen_edge_sets:
                continue
            seen_edge_sets.add(edge_signature)
            candidate_state_list.append(candidate_state)
            candidate_eval_targets.append(eval_targets)
            candidate_min_throughputs.append(value)

        expert_path = self.expert_path_cache.get(target_user.get_id())
        if expert_path is not None:
            expert_candidate, eval_targets, value, edge_signature = self._evaluate_one_candidate(
                state, expert_path
            )
            if edge_signature not in seen_edge_sets:
                expert_index = len(candidate_state_list)
                candidate_state_list.append(expert_candidate)
                candidate_eval_targets.append(eval_targets)
                candidate_min_throughputs.append(value)
            else:
                for idx, candidate in enumerate(candidate_state_list):
                    if self._edge_signature(candidate) == edge_signature:
                        expert_index = idx
                        break

        return (
            candidate_state_list,
            candidate_eval_targets,
            candidate_min_throughputs,
            expert_index,
        )

    def _build_candidate_path_cache(self) -> Dict[int, List[List[int]]]:
        if self.master_graph is None:
            return {}

        all_paths = utils.get_all_shortest_paths(self.master_graph, self.predecessors_list)
        deduped: Dict[int, List[List[int]]] = {}
        for user_id, path_list in all_paths.items():
            unique_paths = []
            seen = set()
            for path in path_list:
                path_key = tuple(path)
                if path_key in seen:
                    continue
                seen.add(path_key)
                unique_paths.append(path)
            deduped[user_id] = unique_paths
        return deduped

    def _build_expert_path_cache(self) -> Dict[int, List[int]]:
        if self.expert_graph is None:
            return {}

        expert_paths: Dict[int, List[int]] = {}
        for user in self.expert_graph.users:
            if not user.has_parent():
                continue
            current_node = self.expert_graph.nodes[user.get_id()]
            visited = set()
            path = [user.get_id()]
            while current_node.has_parent():
                if current_node.get_id() in visited:
                    path = []
                    break
                visited.add(current_node.get_id())
                parent = current_node.get_parent()[0]
                path.append(parent.get_id())
                current_node = parent
            if path:
                path.reverse()
                expert_paths[user.get_id()] = path
        return expert_paths

    def _evaluate_one_candidate(self, state: bs.IABRelayGraph, path: List[int]):
        candidate_state = self.get_aborescence_graph(state, list(path))
        edge_signature = self._edge_signature(candidate_state)
        eval_targets = _candidate_eval_targets_from_path(candidate_state, path)
        if eval_targets:
            value = float(candidate_state.compute_network_throughput(eval_targets))
            if not np.isfinite(value):
                value = float(candidate_state.compute_network_throughput())
        else:
            value = float(candidate_state.compute_network_throughput())
        if not np.isfinite(value):
            value = 0.0
        return candidate_state, eval_targets, value, edge_signature

    def _get_candidate_pool(self):
        if self.num_candidate_workers <= 1:
            return None
        if self._candidate_pool is None:
            ctx = mp.get_context("spawn")
            self._candidate_pool = ctx.Pool(processes=self.num_candidate_workers)
        return self._candidate_pool

    def _evaluate_candidates(self, state: bs.IABRelayGraph, paths: List[List[int]]):
        pool = self._get_candidate_pool()
        if pool is None:
            return [self._evaluate_one_candidate(state, path) for path in paths]
        payloads = [(state, path) for path in paths]
        return pool.map(_evaluate_candidate_worker, payloads)

    def close(self):
        if self._candidate_pool is not None:
            self._candidate_pool.close()
            self._candidate_pool.join()
            self._candidate_pool = None

    def _edge_signature(self, graph: bs.IABRelayGraph):
        return tuple(
            sorted(
                (from_node_id, to_node_id)
                for from_node_id, neighbors in graph.adjacency_list.items()
                for to_node_id in neighbors
            )
        )

    def learn(self, experiences):
        """
        Update value parameters using a batch of experience tuples.

        For each experience tuple (s, r, s', d), we define the target value as:
            V_target(s) = r + γ * V_target(s') * (1 - d)
        and the local network predicts:
            V_local(s) = self.local_q_network(embedding(s))
        The loss is the mean squared error between V_local(s) and V_target(s).
        """
        # train mode
        self.local_q_network.train()

        # Unpack experiences including PER-related indices and weights
        states, actions, rewards, next_states, dones, indices, weights = experiences

        # Convert lists of torch_geometric Data objects to batched graphs
        batched_states = Batch.from_data_list(states).to(self.device)  # type: ignore
        batched_next_states = Batch.from_data_list(next_states).to(self.device)  # type: ignore

        # Obtain embeddings for states and next states via the graph embedder
        # state_embeddings = self.graph_embedder(batched_states)
        # next_state_embeddings = self.graph_embedder(batched_next_states)

        # Compute the target state value for the next states using the target Q network
        # Note: no max over actions is needed because the network outputs a single scalar per state
        V_targets_next = self.target_q_network(
            batched_next_states
        ).detach()  # shape: (batch_size, 1)

        # Compute Q_targets using the Bellman update for state values:
        # V_target(s) = r + γ * V_target(s') * (1 - done)
        Q_targets = rewards + self.gamma**self.memory.n_step * V_targets_next * (
            1 - dones
        )

        # Evaluate expected state values from the local Q network
        Q_expected = self.local_q_network(batched_states)

        # Compute the mean squared error loss
        loss = (weights.to(self.device) * self.criterion(Q_expected, Q_targets)).mean()
        self.latest_loss = loss.item()

        # Backpropagation and optimization step
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.local_q_network.parameters(), 1)
        self.optimizer.step()
        self.scheduler.step()

        # Compute TD errors for priority update and update the buffer priorities
        with torch.no_grad():
            TD_errors = (
                torch.abs(Q_expected - Q_targets).detach().cpu().numpy().squeeze()
            )  # shape: (batch_size,)
        new_priorities = TD_errors + 1e-6  # small constant for stability
        self.memory.update_priorities(indices, new_priorities)

        # ------------------- update target network ------------------- #
        self.update_count += 1
        if self.update_count % self.soft_update_period == 0:
            self.soft_update(self.target_q_network, self.local_q_network)

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_aborescence_graph(self, state: bs.IABRelayGraph, path: List[int]):
        """
        Get the aborescence graph from the path.

        args:
        - graph [IABRelayGraph]: the graph to which the path will be added
        - path [List[int]]: the path to be added
        """
        aborescence_graph = state.copy()
        utils.get_aborescence_graph(aborescence_graph, list(path))
        return aborescence_graph

    def _convert_state_to_data(self, state: bs.IABRelayGraph):
        state.compute_hops()

        hop_list = []
        for node in state.nodes.values():
            if isinstance(node, bs.User):
                hop_list.append(node.hops)
            elif isinstance(node, bs.BaseStation):
                if node.get_id() == 0:
                    hop_list.append(0)
                else:
                    hop_list.append(sum([u.hops for u in node.connected_user]))
            else:
                hop_list.append(0)
        hop_list = torch.tensor(hop_list).view(-1, 1).float()
        tau_value = getattr(
            state.environmental_variables,
            "SPSC_probability",
            bs.environmental_variables.SPSC_probability,
        )
        tau_list = torch.full((len(state.nodes), 1), float(tau_value)).float()
        # append hop_list to Data.x
        state_data = state.to_torch_geometric()
        state_data.x = torch.cat((state_data.x, hop_list, tau_list), dim=1)
        reversed_edge_index = state_data.edge_index.flip(0)
        state_data.edge_index = reversed_edge_index

        return state_data


class PrioritizedNStepReplayBuffer:
    """
    Prioritized replay buffer supporting n-step returns.
    Each experience is stored along with a priority.
    """

    def __init__(
        self,
        buffer_size,
        batch_size,
        n_step,
        gamma,
        alpha=0.6,
        beta=0.4,
        beta_increment=0.001,
        device="",
    ):
        self.buffer = []  # List to store experiences
        self.priorities = []  # List to store corresponding priorities
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)
        self.alpha = (
            alpha  # Degree of prioritization (0: uniform, 1: full prioritization)
        )
        self.beta = beta  # Importance sampling correction factor
        self.beta_increment = beta_increment
        self.device = device

    def add(self, state, action, reward, next_state, done):
        # Append current transition to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # Only proceed if n-step buffer is full or if episode terminates
        if len(self.n_step_buffer) < self.n_step and not done:
            return

        # Compute n-step return: R = r_t + gamma*r_{t+1} + ... + gamma^(n-1)*r_{t+n-1}
        R = 0.0
        for idx, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            R += (self.gamma**idx) * r
            if d:
                break

        # Use the state and action from the first transition, and next_state from the last one
        state_n, action_n, _, _, done_n = self.n_step_buffer[0]
        next_state_n = self.n_step_buffer[-1][3]
        e = Experience(state_n, action_n, R, next_state_n, done_n)

        # Assign maximum priority to new experience to ensure it is sampled at least once
        max_priority = max(self.priorities) if self.priorities else 1.0
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(e)
            self.priorities.append(max_priority)
        else:
            # FIFO replacement: remove oldest experience and its priority
            self.buffer.pop(0)
            self.buffer.append(e)
            self.priorities.pop(0)
            self.priorities.append(max_priority)

        # If the first transition was terminal, clear the n-step buffer; otherwise, remove the oldest
        if self.n_step_buffer[0][4]:
            self.n_step_buffer.clear()
        else:
            self.n_step_buffer.popleft()

    def sample(self):
        # Compute sampling probabilities: p_i^alpha / sum_j(p_j^alpha)
        priorities_np = np.array(self.priorities, dtype=np.float32)
        probs = priorities_np**self.alpha
        probs /= probs.sum()

        # Sample indices according to computed probabilities
        indices = np.random.choice(len(self.buffer), self.batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]

        # Unpack experiences
        states = [e.state for e in experiences]
        actions = [e.action for e in experiences]
        rewards = (
            torch.from_numpy(np.vstack([e.reward for e in experiences]))
            .float()
            .to(self.device)
        )
        next_states = [e.next_state for e in experiences]
        dones = (
            torch.from_numpy(np.vstack([e.done for e in experiences]))
            .float()
            .to(self.device)
        )

        # Compute importance sampling (IS) weights:
        # w_i = (N * P(i))^(-beta) normalized by max weight in the batch
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # normalize for stability
        weights = torch.tensor(weights, dtype=torch.float32)

        # Increment beta towards 1 over time for full correction
        self.beta = min(1.0, self.beta + self.beta_increment)

        return (states, actions, rewards, next_states, dones, indices, weights)

    def update_priorities(self, indices, new_priorities):
        # Update priorities for the sampled transitions based on their new TD error
        for idx, priority in zip(indices, new_priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)
