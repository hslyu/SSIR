from collections import deque

import torch

from ssir.pathfinder.rl import agent, environment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    state_channels = 15
    action_dim = 50
    window_size = 30
    num_episodes = 10000
    warmup_episodes = 0
    training_updates_per_episode = 10

    data_dir = "/fast/hslyu/train"
    env = environment.IABRelayEnvironment(
        state_channels, action_dim, data_dir, total_files=1
    )
    IABagent = agent.Agent(
        input_channels=state_channels,
        num_action=action_dim,
        embedding_channels=128,
        criterion=torch.nn.HuberLoss(reduction="none"),
        lr=1e-4,
        gamma=0.99,
        tau=1e-3,
        epsilon=0.00,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=128,
        device=device,
        deterministic=True,
    )

    performance_list = deque(maxlen=window_size)
    best_performance = 0
    for i in range(num_episodes):
        state = env.reset()
        IABagent.set_master_graph(env.master_graph)

        done = False
        reward = 0
        while not done:
            action = IABagent.act(state)
            next_state, reward, done, _ = env.step(action)
            IABagent.step(state, action, reward, next_state, done)
            state = next_state

        if i >= warmup_episodes:
            for _ in range(training_updates_per_episode):
                if len(IABagent.memory) > IABagent.batch_size:
                    experiences = IABagent.memory.sample()
                    IABagent.learn(experiences)
        total_user = len(env.state.users)
        served_user = len(env.state.basestations[0].connected_user)

        throughput = env.state.compute_network_throughput()
        genetic_throughput = env.genetic_graph.compute_network_throughput()
        performance_list.append(throughput / genetic_throughput * 100)
        avg_performance = sum(performance_list) / len(performance_list)
        print(
            f"Episode {i+1}/{num_episodes}, loss: {IABagent.latest_loss:.2f}, moving average: {avg_performance:.1f}%, "
            + f"throughput: {throughput:.2f}, genetic: {throughput/genetic_throughput*100:.1f}%, "
            + f"epsilon: {IABagent.epsilon:.2f}"  # , Unserved user: {total_user - served_user}"
        )
        if avg_performance > best_performance:
            best_performance = avg_performance


if __name__ == "__main__":
    train()
