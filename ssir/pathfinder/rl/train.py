from collections import deque

import torch
import wandb

from ssir.pathfinder.rl import agent, environment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    # Define hyperparameters in a dictionary
    config = {
        "state_channels": 15,
        "action_dim": 20,
        "window_size": 30,
        "num_episodes": 10000,
        "warmup_episodes": 0,
        "training_updates_per_episode": 10,
        "data_dir": "/fast/hslyu/train",
        "total_files": 50000,
        "embedding_channels": 128,
        "lr": 1e-4,
        "gamma": 0.99,
        "tau": 1e-3,
        "epsilon": 1.00,
        "epsilon_decay": 0.995,
        "buffer_size": 10000,
        "batch_size": 256,
        "scheduler": "cosine",
    }
    # Initialize wandb run with configuration
    run = wandb.init(
        entity="hslyu",
        project="SSIR",
        config=config,
    )

    # Use wandb.config to get hyperparameters
    state_channels = wandb.config.state_channels
    action_dim = wandb.config.action_dim
    window_size = wandb.config.window_size
    num_episodes = wandb.config.num_episodes
    warmup_episodes = wandb.config.warmup_episodes
    training_updates_per_episode = wandb.config.training_updates_per_episode
    data_dir = wandb.config.data_dir

    env = environment.IABRelayEnvironment(
        state_channels, action_dim, data_dir, total_files=wandb.config.total_files
    )
    IABagent = agent.Agent(
        input_channels=state_channels,
        num_action=action_dim,
        embedding_channels=wandb.config.embedding_channels,
        criterion=torch.nn.HuberLoss(reduction="none"),
        lr=wandb.config.lr,
        gamma=wandb.config.gamma,
        tau=wandb.config.tau,
        epsilon=wandb.config.epsilon,
        epsilon_decay=wandb.config.epsilon_decay,
        buffer_size=wandb.config.buffer_size,
        batch_size=wandb.config.batch_size,
        device=device,
        deterministic=False,
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
        performance = throughput / genetic_throughput * 100
        performance_list.append(performance)
        avg_performance = sum(performance_list) / len(performance_list)

        # Log metrics to wandb
        wandb.log(
            {
                "episode": i + 1,
                "loss": IABagent.latest_loss,
                "moving_average": avg_performance,
                "throughput": throughput,
                "genetic": performance,
                "epsilon": IABagent.epsilon,
                "served_user": served_user,
                "total_user": total_user,
                "best_performance": best_performance,
            }
        )

        print(
            f"Episode {i+1}/{num_episodes}, loss: {IABagent.latest_loss:.2f}, moving average: {avg_performance:.1f}%, "
            + f"throughput: {throughput:.2f}, genetic: {performance:.1f}%, "
            + f"epsilon: {IABagent.epsilon:.2f}"
        )
        if avg_performance > best_performance:
            best_performance = avg_performance
            IABagent.save_network("./models/", f"{run.name}.pth")

    wandb.finish()


if __name__ == "__main__":
    train()
