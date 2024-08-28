import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, env):
        super(Critic, self).__init__()
        # Define the value network (critic) with similar architecture
        self.env = env
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_space = env.action_space
        hidden_dim = 256
        self.critic = nn.Sequential(
            nn.Linear(self.observation_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Batch Normalization
            nn.SiLU(),  # Advanced Activation Function
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Layer Normalization
            nn.SiLU(),  # Advanced Activation Function
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.critic(state)
