import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

class Actor(nn.Module):
    def __init__(self, env):
        super(Actor, self).__init__()
        self.env = env
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_space = env.action_space
        hidden_dim = 256
        # Define the policy network (actor)
        self.fc1 = nn.Linear(self.observation_dim, hidden_dim)
        self.fc2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.LayerNorm(hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, self.action_dim)
        self.cholesky_layer = nn.Linear(hidden_dim, (self.action_dim * (self.action_dim + 1)) // 2)

    def forward(self, state):
        device = state.device

        # Activation Functions
        x = self.fc1(state)
        x = self.fc2(x)
        x = F.silu(x)
        x = self.fc3(x)
        x = F.silu(x)
        mean = self.calculate_mean(x, device)

        cholesky_vector = self.cholesky_layer(x)  # (B, (da*(da+1))//2)
        cholesky_diag_index = torch.arange(self.action_dim, dtype=torch.long) + 1
        cholesky_diag_index = (cholesky_diag_index * (cholesky_diag_index + 1)) // 2 - 1
        cholesky_vector[:, cholesky_diag_index] = F.softplus(cholesky_vector[:, cholesky_diag_index])
        tril_indices = torch.tril_indices(row=self.action_dim, col=self.action_dim, offset=0)
        cholesky = torch.zeros(size=(state.size(0), self.action_dim, self.action_dim), dtype=torch.float32).to(device)
        cholesky[:, tril_indices[0], tril_indices[1]] = cholesky_vector

        return mean, cholesky

    def calculate_mean(self, x, device):
        """
        :param x: Layer to calculate mean
        :param device: Device to send tensors: cpu or GPU
        :return: mean of the action space
        """
        action_low = torch.from_numpy(self.env.action_space.low)[None, ...].to(device)  # (1, da)
        action_high = torch.from_numpy(self.env.action_space.high)[None, ...].to(device)  # (1, da)
        mean = F.sigmoid(self.mean_layer(x))
        mean = action_low + (action_high - action_low) * mean

        return mean

    def select_action(self, state):
        """
                Selects an action based on the current policy.

                Args:
                    state (torch.Tensor): The current state.

                Returns:
                    action (torch.Tensor): The selected continuous action.
                """
        with torch.no_grad():
            mean, cholesky = self.forward(state[None, ...])
            action_distribution = MultivariateNormal(mean, scale_tril=cholesky)
            action = action_distribution.sample()

        return action[0]

