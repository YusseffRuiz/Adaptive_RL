import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, env):
        super(Actor, self).__init__()
        self.env = env
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_space = env.action_space
        hidden_dim = 256
        # Define the policy network (actor)
        self.actor = nn.Sequential(
            nn.Linear(self.observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Layer Normalization
            nn.SiLU(),
            nn.Linear(hidden_dim, self.action_dim * 2)  # Outputs mean and log_std for each action dimension
        )

    def forward(self, state):
        device = state.device
        action_low = torch.from_numpy(self.env.action_space.low)[None, ...].to(device)  # (1, da)
        action_high = torch.from_numpy(self.env.action_space.high)[None, ...].to(device)  # (1, da)
        params = self.actor(state)
        mean, output = params[:, :params.shape[0] // 2], params[:, params.shape[1] // 2:]
        mean = action_low + (action_high - action_low) * mean

        return mean, output

    def action(self, state):
        """
                Selects an action based on the current policy.

                Args:
                    state (torch.Tensor): The current state.

                Returns:
                    action (torch.Tensor): The selected continuous action.
                    log_prob (torch.Tensor): Log probability of the selected action.
                """
        device = state.device
        # Split the output into mean and log_std for each action dimension
        mean, log_std = self.actor(state).to(device)
        std = torch.exp(log_std)

        # Create a normal distribution and sample a continuous action
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()

        # Clamp the actions to the valid range
        if self.action_space is not None:
            action = torch.clamp(action, self.action_space.low[0], self.action_space.high[0])

        entropy = dist.entropy().sum(dim=-1)

        return action, entropy