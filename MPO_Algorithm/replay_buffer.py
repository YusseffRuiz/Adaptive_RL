import random
import torch
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, param_dim=None):
        """
        Initialize the Replay Buffer.

        Args:
            capacity (int): Maximum number of experiences to store in the buffer. When the buffer overflows, old experiences are discarded.
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size = 0

        # Pre-allocate memory for states, actions, rewards, next states, dones
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
        self.values = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)

        if param_dim is not None:
            self.optimized_params = torch.zeros((capacity, param_dim//2, param_dim//2), dtype=torch.float32, device=self.device)
            self.log_probs = torch.zeros((capacity, param_dim//2), dtype=torch.float32, device=self.device)
            self.entropies = torch.zeros((capacity, param_dim//2), dtype=torch.float32, device=self.device)
        else:
            self.optimized_params = None
            self.log_probs = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
            self.entropies = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)

    @classmethod
    def from_mpo(cls, capacity, state_dim, action_dim):
        """
        Class method to initialize ReplayBuffer for use with standard MPO.

        Args:
            capacity (int): Maximum number of experiences to store in the buffer.
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.

        Returns:
            ReplayBuffer: Initialized ReplayBuffer instance for MPO.
        """
        return cls(capacity, state_dim, action_dim)

    @classmethod
    def from_matsuoka(cls, capacity, state_dim, action_dim, param_dim):
        """
        Class method to initialize ReplayBuffer for use with Matsuoka Oscillators.

        Args:
            capacity (int): Maximum number of experiences to store in the buffer.
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            param_dim (int): Dimension of the parameters for the Matsuoka oscillator.

        Returns:
            ReplayBuffer: Initialized ReplayBuffer instance with support for Matsuoka optimized parameters.
        """
        return cls(capacity, state_dim, action_dim, param_dim=param_dim)

    def push(self, state, action, reward, done, log_probs, values, entropy, optimized_params=None):
        """Store an experience in the buffer."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = done
        self.log_probs[self.position] = log_probs
        self.values[self.position] = values
        self.entropies[self.position] = entropy

        if self.optimized_params is not None and optimized_params is not None:
            self.optimized_params[self.position] = optimized_params

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer."""
        indices = np.random.randint(0, self.size, size=batch_size)
        batch_states = self.states[indices]
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_dones = torch.stack([self.dones[i] for i in indices])
        batch_log_probs = self.log_probs[indices]
        batch_values = self.values[indices]
        batch_entropies = self.entropies[indices]

        if self.optimized_params is not None:
            batch_optimized_params = self.optimized_params[indices]
            return (batch_states, batch_actions, batch_rewards, batch_dones, batch_log_probs, batch_values,
                    batch_entropies, batch_optimized_params)
        else:
            return (batch_states, batch_actions, batch_rewards, batch_dones, batch_log_probs, batch_values,
                    batch_entropies)

    def __len__(self):
        return self.size
