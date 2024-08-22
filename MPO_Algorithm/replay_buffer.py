import random
import torch
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
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
        self.log_probs = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
        self.values = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
        self.entropies = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)

    def push(self, state, action, reward, done, log_probs, values, entropy):
        """Store an experience in the buffer."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = done
        self.log_probs[self.position] = log_probs
        self.values[self.position] = values
        self.entropies[self.position] = entropy

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

        return batch_states, batch_actions, batch_rewards, batch_dones, batch_log_probs, batch_values, batch_entropies

    def __len__(self):
        return self.size
