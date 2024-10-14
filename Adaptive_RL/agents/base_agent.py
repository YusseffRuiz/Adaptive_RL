import abc
import torch
import os

from setuptools.sandbox import save_path

from Adaptive_RL import logger
import re

class BaseAgent(abc.ABC):
    """
    Abstract base class used to build agents.
    These are the required methods used to build any agent.
    """

    def initialize(self, observation_space, action_space, seed=None):
        self.model = None
        self.config = None
        self.replay_buffer = None

    @abc.abstractmethod
    def step(self, observations, steps):
        """
        Returns actions during training.
        """
        pass

    def update(self, observations, rewards, resets, terminations, steps):
        """
        Informs the agent of the latest transitions during training.
        """
        pass

    @abc.abstractmethod
    def test_step(self, observations):
        """Returns actions during testing."""
        pass

    def test_update(self, observations, rewards, resets, terminations, steps):
        """Informs the agent of the latest transitions during testing."""
        pass

    def save(self, path, step, save_path=None):
        """Saves the agent weights during training."""
        path = path + '.pt'
        logger.log(f'\nSaving weights to {path}')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

        # Save replay buffer
        if save_path is not None:
            buffer_path = save_path + '/replay_buffer.pth'
            keys = ('observations', 'actions', 'next_observations', 'rewards',
                    'discounts')
            # Retrieve the last batch from the buffer (if it's large enough)
            last_batch = list(self.replay_buffer.get(*keys, steps=step))[-1]

            # Convert the batch to tensor format (if not already)
            last_batch = {k: torch.as_tensor(v) for k, v in last_batch.items()}
            torch.save(last_batch, buffer_path)

    def load(self, path, load_path=None):
        """Reloads the agent weights from a checkpoint, and returns the step number."""
        if not path[-3:] == '.pt':
            path = path + '.pt'
        logger.log(f'\nLoading weights from {path}')
        match = re.search(r'step_(\d+)\.pt', path) # With regex catch the step saved
        step_number = int(match.group(1))

        self.model.load_state_dict(torch.load(path, weights_only=True))

        if load_path is not None:
            # Load replay buffer
            buffer_load_path = os.path.join(load_path, "replay_buffer.pth")
            if os.path.exists(buffer_load_path):
                buffer_data = torch.load(buffer_load_path, weights_only=True)
                self.replay_buffer = buffer_data
                print(f"Replay buffer loaded from {buffer_load_path}")
            else:
                print(f"No replay buffer found at {buffer_load_path}")

        return step_number

    def get_config(self, print_conf=False):
        """
        Print all configuration, if required, can be saved in variable
        """
        if print_conf:
            for key, value in self.config.items():
                print(f"{key}: {value}")
        return self.config