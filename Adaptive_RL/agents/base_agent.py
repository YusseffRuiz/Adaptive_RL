import abc
import torch
import os
from Adaptive_RL import logger
import re
import numpy as np
import random


class BaseAgent(abc.ABC):
    """
    Abstract base class used to build agents.
    These are the required methods used to build any agent.
    """

    def initialize(self, observation_space, action_space, seed=None):
        if seed is None:
            seed = random.randint(1, 10000)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

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

    def save(self, path, full_save=False):
        """Saves the agent weights during training."""
        path = path + '.pt'
        logger.log(f'\nSaving weights to {path}')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if full_save:
            torch.save(self.model, path)
        else:
            torch.save(self.model.state_dict(), path)

    def load(self, path):
        """Reloads the agent weights from a checkpoint, and returns the step number."""
        if not path[-3:] == '.pt':
            path = path + '.pt'
        logger.log(f'\nLoading weights and from {path}')
        match = re.search(r'step_(\d+)\.pt', path)  # With regex catch the step saved
        step_number = 0
        if match is not None:
            step_number = int(match.group(1))

        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()

        return step_number

    def get_config(self, print_conf=False):
        """
        Print all configuration, if required, can be saved in variable
        """
        if print_conf:
            print("######## AGENT CONFIGURATION ########")
            for key, value in self.config.items():
                print(f"{key}: {value}")
        return self.config


