import numpy as np
import torch
import torch
import os
from Adaptive_RL import logger, ReplayBuffer, neural_networks
from Adaptive_RL.agents import base_agent
from Adaptive_RL.neural_networks.utils import trainable_variables

class ARS(base_agent.BaseAgent):
    """
    Augmented Random Search (ARS) Agent

    Implements ARS which perturbs the policy in random directions and evaluates its performance.
    """

    def __init__(self, model, lr=0.02, step_size=0.03, n_directions=16, n_top_directions=8):
        """
        Initializes the ARS Agent.

        Args:
        - policy: The policy network to be perturbed and updated.
        - lr: Learning rate for policy updates.
        - step_size: Step size for perturbations.
        - n_directions: Number of random directions to explore.
        - n_top_directions: Number of top-performing directions used for updates.
        """
        super().__init__()
        self.model = model or neural_networks.BaseModel(hidden_size=256).get_model()
        self.model_updater = neural_networks.StochasticPolicyGradient()
        self.lr = lr
        self.step_size = step_size
        self.n_directions = n_directions
        self.n_top_directions = n_top_directions

        # Initialize the noise for perturbation and policy weights
        self.noise = None
        self.policy_weights = None


    def initialize(self, observation_space, action_space, seed=None):
        """
        Initializes the agent with observation and action spaces.
        """
        self.model.initialize(observation_space, action_space)
        self.model_updater.initialize(self.model)
        self.policy_weights = trainable_variables(self.model)
        np.random.seed(seed)
        torch.manual_seed(seed)


    def step(self, observations, steps):
        """
        Returns the action taken during training.
        """
        # Use the current policy to compute the action
        with torch.no_grad():
            actions = self.model.forward(observations)
        return actions.cpu().numpy()

    def update(self, observations, rewards, resets, terminations, steps):
        """
        Performs the ARS update by generating perturbations and updating policy weights.
        """
        # Generate random directions for perturbation
        self.noise = np.random.randn(self.n_directions, *self.policy_weights.shape)
        rewards_pos = []
        rewards_neg = []

        # Evaluate rewards for positive and negative perturbations
        for noise in self.noise:
            reward_pos = self.evaluate_policy(self.policy_weights + self.step_size * noise, observations)
            reward_neg = self.evaluate_policy(self.policy_weights - self.step_size * noise, observations)
            rewards_pos.append(reward_pos)
            rewards_neg.append(reward_neg)

        # Compute the gradient and update the policy weights
        rewards_pos, rewards_neg = np.array(rewards_pos), np.array(rewards_neg)
        top_rewards = np.argsort(rewards_pos - rewards_neg)[-self.n_top_directions:]
        gradient = np.sum([self.noise[top] * (rewards_pos[top] - rewards_neg[top]) for top in top_rewards], axis=0)

        # Update policy weights
        self.policy_weights += self.lr / (self.n_top_directions * np.std(rewards_pos + rewards_neg)) * gradient
        self.apply_policy_weights()

    def test_step(self, observations):
        pass

    def test_update(self, observations, rewards, resets, terminations, steps):
        pass

    def evaluate_policy(self, perturbed_weights, observations):
        """
        Evaluates the reward obtained with perturbed policy weights.
        """
        self.apply_policy_weights(perturbed_weights)
        actions = self.step(observations, steps=None)
        # Here you would return the reward from the environment, for simplicity we assume it's predefined.
        reward = ...  # Replace with actual reward calculation
        return reward

    def apply_policy_weights(self, weights=None):
        """
        Applies the perturbed policy weights to the policy network.
        """
        weights = weights if weights is not None else self.policy_weights
        self.model.load_state_dict(weights)

    def save(self, path):
        pass

    def load(self, path):
        pass