import torch
import numpy as np
from Adaptive_RL import logger, neural_networks
from Adaptive_RL.agents import base_agent
from Adaptive_RL.neural_networks import DeterministicPolicyGradient, DeterministicQLearning
from Adaptive_RL.utils import explorations, ReplayBuffer, utils


class DDPG(base_agent.BaseAgent):
    """
    Deep Deterministic Policy Gradient.
    DDPG: https://arxiv.org/pdf/1509.02971.pdf
    """

    def __init__(self, hidden_size=256, hidden_layers=2, learning_rate=3e-4, lr_critic=None,
                 batch_size=512, return_step=5,
                 discount_factor=0.99, steps_between_batches=20, replay_buffer_size=10e5, noise_std=0.1,
                 decay_lr=0.98, learning_starts=20000):
        # Store all the inputs in a dictionary
        self.config = {
            "agent" : "DDPG",
            "learning_rate": learning_rate,
            "lr_critic": lr_critic,
            "noise_std": noise_std,
            "decay_lr": decay_lr,
            "learning_starts": learning_starts,
            "hidden_size": hidden_size,
            "hidden_layers": hidden_layers,
            "discount_factor": discount_factor,
            "batch_size": batch_size,
            "return_step": return_step,
            "steps_between_batches": steps_between_batches,
            "replay_buffer_size": replay_buffer_size,
        }
        if lr_critic is None:
            lr_critic = learning_rate
        self.model = neural_networks.ActorCriticDeterministic(hidden_size=hidden_size, hidden_layers=hidden_layers).get_model()
        self.replay = ReplayBuffer(return_steps=return_step, discount_factor=discount_factor,
                                          batch_size=batch_size, steps_between_batches=steps_between_batches,
                                          size=int(replay_buffer_size))
        self.exploration = explorations.NormalNoiseExploration(scale=noise_std, start_steps=learning_starts)
        self.actor_updater = DeterministicPolicyGradient(lr_actor=learning_rate)
        self.critic_updater = DeterministicQLearning(lr_critic=lr_critic)
        self.decay_lr = decay_lr

    def initialize(self, observation_space, action_space, seed=None):
        super().initialize(observation_space, action_space, seed=seed)
        self.model.initialize(observation_space, action_space)
        self.replay.initialize()
        self.exploration.initialize(self._policy, action_space, seed)
        self.actor_updater.initialize(self.model)
        self.critic_updater.initialize(self.model)
        self.decay_flag = False

    def step(self, observations, steps):
        # Get actions from the actor and exploration method.
        actions = self.exploration(observations, steps)

        # Keep some values for the next update.
        self.last_observations = observations.copy()
        self.last_actions = actions.copy()

        return actions

    def update(self, observations, rewards, resets, terminations, steps):
        # Verify if data is tensor already, otherwise, change it
        # Store last transition in the replay buffer
        # Store the last transitions in the replay.
        if np.any(np.isnan(observations)):
            print("NaN detected in output_actions:", observations)
            observations[np.isnan(observations)] = 0.0
        if np.any(np.isnan(rewards)):
            print("NaN detected in output_rewards:", rewards)
            rewards[np.isnan(rewards)] = 0.0
        if np.any(np.isnan(resets)):
            print("NaN detected in output_resets:", resets)
            resets[np.isnan(resets)] = 0.0
        if np.any(np.isnan(terminations)):
            print("NaN detected in output_terminations:", terminations)
            terminations[np.isnan(terminations)] = 0.0

        self.replay.push(observations=self.last_observations, actions=self.last_actions,
                                next_observations=observations, rewards=rewards, resets=resets,
                                terminations=terminations)

        # Update the normalizers
        if self.model.observation_normalizer:
            self.model.observation_normalizer.record(self.last_observations)
        if self.model.return_normalizer:
            self.model.return_normalizer.record(rewards)

        if self.replay.ready(steps):
            self._update(steps)

        self.exploration.update(resets)
        if self.decay_flag:  # Reducing noise to stabilize training
            self.exploration.scale *= self.decay_lr
            self.actor_updater.lr_actor *= self.decay_lr
            self.critic_updater.lr_critic *= self.decay_lr


    def test_step(self, observations, steps=None):
        # Greedy actions for testing.
        return self._greedy_actions(observations).cpu().numpy()


    def _policy(self, observations):
        return self._greedy_actions(observations).cpu().numpy()

    def _greedy_actions(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            return self.model.actor(observations)

    def _update(self, steps):
        keys = ('observations', 'actions', 'next_observations', 'rewards',
                'discounts')

        # Update both the actor and the critic multiple times.
        for batch in self.replay.get(*keys, steps=steps):
            # Batch data is already in tensor form, so no need to convert again
            batch = {k: torch.as_tensor(v) for k, v in batch.items()}
            infos = self._update_actor_critic(**batch)

            for key in infos:
                for k, v in infos[key].items():
                    logger.store(key + '/' + k, v.cpu().numpy())  # Convert back to numpy for logging

        # Update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.update()
        if self.model.return_normalizer:
            self.model.return_normalizer.update()

    def _update_actor_critic(
            self, observations, actions, next_observations, rewards, discounts
    ):
        critic_infos = self.critic_updater(observations, actions, next_observations, rewards, discounts)
        actor_infos = self.actor_updater(observations)
        self.model.update_targets()
        return dict(critic=critic_infos, actor=actor_infos)
