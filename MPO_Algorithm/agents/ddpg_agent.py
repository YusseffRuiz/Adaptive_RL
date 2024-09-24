import torch
import os
from MPO_Algorithm import logger, ReplayBuffer, neural_networks
from MPO_Algorithm.agents import base_agent
from MPO_Algorithm.neural_networks import DeterministicPolicyGradient, DeterministicQLearning
from MPO_Algorithm.utils import noise

class DDPG(base_agent.BaseAgent):
    """
    Deep Deterministic Policy Gradient.
    DDPG: https://arxiv.org/pdf/1509.02971.pdf
    """
    def __init__(self, model=None, hidden_size=256, replay_buffer=None, exploration=None, actor_updater=None,
                 critic_updater=None):
        self.model = model or neural_networks.BaseModel(hidden_size=hidden_size).get_model()
        self.replay_buffer = replay_buffer or ReplayBuffer()
        self.exploration = exploration or noise.NormalActionNoise()
        self.actor_updater = actor_updater or DeterministicPolicyGradient()
        self.critic_updater = critic_updater or DeterministicQLearning()

    def initialize(self, observation_space, action_space, seed=None):
        super().initialize(observation_space, action_space, seed)
        self.model.initialize(observation_space, action_space)
        self.exploration.initialize(self._policy, action_space, seed)
        self.actor_updater.initialize(self.model)
        self.critic_updater.initialize(self.model)


    def step(self, observations, steps):
        # Get actions from the actor and exploration method.
        actions = self.exploration(observations, steps)

        # Keep some values for the next update.
        self.last_observations = observations.copy()
        self.last_actions = actions.copy()

        return actions

    def update(self, observations, rewards, resets, terminations, steps):
        # Store last transition in the replay buffer
        self.replay_buffer.push(observations=self.last_observations, actions=self.last_actions,
            next_observations=observations, rewards=rewards, resets=resets,
            terminations=terminations)

        # Update the normalizers
        if self.model.observation_normalizer:
            self.model.observation_normalizer.record(self.last_observations)
        if self.model.return_normalizer:
            self.model.return_normalizer.record(rewards)

        self.exploration.update(resets)


    def test_update(self, observations, rewards, resets, terminations, steps):
        pass

    def test_step(self, observations):
        # Greedy actions for testing.
        return self._greedy_actions(observations).numpy()

    def save(self, path):
        path = path + '.pt'
        logger.log(f'\nSaving weights to {path}')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        if not path[-3:] == '.pt':
            path = path + '.pt'
        logger.log(f'\nLoading weights from {path}')
        self.model.load_state_dict(torch.load(path, weights_only=True))

    def _policy(self, observations):
        return self._greedy_actions(observations).numpy()

    def _greedy_actions(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            return self.model.actor(observations)

    def _update(self, steps):
        keys = ('observations', 'actions', 'next_observations', 'rewards',
                'discounts')

        # Update both the actor and the critic multiple times.
        for batch in self.replay_buffer.get(*keys, steps=steps):
            batch = {k: torch.as_tensor(v) for k, v in batch.items()}
            infos = self._update_actor_critic(**batch)

            for key in infos:
                for k, v in infos[key].items():
                    logger.store(key + '/' + k, v.numpy())

        # Update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.update()
        if self.model.return_normalizer:
            self.model.return_normalizer.update()

    def _update_actor_critic(
            self, observations, actions, next_observations, rewards, discounts
    ):
        critic_infos = self.critic_updater(
            observations, actions, next_observations, rewards, discounts)
        actor_infos = self.actor_updater(observations)
        self.model.update_targets()
        return dict(critic=critic_infos, actor=actor_infos)