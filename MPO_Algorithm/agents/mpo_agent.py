import torch
import os
from MPO_Algorithm import logger, ReplayBuffer, neural_networks
from MPO_Algorithm.agents import base_agent
from MPO_Algorithm.neural_networks import MaximumAPosterioriPolicyOptimization


class MPO(base_agent.BaseAgent):
    """
    Maximum a Posteriori Policy Optimisation.
    MPO: https://arxiv.org/pdf/1806.06920.pdf
    MO-MPO: https://arxiv.org/pdf/2005.07513.pdf
    """

    def __init__(
        self, model=None, hidden_size=256, critic_updater=None,
            return_step=5, lr_actor=3e-4, lr_dual=3e-4, lr_critic=3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_actions = None
        self.last_observations = None
        self.model = model or neural_networks.BaseModel(hidden_size=hidden_size).get_model()
        self.replay_buffer = ReplayBuffer(return_steps=return_step)
        self.actor_updater = MaximumAPosterioriPolicyOptimization(lr_actor=lr_actor, lr_dual=lr_dual)
        self.critic_updater = critic_updater or neural_networks.ExpectedSARSA(lr_critic=lr_critic)

    def initialize(self, observation_space, action_space, seed=None):
        super().initialize(observation_space, action_space, seed=seed)
        self.model.initialize(observation_space, action_space)
        self.actor_updater.initialize(self.model, action_space)
        self.critic_updater.initialize(self.model)

    def step(self, observations, steps=None):
        actions = self._step(observations)
        actions = actions.cpu().numpy()

        # Keep some values for the next update.
        self.last_observations = observations.copy()
        self.last_actions = actions.copy()

        return actions

    def test_step(self, observations):
        # Sample actions for testing.
        return self._test_step(observations).cpu().numpy()

    def update(self, observations, rewards, resets, terminations, steps):
        # Store the last transitions in the replay.
        self.replay_buffer.push(
            observations=self.last_observations, actions=self.last_actions,
            next_observations=observations, rewards=rewards, resets=resets,
            terminations=terminations)

        # Prepare to update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.record(self.last_observations)
        if self.model.return_normalizer:
            self.model.return_normalizer.record(rewards)

        # Update the model if the replay is ready.
        if self.replay_buffer.ready(steps):
            self._update(steps)

    def _step(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            return self.model.actor(observations).sample()

    def _test_step(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            return self.model.actor(observations).loc

    def _update(self, steps):
        keys = ('observations', 'actions', 'next_observations', 'rewards',
                'discounts')

        # Update both the actor and the critic multiple times.
        for batch in self.replay_buffer.get(*keys, steps=steps):
            batch = {k: torch.as_tensor(v) for k, v in batch.items()}
            infos = self._update_actor_critic(**batch)

            for key in infos:
                for k, v in infos[key].items():
                    logger.store(key + '/' + k, v.cpu().numpy())

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
