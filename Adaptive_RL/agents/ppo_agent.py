import torch
import os
from Adaptive_RL import logger, Segment, neural_networks
from Adaptive_RL.agents import base_agent


class PPO(base_agent.BaseAgent):
    """
    Implementation of PPO (Proximal Policy Optimization) algorithm.
    https://arxiv.org/pdf/1707.06347.pdf
    """

    def __init__(self, model=None, hidden_size=256, lr_actor=3e-4, lr_critic=3e-4, discount_factor=0.99,
                 replay_buffer=None,
                 actor_updater=None, critic_updater=None, batch_size=None, trace_decay=0.97, batch_iterations=80,
                 replay_buffer_size=4096):
        self.model = model or neural_networks.ActorCriticModelNetwork(hidden_size=hidden_size).get_model()
        self.replay_buffer = replay_buffer or Segment(size=replay_buffer_size, batch_iterations=batch_iterations,
                                                      batch_size=batch_size, discount_factor=discount_factor,
                                                      trace_decay=trace_decay)
        self.actor_updater = actor_updater or neural_networks.ClippedRatio(lr_actor=lr_actor)
        self.critic_updater = critic_updater or neural_networks.VRegression(lr_critic=lr_critic)

    def initialize(self, observation_space, action_space, seed=None):
        super().initialize(observation_space, action_space, seed)
        self.model.initialize(observation_space, action_space)
        self.replay_buffer.initialize(seed)
        self.actor_updater.initialize(self.model)
        self.critic_updater.initialize(self.model)

    def step(self, observations, steps=None):
        # Sample actions and get their log-probabilities for training.
        actions, log_probs = self._step(observations)
        actions = actions.cpu().numpy()
        log_probs = log_probs.cpu().numpy()

        # Keep some values for the next update.
        self.last_observations = observations.copy()
        self.last_actions = actions.copy()
        self.last_log_probs = log_probs.copy()

        return actions

    def update(self, observations, rewards, resets, terminations, steps):
        # Store the last transitions in the replay.
        self.replay_buffer.store(observations=self.last_observations, actions=self.last_actions,
                                 next_observations=observations, rewards=rewards, resets=resets,
                                 terminations=terminations, log_probs=self.last_log_probs)

        # Prepare to update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.record(self.last_observations)
        if self.model.return_normalizer:
            self.model.return_normalizer.record(rewards)

        # Update the model if the replay is ready.
        if self.replay_buffer.ready():
            self._update()

    def test_step(self, observations):
        # Sample actions for testing.
        return self._test_step(observations).cpu().numpy()

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

    def _step(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            distributions = self.model.actor(observations)
            if hasattr(distributions, 'sample_with_log_prob'):
                actions, log_probs = distributions.sample_with_log_prob()
            else:
                actions = distributions.sample()
                log_probs = distributions.log_prob(actions)
            log_probs = log_probs.sum(dim=-1)
        return actions, log_probs

    def _test_step(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            return self.model.actor(observations).sample()

    def _evaluate(self, observations, next_observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        next_observations = torch.as_tensor(next_observations, dtype=torch.float32)
        with torch.no_grad():
            values = self.model.critic(observations)
            next_values = self.model.critic(next_observations)
        return values, next_values

    def _update(self):
        # Compute the lambda-returns.
        batch = self.replay_buffer.get_full('observations', 'next_observations')
        values, next_values = self._evaluate(**batch)
        values, next_values = values.cpu().numpy(), next_values.cpu().numpy()
        self.replay_buffer.compute_returns(values, next_values)

        train_actor = True
        actor_iterations = 0
        critic_iterations = 0
        keys = 'observations', 'actions', 'advantages', 'log_probs', 'returns'

        # Update both the actor and the critic multiple times.
        for batch in self.replay_buffer.get(*keys):
            if train_actor:
                batch = {k: torch.as_tensor(v) for k, v in batch.items()}
                infos = self._update_actor_critic(**batch)
                actor_iterations += 1
            else:
                batch = {k: torch.as_tensor(batch[k])
                         for k in ('observations', 'returns')}
                infos = dict(critic=self.critic_updater(**batch))
            critic_iterations += 1

            # Stop earlier the training of the actor.
            if train_actor:
                train_actor = not infos['actor']['stop'].cpu().numpy()

            for key in infos:
                for k, v in infos[key].items():
                    logger.store(key + '/' + k, v.cpu().numpy())

        logger.store('actor/iterations', actor_iterations)
        logger.store('critic/iterations', critic_iterations)

        # Update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.update()
        if self.model.return_normalizer:
            self.model.return_normalizer.update()

    def _update_actor_critic(self, observations, actions, advantages, log_probs, returns):
        actor_infos = self.actor_updater(observations, actions, advantages, log_probs)
        critic_infos = self.critic_updater(observations, returns)
        return dict(actor=actor_infos, critic=critic_infos)
