import torch
from Adaptive_RL import logger, neural_networks
from Adaptive_RL.agents import base_agent
from Adaptive_RL.neural_networks import DeterministicPolicyGradient, DeterministicQLearning
from Adaptive_RL.utils import explorations, ReplayBuffer, utils


class DDPG(base_agent.BaseAgent):
    """
    Deep Deterministic Policy Gradient.
    DDPG: https://arxiv.org/pdf/1509.02971.pdf
    """

    def __init__(self, hidden_size=256, hidden_layers=2, learning_rate=3e-4, batch_size=512, return_step=5,
                 discount_factor=0.99, steps_between_batches=20, replay_buffer_size=10e5, noise_std=0.1,
                 decay_lr=0.98, learning_starts=20000):
        # Store all the inputs in a dictionary
        self.config = {
            "agent" : "DDPG",
            "learning_rate": learning_rate,
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
        self.model = neural_networks.ActorCriticDeterministic(hidden_size=hidden_size, hidden_layers=hidden_layers).get_model()
        self.replay_buffer = ReplayBuffer(return_steps=return_step, discount_factor=discount_factor,
                                          batch_size=batch_size, steps_between_batches=steps_between_batches,
                                          size=int(replay_buffer_size))
        self.exploration = explorations.NormalNoiseExploration(scale=noise_std, start_steps=learning_starts)
        self.actor_updater = DeterministicPolicyGradient(lr_actor=learning_rate)
        self.critic_updater = DeterministicQLearning(lr_critic=learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.decay_lr = decay_lr

    def initialize(self, observation_space, action_space, seed=None):
        self.model.initialize(observation_space, action_space)
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
        self.replay_buffer.push(observations=utils.to_tensor(self.last_observations, self.device), actions=utils.to_tensor(self.last_actions, self.device),
                                next_observations=utils.to_tensor(observations, self.device), rewards=utils.to_tensor(rewards, self.device), resets=utils.to_tensor(resets, self.device),
                                terminations=utils.to_tensor(terminations, self.device))

        # Update the normalizers
        if self.model.observation_normalizer:
            self.model.observation_normalizer.record(self.last_observations)
        if self.model.return_normalizer:
            self.model.return_normalizer.record(rewards)

        if self.replay_buffer.ready(steps):
            self._update(steps)

        self.exploration.update(resets)
        if self.decay_flag:  # Reducing noise to stabilize training
            self.exploration.scale *= self.decay_lr
            self.actor_updater.lr_actor *= self.decay_lr
            self.critic_updater.lr_critic *= self.decay_lr


    def test_step(self, observations):
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
        for batch in self.replay_buffer.get(*keys, steps=steps):
            # Batch data is already in tensor form, so no need to convert again
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
