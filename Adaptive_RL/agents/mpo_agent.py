import torch
from Adaptive_RL import logger, neural_networks
from Adaptive_RL.agents import base_agent
from Adaptive_RL.neural_networks import MaximumAPosterioriPolicyOptimization
from Adaptive_RL.utils import ReplayBuffer


class MPO(base_agent.BaseAgent):
    """
    Maximum a Posteriori Policy Optimisation.
    MPO: https://arxiv.org/pdf/1806.06920.pdf
    MO-MPO: https://arxiv.org/pdf/2005.07513.pdf
    """

    def __init__(
        self, hidden_size=256, hidden_layers=2, lr_actor=3e-4, lr_dual=3e-4, lr_critic=3e-4,
            discount_factor=0.99, epsilon=0.1, epsilon_mean=1e-3, epsilon_std=1e-5, initial_log_temperature=1.,
            initial_log_alpha_mean=1., initial_log_alpha_std=10., min_log_dual=-18., per_dim_constraining=True,
            action_penalization=True, gradient_clip=0.1, batch_size=256, return_step=3, steps_between_batches=30,
            replay_buffer_size=10e5, decay_lr=0.98):
        self.model = neural_networks.BaseModel(hidden_size=hidden_size, hidden_layers=hidden_layers).get_model()
        self.replay_buffer = ReplayBuffer(return_steps=return_step, discount_factor=discount_factor,
                                          batch_size=batch_size, steps_between_batches=steps_between_batches,
                                          size=int(replay_buffer_size))
        self.actor_updater = MaximumAPosterioriPolicyOptimization(lr_actor=lr_actor, lr_dual=lr_dual, epsilon=epsilon,
                                                                  epsilon_mean=epsilon_mean, epsilon_std=epsilon_std,
                                                                  initial_log_temperature=initial_log_temperature,
                                                                  initial_log_alpha_mean=initial_log_alpha_mean,
                                                                  initial_log_alpha_std=initial_log_alpha_std,
                                                                  min_log_dual=min_log_dual,
                                                                  per_dim_constraining=per_dim_constraining,
                                                                  action_penalization=action_penalization,
                                                                  gradient_clip=gradient_clip)
        self.critic_updater = neural_networks.ExpectedSARSA(lr_critic=lr_critic)
        self.decay_lr = decay_lr
        self.config = {
            "agent": "MPO",
            "lr_actor": lr_actor,
            "lr_critic": lr_critic,
            "lr_dual": lr_dual,
            "decay_lr": decay_lr,
            "hidden_size": hidden_size,
            "hidden_layers": hidden_layers,
            "discount_factor": discount_factor,
            "gamma": discount_factor,
            "batch_size": batch_size,
            "replay_buffer_size": replay_buffer_size,
        }

    def initialize(self, observation_space, action_space, seed=None):
        self.model.initialize(observation_space, action_space)
        self.actor_updater.initialize(self.model, action_space)
        self.critic_updater.initialize(self.model)
        self.decay_flag = False

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
        self.replay_buffer.push(observations=self.last_observations,
                                actions=self.last_actions,
                                next_observations=observations,
                                rewards=rewards,
                                resets=resets,
                                terminations=terminations)
        # Prepare to update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.record(self.last_observations)
        if self.model.return_normalizer:
            self.model.return_normalizer.record(rewards)

        # Update the model if the replay is ready.
        if self.replay_buffer.ready(steps):
            self._update(steps)

        if self.decay_flag:  # Reducing noise to stabilize training
            # self.actor_updater.lr_actor *= self.decay_lr
            self.critic_updater.lr_critic *= self.decay_lr

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
