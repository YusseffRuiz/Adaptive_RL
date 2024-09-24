import torch
import os
from MPO_Algorithm import logger, ReplayBuffer, neural_networks
from MPO_Algorithm.agents import base_agent


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

    def step(self, observations):
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


FLOAT_EPSILON = 1e-8


def actor_optimizer(params, lr_actor=3e-4):
    return torch.optim.AdamW(params, lr=lr_actor)


def dual_optimizer(params, lr_dual=3e-4):
    return torch.optim.AdamW(params, lr=lr_dual)


class MaximumAPosterioriPolicyOptimization:
    def __init__(
        self, num_samples=50, epsilon=5e-2, epsilon_penalty=1e-3,
        epsilon_mean=1e-3, epsilon_std=1e-6, initial_log_temperature=1.,
        initial_log_alpha_mean=1., initial_log_alpha_std=10.,
        min_log_dual=-18., per_dim_constraining=True, action_penalization=True,
        gradient_clip=0.1, lr_actor=3e-4, lr_dual=3e-4,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_alpha_std = None
        self.log_alpha_mean = None
        self.actor_variables = None
        self.model = None
        self.num_samples = num_samples
        self.epsilon = epsilon
        self.epsilon_mean = epsilon_mean
        self.epsilon_std = epsilon_std
        self.initial_log_temperature = initial_log_temperature
        self.initial_log_alpha_mean = initial_log_alpha_mean
        self.initial_log_alpha_std = initial_log_alpha_std
        self.min_log_dual = torch.as_tensor(min_log_dual, dtype=torch.float32)
        self.action_penalization = action_penalization
        self.epsilon_penalty = epsilon_penalty
        self.per_dim_constraining = per_dim_constraining
        self.gradient_clip = gradient_clip
        self.lr_actor = lr_actor
        self.lr_dual = lr_dual

        # Dual Variables
        self.dual_variables = []
        self.log_temperature = torch.nn.Parameter(torch.as_tensor(
            [initial_log_temperature], dtype=torch.float32))
        self.dual_variables.append(self.log_temperature)
        if self.action_penalization:
            self.log_penalty_temperature = torch.nn.Parameter(torch.as_tensor(
                [initial_log_temperature], dtype=torch.float32))
            self.dual_variables.append(self.log_penalty_temperature)

    def initialize(self, model, action_space):
        self.model = model
        self.actor_variables = base_agent.trainable_variables(self.model.actor)
        self.actor_optimizer = actor_optimizer(self.actor_variables, lr_actor=self.lr_actor)

        # Dual variables.
        shape = [action_space.shape[0]] if self.per_dim_constraining else [1]
        self.log_alpha_mean = torch.nn.Parameter(torch.full(
            shape, self.initial_log_alpha_mean, dtype=torch.float32))
        self.dual_variables.append(self.log_alpha_mean)
        self.log_alpha_std = torch.nn.Parameter(torch.full(
            shape, self.initial_log_alpha_std, dtype=torch.float32))
        self.dual_variables.append(self.log_alpha_std)
        self.dual_optimizer = dual_optimizer(self.dual_variables, self.lr_dual)

    def __call__(self, observations):
        def parametric_kl_and_dual_losses(kl, alpha, epsilon):
            kl_mean = kl.mean(dim=0)
            kl_loss = (alpha.detach() * kl_mean).sum()
            alpha_loss = (alpha * (epsilon - kl_mean.detach())).sum()
            return kl_loss, alpha_loss

        def weights_and_temperature_loss(q_values, epsilon, temperature):
            tempered_q_values = q_values.detach() / temperature
            weights = torch.nn.functional.softmax(tempered_q_values, dim=0)
            weights = weights.detach()

            # Temperature loss (dual of the E-step).
            q_log_sum_exp = torch.logsumexp(tempered_q_values, dim=0)
            num_actions = torch.as_tensor(
                q_values.shape[0], dtype=torch.float32)
            log_num_actions = torch.log(num_actions)
            loss = epsilon + (q_log_sum_exp).mean() - log_num_actions
            loss = temperature * loss

            return weights, loss

        # Use independent normals to satisfy KL constraints per-dimension.
        def independent_normals(distribution_1, distribution_2=None):
            distribution_2 = distribution_2 or distribution_1
            return torch.distributions.independent.Independent(
                torch.distributions.normal.Normal(
                    distribution_1.mean, distribution_2.stddev), -1)

        with torch.no_grad():
            self.log_temperature.data.copy_(
                torch.maximum(self.min_log_dual, self.log_temperature))
            self.log_alpha_mean.data.copy_(
                torch.maximum(self.min_log_dual, self.log_alpha_mean))
            self.log_alpha_std.data.copy_(
                torch.maximum(self.min_log_dual, self.log_alpha_std))
            if self.action_penalization:
                self.log_penalty_temperature.data.copy_(torch.maximum(
                    self.min_log_dual, self.log_penalty_temperature))

            target_distributions = self.model.target_actor(observations)
            actions = target_distributions.sample((self.num_samples,))

            # Repeat for the number of samples
            tiled_observations = observations[None].repeat([self.num_samples] + [1] * len(observations.shape))
            flat_observations = tiled_observations.view(tiled_observations.shape[0] * tiled_observations.shape[1],
                                                        *tiled_observations.shape[2:])  # Merging first 2 dimensions

            flat_actions = actions.view(actions.shape[0] * actions.shape[1], *actions.shape[2:])
            values = self.model.target_critic(flat_observations, flat_actions)
            values = values.view(self.num_samples, -1)

            assert isinstance(
                target_distributions, torch.distributions.normal.Normal)
            target_distributions = independent_normals(target_distributions)

        self.actor_optimizer.zero_grad()
        self.dual_optimizer.zero_grad()

        distributions = self.model.actor(observations)
        distributions = independent_normals(distributions)

        temperature = (torch.nn.functional.softplus(
            self.log_temperature) + FLOAT_EPSILON).to(self.device)
        alpha_mean = (torch.nn.functional.softplus(
            self.log_alpha_mean) + FLOAT_EPSILON).to(self.device)
        alpha_std = (torch.nn.functional.softplus(
            self.log_alpha_std) + FLOAT_EPSILON).to(self.device)
        weights, temperature_loss = weights_and_temperature_loss(
            values, self.epsilon, temperature)

        # Action penalization is quadratic beyond [-1, 1].
        penalty_temperature = 0
        if self.action_penalization:
            penalty_temperature = (torch.nn.functional.softplus(
                self.log_penalty_temperature) + FLOAT_EPSILON).to(self.device)
            diff_bounds = actions - torch.clamp(actions, -1, 1)
            action_bound_costs = -torch.norm(diff_bounds, dim=-1)
            penalty_weights, penalty_temperature_loss = \
                weights_and_temperature_loss(
                    action_bound_costs.to(self.device),
                    self.epsilon_penalty, penalty_temperature)
            weights += penalty_weights
            temperature_loss += penalty_temperature_loss

        # Decompose the policy into fixed-mean and fixed-std distributions.
        fixed_std_distribution = independent_normals(
            distributions.base_dist, target_distributions.base_dist)
        fixed_mean_distribution = independent_normals(
            target_distributions.base_dist, distributions.base_dist)

        # Compute the decomposed policy losses.
        policy_mean_losses = (fixed_std_distribution.base_dist.log_prob(
            actions).sum(dim=-1) * weights).sum(dim=0)
        policy_mean_loss = -policy_mean_losses.mean()
        policy_std_losses = (fixed_mean_distribution.base_dist.log_prob(actions).sum(dim=-1) * weights).sum(dim=0)
        policy_std_loss = -policy_std_losses.mean()

        # Compute the decomposed KL between the target and online policies.
        if self.per_dim_constraining:
            kl_mean = torch.distributions.kl.kl_divergence(
                target_distributions.base_dist,
                fixed_std_distribution.base_dist)
            kl_std = torch.distributions.kl.kl_divergence(
                target_distributions.base_dist,
                fixed_mean_distribution.base_dist)
        else:
            kl_mean = torch.distributions.kl.kl_divergence(
                target_distributions, fixed_std_distribution)
            kl_std = torch.distributions.kl.kl_divergence(
                target_distributions, fixed_mean_distribution)

        # Compute the alpha-weighted KL-penalty and dual losses.
        kl_mean_loss, alpha_mean_loss = parametric_kl_and_dual_losses(
            kl_mean, alpha_mean, self.epsilon_mean)
        kl_std_loss, alpha_std_loss = parametric_kl_and_dual_losses(
            kl_std, alpha_std, self.epsilon_std)

        # Combine losses.
        policy_loss = policy_mean_loss + policy_std_loss
        kl_loss = kl_mean_loss + kl_std_loss
        dual_loss = alpha_mean_loss + alpha_std_loss + temperature_loss
        loss = policy_loss + kl_loss + dual_loss

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.actor_variables, self.gradient_clip)
            torch.nn.utils.clip_grad_norm_(
                self.dual_variables, self.gradient_clip)
        self.actor_optimizer.step()
        self.dual_optimizer.step()

        dual_variables = dict(
            temperature=temperature.detach(), alpha_mean=alpha_mean.detach(),
            alpha_std=alpha_std.detach())
        if self.action_penalization:
            dual_variables['penalty_temperature'] = \
                penalty_temperature.detach()

        return dict(
            policy_mean_loss=policy_mean_loss.detach(),
            policy_std_loss=policy_std_loss.detach(),
            kl_mean_loss=kl_mean_loss.detach(),
            kl_std_loss=kl_std_loss.detach(),
            alpha_mean_loss=alpha_mean_loss.detach(),
            alpha_std_loss=alpha_std_loss.detach(),
            temperature_loss=temperature_loss.detach(),
            **dual_variables)
