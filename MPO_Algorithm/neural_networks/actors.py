import torch
import copy
import utils
from MPO_Algorithm.agents.base_agent import trainable_variables


class ActorCriticWithTargets(torch.nn.Module):
    def __init__(
            self, actor, critic, observation_normalizer=None,
            return_normalizer=None, target_coeff=0.005
    ):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.target_actor = copy.deepcopy(actor)
        self.target_critic = copy.deepcopy(critic)
        self.observation_normalizer = observation_normalizer
        self.return_normalizer = return_normalizer
        self.target_coeff = target_coeff

    def initialize(self, observation_space, action_space):
        if self.observation_normalizer:
            self.observation_normalizer.initialize(observation_space.shape)
        self.actor.initialize(observation_space, action_space, self.observation_normalizer)
        self.critic.initialize(observation_space, action_space, self.observation_normalizer, self.return_normalizer)
        self.target_actor.initialize(observation_space, action_space, self.observation_normalizer)
        self.target_critic.initialize(observation_space, action_space, self.observation_normalizer,
                                      self.return_normalizer)
        self.online_variables = trainable_variables(self.actor)
        self.online_variables += trainable_variables(self.critic)
        self.target_variables = trainable_variables(self.target_actor)
        self.target_variables += trainable_variables(self.target_critic)
        for target in self.target_variables:
            target.requires_grad = False
        self.assign_targets()

    def assign_targets(self):
        for o, t in zip(self.online_variables, self.target_variables):
            t.data.copy_(o.data)

    def update_targets(self):
        with torch.no_grad():
            for o, t in zip(self.online_variables, self.target_variables):
                t.data.mul_(1 - self.target_coeff)
                t.data.add_(self.target_coeff * o.data)


class Actor(torch.nn.Module):
    def __init__(self, encoder, torso, head):
        super().__init__()
        self.encoder = encoder
        self.torso = torso
        self.head = head

    def initialize(self, observation_space, action_space, observation_normalizer=None):
        size = self.encoder.initialize(observation_space, observation_normalizer)
        size = self.torso.initialize(size)
        action_size = action_space.shape[0]
        self.head.initialize(size, action_size)

    def forward(self, *inputs):
        out = self.encoder(*inputs)
        out = self.torso(out)
        return self.head(out)


class DeterministicPolicyGradient:
    def __init__(self, lr_actor=3e-4, gradient_clip=0):
        self.gradient_clip = gradient_clip
        self.lr_actor = lr_actor

    def initialize(self, model):
        self.model = model
        self.variables = trainable_variables(model)
        self.optimizer = utils.local_optimizer(params=self.variables, lr=self.lr_actor)

    def __call__(self, observation):
        critic_variables = trainable_variables(self.model.critic)

        for var in critic_variables:
            var.requires_grad = False

        self.optimizer.zero_grad()
        actions = self.model.actor(observation)
        values = self.model.critic(observation, actions)
        loss = -values.mean()

        loss.backward()

        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step()

        for var in critic_variables:
            var.requires_grad = True

        return dict(loss=loss.detach())

class TwinCriticSoftDeterministicPolicyGradient:
    def __init__(self, lr_actor=1e-3, entropy_coeff=0.2, gradient_clip=0):
        self.entropy_coeff = entropy_coeff
        self.gradient_clip = gradient_clip
        self.lr_actor = lr_actor

    def initialize(self, model):
        self.model = model
        self.variables = trainable_variables(self.model.actor)
        self.optimizer = utils.local_optimizer(params=self.variables, lr=self.lr_actor)

    def __call__(self, observations):
        critic_1_variables = trainable_variables(self.model.critic_1)
        critic_2_variables = trainable_variables(self.model.critic_2)
        critic_variables = critic_1_variables + critic_2_variables

        for var in critic_variables:
            var.requires_grad = False

        self.optimizer.zero_grad()
        distributions = self.model.actor(observations)
        if hasattr(distributions, 'rsample_with_log_prob'):
            actions, log_probs = distributions.rsample_with_log_prob()
        else:
            actions = distributions.rsample()
            log_probs = distributions.log_prob(actions)
        log_probs = log_probs.sum(dim=-1)
        values_1 = self.model.critic_1(observations, actions)
        values_2 = self.model.critic_2(observations, actions)
        values = torch.min(values_1, values_2)
        loss = (self.entropy_coeff * log_probs - values).mean()

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step()

        for var in critic_variables:
            var.requires_grad = True

        return dict(loss=loss.detach())


FLOAT_EPSILON = 1e-8


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
        self.actor_variables = trainable_variables(self.model.actor)
        self.actor_optimizer = utils.local_optimizer(params=self.actor_variables, lr=self.lr_actor)

        # Dual variables.
        shape = [action_space.shape[0]] if self.per_dim_constraining else [1]
        self.log_alpha_mean = torch.nn.Parameter(torch.full(
            shape, self.initial_log_alpha_mean, dtype=torch.float32))
        self.dual_variables.append(self.log_alpha_mean)
        self.log_alpha_std = torch.nn.Parameter(torch.full(
            shape, self.initial_log_alpha_std, dtype=torch.float32))
        self.dual_variables.append(self.log_alpha_std)
        self.dual_optimizer = utils.local_optimizer(params=self.dual_variables, lr=self.lr_dual)

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
