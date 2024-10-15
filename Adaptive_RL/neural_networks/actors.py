import torch
import copy
from Adaptive_RL.neural_networks.utils import local_optimizer, trainable_variables, tile, merge_first_two_dims

class ActorCriticWithTargets(torch.nn.Module):
    """
    Actor-Critic Model with Target Networks

    This class implements an actor-critic architecture with target networks for both the actor and critic.
    Target networks are used to stabilize training by providing a slowly updated reference for the critic.

    Attributes:
    ----------
    actor : torch.nn.Module
        The actor network, responsible for determining actions based on observations.
    critic : torch.nn.Module
        The critic network, responsible for evaluating the value of state-action pairs.
    target_actor : torch.nn.Module
        A copy of the actor network used as the target.
    target_critic : torch.nn.Module
        A copy of the critic network used as the target.
    observation_normalizer : object, optional
        An optional normalizer for observations.
    return_normalizer : object, optional
        An optional normalizer for returns.
    target_coeff : float
        The coefficient used for soft updates to the target networks.
    """

    def __init__(self, actor, critic, observation_normalizer=None, return_normalizer=None, target_coeff=0.005):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.target_actor = copy.deepcopy(actor)
        self.target_critic = copy.deepcopy(critic)
        self.observation_normalizer = observation_normalizer
        self.return_normalizer = return_normalizer
        self.target_coeff = target_coeff

    def initialize(self, observation_space, action_space):
        """
        Initializes the actor, critic, and target networks using the observation and action spaces.
        Also sets up the trainable and target variables.

        Parameters:
        ----------
        observation_space : object
            Space of the observations.
        action_space : object
            Space of the actions.
        """
        if self.observation_normalizer:
            self.observation_normalizer.initialize(observation_space.shape)
        self.actor.initialize(observation_space, action_space, self.observation_normalizer)
        self.critic.initialize(observation_space, action_space, self.observation_normalizer, self.return_normalizer)
        self.target_actor.initialize(observation_space, action_space, self.observation_normalizer)
        self.target_critic.initialize(observation_space, action_space, self.observation_normalizer, self.return_normalizer)
        self.online_variables = trainable_variables(self.actor) + trainable_variables(self.critic)
        self.target_variables = trainable_variables(self.target_actor) + trainable_variables(self.target_critic)
        for target in self.target_variables:
            target.requires_grad = False
        self.assign_targets()

    def assign_targets(self):
        """Assigns the online network's weights to the target networks."""
        for o, t in zip(self.online_variables, self.target_variables):
            t.data.copy_(o.data)

    def update_targets(self):
        """
        Performs a soft update of the target networks, blending the current target values with the online values.
        """
        with torch.no_grad():
            for o, t in zip(self.online_variables, self.target_variables):
                t.data.mul_(1 - self.target_coeff)
                t.data.add_(self.target_coeff * o.data)


class Actor(torch.nn.Module):
    """
    Actor Network

    The actor network predicts actions based on the input observations.

    Attributes:
    ----------
    encoder : torch.nn.Module
        Network module responsible for encoding the input observations.
    torso : torch.nn.Module
        Network module responsible for processing the encoded observations.
    head : torch.nn.Module
        The final network module responsible for outputting actions.
    """

    def __init__(self, encoder, torso, head):
        super().__init__()
        self.encoder = encoder
        self.torso = torso
        self.head = head

    def initialize(self, observation_space, action_space, observation_normalizer=None):
        """
        Initializes the actor network based on the observation and action spaces.

        Parameters:
        ----------
        observation_space : object
            Space of the observations.
        action_space : object
            Space of the actions.
        observation_normalizer : object, optional
            Optional normalizer for observations.
        """
        size = self.encoder.initialize(observation_space, observation_normalizer)
        size = self.torso.initialize(size)
        action_size = action_space.shape[0]
        self.head.initialize(size, action_size)

    def forward(self, *inputs):
        """Performs a forward pass through the actor network."""
        out = self.encoder(*inputs)
        out = self.torso(out)
        return self.head(out)


class ActorCritic(torch.nn.Module):
    def __init__(
        self, actor, critic, observation_normalizer=None, return_normalizer=None):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.observation_normalizer = observation_normalizer
        self.return_normalizer = return_normalizer

    def initialize(self, observation_space, action_space):
        if self.observation_normalizer:
            self.observation_normalizer.initialize(observation_space.shape)
        self.actor.initialize(observation_space, action_space, self.observation_normalizer)
        self.critic.initialize(observation_space, action_space, self.observation_normalizer, self.return_normalizer)


class ActorTwinCriticWithTargets(torch.nn.Module):
    """
    Actor-Critic Architecture with Twin Critics and Target Networks

    This class implements an actor-critic architecture with two critic networks (for double Q-learning)
    and target networks for both the actor and critic networks.

    Attributes:
    ----------
    actor : torch.nn.Module
        The actor network.
    critic_1 : torch.nn.Module
        First critic network.
    critic_2 : torch.nn.Module
        Second critic network (twin critic).
    target_actor : torch.nn.Module
        Target network for the actor.
    target_critic_1 : torch.nn.Module
        Target network for the first critic.
    target_critic_2 : torch.nn.Module
        Target network for the second critic.
    observation_normalizer : object, optional
        Optional normalizer for observations.
    return_normalizer : object, optional
        Optional normalizer for returns.
    target_coeff : float
        Coefficient used for soft updates to the target networks.
    """

    def __init__(self, actor, critic, observation_normalizer=None, return_normalizer=None, target_coeff=0.005):
        super().__init__()
        self.actor = actor
        self.critic_1 = critic
        self.critic_2 = copy.deepcopy(critic)
        self.target_actor = copy.deepcopy(actor)
        self.target_critic_1 = copy.deepcopy(critic)
        self.target_critic_2 = copy.deepcopy(critic)
        self.observation_normalizer = observation_normalizer
        self.return_normalizer = return_normalizer
        self.target_coeff = target_coeff

    def initialize(self, observation_space, action_space):
        """
        Initializes the actor, critics, and their target networks using the observation and action spaces.
        """
        if self.observation_normalizer:
            self.observation_normalizer.initialize(observation_space.shape)
        self.actor.initialize(observation_space, action_space, self.observation_normalizer)
        self.critic_1.initialize(observation_space, action_space, self.observation_normalizer, self.return_normalizer)
        self.critic_2.initialize(observation_space, action_space, self.observation_normalizer, self.return_normalizer)
        self.target_actor.initialize(observation_space, action_space, self.observation_normalizer)
        self.target_critic_1.initialize(observation_space, action_space, self.observation_normalizer, self.return_normalizer)
        self.target_critic_2.initialize(observation_space, action_space, self.observation_normalizer, self.return_normalizer)
        self.online_variables = trainable_variables(self.actor) + trainable_variables(self.critic_1) + trainable_variables(self.critic_2)
        self.target_variables = trainable_variables(self.target_actor) + trainable_variables(self.target_critic_1) + trainable_variables(self.target_critic_2)
        for target in self.target_variables:
            target.requires_grad = False
        self.assign_targets()

    def assign_targets(self):
        """Copies the weights from the online networks to the target networks."""
        for o, t in zip(self.online_variables, self.target_variables):
            t.data.copy_(o.data)

    def update_targets(self):
        """
        Performs a soft update of the target networks by blending the target weights with the online weights.
        """
        with torch.no_grad():
            for o, t in zip(self.online_variables, self.target_variables):
                t.data.mul_(1 - self.target_coeff)
                t.data.add_(self.target_coeff * o.data)


class ClippedRatio:
    def __init__(
        self, learning_rate=3e-4, ratio_clip=0.2, kl_threshold=0.015,
        entropy_coeff=0, gradient_clip=0
    ):
        self.learning_rate = learning_rate
        self.ratio_clip = ratio_clip
        self.kl_threshold = kl_threshold
        self.entropy_coeff = entropy_coeff
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = trainable_variables(self.model.actor)
        self.optimizer = local_optimizer(self.variables, lr=self.learning_rate)

    def __call__(self, observations, actions, advantages, log_probs):
        if (advantages == 0.).all():
            loss = torch.as_tensor(0., dtype=torch.float32)
            kl = torch.as_tensor(0., dtype=torch.float32)
            clip_fraction = torch.as_tensor(0., dtype=torch.float32)
            with torch.no_grad():
                distributions = self.model.actor(observations)
                entropy = distributions.entropy().mean()
                std = distributions.stddev.mean()

        else:
            self.optimizer.zero_grad()
            distributions = self.model.actor(observations)
            new_log_probs = distributions.log_prob(actions).sum(dim=-1)
            ratios_1 = torch.exp(new_log_probs - log_probs)
            surrogates_1 = advantages * ratios_1
            ratio_low = 1 - self.ratio_clip
            ratio_high = 1 + self.ratio_clip
            ratios_2 = torch.clamp(ratios_1, ratio_low, ratio_high)
            surrogates_2 = advantages * ratios_2
            loss = -(torch.min(surrogates_1, surrogates_2)).mean()
            entropy = distributions.entropy().mean()
            if self.entropy_coeff != 0:
                loss -= self.entropy_coeff * entropy

            loss.backward()
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.variables, self.gradient_clip)
            self.optimizer.step()

            loss = loss.detach()
            with torch.no_grad():
                kl = (log_probs - new_log_probs).mean()
            entropy = entropy.detach()
            clipped = ratios_1.gt(ratio_high) | ratios_1.lt(ratio_low)
            clip_fraction = torch.as_tensor(
                clipped, dtype=torch.float32).mean()
            std = distributions.stddev.mean().detach()

        return dict(
            loss=loss, kl=kl, entropy=entropy, clip_fraction=clip_fraction,
            std=std, stop=kl > self.kl_threshold)



class StochasticPolicyGradient:
    def __init__(self, lr_actor=3e-4, entropy_coeff=0, gradient_clip=0):
        self.entropy_coeff = entropy_coeff
        self.gradient_clip = gradient_clip
        self.lr_actor = lr_actor

    def initialize(self, model):
        self.model = model
        self.variables = trainable_variables(self.model.actor)
        self.optimizer = local_optimizer(self.variables, lr=self.lr_actor)

    def __call__(self, observations, actions, advantages, log_probs):
        if (advantages == 0.).all():
            loss = torch.as_tensor(0., dtype=torch.float32)
            kl = torch.as_tensor(0., dtype=torch.float32)
            with torch.no_grad():
                distributions = self.model.actor(observations)
                entropy = distributions.entropy().mean()
                std = distributions.stddev.mean()

        else:
            self.optimizer.zero_grad()
            distributions = self.model.actor(observations)
            new_log_probs = distributions.log_prob(actions).sum(dim=-1)
            loss = -(advantages * new_log_probs).mean()
            entropy = distributions.entropy().mean()
            if self.entropy_coeff != 0:
                loss -= self.entropy_coeff * entropy

            loss.backward()
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.variables, self.gradient_clip)
            self.optimizer.step()

            loss = loss.detach()
            kl = (log_probs - new_log_probs).mean().detach()
            entropy = entropy.detach()
            std = distributions.stddev.mean().detach()

        return dict(loss=loss, kl=kl, entropy=entropy, std=std)


class DeterministicPolicyGradient:
    """
    Implements Deterministic Policy Gradient (DPG) for training an actor-critic model.

    Attributes:
    ----------
    lr_actor : float
        Learning rate for the actor optimizer.
    gradient_clip : float
        Maximum value for gradient clipping to avoid exploding gradients (default: 0, meaning no clipping).

    Methods:
    -------
    initialize(model):
        Initializes the optimizer and collects the trainable variables for the actor.
    __call__(observation):
        Updates the actor based on the deterministic policy gradient.
    """

    def __init__(self, lr_actor=3e-4, gradient_clip=0):
        self.gradient_clip = gradient_clip
        self.lr_actor = lr_actor

    def initialize(self, model):
        """Initializes the optimizer for the actor."""
        self.model = model
        self.variables = trainable_variables(model)
        self.optimizer = local_optimizer(params=self.variables, lr=self.lr_actor)

    def __call__(self, observation):
        """Calculates the loss and performs an actor update step."""
        critic_variables = trainable_variables(self.model.critic)

        for var in critic_variables:
            var.requires_grad = False

        self.optimizer.zero_grad()
        actions = self.model.actor(observation)
        values = self.model.critic(observation, actions)
        loss = -values.mean()  # Maximize critic value for the selected actions

        loss.backward()

        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step()

        for var in critic_variables:
            var.requires_grad = True

        return dict(loss=loss.detach())


class TwinCriticSoftDeterministicPolicyGradient:
    """
    Implements Twin-Critic Soft Deterministic Policy Gradient (SAC) for actor-critic training.

    Attributes:
    ----------
    lr_actor : float
        Learning rate for the actor optimizer.
    entropy_coeff : float
        Coefficient for the entropy term to encourage exploration.
    gradient_clip : float
        Maximum value for gradient clipping (default: 0, no clipping).

    Methods:
    -------
    initialize(model):
        Initializes the optimizer and collects trainable variables.
    __call__(observations):
        Updates the actor using soft deterministic policy gradient with entropy regularization.
    """

    def __init__(self, lr_actor=1e-3, entropy_coeff=0.2, gradient_clip=0):
        self.entropy_coeff = entropy_coeff
        self.gradient_clip = gradient_clip
        self.lr_actor = lr_actor

    def initialize(self, model):
        """Initializes the optimizer for the actor and retrieves trainable variables."""
        self.model = model
        self.variables = trainable_variables(self.model.actor)
        self.optimizer = local_optimizer(params=self.variables, lr=self.lr_actor)

    def __call__(self, observations):
        """Updates the actor with respect to the entropy-regularized loss."""
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

        # Entropy-regularized loss
        loss = (self.entropy_coeff * log_probs - values).mean()

        loss.backward()

        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step()

        for var in critic_variables:
            var.requires_grad = True

        return dict(loss=loss.detach())


FLOAT_EPSILON = 1e-8  # Small constant to prevent divide by zero errors


class MaximumAPosterioriPolicyOptimization:
    """
    Maximum A Posteriori Policy Optimization (MPO) Algorithm

    MPO balances policy optimization with constraints on KL divergence to ensure stable learning.
    This class implements the MPO algorithm with dual variables for adjusting the constraints dynamically.

    Attributes:
        num_samples (int): Number of action samples for estimating the expected return.
        epsilon (float): KL divergence constraint to limit policy changes.
        epsilon_penalty (float): KL constraint for action penalization.
        epsilon_mean (float): Mean constraint for KL divergence regularization.
        epsilon_std (float): Standard deviation constraint for KL divergence regularization.
        initial_log_temperature (float): Initial temperature parameter controlling entropy regularization.
        initial_log_alpha_mean (float): Initial value for alpha mean dual variable (for KL).
        initial_log_alpha_std (float): Initial value for alpha std dual variable (for KL).
        min_log_dual (float): Minimum value for dual variables to prevent values from going too low.
        per_dim_constraining (bool): Whether to constrain KL divergence per dimension of action space.
        action_penalization (bool): Whether to penalize actions beyond a defined range.
        gradient_clip (float): Clipping threshold for gradient updates to avoid exploding gradients.
        lr_actor (float): Learning rate for actor policy updates.
        lr_dual (float): Learning rate for dual variable updates.
    """
    def __init__(
        self, num_samples=20, epsilon=0.1, epsilon_penalty=1e-3,
        epsilon_mean=1e-3, epsilon_std=1e-5, initial_log_temperature=1.,
        initial_log_alpha_mean=1., initial_log_alpha_std=10.,
        min_log_dual=-18., per_dim_constraining=True, action_penalization=True,
        gradient_clip=0.1, lr_actor=3e-4, lr_dual=1e-2,
    ):
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


    def initialize(self, model, action_space):
        """
        Initialize the model and dual variables, and set up the optimizers for actor and dual updates.

        Args:
            model (torch.nn.Module): The actor-critic model used in MPO.
            action_space: The action space of the environment (used for dimensional constraints).
        """
        self.model = model
        self.actor_variables = trainable_variables(self.model.actor)
        self.optimizer = local_optimizer(params=self.actor_variables, lr=self.lr_actor)

        # Dual variables.
        self.dual_variables = []
        self.log_temperature = torch.nn.Parameter(torch.as_tensor(
            [self.initial_log_temperature], dtype=torch.float32))
        self.dual_variables.append(self.log_temperature)
        shape = [action_space.shape[0]] if self.per_dim_constraining else [1]
        self.log_alpha_mean = torch.nn.Parameter(torch.full(
            shape, self.initial_log_alpha_mean, dtype=torch.float32))
        self.dual_variables.append(self.log_alpha_mean)
        self.log_alpha_std = torch.nn.Parameter(torch.full(
            shape, self.initial_log_alpha_std, dtype=torch.float32))
        self.dual_variables.append(self.log_alpha_std)
        if self.action_penalization:
            self.log_penalty_temperature = torch.nn.Parameter(torch.as_tensor(
                [self.initial_log_temperature], dtype=torch.float32))
            self.dual_variables.append(self.log_penalty_temperature)
        self.dual_optimizer = local_optimizer(params=self.dual_variables, lr=self.lr_dual)

    def __call__(self, observations):
        """
        Perform a forward pass to optimize the policy based on KL constraints and value function estimates.

        Args:
            observations: The observations from the environment.

        Returns:
            dict: A dictionary of loss values for logging and diagnostics.
        """
        def parametric_kl_and_dual_losses(kl, alpha, epsilon):
            # Compute the KL loss and alpha update based on the constraints
            kl_mean = kl.mean(dim=0)
            kl_loss = (alpha.detach() * kl_mean).sum()
            alpha_loss = (alpha * (epsilon - kl_mean.detach())).sum()
            return kl_loss, alpha_loss

        def weights_and_temperature_loss(q_values, epsilon, temperature):
            # Compute the weights for updating the policy and the temperature loss for regularization
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

        # Begin forward pass and temperature updates
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

            # Sample actions from target actor
            target_distributions = self.model.target_actor(observations)
            actions = target_distributions.sample((self.num_samples,))

            # Repeat for the number of samples
            tiled_observations = tile(observations, self.num_samples)
            flat_observations = merge_first_two_dims(tiled_observations)

            flat_actions = merge_first_two_dims(actions)
            values = self.model.target_critic(flat_observations, flat_actions)
            values = values.view(self.num_samples, -1)

            assert isinstance(
                target_distributions, torch.distributions.normal.Normal)
            target_distributions = independent_normals(target_distributions)

        # Begin gradient update steps
        self.optimizer.zero_grad()
        self.dual_optimizer.zero_grad()

        distributions = self.model.actor(observations)
        distributions = independent_normals(distributions)

        temperature = (torch.nn.functional.softplus(self.log_temperature) + FLOAT_EPSILON)
        alpha_mean = (torch.nn.functional.softplus(self.log_alpha_mean) + FLOAT_EPSILON)
        alpha_std = (torch.nn.functional.softplus(self.log_alpha_std) + FLOAT_EPSILON)

        # Compute weights and temperature loss
        weights, temperature_loss = weights_and_temperature_loss(
            values, self.epsilon, temperature)

        # Action penalization is quadratic beyond [-1, 1].
        if self.action_penalization:
            penalty_temperature = torch.nn.functional.softplus(
                self.log_penalty_temperature) + FLOAT_EPSILON
            diff_bounds = actions - torch.clamp(actions, -1, 1)
            action_bound_costs = -torch.norm(diff_bounds, dim=-1)
            penalty_weights, penalty_temperature_loss = \
                weights_and_temperature_loss(
                    action_bound_costs,
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
        self.optimizer.step()
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
