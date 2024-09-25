import torch
import copy
from MPO_Algorithm.neural_networks.utils import local_optimizer, trainable_variables

class Critic(torch.nn.Module):
    def __init__(self, encoder, torso, head):
        super().__init__()
        self.encoder = encoder
        self.torso = torso
        self.head = head

    def initialize(self, observation_space, action_space, observation_normalizer=None,
                   return_normalizer=None):
        size = self.encoder.initialize(
            observation_space=observation_space, action_space=action_space,
            observation_normalizer=observation_normalizer)
        size = self.torso.initialize(size)
        self.head.initialize(size, return_normalizer)

    def forward(self, *inputs):
        out = self.encoder(*inputs)
        out = self.torso(out)
        return self.head(out)


class ValueHead(torch.nn.Module):
    def __init__(self, fn=None):
        super().__init__()
        self.v_layer = None
        self.return_normalizer = None
        self.fn = fn

    def initialize(self, input_size, return_normalizer=None):
        self.return_normalizer = return_normalizer
        self.v_layer = torch.nn.Linear(input_size, 1)
        if self.fn:
            self.v_layer.apply(self.fn)

    def forward(self, inputs):
        out = self.v_layer(inputs)
        out = torch.squeeze(out, -1)
        if self.return_normalizer:
            out = self.return_normalizer(out)
        return out



class DeterministicQLearning:
    def __init__(self, loss=None, lr_critic=3e-4, gradient_clip=0):
        self.loss = loss or torch.nn.MSELoss()
        self.gradient_clip = gradient_clip
        self.lr_critic = lr_critic

    def initialize(self, model):
        self.model = model
        self.variables = trainable_variables(self.model.critic)
        self.optimizer = local_optimizer(params=self.variables, lr=self.lr_critic)

    def __call__(
        self, observations, actions, next_observations, rewards, discounts
    ):
        with torch.no_grad():
            next_actions = self.model.target_actor(next_observations)
            next_values = self.model.target_critic(next_observations, next_actions)
            returns = rewards + discounts * next_values

        self.optimizer.zero_grad()
        values = self.model.critic(observations, actions)
        loss = self.loss(values, returns)

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step()

        return dict(loss=loss.detach(), q=values.detach())


class TwinCriticSoftQLearning:
    def __init__(
        self, loss=None, lr_critic=3e-4, entropy_coeff=0.2, gradient_clip=0
    ):
        self.loss = loss or torch.nn.MSELoss()
        self.entropy_coeff = entropy_coeff
        self.gradient_clip = gradient_clip
        self.lr_critic = lr_critic

    def initialize(self, model):
        self.model = model
        variables_1 = trainable_variables(self.model.critic_1)
        variables_2 = trainable_variables(self.model.critic_2)
        self.variables = variables_1 + variables_2
        self.optimizer = local_optimizer(params=self.variables, lr=self.lr_critic)

    def __call__(
        self, observations, actions, next_observations, rewards, discounts
    ):
        with torch.no_grad():
            next_distributions = self.model.actor(next_observations)
            if hasattr(next_distributions, 'rsample_with_log_prob'):
                outs = next_distributions.rsample_with_log_prob()
                next_actions, next_log_probs = outs
            else:
                next_actions = next_distributions.rsample()
                next_log_probs = next_distributions.log_prob(next_actions)
            next_log_probs = next_log_probs.sum(dim=-1)
            next_values_1 = self.model.target_critic_1(
                next_observations, next_actions)
            next_values_2 = self.model.target_critic_2(
                next_observations, next_actions)
            next_values = torch.min(next_values_1, next_values_2)
            returns = rewards + discounts * (
                next_values - self.entropy_coeff * next_log_probs)

        self.optimizer.zero_grad()
        values_1 = self.model.critic_1(observations, actions)
        values_2 = self.model.critic_2(observations, actions)
        loss_1 = self.loss(values_1, returns)
        loss_2 = self.loss(values_2, returns)
        loss = loss_1 + loss_2

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step()

        return dict(
            loss=loss.detach(), q1=values_1.detach(), q2=values_2.detach())





class ExpectedSARSA:
    def __init__(
        self, num_samples=20, gradient_clip=0, lr_critic=3e-4
    ):
        self.num_samples = num_samples
        self.loss = torch.nn.MSELoss()
        self.gradient_clip = gradient_clip
        self.lr_critic = lr_critic

    def initialize(self, model):
        self.model = model
        self.variables = trainable_variables(self.model.critic)
        self.optimizer = local_optimizer(self.variables, self.lr_critic)

    def __call__(
        self, observations, actions, next_observations, rewards, discounts
    ):
        # Approximate the expected next values.
        with torch.no_grad():
            next_target_distributions = self.model.target_actor(
                next_observations)
            next_actions = next_target_distributions.rsample(
                (self.num_samples,))
            next_actions = next_actions.view(next_actions.shape[0] * next_actions.shape[1], *next_actions.shape[2:])
            next_observations = next_observations[None].repeat([self.num_samples] + [1] * len(next_observations.shape))
            next_observations = next_observations.view(next_observations.shape[0] * next_observations.shape[1],
                                                       *next_observations.shape[2:])
            next_values = self.model.target_critic(next_observations, next_actions)
            next_values = next_values.view(self.num_samples, -1)
            next_values = next_values.mean(dim=0)
            returns = rewards + discounts * next_values

        self.optimizer.zero_grad()
        values = self.model.critic(observations, actions)
        loss = self.loss(returns, values)

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step()

        return dict(loss=loss.detach(), q=values.detach())
