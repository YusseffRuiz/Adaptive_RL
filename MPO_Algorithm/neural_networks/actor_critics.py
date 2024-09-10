import torch
import copy
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


class ExpectedSARSA:
    def __init__(
        self, num_samples=20, gradient_clip=0, lr_critic=3e-4
    ):
        self.num_samples = num_samples
        self.loss = torch.nn.MSELoss()
        self.optimizer_fun = lambda params: torch.optim.AdamW(params, lr=lr_critic)
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = trainable_variables(self.model.critic)
        self.optimizer = self.optimizer_fun(self.variables)

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
            next_observations = next_observations.repeat([self.num_samples] + [1] * len(next_observations))
            next_observations = next_observations.view(next_observations.shape[0] * next_observations.shape[1],
                                                       *next_observations.shape[2:])
            next_values = self.model.target_critic(
                next_observations, next_actions)
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