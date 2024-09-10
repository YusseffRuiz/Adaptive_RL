import torch
from .actor_critics import ActorCriticWithTargets, Actor, Critic, ValueHead
from MPO_Algorithm import normalizers


class ObservationEncoder(torch.nn.Module):
    def initialize(
        self, observation_space, action_space=None,
        observation_normalizer=None,
    ):
        self.observation_normalizer = observation_normalizer
        observation_size = observation_space.shape[0]
        return observation_size

    def forward(self, observations):
        if self.observation_normalizer:
            observations = self.observation_normalizer(observations)
        return observations


class ObservationActionEncoder(torch.nn.Module):
    def initialize(
        self, observation_space, action_space, observation_normalizer=None
    ):
        self.observation_normalizer = observation_normalizer
        observation_size = observation_space.shape[0]
        action_size = action_space.shape[0]
        return observation_size + action_size

    def forward(self, observations, actions):
        if self.observation_normalizer:
            observations = self.observation_normalizer(observations)
        return torch.cat([observations, actions], dim=-1)


class MLP(torch.nn.Module):
    def __init__(self, size, activation, fn=None):
        super().__init__()
        self.model = None
        self.sizes = [size, size]
        self.activation = activation
        self.fn = fn

    def initialize(self, input_size):
        sizes = [input_size] + list(self.sizes)
        layers = []
        for i in range(len(sizes) - 1):
            layers += [torch.nn.Linear(sizes[i], sizes[i + 1]),
                       self.activation()]
        self.model = torch.nn.Sequential(*layers)
        if self.fn is not None:
            self.model.apply(self.fn)
        return sizes[-1]

    def forward(self, inputs):
        return self.model(inputs)


class GaussianPolicyHead(torch.nn.Module):
    def __init__(
        self, loc_activation=torch.nn.Tanh, loc_fn=None,
        scale_activation=torch.nn.Softplus, scale_min=1e-4, scale_max=1,
        scale_fn=None, distribution=torch.distributions.normal.Normal
    ):
        super().__init__()
        self.loc_activation = loc_activation
        self.loc_fn = loc_fn
        self.scale_activation = scale_activation
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.scale_fn = scale_fn
        self.distribution = distribution

        self.scale_layer = None
        self.loc_layer = None

    def initialize(self, input_size, action_size):
        self.loc_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, action_size), self.loc_activation())
        if self.loc_fn:
            self.loc_layer.apply(self.loc_fn)
        self.scale_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, action_size), self.scale_activation())
        if self.scale_fn:
            self.scale_layer.apply(self.scale_fn)

    def forward(self, inputs):
        loc = self.loc_layer(inputs)
        scale = self.scale_layer(inputs)
        scale = torch.clamp(scale, self.scale_min, self.scale_max)
        return self.distribution(loc, scale)


class BaseModel(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def get_model(self):
        return ActorCriticWithTargets(
            actor=Actor(
                encoder=ObservationEncoder(),
                torso=MLP(self.hidden_size, torch.nn.SiLU),
                head=GaussianPolicyHead()),
            critic=Critic(
                encoder=ObservationActionEncoder(),
                torso=MLP(self.hidden_size, torch.nn.SiLU),
                head=ValueHead()),
            observation_normalizer=normalizers.MeanStd()
        )
