import numpy as np
import torch
from .actors import ActorCriticWithTargets, Actor, ActorTwinCriticWithTargets, ActorCritic
from .critics import Critic, ValueHead
from Adaptive_RL import normalizers


class ObservationEncoder(torch.nn.Module):
    """
    Encodes raw observations by applying an optional normalizer if provided.

    Attributes:
    ----------
    observation_normalizer : object, optional
        Normalizes the observations if provided.

    Methods:
    -------
    initialize(observation_space, action_space=None, observation_normalizer=None):
        Initializes the encoder with the observation space and normalizer.
    forward(observations):
        Applies the normalizer (if available) to the observations and returns the result.
    """

    def initialize(self, observation_space, action_space, observation_normalizer=None):
        self.observation_normalizer = observation_normalizer
        observation_size = observation_space.shape[0]
        return observation_size

    def forward(self, observations):
        if self.observation_normalizer:
            observations = self.observation_normalizer(observations)
        return observations


class ObservationActionEncoder(torch.nn.Module):
    """
    Encodes both observations and actions, with optional observation normalization.

    Attributes:
    ----------
    observation_normalizer : object, optional
        Normalizes the observations if provided.

    Methods:
    -------
    initialize(observation_space, action_space, observation_normalizer=None):
        Initializes the encoder with observation and action spaces, returning their combined size.
    forward(observations, actions):
        Normalizes the observations and concatenates them with the actions.
    """

    def initialize(self, observation_space, action_space, observation_normalizer=None):
        self.observation_normalizer = observation_normalizer
        observation_size = observation_space.shape[0]
        action_size = action_space.shape[0]
        return observation_size + action_size

    def forward(self, observations, actions):
        if self.observation_normalizer:
            observations = self.observation_normalizer(observations)
        return torch.cat([observations, actions], dim=-1)


class NoisyLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Initializing weights
        self.weight_mu = torch.nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-1, 1))
        self.weight_sigma = torch.nn.Parameter(torch.Tensor(out_features, in_features).fill_(0.017))

    def forward(self, x):
        weight = self.weight_mu + self.weight_sigma * torch.randn_like(self.weight_sigma)
        return torch.nn.functional.linear(x, weight)


class MLP(torch.nn.Module):
    """
    A multi-layer perceptron (MLP) used as a torso for neural networks in the policy or value function.

    Attributes:
    ----------
    size : int
        The number of units in each hidden layer.
    activation : torch.nn.Module
        The activation function used after each layer.
    fn : callable, optional
        Initialization function for layers.

    Methods:
    -------
    initialize(input_size):
        Builds the MLP with the given input size and specified layers and activations.
    forward(inputs):
        Passes the inputs through the network.
    """

    def __init__(self, sizes, activation, fn=None, noise=False):
        super().__init__()
        self.sizes = sizes
        self.activation = activation
        self.fn = fn
        self.noise = noise

    def initialize(self, input_size):
        sizes = [input_size] + list(self.sizes)
        layers = []
        for i in range(len(sizes) - 1):
            if self.noise:
                layers += [NoisyLinear(sizes[i], sizes[i + 1]), self.activation(), torch.nn.LayerNorm(sizes[i + 1])]
            else:
                layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), self.activation(), torch.nn.LayerNorm(sizes[i+1])]
        self.model = torch.nn.Sequential(*layers)
        if self.fn is not None:
            self.model.apply(self.fn)
        return sizes[-1]

    def forward(self, inputs):
        return self.model(inputs)


class GaussianPolicyHead(torch.nn.Module):
    """
    Implements a Gaussian policy head that outputs actions based on a normal distribution.

    Attributes:
    ----------
    loc_activation : torch.nn.Module
        Activation function for the mean (loc).
    scale_activation : torch.nn.Module
        Activation function for the scale (variance).
    distribution : torch.distributions.Distribution
        The distribution used for sampling actions.

    Methods:
    -------
    initialize(input_size, action_size):
        Initializes the head with the sizes of inputs and actions.
    forward(inputs):
        Outputs the mean (loc) and variance (scale) for the action distribution.
    """

    def __init__(self, loc_activation=torch.nn.Tanh, loc_fn=None, scale_activation=torch.nn.Softplus, scale_min=1e-4,
                 scale_max=1, scale_fn=None, distribution=torch.distributions.normal.Normal):
        super().__init__()
        self.loc_activation = loc_activation
        self.loc_fn = loc_fn
        self.scale_activation = scale_activation
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.scale_fn = scale_fn
        self.distribution = distribution

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



class DeterministicPolicyHead(torch.nn.Module):
    def __init__(self, activation=torch.nn.Tanh, fn=None):
        super().__init__()
        self.activation = activation
        self.fn = fn

    def initialize(self, input_size, action_size):
        self.action_layer = torch.nn.Sequential(torch.nn.Linear(input_size, action_size), self.activation())
        if self.fn is not None:
            self.action_layer.apply(self.fn)

    def forward(self, inputs):
        return self.action_layer(inputs)


FLOAT_EPSILON = 1e-8  # Small constant to prevent divide by zero errors


class DetachedScaleGaussianPolicyHead(torch.nn.Module):
    def __init__(
        self, loc_activation=torch.nn.Tanh, loc_fn=None, log_scale_init=0.,
        scale_min=1e-4, scale_max=1.,
        distribution=torch.distributions.normal.Normal
    ):
        super().__init__()
        self.loc_activation = loc_activation
        self.loc_fn = loc_fn
        self.log_scale_init = log_scale_init
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.distribution = distribution

    def initialize(self, input_size, action_size):
        self.loc_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, action_size), self.loc_activation())
        if self.loc_fn:
            self.loc_layer.apply(self.loc_fn)
        log_scale = [[self.log_scale_init] * action_size]
        self.log_scale = torch.nn.Parameter(
            torch.as_tensor(log_scale, dtype=torch.float32))

    def forward(self, inputs):
        loc = self.loc_layer(inputs)
        batch_size = inputs.shape[0]
        scale = torch.nn.functional.softplus(self.log_scale) + FLOAT_EPSILON
        scale = torch.clamp(scale, self.scale_min, self.scale_max)
        scale = scale.repeat(batch_size, 1)
        return self.distribution(loc, scale)


class SquashedMultivariateNormalDiag:
    """
    Implements a squashed multivariate normal distribution for continuous action spaces.

    Attributes:
    ----------
    loc : torch.Tensor
        Mean of the distribution.
    scale : torch.Tensor
        Scale (variance) of the distribution.

    Methods:
    -------
    rsample_with_log_prob(shape=()):
        Samples from the distribution, applies the tanh squashing function, and computes log probabilities.
    rsample(shape=()):
        Samples from the distribution and applies the tanh squashing function.
    sample(shape=()):
        Samples from the distribution.
    """

    def __init__(self, loc, scale):
        self._distribution = torch.distributions.normal.Normal(loc, scale)

    def rsample_with_log_prob(self, shape=()):
        samples = self._distribution.rsample(shape)
        squashed_samples = torch.tanh(samples)
        log_probs = self._distribution.log_prob(samples)
        log_probs -= torch.log(1 - squashed_samples ** 2 + 1e-6)
        return squashed_samples, log_probs

    def rsample(self, shape=()):
        samples = self._distribution.rsample(shape)
        return torch.tanh(samples)

    def sample(self, shape=()):
        samples = self._distribution.sample(shape)
        return torch.tanh(samples)

    @property
    def loc(self):
        return torch.tanh(self._distribution.mean)


class BaseModel(torch.nn.Module):
    """
    Base Model for Actor-Critic architecture using soft updates for stabilization.

    Attributes:
    ----------
    hidden_size : int
        Number of hidden units in each layer.

    Methods:
    -------
    get_model():
        Returns the full Actor-Critic model with target networks.
    """

    def __init__(self, hidden_size=(64, 64), hidden_layers=1, activation_fn=torch.nn.ReLU):
        super().__init__()
        if hidden_layers > 1:
            if isinstance(hidden_size, int):
                self.neuron_shape = [hidden_size] * hidden_layers
            else:
                self.neuron_shape = hidden_size
        else:
            if len(hidden_size > 1 and hidden_layers == 1):
                self.neuron_shape = hidden_size*2
            self.neuron_shape = hidden_size
        self.activation_fn = activation_fn
    def get_model(self):
        return ActorCriticWithTargets(
            actor=Actor(
                encoder=ObservationEncoder(),
                torso=MLP(self.neuron_shape, self.activation_fn),
                head=GaussianPolicyHead()),
            critic=Critic(
                encoder=ObservationActionEncoder(),
                torso=MLP(self.neuron_shape, self.activation_fn),
                head=ValueHead()),
            observation_normalizer=normalizers.MeanStd()
        )


class ActorCriticDeterministic(BaseModel):

    def __init__(self, hidden_size=(64, 64), hidden_layers=1, activation_fn=torch.nn.ReLU):
        super().__init__(hidden_size, hidden_layers, activation_fn)

    def get_model(self):
        return ActorCriticWithTargets(
        actor=Actor(
            encoder=ObservationEncoder(),
            torso=MLP(self.neuron_shape, self.activation_fn),
            head=DeterministicPolicyHead()),
        critic=Critic(
                encoder=ObservationActionEncoder(),
                torso=MLP(self.neuron_shape, self.activation_fn),
                head=ValueHead()),
            observation_normalizer=normalizers.MeanStd())


class ActorTwinCriticsModelNetwork(BaseModel):
    """
    Actor-Twin-Critic Model Network with soft updates for actor and twin critics.

    Attributes:
    ----------
    hidden_size : int
        Number of hidden units in each layer.

    Methods:
    -------
    get_model():
        Returns the actor-twin-critic model with target networks.
    """

    def __init__(self, hidden_size=(64, 64), hidden_layers=1, activation_fn=torch.nn.ReLU):
        super().__init__(hidden_size, hidden_layers, activation_fn)

    def get_model(self):
        return ActorTwinCriticWithTargets(
            actor=Actor(
                encoder=ObservationEncoder(),
                torso=MLP(self.neuron_shape, self.activation_fn),
                head=GaussianPolicyHead(
                    loc_activation=torch.nn.Identity,
                    distribution=SquashedMultivariateNormalDiag)),
            critic=Critic(
                encoder=ObservationActionEncoder(),
                torso=MLP(self.neuron_shape, self.activation_fn),
                head=ValueHead()),
            observation_normalizer=normalizers.MeanStd())


class ActorCriticModelNetwork(BaseModel):
    def __init__(self, hidden_size=(64, 64), hidden_layers=1, activation_fn=torch.nn.Tanh, return_normalizer=False,
                 discount_factor=0.95):
        super().__init__(hidden_size, hidden_layers, activation_fn)
        self.return_normalizer = return_normalizer
        self.discount_factor = discount_factor

    def get_model(self):
        if self.return_normalizer:
            return ActorCritic(
            actor=Actor(
                encoder=ObservationEncoder(),
                torso=MLP(self.neuron_shape, self.activation_fn),
                head=DetachedScaleGaussianPolicyHead()),
            critic=Critic(
                encoder=ObservationEncoder(),
                torso=MLP(self.neuron_shape, self.activation_fn),
                head=ValueHead()),
            observation_normalizer=normalizers.MeanStd(),
            return_normalizer=normalizers.Return(self.discount_factor)
        )
        else:
            return ActorCritic(
            actor=Actor(
                encoder=ObservationEncoder(),
                torso=MLP(self.neuron_shape, self.activation_fn),
                head=DetachedScaleGaussianPolicyHead()),
            critic=Critic(
                encoder=ObservationEncoder(),
                torso=MLP(self.neuron_shape, self.activation_fn),
                head=ValueHead()),
            observation_normalizer=normalizers.MeanStd())
