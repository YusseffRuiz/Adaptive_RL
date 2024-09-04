import torch
from torch import nn
import torch.nn.functional as F
from MPO_Algorithm import Actor


class MatsuokaActor(Actor):
    def __init__(self, env, neuron_number, num_oscillators):
        """
        :param env
        """
        super().__init__(env)

        self.env = env
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space
        self.neuron_number = neuron_number
        self.num_oscillators = num_oscillators
        self.hidden_size = 256
        param_dim = self.num_oscillators * self.neuron_number
        self.action_dim = param_dim
        self.real_action_dim = env.action_space.shape[0]
        self.mean_layer = nn.Linear(self.hidden_size, param_dim)
        self.cholesky_layer = nn.Linear(self.hidden_size, (param_dim * (param_dim + 1)) // 2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.output_neuron = nn.Sequential(
            nn.Linear(param_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.real_action_dim)  # output to the real actions from the environment
        ).to(device)

    def calculate_mean(self, x, device):
        """
        mean recalculation without action space for params
        :param x: layer to calculate mean of the params
        :param device: Not in use at the moment
        :return: mean
        """
        mean = F.sigmoid(self.mean_layer(x)).to(device)
        return mean
