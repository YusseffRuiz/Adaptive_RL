import torch
import os
from MPO_Algorithm import ReplayBuffer, neural_networks
from MPO_Algorithm.agents import DDPG
from MPO_Algorithm.neural_networks import TwinCriticSoftDeterministicPolicyGradient, TwinCriticSoftQLearning
from MPO_Algorithm.utils import noise

class SAC(DDPG):
    """
    Soft Actor Critic
    https://arxiv.org/pdf/1801.01290.pdf
    """
    def __init__(self, model=None, hidden_size=256, replay_buffer=None, exploration=None, actor_updater=None, critic_updater=None):
        model = model or neural_networks.BaseModel(hidden_size=hidden_size).get_model()
        exploration = exploration or noise.NormalActionNoise
        replay_buffer = replay_buffer or ReplayBuffer()
        actor_updater = actor_updater or TwinCriticSoftDeterministicPolicyGradient()
        critic_updater = critic_updater or TwinCriticSoftQLearning()
        super().__init__(model,hidden_size, replay_buffer, exploration, actor_updater, critic_updater)

    def _stochastic_actions(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            return self.model.actor(observations).sample()

    def _policy(self, observations):
        return self._stochastic_actions(observations).numpy()

    def _greedy_actions(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            return self.model.actor(observations).loc