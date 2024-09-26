import torch
import os
from RL_Adaptive import ReplayBuffer, neural_networks
from RL_Adaptive.agents import DDPG
from RL_Adaptive.neural_networks import TwinCriticSoftDeterministicPolicyGradient, TwinCriticSoftQLearning
from RL_Adaptive.utils import explorations


class SAC(DDPG):
    """
    The SAC algorithm is an off-policy, entropy-regularized reinforcement learning algorithm designed
    to improve exploration in continuous action spaces. It allows for stochastic action selection by
    optimizing the maximum entropy objective, which encourages exploration by favoring actions with
    high entropy (randomness) while still aiming for high reward.

    Reference:
    - Paper: https://arxiv.org/pdf/1801.01290.pdf

    Attributes:
    ----------
    model : torch.nn.Module
        Neural network model representing the actor and twin critics. If not provided, it defaults to
        an `ActorTwinCriticsModelNetwork` with a specified hidden size.
    hidden_size : int
        The number of neurons in the hidden layers of the actor-critic networks. Defaults to 256.
    lr_actor : float
        Learning rate for the actor's optimizer. Defaults to 3e-4.
    lr_critic : float
        Learning rate for the critic's optimizer. Defaults to 3e-4.
    replay_buffer : ReplayBuffer
        Buffer to store experience tuples (state, action, reward, next_state). Defaults to a new `ReplayBuffer` instance.
    exploration : explorations.NormalNoiseExploration
        Exploration strategy that applies noise to the actions. Defaults to `NormalNoiseExploration`.
    actor_updater : TwinCriticSoftDeterministicPolicyGradient
        Update method for the actor network, using the Soft Deterministic Policy Gradient technique. Defaults to `TwinCriticSoftDeterministicPolicyGradient`.
    critic_updater : TwinCriticSoftQLearning
        Update method for the critic network, using the Soft Q-Learning technique. Defaults to `TwinCriticSoftQLearning`.
    """
    def __init__(self, model=None, hidden_size=256, lr_actor=3e-4, lr_critic=3e-4, discount_factor=0.99,
                 replay_buffer=None, exploration=None, actor_updater=None, critic_updater=None,
                 batch_size=512, return_step=5, steps_between_batches=20, replay_buffer_size=10e6
                 ):
        model = model or neural_networks.ActorTwinCriticsModelNetwork(hidden_size=hidden_size).get_model()
        exploration = exploration or explorations.NormalNoiseExploration()
        replay_buffer = replay_buffer or ReplayBuffer(return_steps=return_step, discount_factor=discount_factor,
                                                      batch_size=batch_size, steps_between_batches=steps_between_batches
                                                      , size=replay_buffer_size)
        actor_updater = actor_updater or TwinCriticSoftDeterministicPolicyGradient(lr_actor=lr_actor)
        critic_updater = critic_updater or TwinCriticSoftQLearning(lr_critic=lr_critic)
        super().__init__(model, hidden_size, discount_factor, replay_buffer_size, exploration,
                         actor_updater, critic_updater, batch_size, return_step, steps_between_batches,
                         replay_buffer_size)

    def _stochastic_actions(self, observations):
        """
        Samples stochastic actions from the actor network given the current observations.

        Parameters:
        ----------
        observations : np.array or torch.Tensor
        The current state/observations from the environment.

        Returns:
        -------
        torch.Tensor
        A tensor representing the sampled actions for the given observations.
        """
        observations = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            return self.model.actor(observations).sample()

    def _policy(self, observations):
        """
        Returns the policy action by sampling stochastic actions from the actor model.
        Already as numppy array
        """
        return self._stochastic_actions(observations).cpu().numpy()

    def _greedy_actions(self, observations):
        """
        Returns the greedy (mean) action from the actor network for the given observations.
        """
        observations = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            return self.model.actor(observations).loc