import torch
from Adaptive_RL import neural_networks
from Adaptive_RL.agents import DDPG
from Adaptive_RL.neural_networks import TwinCriticSoftDeterministicPolicyGradient, TwinCriticSoftQLearning, ResidualSACModel
from Adaptive_RL.utils import explorations, ReplayBuffer


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
    def __init__(self, hidden_size=256, hidden_layers=2, learning_rate=3e-4, lr_critic=None,
                 entropy_coeff=0.001, tau=0.005,
                 batch_size=256, return_step=3,
                 discount_factor=0.99, steps_between_batches=30, replay_buffer_size=10e5, noise_std=0.1,
                 learning_starts=20000, residual=False):
        super().__init__(hidden_size, hidden_layers, learning_rate, lr_critic, batch_size, return_step,
                         discount_factor, steps_between_batches, replay_buffer_size, noise_std, learning_starts)
        if lr_critic is None:
            lr_critic = learning_rate
        self.model = neural_networks.ActorTwinCriticsModelNetwork(hidden_size=hidden_size,
                                                                 hidden_layers=hidden_layers).get_model()
        if residual:
            self.model = ResidualSACModel(self.model).get_model()
        self.exploration = explorations.NoNoiseExploration(start_steps=learning_starts)
        self.replay = ReplayBuffer(return_steps=return_step, discount_factor=discount_factor,
                                          batch_size=batch_size, steps_between_batches=steps_between_batches
                                          , size=replay_buffer_size)
        self.actor_updater = TwinCriticSoftDeterministicPolicyGradient(lr_actor=learning_rate,
                                                                       entropy_coeff=entropy_coeff)
        self.critic_updater = TwinCriticSoftQLearning(lr_critic=lr_critic, entropy_coeff=tau)

        self.config = {
            "agent": "SAC",
            "learning_rate": learning_rate,
            "lr_critic": lr_critic,
            "noise_std": noise_std,
            "tau": tau,
            "learning_starts": learning_starts,
            "hidden_size": hidden_size,
            "hidden_layers": hidden_layers,
            "discount_factor": discount_factor,
            "batch_size": batch_size,
            "return_step": return_step,
            "steps_between_batches": steps_between_batches,
            "replay_buffer_size": replay_buffer_size,
        }

    def _stochastic_actions(self, observations):
        """
        Samples stochastic actions from the actor network given the current observations.

        Parameters:
        ----------
        observations : np array or torch.Tensor
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


