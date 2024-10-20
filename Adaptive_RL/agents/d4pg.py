from Adaptive_RL import neural_networks
from Adaptive_RL.agents import DDPG



class D4PG(DDPG):
    """
    Distributed Distributional Deterministic Policy Gradients.
    D4PG: https://arxiv.org/pdf/1804.08617.pdf
    """

    def __init__(self, hidden_size=256, hidden_layers=2, learning_rate=3e-4, batch_size=512, return_step=5,
                 discount_factor=0.99, steps_between_batches=20, replay_buffer_size=10e5, noise_std=0.1,
                learning_starts=20000):
        super().__init__(hidden_size, hidden_layers, learning_rate, batch_size, return_step,
                         discount_factor, steps_between_batches, replay_buffer_size, noise_std, learning_starts)
        self.model = neural_networks.ActorCriticDistributional(hidden_size=hidden_size, hidden_layers=hidden_layers).get_model()
        self.actor_updater = neural_networks.DistributionalDeterministicPolicyGradient(lr_actor=learning_rate)
        self.critic_updater = neural_networks.DistributionalDeterministicQLearning(lr_critic=learning_rate)
