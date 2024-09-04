from MPO_Algorithm import Critic


class MatsuokaCritic(Critic):
    def __init__(self, env, num_oscillators=2, neuron_num=2):
        self.action_dim = num_oscillators*neuron_num
