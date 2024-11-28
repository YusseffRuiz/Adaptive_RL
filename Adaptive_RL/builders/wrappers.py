import gymnasium as gym
import numpy as np


class ActionRescaler(gym.ActionWrapper):
    """Rescales actions from [-1, 1]^n to the true action space.
    The baseline agents return actions in [-1, 1]^n."""

    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Box)
        super().__init__(env)
        high = np.ones(env.action_space.shape, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-high, high=high)
        true_low = env.action_space.low
        true_high = env.action_space.high
        self.bias = (true_high + true_low) / 2
        self.scale = (true_high - true_low) / 2

    def action(self, action):
        return self.bias + self.scale * np.clip(action, -1, 1)


class TimeFeature(gym.Wrapper):
    """Adds a notion of time in the observations.
    It can be used in terminal timeout settings to get Markovian MDPs.
    """

    def __init__(self, env, max_steps, low=-1, high=1):
        super().__init__(env)
        dtype = self.observation_space.dtype

        self.observation_space = gym.spaces.Box(
            low=np.append(self.observation_space.low, low).astype(dtype),
            high=np.append(self.observation_space.high, high).astype(dtype))
        self.max_episode_steps = max_steps
        self.steps = 0
        self.low = low
        self.high = high

    def reset(self, **kwargs):
        self.steps = 0
        observation = self.env.reset(**kwargs)
        observation = np.append(observation, self.low)
        return observation

    def step(self, action):
        assert self.steps < self.max_episode_steps
        observation, reward, done, info, _ = self.env.step(action)
        self.steps += 1
        prop = self.steps / self.max_episode_steps
        v = self.low + (self.high - self.low) * prop
        observation = np.append(observation, v)
        return observation, reward, done, info


class CPGWrapper(gym.Wrapper):
    def __init__(self, env, cpg_model=None, use_cpg=False):
        super(CPGWrapper, self).__init__(env)
        self.use_cpg = use_cpg
        if use_cpg:
            self.cpg_model = cpg_model  # The CPG model should be passed in as an argument
            self.da = len(env.action_space.low)
            self.original_action_space = env.action_space  # Store the original action space
            oscillators = cpg_model.num_oscillators
            neurons = cpg_model.neuron_number
            self.params = oscillators*neurons
            # Extend the action space
            low = np.concatenate([env.action_space.low, np.full((self.params,), -1)])
            high = np.concatenate([env.action_space.high, np.full((self.params,), 1)])
            self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)


    def step(self, action):
        if self.use_cpg:
            osc_weights, u_value, action = env_selection(self.da, action, self.params, self.public_obs())
            action = self.cpg_model.step(action, osc_weights, u_value)
        return self.env.step(action)

    def get_error_data(self):
        pass
        # if self.use_cpg:
            # self.phase_error = self.cpg_model.phase_error.cpu().numpy()
            # self.control_signal = self.cpg_model.control_signal.cpu().numpy()
            # self.phase_1 = self.cpg_model.phase_1
            # self.phase_2 = self.cpg_model.phase_2
            # return self.phase_1, self.phase_2

    def reset(self, **kwargs):
        # Pass seed and other arguments down to the wrapped environment
        if self.use_cpg:
            self.cpg_model.reset()
        return self.env.reset(**kwargs)


def env_selection(action_dim, weights, params, obs):
    """
    Returns the separation into oscillator weights, original actions and u_feedback coming from the movement on the legs
    :param action_dim: action dimension for the different enviroments
    :param weights: weights comming from the DRL algorithm
    :param params: parameters of the CPG
    :param obs: observations from the environment for the u_feedback
    :return: weights for oscillator, the u_feedback and the weights for the actual actions.
    """
    osc_weights = weights[-params:]
    action_weights = weights[:-params]
    if action_dim == 6:
        u_values = weight_conversion_walker(obs)
    elif action_dim == 17:
        u_values = weight_conversion_humanoid(obs)
    elif action_dim == 70:
        u_values = weight_conversion_myoleg(obs)
    else:
        print("Not an implemented environment")
        return None

    return osc_weights, u_values, action_weights


def weight_conversion_walker(observation):
    return np.concatenate([observation[0:2], observation[3:5]])


def weight_conversion_humanoid(obs):
    return np.concatenate([obs[10:12], obs[14:16]])


# Define muscle groups and their corresponding neurons/oscillators
MUSCLE_GROUP_MYOSIM = {
    'quadriceps_right': [21, 36, 27, 28],  # recfem_r, vasint_r, vaslat_r, vasmed_r
    'hamstrings_right': [23, 24],  # semimem_r, semiten_r
    'hip_flexors_right': [18, 20, 22],  # iliacus_r, psoas_r, sart_r
    'hip_extensors_right': [0, 1, 8, 9, 10],  # addbrev_r, addlong_r, glmax1_r, glmax2_r, glmax3_r
    'ankle_motor_right': [69],  #

    'quadriceps_left': [58, 66, 67, 68],  # recfem_l, vasint_l, vaslat_l, vasmed_l
    'hamstrings_left': [35, 36, 60, 61],  # bflh_l, bfsh_l, semimem_l, semiten_l
    'hip_flexors_left': [53, 57, 59],  # iliacus_l, psoas_l, sart_l
    'hip_extensors_left': [29, 30, 43, 44, 45],  # addbrev_l, addlong_l, glmax1_l, glmax2_l, glmax3_l
    'ankle_dorsiflexors_left': [37, 38, 64],  # edl_l, ehl_l, fdl_l (right dorsiflexors)
    'ankle_plantarflexors_left': [39, 40, 41, 42, 62, 65],  # fhl_r, soleus_r (right plantarflexors)
}

def weight_conversion_myoleg(obs):
    u_feedback = None

    return u_feedback

