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
            self.original_action_space = env.action_space  # Store the original action space
            oscillators = cpg_model.num_oscillators
            neurons = cpg_model.neuron_number
            self.params = oscillators*neurons
            # Extend the action space
            low = np.concatenate([env.action_space.low, np.full((self.params,), -1)])
            high = np.concatenate([env.action_space.high, np.full((self.params,), 1)])
            self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
            # self.phase_error = cpg_model.phase_error
            # self.control_signal = cpg_model.control_signal
            # self.phase_1 = cpg_model.phase_1
            # self.phase_2 = cpg_model.phase_2


    def step(self, action):
        if self.use_cpg:
            osc_weights = action[-self.params:]
            # osc_weights = np.clip(osc_weights, -1, 1)
            u_tmp = self.observation[0:2]
            u_value = np.concatenate([self.observation[0:2], self.observation[3:5]])
            action = action[:-self.params]
            action = self.cpg_model.step(action, osc_weights, u_value)
        return self.env.step(action)

    def get_error_data(self):
        if self.use_cpg:
            # self.phase_error = self.cpg_model.phase_error.cpu().numpy()
            # self.control_signal = self.cpg_model.control_signal.cpu().numpy()
            self.phase_1 = self.cpg_model.phase_1
            self.phase_2 = self.cpg_model.phase_2
            return self.phase_1, self.phase_2

    def reset(self, **kwargs):
        # Pass seed and other arguments down to the wrapped environment
        if self.use_cpg:
            self.cpg_model.reset()
        return self.env.reset(**kwargs)


