import gymnasium as gym
import numpy as np

class CPGWrapper(gym.Wrapper):
    def __init__(self, env, cpg_model=None, use_cpg=False):
        super(CPGWrapper, self).__init__(env)
        self.use_cpg = use_cpg
        self.muscle_flag = hasattr(self.env, "muscle_states")
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
            osc_weights, u_value, action = env_selection(self.da, action, self.params, self.unwrapped.public_joints())
            action = self.cpg_model.step(action, osc_weights, u_value)
        return self.env.step(action)

    def reset(self, seed=None, options=None):
        # Pass seed and other arguments down to the wrapped environment
        if self.use_cpg:
            self.cpg_model.reset()
        return super().reset(seed=None, options=options)

    def get_osc_output(self):
        # 2 values: left and right
        return self.cpg_model.get_osc_states()

    def _max_episode_steps(self):
        return self.env.unwrapped.max_episode_steps


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
        u_values = obs
    else:
        print("Not an implemented environment")
        return None

    return osc_weights, u_values, action_weights


def weight_conversion_walker(observation):
    return np.array([observation[3], observation[5]])  #[observation[0], observation[2],


def weight_conversion_humanoid(obs):
    return np.array([obs[10], obs[14]])


# Define muscle groups and their corresponding neurons/oscillators
def weight_conversion_myoleg(obs):
    # Placeholder in case we need some conversion
    # hip_flex_r = np.array([obs[5], obs[21]])  # Feedback to muscles controlling hip
    # knee_rot_r = np.array([obs[9], obs[26]])  # Feedback to muscles controlling knee
    # ankle_flex_r = np.array([obs[17], obs[29]])  # Feedback to muscles controlling ankle right

    # u_feedback = np.concatenate([hip_flex_r, ankle_flex_r])
    # u_feedback = np.array(ankle_flex_r)
    return obs

