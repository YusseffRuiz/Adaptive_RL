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
            osc_weights = action[-self.params:]
            action = action[:-self.params]
            u_values = self.unwrapped.public_joints()
            # 0,0 is left Hip
            # 0,1 is right Hip
            # 1,0 is left Ankle
            # 1,1 is right Ankle
            action = self.cpg_model.step(action, osc_weights, u_values)
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


