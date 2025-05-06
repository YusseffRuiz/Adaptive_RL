import traceback

import gymnasium as gym
import numpy as np
from abc import ABC, abstractmethod



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


class AbstractWrapper(gym.Wrapper, ABC):
    def merge_args(self, args):
        if args is not None:
            for k, v in args.items():
                setattr(self.unwrapped, k, v)

    def apply_args(self):
        pass

    def render(self, *args, **kwargs):
        pass

    @property
    def force_scale(self):
        if not hasattr(self, "_force_scale"):
            self._force_scale = 0
        return self._force_scale

    @force_scale.setter
    def force_scale(self, force_scale):
        assert force_scale >= 0, f"expected positive value, got {force_scale}"
        self._force_scale = force_scale

    @abstractmethod
    def muscle_lengths(self):
        pass

    @abstractmethod
    def muscle_forces(self):
        pass

    @property
    def muscle_states(self):
        """
        Computes the DEP input. We assume an input
        muscle_length + force_scale * muscle_force
        where force_scale is chosen by the user and the other
        variables are normalized by the encountered max and min
        values.
        """
        lce = self.muscle_lengths()
        f = self.muscle_forces()
        if not hasattr(self, "max_muscle"):
            self.max_muscle = np.zeros_like(lce)
            self.min_muscle = np.ones_like(lce) * 100.0
            self.max_force = -np.ones_like(f) * 100.0
            self.min_force = np.ones_like(f) * 100.0
        if not np.any(np.isnan(lce)):
            self.max_muscle = np.maximum(lce, self.max_muscle)
            self.min_muscle = np.minimum(lce, self.min_muscle)
        if not np.any(np.isnan(f)):
            self.max_force = np.maximum(f, self.max_force)
            self.min_force = np.minimum(f, self.min_force)
        return (
            1.0
            * (
                (
                    (lce - self.min_muscle)
                    / (self.max_muscle - self.min_muscle + 0.1)
                )
                - 0.5
            )
            * 2.0
            + self.force_scale
            * ((f - self.min_force) / (self.max_force - self.min_force + 0.1))
        ).copy()


class ExceptionWrapper(AbstractWrapper):
    """
    Catches MuJoCo related exception thrown mostly by instabilities in the simulation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, seed=None, options=None):
        observation = super().reset(seed=seed, options=options)[0]
        if not np.any(np.isnan(observation)):
            self.last_observation = observation.copy()
        else:
            return self.reset(seed=seed, options=options)
        return observation

    def step(self, action):
        try:
            observation, reward, done, info, extras = self._inner_step(action)
            if np.any(np.isnan(observation)):
                observation[np.isnan(observation)] = 0.0
                reward = 0
                done = 1
                info = {}
                extras = {}
                self.reset()
                print("NaN detected! Resetting.")

        except Exception as e:
            # logger.log(f"Simulator exception thrown: {e}")
            print(f"Simulator exception thrown: {e}")
            traceback.print_exc()  # 🔥 Show full traceback
            observation = self.last_observation
            reward = 0
            done = 1
            info = {}
            extras = {}
            self.reset()
        return observation, reward, done, info, extras

    def _inner_step(self, action):
        return super().step(action)


class GymWrapper(ExceptionWrapper):
    """Wrapper for OpenAI Gym and MuJoCo, compatible with
    gym=0.13.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.muscles_enable=True

    def remove_action_osl(self, osc_size=0):
        if osc_size>0:
            self.action_space = gym.spaces.Box(
                low=np.concatenate([self.unwrapped.sim.model.actuator_ctrlrange[:self.unwrapped.sim.model.na, 0], np.full((osc_size,), -1)]),
                high=np.concatenate([self.unwrapped.sim.model.actuator_ctrlrange[:self.unwrapped.sim.model.na, 1], np.full((osc_size,), -1)]),
                dtype=np.float32
            )

    def render(self, *args, **kwargs):
        kwargs["mode"] = "window"
        self.unwrapped.sim.render(*args, **kwargs)

    def muscle_lengths(self):
        length = self.unwrapped.sim.data.actuator_length

        return length

    def muscle_forces(self):
        return self.unwrapped.sim.data.actuator_force

    def muscle_velocities(self):
        return self.unwrapped.sim.data.actuator_velocity

    def muscle_activity(self):
        return self.unwrapped.sim.data.act

    @property
    def _max_episode_steps(self):
        return self.unwrapped.max_episode_steps
