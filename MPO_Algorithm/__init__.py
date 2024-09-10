from .utils import logger, normalizers
from .utils.utils import load_checkpoint
from .utils.replay_buffer import ReplayBuffer
from .agents import MPO
from .trainer import Trainer
from .builders import environments, parallelize

from gymnasium.envs.registration import register

__all__ = [logger, normalizers, utils, ReplayBuffer, MPO, Trainer]


def register_new_env():
    register(
        # unique identifier for the env `name-version`
        id="Hopper_CPG_v1",
        # path to the class for creating the env
        # Note: entry_point also accept a class as input (and not only a string)
        entry_point="gymnasium.envs.mujoco:HopperCPG",
        # Max number of steps per episode, using a `TimeLimitWrapper`
        max_episode_steps=1000,
    )
    print("Registered new env Hopper_CPG_v1")

    register(
        # unique identifier for the env `name-version`
        id="Walker2d-v4-CPG",
        # path to the class for creating the env
        # Note: entry_point also accept a class as input (and not only a string)
        entry_point="gymnasium.envs.mujoco:Walker2dCPGEnv",
        # Max number of steps per episode, using a `TimeLimitWrapper`
        max_episode_steps=1000,
    )
    print("Registered new env Walker2d-v4-CPG")

register_new_env()