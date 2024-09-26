import gymnasium as gym
from RL_Adaptive.builders import wrappers
import numpy as np


def gym_environment(*args, **kwargs):
    """Returns a wrapped Gym environment."""

    def _builder(*args, **kwargs):
        return gym.make(*args, **kwargs)

    return build_environment(_builder, *args, **kwargs)


def myosuite_environment(*args, **kwargs):
    """
    Returns a wrapped Myosuite environment.
    In development
    """
    def _builder(*args, **kwargs):
        pass

    return build_environment(_builder, *args, **kwargs)


def build_environment(
    builder, name, terminal_timeouts=False, time_feature=False,
    max_episode_steps='default', scaled_actions=True, *args, **kwargs
):
    """Builds and wrap an environment.
    Time limits can be properly handled with terminal_timeouts=False or
    time_feature=True, see https://arxiv.org/pdf/1712.00378.pdf for more
    details.
    """

    # Build the environment.
    environment = builder(name, *args, **kwargs)

    # Get the default time limit.
    if max_episode_steps == 'default':
        max_episode_steps = environment._max_episode_steps

    # Remove the TimeLimit wrapper if needed.
    if not terminal_timeouts:
        assert type(environment) == gym.wrappers.TimeLimit, environment
        environment = environment.env

    # Add time as a feature if needed.
    if time_feature:
        environment = wrappers.TimeFeature(environment, max_episode_steps)

    # Scale actions from [-1, 1]^n to the true action space if needed.
    if scaled_actions:
        environment = wrappers.ActionRescaler(environment)

    environment.name = name
    environment.max_episode_steps = max_episode_steps

    return environment


def _flatten_observation(observation):
    """Turns OrderedDict observations into vectors."""
    observation = [np.array([o]) if np.isscalar(o) else o.ravel()
                   for o in observation.values()]
    return np.concatenate(observation, axis=0)


Gym = gym_environment
MyoSuite = myosuite_environment
