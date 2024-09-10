"""
Python wall designed to create the functions used to run the experiments.
Comparison of different DRL methods with and without CPG.
"""


def get_velocity(obs):
    """
    Calculation of the velocity of the agent in meters/sec across the x axis
    :param obs: environment base
    :return: velocity of the agent
    """
    # Extract velocity of the agent
    velocity = obs[8]
    return velocity


def get_motion(env):
    pass


def get_energy_consumption(env):
    pass


def compare_velocity(envs_list):
    pass


def compare_motion(envs_list):
    pass


def compare_energy_consumption(envs_list):
    pass
