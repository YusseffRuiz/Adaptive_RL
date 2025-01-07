"""
Multiprocessing for parallelized computation.
Using MP to parallelize training.
All methods to start computation
"""

import torch.multiprocessing as mp
import numpy as np


class Sequential:
    """Environment Vectorization"""

    def __init__(self, environment, max_episode_steps, workers):
        self.environments = [environment for _ in range(workers)]
        self.max_episode_steps = max_episode_steps
        self.observation_space = self.environments[0].observation_space
        self.action_space = self.environments[0].action_space
        self.name = self.environments[0].get_wrapper_attr('name')
        self.num_workers = workers
        self.use_cpg = False
        self.muscles = False

    def initialize(self, muscles=False):
        if hasattr(self.environments[0], "use_cpg"):
            self.use_cpg = True
        if muscles:
            self.muscles = True

    def start(self):
        """
        Resets all environments and returns their initial observations.
        Returns:
        - np.array: An array of initial observations from all environments, formatted as float32.
        """
        self.lengths = np.zeros(len(self.environments), int)
        if self.muscles:
            observations = [env.reset() for env in self.environments]
            muscle_states = [env.get_wrapper_attr('muscle_states') for env in self.environments]
            if self.use_cpg:
                osc_states = [env.get_osc_output() for env in self.environments]
                muscle_states = np.concatenate((np.array(muscle_states), osc_states), axis=1)
            return np.array(observations, np.float32), muscle_states
        else:
            observations = [env.reset()[0] for env in self.environments]
            return np.array(observations, np.float32)


    def step(self, actions):
        """
        Takes a step in each environment using the provided actions.

        Parameters:
        - actions (list): A list of actions to be executed in each environment.

        Returns:
        - observations (np.array): An array of observations for action selection after stepping.
        - infos (dict): A dictionary containing:
            - 'observations': Array of next observations after the step.
            - 'rewards': Array of rewards received after the step.
            - 'resets': Boolean array indicating if environments were reset.
            - 'terminations': Boolean array indicating if environments reached a terminal state.

        """
        next_observations = []  # Observations for the transitions.
        rewards = []
        resets = []
        terminations = []
        observations = []  # Observations for the actions selection.
        muscle_states = [] # In case muscle states are gathered

        for i in range(len(self.environments)):
            ob, rew, term, *_ = self.environments[i].step(actions[i])
            if self.muscles:
                muscle = self.environments[i].get_wrapper_attr('muscle_states')
            if self.use_cpg:
                osc = self.environments[i].get_osc_output()

            self.lengths[i] += 1
            # Timeouts trigger resets but are not true terminations.
            reset = term or self.lengths[i] == self.max_episode_steps
            next_observations.append(ob)
            rewards.append(rew)
            resets.append(reset)
            terminations.append(term)

            if reset:
                ob = self.environments[i].reset()
                if self.muscles:
                    muscle = self.environments[i].get_wrapper_attr('muscle_states')
                if self.use_cpg:
                    osc = self.environments[i].get_osc_output()
                self.lengths[i] = 0

            observations.append(ob)

            if self.muscles:
                if self.use_cpg:
                    muscle_states.append(np.concatenate((muscle, osc)))
                else:
                    muscle_states.append(muscle)

        observations = np.array(observations, np.float32)
        if self.muscles:
            muscle_states = np.array(muscle_states, np.float32)
        infos = dict(
            observations=np.array(next_observations, np.float32),
            rewards=np.array(rewards, np.float32),
            resets=np.array(resets, bool),
            terminations=np.array(terminations, bool))
        if self.muscles:
            return observations, muscle_states, infos
        else:
            return observations, infos

    def render(self, mode='human', *args, **kwargs):
        """
        Renders all environments in the specified mode.

        Parameters:
        - mode (str): The mode in which to render the environments (default is 'human').
        - *args, **kwargs: Additional arguments passed to the render method of each environment.

        Returns:
        - np.array: An array of render outputs if the mode is not 'human'.
        """
        outs = []
        for env in self.environments:
            out = env.render(mode=mode, *args, **kwargs)
            outs.append(out)
        if mode != 'human':
            return np.array(outs)

def proc(action_pipe, index, environment, max_episode_steps, workers_per_group, output_queue, muscles_flag=False):
    """Process holding a sequential group of environments.
    Parameters:
    :param action_pipe: actions being processed
    :param index: number of observation being processed to the queue
    """
    envs = Sequential(environment, max_episode_steps, workers_per_group)
    envs.initialize(muscles=muscles_flag)

    observations = envs.start()
    output_queue.put((index, observations))

    while True:
        actions = action_pipe.recv()
        out = envs.step(actions)
        output_queue.put((index, out))

class Parallel:
    """A group of sequential environments used in parallel for GPU performance."""

    def __init__(self, environment, worker_groups, workers_per_group, max_episode_steps):
        """
        Initializes a Parallel group of environments that are executed in parallel.

        Parameters:
        - environment_builder (callable): A function or callable object that creates an environment instance.
        - worker_groups (int): The number of worker groups running in parallel.
        - workers_per_group (int): The number of environments (or workers) per group.
        - max_episode_steps (int): The maximum number of steps allowed per episode.
        """
        self.environment = environment
        self.worker_groups = worker_groups
        self.workers_per_group = workers_per_group
        self.max_episode_steps = max_episode_steps
        self.muscles = False

    def initialize(self, muscles=False):
        """
        Initializes the parallel environments by creating processes for each group of workers.
        """
        mp.set_start_method('spawn')
        self.muscles = muscles

        dummy_environment = self.environment
        self.observation_space = dummy_environment.observation_space
        self.action_space = dummy_environment.action_space
        del dummy_environment
        self.started = False

        self.output_queue = mp.Queue()
        self.action_pipes = []
        self.processes = []

        for i in range(self.worker_groups):
            pipe, worker_end = mp.Pipe()
            self.action_pipes.append(pipe)
            self.processes.append(mp.Process(target=proc, args=(worker_end, i, self.environment, self.max_episode_steps,
                                                    self.workers_per_group, self.output_queue, muscles)
            ))
            self.processes[-1].daemon = True
            self.processes[-1].start()

    def start(self):
        """
        Resets all environments in parallel and returns their initial observations.

        Returns:
        - np.array: An array of initial observations from all environments, formatted as float32.
        """
        assert not self.started
        self.started = True
        observations_list = [None for _ in range(self.worker_groups)]
        muscle_states_list = [None for _ in range(self.worker_groups)]

        for _ in range(self.worker_groups):
            if self.muscles:
                index, (observations, muscle_states) = self.output_queue.get()
                muscle_states_list[index] = muscle_states
            else:
                index, observations = self.output_queue.get()
            observations_list[index] = observations

        self.observations_list = np.array(observations_list)
        if self.muscles:
            self.muscle_states_list = np.array(muscle_states_list)
        self.next_observations_list = np.zeros_like(self.observations_list)
        self.rewards_list = np.zeros(
            (self.worker_groups, self.workers_per_group), np.float32)
        self.resets_list = np.zeros(
            (self.worker_groups, self.workers_per_group), bool)
        self.terminations_list = np.zeros(
            (self.worker_groups, self.workers_per_group), bool)

        if self.muscles:
            return np.concatenate(self.observations_list), np.concatenate(self.muscle_states_list)
        else:
            return np.concatenate(self.observations_list)

    def step(self, actions):
        """
        Takes a step in each environment group in parallel using the provided actions.

        Parameters:
        - actions (list): A list of actions to be executed in each environment group.

        Returns:
        - observations (np.array): An array of observations for action selection after stepping.
        - infos (dict): A dictionary containing:
            - 'observations': Array of next observations after the step.
            - 'rewards': Array of rewards received after the step.
            - 'resets': Boolean array indicating if environments were reset.
            - 'terminations': Boolean array indicating if environments reached a terminal state.
        """
        actions_list = np.split(actions, self.worker_groups)
        for actions, pipe in zip(actions_list, self.action_pipes):
            pipe.send(actions)

        for _ in range(self.worker_groups):
            if self.muscles:
                index, (observations, muscle_states,infos) = self.output_queue.get()
                self.muscle_states_list[index] = muscle_states
            else:
                index, (observations, infos) = self.output_queue.get()
            # obs_inf = [observations, rewards, resets, terminations
            self.observations_list[index] = observations
            self.next_observations_list[index] = infos['observations']
            self.rewards_list[index] = infos['rewards']
            self.resets_list[index] = infos['resets']
            self.terminations_list[index] = infos['terminations']

        observations = np.concatenate(self.observations_list)
        if self.muscles:
            muscle_states = np.concatenate(self.muscle_states_list)
        infos = dict(
            observations=np.concatenate(self.next_observations_list),
            rewards=np.concatenate(self.rewards_list),
            resets=np.concatenate(self.resets_list),
            terminations=np.concatenate(self.terminations_list))
        if self.muscles:
            return observations, muscle_states, infos
        else:
            return observations, infos


def distribute(environment, worker_groups=1, workers_per_group=1):
    """
    Distributes workers over parallel and sequential groups.

    Parameters:
    - environment_builder (callable): A function or callable object that creates an environment instance.
    - worker_groups (int): The number of worker groups running in parallel. Default is 1.
    - workers_per_group (int): The number of environments (or workers) per group. Default is 1.

    Returns:
    - Parallel or Sequential: An instance of either the Parallel or Sequential class based on the number of worker groups.
    """
    dummy_environment = environment
    max_episode_steps = dummy_environment.get_wrapper_attr('max_episode_steps')
    del dummy_environment
    if worker_groups < 2:
        return Sequential(
            environment, max_episode_steps=max_episode_steps,
            workers=workers_per_group)

    return Parallel(
        environment, worker_groups=worker_groups,
        workers_per_group=workers_per_group,
        max_episode_steps=max_episode_steps)

