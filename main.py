import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecMonitor

from MatsuokaOscillator import MatsuokaOscillator, MatsuokaNetwork, MatsuokaNetworkWithNN

import gymnasium as gym
from tqdm import tqdm
import os

from gymnasium.envs.registration import register
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
from stable_baselines3.common.callbacks import BaseCallback

import MPO_Algorithm
import yaml
import argparse


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, f"best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True


# Basic Matsuoka Oscillator Implementation
def matsuoka_main():
    # Parameters
    neural_net = False
    num_oscillators = 2
    neuron_number = 2
    tau_r = 2
    tau_a = 12
    w12 = 2.5
    u1 = 2.5
    beta = 2.5
    dt = 1
    steps = 1000
    weights = np.full(neuron_number, w12)
    time = np.linspace(0, steps * dt, steps)

    if neural_net is True:
        # Neural Network Implementation
        input_size = num_oscillators  # Example input size
        hidden_size = 10  # Hidden layer size
        output_size = 3  # tau_r, weights, and beta for each oscillator

        matsuoka_network = MatsuokaNetworkWithNN(num_oscillators=num_oscillators,
                                                 env=[1, neuron_number * num_oscillators], neuron_number=neuron_number,
                                                 tau_r=tau_r, tau_a=tau_a)
        # Create a sample sensory input sequence
        # sensory_input_seq = torch.rand(steps, num_oscillators, input_size, dtype=torch.float32, device="cuda")

        # Run the coupled system with NN control
        outputs = matsuoka_network.run(steps=steps)

        for i in range(num_oscillators):
            plt.plot(time, outputs[:, i, 0], label=f'Oscillator {i + 1} Neuron 1')
            plt.plot(time, outputs[:, i, 1], label=f'Oscillator {i + 1} Neuron 2')

        plt.xlabel('Time step')
        plt.ylabel('Output')
        plt.title('Outputs of Coupled Matsuoka Oscillators Controlled by NN')
        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        # Run of the events
        if num_oscillators == 1:
            # Create Matsuoka Oscillator with N neurons
            oscillator = MatsuokaOscillator(neuron_number=neuron_number, tau_r=tau_r, tau_a=tau_a,
                                            beta=beta, dt=dt, action_space=neuron_number * num_oscillators,
                                            num_oscillators=num_oscillators)
            y_output = oscillator.run(steps=steps)

            for i in range(y_output.shape[1]):
                plt.plot(time, y_output[:, i], label=f'y{i + 1} (Neuron {i + 1})')
            plt.xlabel('Time')
            plt.ylabel('Output')
            plt.title('Matsuoka Oscillator Output')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            # Coupled System
            coupled_system = MatsuokaNetwork(num_oscillators=num_oscillators, neuron_number=neuron_number, tau_r=tau_r,
                                             tau_a=tau_a, weights=weights, beta=beta, dt=dt,
                                             action_space=neuron_number * num_oscillators)
            y_output = coupled_system.run(steps=steps)

            # Coupled Oscillators
            for i in range(num_oscillators):
                for j in range(neuron_number):
                    plt.plot(time, y_output[i][:, j], label=f'Oscillator {i + 1} Neuron {j + 1}')

            plt.xlabel('Time step')
            plt.ylabel('Output')
            plt.title('Outputs of Coupled Matsuoka Oscillators')
            plt.legend()
            plt.grid(True)
            plt.show()


def get_name_environment(name, cpg_flag=False, algorithm=None, experiment_number=0):
    """
    :param algorithm: algorithm being used to create the required folder
    :param name: of the environment
    :param cpg_flag: either we are looking for the cpg env or not
    :param experiment_number: if required, we can create a subfolder
    :return: env_name, save_folder, log_dir
    """
    env_name = name
    cpg = cpg_flag

    if cpg:
        env_name = env_name + "-CPG"

    print(f"Creating env {env_name}")
    # Create log dir
    if algorithm is not None:
        save_folder = f"{env_name}-{algorithm}"
    else:
        save_folder = f"{env_name}"
    if experiment_number > 0:
        log_dir = f"{env_name}/logs/{save_folder}/{experiment_number}"
    else:
        log_dir = f"{env_name}/logs/{save_folder}"
    os.makedirs(log_dir, exist_ok=True)
    return env_name, save_folder, log_dir


def evaluate(model, env, algorithm):
    total_rewards = []
    range_episodes = 3
    for i in range(range_episodes):
        obs, *_ = env.reset()
        done = False
        episode_reward = 0
        cnt = 0
        while not done:
            with torch.no_grad():
                if algorithm == "mpo":
                    action = model.test_step(obs, _)
                elif algorithm == "sac":
                    action, *_ = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()
            obs, reward, done, *_ = env.step(action)
            episode_reward += reward
            cnt += 1
            if cnt >= 2000:
                done = True

        total_rewards.append(episode_reward)
        print(f"Episode {i + 1}/{range_episodes}: Reward = {episode_reward}")
    average_reward = np.mean(total_rewards)
    print(f"Average Reward over {range_episodes} episodes: {average_reward}")


def training_stable():
    env_name, save_folder, log_dir = get_name_environment("Walker2d-v4", cpg_flag=True, algorithm="SAC")

    train = True
    time_steps = 5e6

    if train:
        save = True
        plot = True
        # env = gym.make(env_name, render_mode="rgb_array", forward_reward_weight=5.0, healthy_reward=0.0)
        # env = Monitor(env, log_dir)
        n_envs = 4

        vec_env = make_vec_env(env_name, n_envs=n_envs, seed=0)
        vec_env = VecMonitor(vec_env, filename=log_dir)

        # Noise Definition
        n_actions = vec_env.action_space.shape[-1]  # Starting noise
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        action_noise = VectorizedActionNoise(base_noise=action_noise, n_envs=n_envs)

        model = SAC("MlpPolicy", vec_env, verbose=1, gamma=0.9999, batch_size=128, device="cuda:0",
                    train_freq=1, gradient_steps=2, learning_starts=0, tensorboard_log=log_dir,
                    action_noise=action_noise)
        # Create the callback: check every 5000 steps
        callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir)

        model.learn(total_timesteps=int(time_steps), progress_bar=True, callback=callback)

        if save:
            print("saving")
            model.save(f"{save_folder}/{env_name}-SAC-top")
            model.policy.save(f"{save_folder}/{env_name}-SAC-policy.pkl")
            model.save_replay_buffer("sac_replay_buffer")
    else:
        plot = False

    if plot:
        plot_results([log_dir], int(time_steps), results_plotter.X_TIMESTEPS, f"{env_name}-SAC")
        plt.show()

    env = gym.make(env_name, render_mode="human", max_episode_steps=1500)
    model = SAC("MlpPolicy", env, verbose=1)

    if os.path.exists(f"{save_folder}"):
        path = f"{save_folder}/{env_name}-SAC-top"
        # path = "walker_sac/Walker2d-v4-CPG-SAC.zip" # Testing path
        model = SAC.load(path)
        print("model loaded")

    print("Starting")

    evaluate(model, env, algorithm="sac")
    env.close()


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


def train_mpo(
        agent, environment, trainer=MPO_Algorithm.Trainer(), parallel=1, sequential=1, seed=0,
        checkpoint="last", path=None, log_dir=None):
    """
    :param agent: Agent and algorithm to be trained.
    :param environment: Environment name
    :param trainer: Trainer to be used, at this moment, the default from tonic
    :param parallel: Parallel Processes
    :param sequential: Vector Environments.
    :param seed: random seed
    :param checkpoint: checkpoint to verify existence.
    :param path: Path where the experiment to check for checkpoints
    :param log_dir:: Path to add the logs of the experiment
    """
    torch.set_default_device('cuda')
    path = log_dir
    args = dict(locals())
    checkpoint_path = None
    config = None
    # Process the checkpoint path same way as in tonic.play
    if path:
        checkpoint_path = MPO_Algorithm.load_checkpoint(checkpoint, path)
        if checkpoint_path is not None:
            # Load the experiment configuration.
            arguments_path = os.path.join(path, 'config.yaml')
            with open(arguments_path, 'r') as config_file:
                config = yaml.load(config_file, Loader=yaml.Loader)
            config = argparse.Namespace(**config)

            agent = agent or config.agent
            environment = environment or config.test_environment
            environment = environment or config.environment
            trainer = trainer or config.trainer

    # Build the training environment.

    _environment = MPO_Algorithm.environments.Gym(environment)
    environment = MPO_Algorithm.parallelize.distribute(
        lambda: _environment, parallel, sequential)
    environment.initialize() if parallel > 1 else 0

    # Build the testing environment.
    test_environment = MPO_Algorithm.parallelize.distribute(
        lambda: _environment)

    # Build the agent.
    if not agent:
        raise ValueError('No agent specified.')

    agent.initialize(observation_space=environment.observation_space, action_space=environment.action_space,
                     seed=seed)

    # Load the weights of the agent form a checkpoint.
    if checkpoint_path:
        agent.load(checkpoint_path)
        print(f"Checkpoint found in {checkpoint_path}")

    # Initialize the logger to save data to the path
    MPO_Algorithm.logger.initialize(path=log_dir, config=args)

    # Build the trainer.
    trainer.initialize(
        agent=agent, environment=environment,
        test_environment=test_environment)

    # Train.
    trainer.run()


def mpo_tonic_train_main():
    env_name = "Ant-v4"
    env_name, save_folder, log_dir = get_name_environment(env_name, cpg_flag=True, algorithm="MPO")
    max_steps = int(5e6)
    epochs = max_steps / 500
    save_steps = max_steps / 200
    agent = MPO_Algorithm.agents.MPO(lr_actor=3.53e-5, lr_critic=6.081e-5, lr_dual=0.00213, hidden_size=512)
    train_mpo(agent=agent,
              environment=env_name,
              sequential=5, parallel=5,
              trainer=MPO_Algorithm.Trainer(steps=max_steps, epoch_steps=epochs, save_steps=save_steps),
              log_dir=log_dir)
    env = gym.make(env_name, render_mode="human", max_episode_steps=1500)

    evaluate(agent, env, algorithm="mpo")
    env.close()


# Training of MPO method
if __name__ == "__main__":
    # register_new_env()
    mpo_tonic_train_main()
