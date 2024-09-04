import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from MatsuokaOscillator import MatsuokaOscillator, MatsuokaNetwork, MatsuokaNetworkWithNN, MpoMatsuokaTrainer
from MPO_Algorithm import MPO

import gymnasium as gym
from reinforce import A2C
from tqdm import tqdm
import os

import torch.multiprocessing as mp
import torch.distributed as dist

from gymnasium.envs.registration import register
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback


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
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
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

        matsuoka_network = MatsuokaNetworkWithNN(num_oscillators=num_oscillators, env=[1, neuron_number*num_oscillators], neuron_number=neuron_number, tau_r=tau_r, tau_a=tau_a)
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
                                            beta=beta, dt=dt, action_space=neuron_number*num_oscillators, num_oscillators=num_oscillators)
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
                                             tau_a=tau_a, weights=weights, beta=beta, dt=dt, action_space=neuron_number*num_oscillators)
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


def training_stable():
    #env_name = "Walker2d-v4"
    env_name = "Walker2d-CPG_v1"
    save_folder = "walker_sac"
    # Create log dir
    log_dir = f"{save_folder}/logs/{env_name}"
    os.makedirs(log_dir, exist_ok=True)

    save = True
    train = True

    if train:
        vec_env = make_vec_env(env_name, n_envs=4, seed=0)
        # env = Monitor(vec_env, log_dir)
        # env = gym.make(env_name, render_mode="rgb_array")
        model = SAC("MlpPolicy", vec_env, verbose=1, gamma=0.9999, batch_size=64, device="cuda:0",
                    train_freq=1, gradient_steps=2, learning_starts=0, tensorboard_log=log_dir)
        time_steps = 1e6
        model.learn(total_timesteps=int(time_steps), progress_bar=True, log_interval=100)

        if save:
            print("saving")
            model.save(f"{save_folder}/sac_walker_cpg_par")

    env = gym.make(env_name, render_mode="human", max_episode_steps=1500)
    model = SAC("MlpPolicy", env, verbose=1)


    if os.path.exists(f"{save_folder}"):
        model = SAC.load(f"{save_folder}/sac_walker_cpg_par")
        print("model loaded")

    print("Starting")

    total_rewards = []
    for i in range(10):
        obs, *_ = env.reset()
        done = False
        episode_reward = 0
        cnt = 0
        while not done:
            #action = env.action_space.sample()
            action, *_ = model.predict(obs, deterministic=True)
            obs, reward, done, *_ = env.step(action)
            episode_reward += reward
            cnt += 1
            if cnt >= 5000:
                done = True


        total_rewards.append(episode_reward)
        print(f"Episode {i + 1}/{10}: Reward = {episode_reward}")

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

    register(
        # unique identifier for the env `name-version`
        id="Walker2d-CPG_v1",
        # path to the class for creating the env
        # Note: entry_point also accept a class as input (and not only a string)
        entry_point="gymnasium.envs.mujoco:Walker2dCPGEnv",
        # Max number of steps per episode, using a `TimeLimitWrapper`
        max_episode_steps=1000,
    )


# In development, DO NOT USE
def matsuoka_mpo_main(shared_best_rw, world_size):
    env_name = 'Walker2d-v4'
    env = gym.make(env_name)
    num_envs = 4
    state_dim = env.observation_space.shape[0]
    action_space = env.action_space
    params_dim = 2  # Weights number, 1 per neuron

    setup()

    # Initialize MPOTrainer
    trainer = MpoMatsuokaTrainer(env=env_name, device="cuda")

    # Run the training loop
    trainer.train()
    cleanup()

    # After training, evaluate the agent
    # Implement an evaluation loop or method in MPOTrainer if needed

    save_weights = True
    load_weights = False

    actor_weights_path = "weights_2/FinalWeights/actor_weights_MPO.h5"
    critic_weights_path = "weights_2/FinalWeights/critic_weights_MPO.h5"

    if not os.path.exists("weights_2"):
        os.mkdir("weights_2")
        if not os.path.exists("weights_2/FinalWeights"):
            os.mkdir("weights_2/FinalWeights")

    print("Done")

# Training of the A2C algorithm
def a2c_train_main():
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # environment hyperparams
    n_envs = 3
    n_updates = 1000
    n_steps_per_update = 128
    randomize_domain = False

    # agent hyperparams
    gamma = 0.999
    lam = 0.95  # hyperparameter for GAE
    ent_coef = 0.01  # coefficient for the entropy bonus (to encourage exploration)
    actor_lr = 0.001
    critic_lr = 0.005

    # Note: the actor has a slower learning rate so that the value targets become
    # more stationary and are therefore easier to estimate for the critic

    # environment setup

    tot_episodes = 10
    # Create and wrap the environment
    env = gym.make_vec("Walker2d-v4", num_envs=n_envs)

    envs = gym.vector.AsyncVectorEnv(
        [
            lambda: gym.make(
                "Walker2d-v4",
            ),
            lambda: gym.make(
                "Walker2d-v4",
            ),
            lambda: gym.make(
                "Walker2d-v4",
            ),
        ]
    )

    obs_shape = envs.single_observation_space.shape[0]
    action_shape = envs.single_action_space.shape[0]
    agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, n_envs)

    # create a wrapper environment to save episode returns and episode lengths
    envs_wrapper = gym.wrappers.RecordEpisodeStatistics(envs, deque_size=n_envs * n_updates)

    critic_losses = []
    actor_losses = []
    entropies = []

    # use tqdm to get a progress bar for training
    for sample_phase in tqdm(range(n_updates)):
        # we don't have to reset the envs, they just continue playing
        # until the episode is over and then reset automatically

        # reset lists that collect experiences of an episode (sample phase)
        ep_value_preds = torch.zeros(n_steps_per_update, n_envs, device=device)
        ep_rewards = torch.zeros(n_steps_per_update, n_envs, device=device)
        ep_action_log_probs = torch.zeros(n_steps_per_update, n_envs, device=device)
        masks = torch.zeros(n_steps_per_update, n_envs, device=device)

        # at the start of training reset all envs to get an initial state
        if sample_phase == 0:
            states, info = envs_wrapper.reset(seed=42)

        # play n steps in our parallel environments to collect data
        for step in range(n_steps_per_update):
            # select an action A_{t} using S_{t} as input for the agent
            actions, action_log_probs, state_value_preds, entropy = agent.select_action(
                states
            )

            # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
            states, rewards, terminated, truncated, infos = envs_wrapper.step(
                actions.cpu().numpy()
            )

            ep_value_preds[step] = torch.squeeze(state_value_preds)
            ep_rewards[step] = torch.tensor(rewards, device=device)
            ep_action_log_probs[step] = action_log_probs

            # add a mask (for the return calculation later);
            # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
            masks[step] = torch.tensor([not term for term in terminated])

        # calculate the losses for actor and critic
        critic_loss, actor_loss = agent.get_losses(
            ep_rewards,
            ep_action_log_probs,
            ep_value_preds,
            entropy,
            masks,
            gamma,
            lam,
            ent_coef,
            device,
        )

        # update the actor and critic networks
        agent.update_parameters(critic_loss, actor_loss)

        # log the losses and entropy
        critic_losses.append(critic_loss.detach().cpu().numpy())
        actor_losses.append(actor_loss.detach().cpu().numpy())
        entropies.append(entropy.detach().mean().cpu().numpy())

    """ plot the results """

    # %matplotlib inline

    rolling_length = 20
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
    fig.suptitle(
        f"Training plots for {agent.__class__.__name__} in the Walker2d-v4 environment \n \
                     (n_envs={n_envs}, n_steps_per_update={n_steps_per_update}, randomize_domain={randomize_domain})"
    )

    # episode return
    axs[0][0].set_title("Episode Returns")
    episode_returns_moving_average = (
            np.convolve(
                np.array(envs_wrapper.return_queue).flatten(),
                np.ones(rolling_length),
                mode="valid",
            )
            / rolling_length
    )
    axs[0][0].plot(
        np.arange(len(episode_returns_moving_average)) / n_envs,
        episode_returns_moving_average,
    )
    axs[0][0].set_xlabel("Number of episodes")

    # entropy
    axs[1][0].set_title("Entropy")
    entropy_moving_average = (
            np.convolve(np.array(entropies), np.ones(rolling_length), mode="valid")
            / rolling_length
    )
    axs[1][0].plot(entropy_moving_average)
    axs[1][0].set_xlabel("Number of updates")

    # critic loss
    axs[0][1].set_title("Critic Loss")
    critic_losses_moving_average = (
            np.convolve(
                np.array(critic_losses).flatten(), np.ones(rolling_length), mode="valid"
            )
            / rolling_length
    )
    axs[0][1].plot(critic_losses_moving_average)
    axs[0][1].set_xlabel("Number of updates")

    # actor loss
    axs[1][1].set_title("Actor Loss")
    actor_losses_moving_average = (
            np.convolve(np.array(actor_losses).flatten(), np.ones(rolling_length), mode="valid")
            / rolling_length
    )
    axs[1][1].plot(actor_losses_moving_average)
    axs[1][1].set_xlabel("Number of updates")

    plt.tight_layout()
    plt.show()

    save_weights = False
    load_weights = False

    actor_weights_path = "weights/actor_weights.h5"
    critic_weights_path = "weights/critic_weights.h5"

    if not os.path.exists("weights"):
        os.mkdir("weights")

    """ save network weights """
    if save_weights:
        torch.save(agent.actor.state_dict(), actor_weights_path)
        torch.save(agent.critic.state_dict(), critic_weights_path)

    """
    rewards_over_seeds = []
    obs, *_ = env.reset()
    for _ in range(500):
        done = False
        while not done:
            action = env.action_space.sample()
            env.render()
            next_state, reward, done, info, extra = env.step(action)
            obs = next_state
        print("Reward: ", reward)
        env.reset()
    env.close()
    """


def mpo_ext_train_main():
    env = gym.make('Walker2d-CPG_v1')
    save_weights = True
    save_folder = "walker_mpo"

    model = MPO(device="cuda:0", env=env)

    if os.path.exists(f"{save_folder}/model/model_latest.pt"):
        model.load_model(f"{save_folder}/model/model_latest.pt")
        print("Loaded weights")
    else:
        print("No weights found, training from scratch")

    model.train(iteration_num=10000, log_dir=save_folder)

    env.close()

    """ save network weights """
    actor_weights_path = f"{save_folder}/FinalWeights/actor_weights_MPO.h5"
    critic_weights_path = f"{save_folder}/FinalWeights/critic_weights_MPO.h5"
    if save_weights:
        torch.save(model.target_actor.state_dict(), actor_weights_path)
        torch.save(model.critic.state_dict(), critic_weights_path)
        print("saved weights")

    print("Done")

    print("Starting")

    total_rewards = []
    for i in range(10):
        obs, *_ = env.reset()
        done = False
        episode_reward = 0
        cnt = 0
        while not done:
            action, *_ = model.actor.select_action(obs)
            obs, reward, done, *_ = env.step(action)
            episode_reward += reward
            cnt += 1
            if cnt >= 5000:
                done = True

        total_rewards.append(episode_reward)
        print(f"Episode {i + 1}/{10}: Reward = {episode_reward}")

    env.close()


def setup():
    world_size = 1
    os.environ["USE_LIBUV"] = "0"
    env_master_addr = os.environ.get("MASTER_ADDR", "localhost")
    env_master_port = os.environ.get("MASTER_PORT", "234") + str(random.randint(1,99))
    env_rank = os.environ.get("RANK", 0)
    print(f"MASTER_ADDR: {env_master_addr}")
    print(f"MASTER_PORT: {env_master_port}")
    print(f"RANK: {env_rank}")
    print(f"WORLD_SIZE: {world_size}")

    # initialize the process group
    dist.init_process_group("cuda:gloo", init_method=f"tcp://{env_master_addr}:{env_master_port}", rank=env_rank,
                            world_size=world_size)
    print("Rank {} initialized".format(env_rank))


def cleanup():
    dist.destroy_process_group()


def run_mp():
    num_processes = 4
    shared_best_reward = mp.Value('d', -float('inf'))  # Shared variable to store the best reward
    mp.spawn(matsuoka_mpo_main, args=(shared_best_reward,), nprocs=num_processes, join=True)


# Training of MPO method
if __name__ == "__main__":
    register_new_env()
    # mpo_train_main(1)
    # matsuoka_mpo_main(1)
    # matsuoka_env_main()
    # run_mp()
    mpo_ext_train_main()
    # training_stable()
