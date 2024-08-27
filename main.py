import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from MatsuokaOscillator import MatsuokaOscillator, MatsuokaNetwork, MatsuokaAgent, MatsuokaNetworkWithNN, MpoMatsuokaTrainer
from MPO_Algorithm import MPOAgent, MPOTrainer

import gymnasium as gym
from reinforce import A2C
from tqdm import tqdm
import os

import torch.multiprocessing as mp
import torch.distributed as dist


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
    u = np.full(neuron_number, u1)
    time = np.linspace(0, steps * dt, steps)

    if neural_net is True:
        # Neural Network Implementation
        input_size = num_oscillators  # Example input size
        hidden_size = 10  # Hidden layer size
        output_size = 3  # tau_r, weights, and beta for each oscillator

        nn_model = MatsuokaAgent(input_size, hidden_size, 1, output_size, 1, device="cuda")
        matsuoka_network = MatsuokaNetworkWithNN(num_oscillators, nn_model, action_space=None, action_dim=6)
        # Create a sample sensory input sequence
        sensory_input_seq = torch.rand(steps, num_oscillators, input_size, dtype=torch.float32, device="cuda")

        # Run the coupled system with NN control
        outputs = matsuoka_network.run(steps=steps, sensory_input_seq=sensory_input_seq)

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
            oscillator = MatsuokaOscillator(neuron_number=neuron_number, tau_r=tau_r, tau_a=tau_a, weights=weights, u=u,
                                            beta=beta, dt=dt, action_space=None)
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
                                             tau_a=tau_a, weights=weights, u=u, beta=beta, dt=dt)
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


def matsuoka_env_main():
    num_oscillators = 2
    neuron_number = 2
    dt = 0.1
    steps = 1000
    time = np.linspace(0, steps * dt, steps)

    env = gym.make('Walker2d-v4', render_mode='human')


    matsuoka_network = MatsuokaNetworkWithNN(num_oscillators=num_oscillators, observation_space=env.observation_space.shape[0],
                                             action_dim=env.action_space.shape[0], action_space=env.action_space,
                                             neuron_number=neuron_number)

    # Run the environment in a loop

    total_rewards = []
    state, *_ = env.reset()
    for episode in range(5):
        episode_reward = 0
        done = False
        state, *_ = env.reset()
        while not done:
            # Get the output from the oscillator network
            state_tensor = torch.tensor(state, dtype=torch.float32, device="cuda").unsqueeze(0)  # Convert state to tensor
            # sensory_input = torch.rand(num_oscillators, 2, dtype=torch.float32, device="cuda")
            actions = matsuoka_network.step(state_tensor)  # Update the oscillator network

            # Step the environment using the actions
            next_state, reward, done, *_ = env.step(actions.detach().cpu().numpy())
            state = next_state

            episode_reward += reward
            # Render the environment
            env.render()
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{5}: Reward = {episode_reward}")
    env.close()


def matsuoka_mpo_main(world_size):
    env_name = 'Walker2d-v4'
    env = gym.make(env_name)
    num_envs = 4
    state_dim = env.observation_space.shape[0]
    action_space = env.action_space
    params_dim = 2  # Weights number, 1 per neuron

    setup()

    # Initialize MPOAgent
    agent = MPOAgent(state_dim=state_dim, action_dim=params_dim, action_space=action_space, hidden_dim=128,
                     kl_epsilon=0.1, actor_lr=3e-4, critic_lr=3e-4,
                     device="cuda")

    # Initialize MPOTrainer
    trainer = MpoMatsuokaTrainer(env=env_name, n_envs=num_envs, agent=agent, n_steps_per_update=2048, gamma=0.99,
                                 lam=0.95, device="cuda", weights_path="weights_2")

    # Run the training loop
    trainer.train(n_updates=1000)
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

    """ save network weights """
    if save_weights:
        torch.save(agent.actor.state_dict(), actor_weights_path)
        torch.save(agent.critic.state_dict(), critic_weights_path)
        print("saved weights")

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


def mpo_train_main(world_size):

    env_name = 'Walker2d-v4'
    env = gym.make(env_name)
    num_envs = 4
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_space = env.action_space

    #setup()

    # Initialize MPOAgent
    agent = MPOAgent(state_dim=state_dim, action_dim=action_dim, action_space=action_space, hidden_dim=128,
                     kl_epsilon=0.1, actor_lr=3e-4, critic_lr=3e-4,
                     device="cuda")

    # Initialize MPOTrainer
    trainer = MPOTrainer(env=env_name, n_envs=num_envs, agent=agent, n_steps_per_update=2048, gamma=0.99, lam=0.95,
                         device="cuda", weights_path="weights_1")


    # Run the training loop
    trainer.train(n_updates=10001)


    # After training, evaluate the agent
    # Implement an evaluation loop or method in MPOTrainer if needed

    save_weights = True
    load_weights = False

    actor_weights_path = "weights_1/FinalWeights/actor_weights_MPO.h5"
    critic_weights_path = "weights_1/FinalWeights/critic_weights_MPO.h5"

    if not os.path.exists("weights_1"):
        os.mkdir("weights_1")
        if not os.path.exists("weights_1/FinalWeights"):
            os.mkdir("weights_1/FinalWeights")


    """ save network weights """
    if save_weights:
        torch.save(agent.actor.state_dict(), actor_weights_path)
        torch.save(agent.critic.state_dict(), critic_weights_path)
        print("saved weights")

    print("Done")




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
    mp.spawn(matsuoka_mpo_main, nprocs=1, join=True)


# Training of MPO method
if __name__ == "__main__":
    # mpo_train_main(1)
    # matsuoka_mpo_main(1)
    # matsuoka_env_main()
    run_mp()
