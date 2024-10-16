import matplotlib.pyplot as plt
import numpy as np
import torch
from MatsuokaOscillator import MatsuokaOscillator, MatsuokaNetwork, MatsuokaNetworkWithNN
import Adaptive_RL
from Adaptive_RL import SAC, DDPG, MPO, PPO, plot
import Experiments.experiments_utils as trials
from myosuite.utils import gym
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run RL training with hyperparameters and CPG environment")

    # Algorithm and environment
    parser.add_argument('--algorithm', type=str, default='PPO',
                        choices=['PPO', 'SAC', 'MPO', 'DDPG', 'ppo', 'sac', 'mpo', 'ddpg'],
                        help='Choose the RL algorithm to use (PPO, SAC, MPO, DDPG).')
    parser.add_argument('--env', type=str, default='Humanoid-v4', help='Name of the environment to train on.')
    parser.add_argument('--cpg', action='store_true', help='Whether to enable CPG flag.')
    parser.add_argument('--f', type=str, default=None, help='Folder to save logs, models, and results.')

    # General Paramenters
    parser.add_argument('--experiment_number', type=int, default=0, help='Experiment number for logging.')
    parser.add_argument('--steps', type=int, default=int(1e7), help='Maximum steps for training.')
    parser.add_argument('--seq', type=int, default=1, help='Number of sequential environments.')
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel environments.')

    # Hyperparameters
    parser.add_argument('--learning_rate', type=float, default=3.56987e-05, help='Learning rate for the actor.')
    parser.add_argument('--lr_critic', type=float, default=3e-4, help='Learning rate for the critic.')
    parser.add_argument('--ent_coeff', type=float, default=0.00238306, help='Entropy coefficient for PPO or SAC.')
    parser.add_argument('--clip_range', type=float, default=0.3, help='Clip range for PPO.')
    parser.add_argument('--lr_dual', type=float, default=3.56987e-04, help='Learning rate for dual variables (MPO).')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor.')
    parser.add_argument('--neuron_number', type=int, nargs='+', default=[256],
                        help='Number of neurons in hidden layers. Can be a single integer or a list of integers.')
    parser.add_argument('--layers_number', type=int, default=2, help='Number of hidden layers.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training.')
    parser.add_argument('--replay_buffer_size', type=int, default=int(10e5), help='Replay buffer size.')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Exploration rate (epsilon-greedy).')
    parser.add_argument('--learning_starts', type=int, default=10000, help='Number of steps before learning starts.')
    parser.add_argument('--noise', type=float, default=0.01, help='Noise added to future rewards.')

    return parser.parse_args()


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


def train_agent(
        agent, environment, trainer=Adaptive_RL.Trainer(), parallel=1, sequential=1, seed=0,
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
    # if torch.cuda.is_available():
    #     device = torch.cuda.get_device_name(torch.cuda.current_device())
    #     torch.set_default_device('cuda')
    #     print(f"Runing with {device}")
    # else:
    #     print("Running with CPU")
    path = log_dir
    args = dict(locals())
    # Create a new dictionary excluding 'agent' and 'trainer'
    args = {k: v for k, v in args.items() if k not in ['agent', 'trainer']}


    checkpoint_path = None
    checkpoint_folder = None
    config = None

    # Build the training environment.

    _environment = Adaptive_RL.Gym(environment)
    environment = Adaptive_RL.parallelize.distribute(
        lambda: _environment, parallel, sequential)
    environment.initialize() if parallel > 1 else 0

    # Build the testing environment.
    test_environment = Adaptive_RL.parallelize.distribute(
        lambda: _environment)

    # Process the checkpoint path same way as in tonic.play
    if path:
        checkpoint_path, config, checkpoint_folder = Adaptive_RL.get_last_checkpoint(path)
        if config is not None:
            # Load the experiment configuration.
            trainer = trainer or config.trainer
            print("Loaded Config")

    # Build the agent.
    if not agent:
        raise ValueError('No agent specified.')


    # Load the weights of the agent form a checkpoint.
    step_number = 0
    buffer_data = None
    if checkpoint_path:
        agent, step_number = Adaptive_RL.load_agent(config, checkpoint_path, env=_environment)
    else:
        agent.initialize(observation_space=environment.observation_space, action_space=environment.action_space,
                         seed=seed)
    args['agent'] = agent.get_config()

    agent.get_config(print_conf=True)
    # Initialize the logger to save data to the path
    Adaptive_RL.logger.initialize(path=log_dir, config=args)

    # Build the trainer.
    trainer.initialize(
        agent=agent, environment=environment,
        test_environment=test_environment, step_saved=step_number)
    trainer.load_model(agent=agent, actor_updater=agent.actor_updater, replay_buffer=agent.replay_buffer, save_path=checkpoint_folder)
    # Train.
    trainer.run()


if __name__ == "__main__":

    args = parse_args()

    training_algorithm = args.algorithm.upper()

    # env_name = "Walker2d-v4"
    env_name = args.env
    cpg_flag = args.cpg
    experiment_number = args.experiment_number

    # Steps to train
    max_steps = args.steps
    epochs = int(max_steps / 1000)
    save_steps = int(max_steps / 500)

    save_folder = args.f

    # Training Mode
    sequential = args.seq
    parallel = args.parallel

    # Hyperparameters
    learning_rate = args.learning_rate
    lr_critic = args.lr_critic
    ent_coeff = args.ent_coeff
    clip_range = args.clip_range
    lr_dual = args.lr_dual
    gamma = args.gamma
    neuron_number = args.neuron_number
    layers_number = args.layers_number
    batch_size = args.batch_size
    replay_buffer_size = args.replay_buffer_size
    epsilon = args.epsilon
    learning_starts = args.learning_starts
    noise_std = args.noise

    env_name, save_folder, log_dir = trials.get_name_environment(env_name, cpg_flag=cpg_flag,
                                                                 algorithm=training_algorithm, create=True,
                                                                 experiment_number=experiment_number,
                                                                 external_folder=save_folder)

    if training_algorithm == "MPO":
        agent = MPO(lr_actor=learning_rate, lr_critic=lr_critic, lr_dual=lr_dual, hidden_size=neuron_number,
                    discount_factor=gamma, replay_buffer_size=replay_buffer_size, hidden_layers=layers_number)
    elif training_algorithm == "SAC":
        agent = SAC(learning_rate=learning_rate, hidden_size=neuron_number, discount_factor=gamma,
                    hidden_layers=layers_number, replay_buffer_size=replay_buffer_size, batch_size=batch_size,
                    learning_starts=learning_starts)
    elif training_algorithm == "PPO":
        agent = PPO(learning_rate=learning_rate, hidden_size=neuron_number, hidden_layers=layers_number,
                    discount_factor=gamma,
                    batch_size=batch_size, entropy_coeff=ent_coeff, clip_range=clip_range)
    elif training_algorithm == "DDPG":
        agent = DDPG(learning_rate=learning_rate, batch_size=batch_size, learning_starts=learning_starts,
                     noise_std=noise_std, hidden_size=neuron_number, hidden_layers=layers_number, replay_buffer_size=replay_buffer_size)
    else:
        agent = None

    if agent is not None:
        print("Max Steps: ", max_steps)
        train_agent(agent=agent,
                    environment=env_name,
                    sequential=1, parallel=1,
                    trainer=Adaptive_RL.Trainer(steps=max_steps, epoch_steps=epochs, save_steps=save_steps),
                    log_dir=log_dir)

        env = gym.make(env_name, render_mode="human", max_episode_steps=1500)

        print("Starting Evaluation")
        trials.evaluate(agent, env, algorithm=training_algorithm, num_episodes=5)

        env.close()
    else:
        print("No agent specified, finishing program")
        exit()
