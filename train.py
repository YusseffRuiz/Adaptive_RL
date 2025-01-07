import torch
from MatsuokaOscillator import MatsuokaNetworkWithNN
import Adaptive_RL
from Adaptive_RL import SAC, DDPG, MPO, PPO
import Experiments.experiments_utils as trials
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
    parser.add_argument('--params', type=str, default=None, help='Parameters to load from a file.')
    parser.add_argument('-hh', action='store_true', help='Whether to enable HH Neurons, hidden.')
    parser.add_argument('--muscles', action='store_true', help='The use of DEP to map and create muscles.')
    parser.add_argument('--P', action='store_true', help='Whether to show complete progress bar.')

    # General Paramenters
    parser.add_argument('--experiment_number', type=int, default=0, help='Experiment number for logging.')
    parser.add_argument('--steps', type=int, default=int(1e6), help='Maximum steps for training.')
    parser.add_argument('--seq', type=int, default=1, help='Number of sequential environments.')
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel environments.')
    parser.add_argument('--early', action='store_true', help='Early stopping if we do not have rewards improvement')

    # Hyperparameters
    parser.add_argument('--learning_rate', type=float, default=3.56987e-05, help='Learning rate for the actor.')
    parser.add_argument('--lr_critic', type=float, default=3e-4, help='Learning rate for the critic.')
    parser.add_argument('--ent_coeff', type=float, default=0.00238306, help='Entropy coefficient for PPO or SAC.')
    parser.add_argument('--gamma', type=float, default=0.98, help='Discount factor for replay buffer')
    parser.add_argument('--decay_lr', type=float, default=None, help='Discount factor for normalizer')
    parser.add_argument('--clip_range', type=float, default=0.3, help='Clip range for PPO and MPO.')
    parser.add_argument('--lr_dual', type=float, default=3.56987e-04, help='Learning rate for dual variables (MPO).')
    parser.add_argument('--neuron_number', type=int, nargs='+', default=[256],
                        help='Number of neurons in hidden layers. Can be a single integer or a list of integers.')
    parser.add_argument('--layers_number', type=int, default=2, help='Number of hidden layers.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training.')
    parser.add_argument('--replay_buffer_size', type=int, default=int(10e5), help='Replay buffer size.')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Exploration rate (epsilon-greedy) (MPO).')
    parser.add_argument('--learning_starts', type=int, default=10000, help='Number of steps before learning starts.')
    parser.add_argument('--noise', type=float, default=0.01, help='Noise added to future rewards.')

    # CPG Hyperparameters
    parser.add_argument('--cpg_oscillators', type=int, default=2, help='Number of CPG oscillators.')
    parser.add_argument('--cpg_neurons', type=int, default=2, help='Number of CPG neurons.')
    parser.add_argument('--cpg_tau_r', type=float, default=32.0, help='tau r for CPG')
    parser.add_argument('--cpg_tau_a', type=float, default=96.0, help='tau a for CPG')

    return parser.parse_args()


def train_agent(
        agent, environment, trainer=Adaptive_RL.Trainer(), parallel=1, sequential=1, seed=0,
        checkpoint="last", path=None, cpg_flag=False, hh=False, cpg_oscillators=2,
        cpg_neurons=2, cpg_tau_r=32.0, cpg_tau_a=96.0, progress=False, muscle_flag=False, device='cuda'):
    """
    :param progress: Show or not progress bar
    :param cpg_amplitude: Amplitud for the Matsuoka Calculations
    :param cpg_tau_a: object learning component of the CPGs
    :param cpg_tau_r: learning component of the CPGs
    :param cpg_neurons: Number of neurons to control actuators on the model
    :param cpg_oscillators: number of oscillators to be in synchrony
    :param hh: If we want to use Hudkin-Huxley Experimental Neurons
    :param cpg_flag: The use of oscillators in the training
    :param agent: Agent and algorithm to be trained.
    :param environment: Environment name
    :param trainer: Trainer to be used, at this moment, the default from tonic
    :param parallel: Parallel Processes
    :param sequential: Vector Environments.
    :param seed: random seed
    :param checkpoint: checkpoint to verify existence.
    :param path: Path where the experiment to check for checkpoints
    :param device:
    """
    if device == 'cuda':
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(torch.cuda.current_device())
            torch.set_default_device('cuda')
            print(f"Runing with {device}")
        else:
            print("Running with CPU, no CUDA available")
    else:
        print("Running with CPU")

    args = dict(locals())
    # Create a new dictionary excluding 'agent'

    checkpoint_path = None
    checkpoint_folder = None
    config = {}

    if path:
        # Load last checkpoint, not best
        checkpoint_path, config, checkpoint_folder = Adaptive_RL.get_last_checkpoint(path, best=False)
        if config is not None:
            # Load the experiment configuration.
            trainer_config = config.trainer
            # Loading trainer config
            trainer.max_steps = int(float(trainer_config['max_steps'])) or trainer.max_steps
            trainer.epoch_steps = trainer_config['epoch_steps'] or trainer.epoch_steps
            trainer.save_steps = trainer_config['save_steps'] or trainer.save_steps
            trainer.early_stopping = trainer_config['early_stopping'] or trainer.early_stopping
            trainer.test_episodes = trainer_config['test_episodes'] or trainer.test_episodes

            # Loading CPG configuration
            cpg_oscillators = config.cpg_oscillators or cpg_oscillators
            cpg_neurons = config.cpg_neurons or cpg_neurons
            cpg_tau_r = config.cpg_tau_r or cpg_tau_r
            cpg_tau_a = config.cpg_tau_a or cpg_tau_a

            # Set DEP parameters
            # if hasattr(agent, "expl") and "DEP" in config:
            #     agent.expl.set_params(config["DEP"])

            print("Loaded Config")

    # Build the training environment.
    if 'myo' in env_name:
        _environment = Adaptive_RL.MyoSuite(environment)
    else:
        _environment = Adaptive_RL.Gym(environment)
    if muscle_flag:
        _environment = Adaptive_RL.apply_wrapper(_environment)
    cpg_model = None
    if cpg_flag:
        _environment = Adaptive_RL.wrap_cpg(_environment, env_name, cpg_oscillators, cpg_neurons, cpg_tau_r,
                                   cpg_tau_a, hh)
        print(_environment.cpg_model.print_characteristics())
    environment = Adaptive_RL.parallelize.distribute(_environment, parallel, sequential)
    environment.initialize(muscles=muscle_flag)

    # Build the testing environment.
    test_environment = Adaptive_RL.parallelize.distribute(_environment)
    test_environment.initialize(muscles=muscle_flag)


    # Build the agent.
    if not agent:
        raise ValueError('No agent specified.')

    # Load the weights of the agent form a checkpoint.
    step_number = 0
    if checkpoint_path:
        agent, step_number = Adaptive_RL.load_agent(config, checkpoint_path, env=_environment, muscle_flag=muscle_flag)
        agent.initialize(observation_space=environment.observation_space, action_space=environment.action_space,
                             seed=seed)
    else:
        agent.initialize(observation_space=environment.observation_space, action_space=environment.action_space,
                         seed=seed)

    args['agent'] = agent.get_config(print_conf=True)
    args['trainer'] = trainer.dump_trainer()
    # Initialize the logger to save data to the path
    Adaptive_RL.logger.initialize(path=path, config=args, progress=progress)

    # Build the trainer.
    trainer.initialize(
        agent=agent, environment=environment,
        test_environment=test_environment, step_saved=step_number, muscle_flag=muscle_flag)
    trainer.load_model(agent=agent, actor_updater=agent.actor_updater, replay_buffer=agent.replay,
                       save_path=checkpoint_folder)
    # Train.
    trainer.run()
    agent = trainer.agent
    return agent


if __name__ == "__main__":

    args = parse_args()

    training_algorithm = args.algorithm.upper()

    env_name = args.env

    if 'myo' in env_name:  # Register environments if using myosuite environment
        import myosuite

    cpg_flag = args.cpg
    hh = args.hh
    experiment_number = args.experiment_number
    muscle_flag = args.muscles
    progress = args.P

    save_folder = args.f

    # Training Mode
    sequential = args.seq
    parallel = args.parallel
    early_stopping = args.early

    if args.params is not None:
        args, cpg_args = Adaptive_RL.file_to_hyperparameters(args.params, env_name, training_algorithm)
        # Hyperparameters
        learning_rate = args['training']['learning_rate']
        lr_critic = args['training'].get('lr_critic', 0.0)
        ent_coeff = args['training'].get('ent_coeff', 0.0)
        clip_range = args['training'].get('clip_range', 0.0)
        lr_dual = args['training'].get('lr_dual', 0.0)
        gamma = args['training']['gamma']
        decay_lr = args['training'].get('decay_lr', None)
        neuron_number = args['model']['neuron_number']
        layers_number = args['model']['layers_number']
        batch_size = args['training']['batch_size']
        replay_buffer_size = args['training']['replay_buffer_size']
        epsilon = args['training'].get('epsilon', 0.0)
        learning_starts = args['training'].get('learning_starts', 0.0)
        noise_std = args['training'].get('noise_std', 0.0)

        # CPG params
        cpg_oscillator = cpg_args['num_oscillators']
        cpg_neurons = cpg_args['neuron_number']
        cpg_tau_r = cpg_args['tau_r'] or 32.0
        cpg_tau_a = cpg_args['tau_a'] or 96.0

        max_steps = int(float(args['training']['steps']))

    else:
        # Steps to train
        max_steps = args.steps

        # Hyperparameters
        learning_rate = args.learning_rate
        lr_critic = args.lr_critic
        ent_coeff = args.ent_coeff
        clip_range = args.clip_range
        lr_dual = args.lr_dual
        gamma = args.gamma
        decay_lr = args.decay_lr
        neuron_number = args.neuron_number
        layers_number = args.layers_number
        batch_size = args.batch_size
        replay_buffer_size = args.replay_buffer_size
        epsilon = args.epsilon
        learning_starts = args.learning_starts
        noise_std = args.noise

        # cpg
        cpg_oscillator = args.cpg_oscillators
        cpg_neurons = args.cpg_neurons
        cpg_tau_r = args.cpg_tau_r
        cpg_tau_a = args.cpg_tau_a
    save_steps = int(max_steps / 200)
    epochs = save_steps//2

    normalizer_flag = False
    if decay_lr is not None:
        normalizer_flag = True

    env_name, save_folder, log_dir = trials.get_name_environment(env_name, cpg_flag=cpg_flag,
                                                                 algorithm=training_algorithm, create=True,
                                                                 experiment_number=experiment_number,
                                                                 external_folder=save_folder, hh_neuron=hh)

    if training_algorithm == "MPO":
        agent = MPO(lr_actor=learning_rate, lr_critic=lr_critic, lr_dual=lr_dual, hidden_size=neuron_number,
                    discount_factor=gamma, replay_buffer_size=replay_buffer_size, hidden_layers=layers_number,
                    batch_size=batch_size, epsilon=epsilon, gradient_clip=clip_range)
    elif training_algorithm == "SAC":
        agent = SAC(learning_rate=learning_rate, hidden_size=neuron_number, discount_factor=gamma,
                    hidden_layers=layers_number, replay_buffer_size=replay_buffer_size, batch_size=batch_size,
                    learning_starts=learning_starts)
    elif training_algorithm == "PPO":
        agent = PPO(learning_rate=learning_rate, hidden_size=neuron_number, hidden_layers=layers_number,
                    gamma=gamma, decay_lr=decay_lr, normalizer=normalizer_flag,
                    batch_size=batch_size, entropy_coeff=ent_coeff, clip_range=clip_range)
    elif training_algorithm == "DDPG":
        agent = DDPG(learning_rate=learning_rate, batch_size=batch_size, learning_starts=learning_starts,
                     noise_std=noise_std, hidden_size=neuron_number, hidden_layers=layers_number,
                     replay_buffer_size=replay_buffer_size)
    else:
        agent = None

    if agent is not None:
        if muscle_flag:
            agent = Adaptive_RL.dep_agents.dep_factory(3, agent)()
        agent = train_agent(agent=agent,
                            environment=env_name,
                            sequential=sequential, parallel=parallel,
                            trainer=Adaptive_RL.Trainer(steps=max_steps, epoch_steps=epochs, save_steps=save_steps,
                                                        early_stopping=early_stopping),
                            path=log_dir, cpg_flag=cpg_flag, hh=hh, progress=progress, cpg_oscillators=cpg_oscillator,
                            cpg_neurons=cpg_neurons, cpg_tau_a=cpg_tau_a, cpg_tau_r=cpg_tau_r, muscle_flag=muscle_flag)

        print("Training Done")
    else:
        print("No agent specified, finishing program")
    exit()
