import Adaptive_RL
import Experiments.experiments_utils as trials
import warnings
import argparse
import os
from Adaptive_RL import logger
from MatsuokaOscillator import MatsuokaNetworkWithNN


warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="Load Experiment")

    # Algorithm and environment
    parser.add_argument('--algorithm', type=str, default='PPO',
                        choices=['PPO', 'SAC', 'MPO', 'DDPG', 'ppo', 'sac', 'mpo', 'ddpg'],
                        help='Choose the RL algorithm to use (PPO, SAC, MPO, DDPG).')
    parser.add_argument('--env', type=str, default='Humanoid-v4', help='Name of the environment to train on.')
    parser.add_argument('--cpg', action='store_true', help='Whether to enable CPG flag.')
    parser.add_argument('--f', type=str, default=None, help='Folder to load weights, models, and results.')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes.')
    parser.add_argument('--E', action='store_true', default=False, help='Whether to enable experimentation mode.')
    parser.add_argument('--V', action='store_true', default=False, help='Whether to record video.')
    parser.add_argument('--R', action='store_true', default=False, help='Run random actions.')
    parser.add_argument('-hh', action='store_true', help='Whether to enable HH Neurons, hidden.')

    return parser.parse_args()


def main_running():
    args = parse_args()
    """ play a couple of showcase episodes """
    num_episodes = args.episodes

    # env_name = "Ant-v4"
    env_name = args.env
    if 'myo' in env_name:  # Register environments if using myosuite environment
        import myosuite

    video_record = args.V
    experiment = args.E
    cpg_flag = args.cpg
    hh = args.hh
    random = args.R
    algorithm = args.algorithm.upper()

    env_name, save_folder, log_dir = trials.get_name_environment(env_name, cpg_flag=cpg_flag, algorithm=algorithm,
                                                                 experiment_number=0, external_folder=args.f,
                                                                 hh_neuron=hh)

    if experiment or video_record:
        if 'myo' in env_name:
            env = Adaptive_RL.MyoSuite(env_name, render_mode="rgb_array")
        else:
            env = Adaptive_RL.Gym(env_name, render_mode="rgb_array")
    else:
        if 'myo' in env_name:
            env = Adaptive_RL.MyoSuite(env_name, render_mode="human")
        else:
            env = Adaptive_RL.Gym(env_name, render_mode="human")

    path, config, _ = Adaptive_RL.get_last_checkpoint(path=log_dir)

    cpg_model = None
    if cpg_flag:
        cpg_oscillators, cpg_neurons, cpg_tau_r, cpg_tau_a, cpg_amplitude = trials.retrieve_cpg(config)
        cpg_model = MatsuokaNetworkWithNN(num_oscillators=cpg_oscillators,
                                          da=env.action_space.shape[0],
                                          neuron_number=cpg_neurons, tau_r=cpg_tau_r,
                                          tau_a=cpg_tau_a, hh=hh, amplitude=cpg_amplitude)
    env = Adaptive_RL.CPGWrapper(env, cpg_model=cpg_model, use_cpg=cpg_flag)

    if not random:

        if video_record:
            video_folder = "videos/" + env_name
            agent, _ = Adaptive_RL.load_agent(config, path, env)

            print("Video Recording with loaded weights from {} algorithm, path: {}".format(algorithm, path))

            Adaptive_RL.record_video(env, video_folder, algorithm, agent, env_name)
            print("Video Recorded")

        elif experiment:
            algos = ['PPO', 'MPO', 'DDPG', 'SAC']
            algos_compare = []  # Adding CPG and HH neurons
            for algo in algos:
                algos_compare.append(algo)
                algos_compare.append(f'{algo}-CPG')
                algos_compare.append(f'{algo}-CPG-HH')
            trained_algorithms = trials.search_trained_algorithms(env_name=env_name, algorithms_list=algos_compare)
            results = {}

            for algo, algo_folder in trained_algorithms:
                logger.log(f"\nRunning experiments for algorithm: {algo} in folder: {algo_folder}")

                if 'myo' in env_name:
                    env = Adaptive_RL.MyoSuite(env_name, render_mode="rgb_array", max_episode_steps=1000)
                else:
                    env = Adaptive_RL.Gym(env_name, render_mode="rgb_array", max_episode_steps=1000)
                save_folder = f"{env_name}/{algo}"

                # Get the best checkpoint path and config for the algorithm
                path = os.path.join(algo_folder, 'logs')
                checkpoint_path, config, _ = Adaptive_RL.get_last_checkpoint(path=path)
                cpg_oscillators, cpg_neurons, cpg_tau_r, cpg_tau_a, cpg_amplitude = trials.retrieve_cpg(config)

                cpg_model = MatsuokaNetworkWithNN(num_oscillators=cpg_oscillators,
                                                  da=env.action_space.shape[0],
                                                  neuron_number=cpg_neurons, tau_r=cpg_tau_r,
                                                  tau_a=cpg_tau_a, hh=hh, amplitude=cpg_amplitude)

                if 'CPG' in algo:
                    env = Adaptive_RL.CPGWrapper(env, cpg_model=cpg_model, use_cpg=cpg_flag)

                if checkpoint_path and config:
                    # Load the agent using the config and checkpoint path
                    agent, _ = Adaptive_RL.load_agent(config, checkpoint_path, env)

                    result = trials.evaluate_experiment(agent, env, algorithm, episodes_num=num_episodes,
                                               env_name=save_folder)
                    results[algo] = result

                else:
                    logger.log(f"Checking Folder: {checkpoint_path}")
                    logger.log(f"Folder for {algo} does not exist. Skipping.")

            # Perform comparisons using the collected results
            if len(results) >= 2:
                velocities = []
                energies = []
                distances = []
                rewards = []

                algos_found = []

                for algo in algos_compare:
                    # Collect the velocity, energy, distance, and reward from the results
                    if algo in results:
                        algos_found.append(algo)
                        velocities.append(results[algo]['velocity'])
                        energies.append(results[algo]['energy'])
                        distances.append(results[algo]['distance'])
                        rewards.append(results[algo]['reward'])
                    else:
                        # Handle the case where a result does not exist (e.g., missing algorithm folder)
                        print(f"Results for {algo} not found. Skipping.")

                # Create the directory to save results if it doesn't exist
                save_exp = "Experiments/Results_own/"
                os.makedirs(save_exp, exist_ok=True)

                # Perform velocity comparison
                trials.compare_velocity(velocities=velocities, algos=algos_found, save_folder=save_exp)

                # Perform energy comparison using vertical bars
                trials.compare_vertical(data=energies, algos=algos_found, data_name="Energy per Second",
                                        units="Joules/s", save_folder=save_exp)

                # Perform distance comparison using horizontal bars
                trials.compare_horizontal(data=distances, algos=algos_found, data_name="Distance Travelled",
                                          units="Mts", save_folder=save_exp)

                # Perform reward comparison using vertical bars
                trials.compare_vertical(data=rewards, algos=algos_found, data_name="Rewards", save_folder=save_exp)

            else:
                print(f"Not enough results found for comparison. Expected at least 2 results.")

        else:
            """ load network weights """
            agent, _ = Adaptive_RL.load_agent(config, path, env)

            print("Loaded weights from {} algorithm, path: {}".format(algorithm, path))
            trials.evaluate(agent, env, algorithm, num_episodes)
    else:
        algorithm = "random"
        trials.evaluate(env=env, algorithm=algorithm, num_episodes=3, no_done=True, max_episode_steps=500)
    env.close()


if __name__ == '__main__':
    main_running()

