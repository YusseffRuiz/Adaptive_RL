import Adaptive_RL
import Experiments.experiments_utils as trials
import warnings
import logging
import argparse
import os


logger: logging.Logger = logging.getLogger(__name__)

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

    return parser.parse_args()


def main_running():
    args = parse_args()
    """ play a couple of showcase episodes """
    num_episodes = args.episodes

    # env_name = "Ant-v4"
    env_name = args.env

    video_record = args.V
    experiment = args.E
    cpg_flag = args.cpg
    random = args.R
    algorithm = args.algorithm.upper()

    env_name, save_folder, log_dir = trials.get_name_environment(env_name, cpg_flag=cpg_flag, algorithm=algorithm,
                                                                 experiment_number=0, external_folder=args.f)

    if experiment:
        num_episodes = num_episodes
        env = Adaptive_RL.Gym(env_name, render_mode="rgb_array", max_episode_steps=1000)
    else:
        env = Adaptive_RL.Gym(env_name, render_mode="human")

    path = log_dir
    path, config, _ = Adaptive_RL.get_last_checkpoint(path=path)

    if not random:

        if video_record:
            video_folder = "videos/" + env_name
            agent = Adaptive_RL.load_agent(config, path, env)

            print("Video Recording with loaded weights from {} algorithm, path: {}".format(algorithm, path))

            Adaptive_RL.record_video(env_name, video_folder, algorithm, agent)
            print("Video Recorded")

        elif experiment:
            algos_compare = ['PPO', 'MPO', 'DDPG', 'SAC']
            trained_algorithms = trials.search_trained_algorithms(env_name=env_name, algorithms_list=algos_compare)
            results = {}

            # Iterate over the found algorithms and run evaluations
            for algo, algo_folder in trained_algorithms:
                print(f"Running experiments for algorithm: {algo} in folder: {algo_folder}")

                # Get the last checkpoint path and config for the algorithm
                path = os.path.join(algo_folder, 'logs')
                checkpoint_path, config, _ = Adaptive_RL.get_last_checkpoint(path=path)

                if checkpoint_path and config:
                    # Load the agent using the config and checkpoint path
                    agent, _ = Adaptive_RL.load_agent(config, checkpoint_path, env)
                    print(f"Loaded weights for {algo} from {checkpoint_path}")

                    result = trials.evaluate_experiment(agent, env, algorithm, episodes_num=num_episodes,
                                               env_name=save_folder)
                    results[algo] = result

                else:
                    print(f"Checking Folder: {checkpoint_path}")
                    print(f"Folder for {algo} does not exist. Skipping.")

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
            agent = Adaptive_RL.load_agent(config, path, env)

            print("Loaded weights from {} algorithm, path: {}".format(algorithm, path))
            trials.evaluate(agent, env, algorithm, num_episodes)
    else:
        algorithm = "random"
        trials.evaluate(env=env, algorithm=algorithm, num_episodes=3, no_done=True, max_episode_steps=500)
    env.close()


if __name__ == '__main__':
    # plot.plot(paths="Walker2d-v4/logs/Walker2d-v4-DDPG", x_axis="train/seconds", x_label="Seconds", title=f"_training")
    main_running()

