import Adaptive_RL
import Experiments.experiments_utils as trials
import warnings
import argparse
import os
import numpy as np
from Adaptive_RL import logger
from MatsuokaOscillator import MatsuokaNetworkWithNN


warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="Load Experiment")

    # Algorithm and environment
    parser.add_argument('--algorithm', type=str, default='random',
                        choices=['PPO', 'SAC', 'MPO', 'DDPG', 'ppo', 'sac', 'mpo', 'ddpg', 'random'],
                        help='Choose the RL algorithm to use (PPO, SAC, MPO, DDPG).')
    parser.add_argument('--env', type=str, default=None, help='Name of the environment to train on.')
    parser.add_argument('--cpg', action='store_true', help='Whether to enable CPG flag.')
    parser.add_argument('--experiment_number', type=int, default=0, help='Experiment number for logging.')
    parser.add_argument('--f', type=str, default=None, help='Folder to load weights, models, and results.')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes.')
    parser.add_argument('--E', action='store_true', default=False, help='Whether to enable experimentation mode.')
    parser.add_argument('--V', action='store_true', default=False, help='Whether to record video.')
    parser.add_argument('--R', action='store_true', default=False, help='Run random actions.')
    parser.add_argument('-hh', action='store_true', help='Whether to enable HH Neurons, hidden.')
    parser.add_argument('--muscle_flag', action='store_true', default=False, help='Use muscle configuration')
    parser.add_argument('--last_check', action='store_true', default=False, help='Load last Checkpoint, not best.')


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
    muscle_flag = args.muscle_flag
    cpg_flag = args.cpg
    experiment_number = args.experiment_number
    hh = args.hh
    random = args.R
    algorithm = args.algorithm.upper()
    if algorithm == 'RANDOM' and experiment is not True:
        random = True
    last_checkpoint = args.last_check

    env_name, save_folder, log_dir = trials.get_name_environment(env_name, cpg_flag=cpg_flag, algorithm=algorithm,
                                                                 experiment_number=experiment_number, external_folder=args.f,
                                                                 hh_neuron=hh)

    if experiment or video_record:
        if 'myo' in env_name:
            env = Adaptive_RL.MyoSuite(env_name, render_mode="rgb_array")
        else:
            env = Adaptive_RL.Gym(env_name, render_mode="rgb_array")
    else:
        if 'myo' in env_name:
            env = Adaptive_RL.MyoSuite(env_name)
        else:
            env = Adaptive_RL.Gym(env_name, render_mode="human")

    if not random:
        path, config, _ = Adaptive_RL.get_last_checkpoint(path=log_dir, best=(not last_checkpoint))

        if cpg_flag:
            cpg_oscillators, cpg_neurons, cpg_tau_r, cpg_tau_a = trials.retrieve_cpg(config)
            env = Adaptive_RL.wrap_cpg(env, env_name, cpg_oscillators, cpg_neurons, cpg_tau_r, cpg_tau_a, hh)

        if video_record:
            video_folder = "videos/" + env_name
            agent, _ = Adaptive_RL.load_agent(config, path, env, muscle_flag=muscle_flag)

            print("Video Recording with loaded weights from {} algorithm, path: {}".format(algorithm, path))

            Adaptive_RL.record_video(env, video_folder, algorithm, agent, env_name)
            print("Video Recorded")

        elif experiment:
            print("Initialize Experiment")
            if algorithm == 'RANDOM':  # Meaning to compare all saved data
                algos = ['PPO', 'MPO', 'DDPG', 'SAC']
                algos_compare = []  # Adding CPG and HH neurons
                for algo in algos:
                    algos_compare.append(algo)
                    algos_compare.append(f'{algo}-CPG')
                    algos_compare.append(f'{algo}-CPG-HH')
                trained_algorithms = trials.search_trained_algorithms(env_name=env_name, algorithms_list=algos_compare,
                                                                      save_folder=args.f)
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
                    checkpoint_path, config, _ = Adaptive_RL.get_last_checkpoint(path=path, best=(not last_checkpoint))
                    cpg_flag=False
                    if 'CPG' in algo:
                        cpg_oscillators, cpg_neurons, cpg_tau_r, cpg_tau_a = trials.retrieve_cpg(config)
                        env = Adaptive_RL.wrap_cpg(env, env_name, cpg_oscillators, cpg_neurons, cpg_tau_r,
                                                   cpg_tau_a, hh)
                        cpg_flag=True

                    if checkpoint_path and config:
                        # Load the agent using the config and checkpoint path
                        agent, _ = Adaptive_RL.load_agent(config, checkpoint_path, env, muscle_flag=muscle_flag)

                        result = trials.evaluate_experiment(agent, env, algo, episodes_num=num_episodes,
                                                   env_name=save_folder, cpg=cpg_flag)
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
                    joints = []

                    algos_found = []

                    for algo in algos_compare:
                        # Collect the velocity, energy, distance, and reward from the results
                        if algo in results:
                            algos_found.append(algo)
                            velocities.append(results[algo]['velocity'])
                            energies.append(results[algo]['energy'])
                            distances.append(results[algo]['distance'])
                            rewards.append(results[algo]['reward'])
                            tmp_joints_r = results[algo]['joints'][0]
                            tmp_joints_l = results[algo]['joints'][3]
                            joints.append((tmp_joints_r, tmp_joints_l))
                        else:
                            # Handle the case where a result does not exist (e.g., missing algorithm folder)
                            print(f"Results for {algo} not found. Skipping.")

                    # Create the directory to save results if it doesn't exist
                    save_exp = "Experiments/Results_own/"
                    os.makedirs(save_exp, exist_ok=True)

                    # Perform energy comparison using vertical bars
                    trials.compare_vertical(data=energies, algos=algos_found, data_name="Energy per Second",
                                            units="Joules/s", save_folder=save_exp, auto_close=True)

                    # Perform distance comparison using horizontal bars
                    trials.compare_horizontal(data=distances, algos=algos_found, data_name="Distance Travelled",
                                              units="Mts", save_folder=save_exp, auto_close=True)

                    # Perform reward comparison using vertical bars
                    trials.compare_vertical(data=rewards, algos=algos_found, data_name="Rewards", save_folder=save_exp, auto_close=True)

                    trials.compare_motion_pair(results=results, algos=algos_found, save_folder=save_exp, auto_close=True)

                    # Perform velocity comparison
                    trials.compare_velocity(velocities=velocities, algos=algos_found, save_folder=save_exp,
                                            auto_close=True)

                else:
                    print(f"Not enough results found for comparison. Expected at least 2 results.")
            else:
                logger.log(f"\nRunning experiments for algorithm: {algorithm} in folder: {path}")
                agent, _ = Adaptive_RL.load_agent(config, path, env, muscle_flag=muscle_flag)
                results = trials.evaluate_experiment(agent, env, algorithm, episodes_num=num_episodes,
                                           env_name=save_folder, cpg=cpg_flag)
                velocities=results['velocity']
                energies=results['total_energy']
                avg_energies = results['energy']
                distances=results['distance']
                rewards=results['reward']
                joints=results['joints']
                right_hip_movement=joints[0]
                left_hip_movement=joints[3]
                right_hip_movement_clean = np.mean(right_hip_movement, axis=0)
                left_hip_movement_clean = np.mean(left_hip_movement, axis=0)
                right_hip_movement_clean = trials.cut_values_at_zero(right_hip_movement_clean)
                left_hip_movement_clean = trials.cut_values_at_zero(left_hip_movement_clean)

                trials.get_energy_per_meter(energies, distances, avg_energies, plot_fig=True)
                trials.statistical_analysis(data=velocities, y_axis_name="Velocity(m/s)", x_axis_name="Time",
                                            title="Velocity across time", mean_calc=True)
                trials.plot_phase(right_hip_movement_clean, left_hip_movement_clean, algo=algorithm, name="Joint Motion")
                trials.perform_autocorrelation(right_hip_movement, left_hip_movement, "Hips")
                print("Reward: ", np.mean(rewards), "\n Distance: ", np.mean(distances))



        else:
            """ load network weights """
            agent, _ = Adaptive_RL.load_agent(config, path, env, muscle_flag)
            print(env.cpg_model.print_characteristics())
            print("Loaded weights from {} algorithm, path: {}".format(algorithm, path))
            trials.evaluate(agent, env=env, algorithm=algorithm, num_episodes=num_episodes, max_episode_steps=500, no_done=False)
    else:
        algorithm = "random"
        if cpg_flag:
            amplitude = env.action_space.high
            min_value = env.action_space.low
            cpg_model = MatsuokaNetworkWithNN(num_oscillators=2,
                                              da=env.action_space.shape[0],
                                              neuron_number=2, hh=hh, max_value=amplitude, min_value=min_value)
            env = Adaptive_RL.CPGWrapper(env, cpg_model=cpg_model, use_cpg=cpg_flag)
        trials.evaluate(env=env, algorithm=algorithm, num_episodes=num_episodes, no_done=True, max_episode_steps=500)
    env.close()


if __name__ == '__main__':
    main_running()

