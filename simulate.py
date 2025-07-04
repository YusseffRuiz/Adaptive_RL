import Adaptive_RL
import Experiments.experiments_utils as trials
import warnings
import argparse
import os
import numpy as np
from Adaptive_RL import logger

import utilities_repository.utils_extra as utils_extra


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
    parser.add_argument('--auto', action='store_true', default=False, help='Automatically close experiment windows.')
    parser.add_argument('--separate_action', action='store_true', default=False, help='Automatically close experiment windows.')


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
    auto_close = args.auto
    algorithm = args.algorithm.upper()
    separate_flag = args.separate_action
    if algorithm == 'RANDOM' and experiment is not True:
        random = True
    last_checkpoint = args.last_check

    env_name, save_folder, log_dir = trials.get_name_environment(env_name, cpg_flag=cpg_flag, algorithm=algorithm,
                                                                 experiment_number=experiment_number, external_folder=args.f,
                                                                 hh_neuron=hh)

    if 'myo' in env_name:
        env = Adaptive_RL.MyoSuite(env_name, reset_type='random', scaled_actions=False, max_episode_steps=2000)
    else:
        if experiment or video_record:
            env = Adaptive_RL.Gym(env_name, render_mode="rgb_array")
        else:
            env = Adaptive_RL.Gym(env_name, render_mode="human")

    if muscle_flag:
        env = Adaptive_RL.apply_wrapper(env, direct=True, separate_flag=separate_flag)

    if not random:
        if video_record:
            path, config, _ = Adaptive_RL.get_last_checkpoint(path=log_dir, best=(not last_checkpoint))

            if cpg_flag:
                cpg_oscillators, cpg_neurons, cpg_tau_r, cpg_tau_a = trials.retrieve_cpg(config)
                env = Adaptive_RL.wrap_cpg(env, env_name, cpg_oscillators, cpg_neurons, cpg_tau_r, cpg_tau_a, hh)

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
                        env = Adaptive_RL.MyoSuite(env_name, reset_type='random', scaled_actions=False,
                                                   max_episode_steps=1500)
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

                    if muscle_flag:
                        env = Adaptive_RL.apply_wrapper(env, direct=True)

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
                    joints_hip = []
                    joints_ankle = []
                    energy_per_meter = []

                    algos_found = []

                    for algo in algos_compare:
                        # Collect the velocity, energy, distance, and reward from the results
                        if algo in results:
                            algos_found.append(algo)
                            velocities.append(results[algo]['velocity'])
                            energies.append(results[algo]['energy'])
                            energy_per_meter.append(results[algo]['total_energy_per_meter'])
                            distances.append(results[algo]['distance'])
                            rewards.append(results[algo]['reward'])
                            tmp_joints_r = results[algo]['joints'][0]
                            tmp_joints_l = results[algo]['joints'][3]
                            joints_hip.append((tmp_joints_r, tmp_joints_l))
                            if muscle_flag:
                                tmp_joints_r = results[algo]['joints'][2]
                                tmp_joints_l = results[algo]['joints'][5]
                                joints_ankle.append((tmp_joints_r, tmp_joints_l))
                            print(f"Falls in {algo} algorithm", results[algo]['falls'])
                        else:
                            # Handle the case where a result does not exist (e.g., missing algorithm folder)
                            print(f"Results for {algo} not found. Skipping.")

                    # Create the directory to save results if it doesn't exist
                    save_exp = f"Experiments/Results_own/{env_name}"
                    os.makedirs(save_exp, exist_ok=True)

                    # Perform energy comparison using vertical bars
                    # trials.compare_vertical(data=energies, algos=algos_found, data_name="Energy per Second",
                    #                         units="Joules/s", save_folder=save_exp, auto_close=auto_close)

                    trials.compare_vertical(data=energy_per_meter, algos=algos_found, data_name="Energy per Meter",
                                            units="Joules/s", save_folder=save_exp, auto_close=auto_close)

                    # Perform distance comparison using horizontal bars
                    trials.compare_horizontal(data=distances, algos=algos_found, data_name="Distance Travelled",
                                              units="Mts", save_folder=save_exp, auto_close=auto_close)

                    # Perform reward comparison using vertical bars
                    trials.compare_vertical(data=rewards, algos=algos_found, data_name="Rewards", save_folder=save_exp, auto_close=auto_close)

                    trials.compare_motion_pair(results=results, algos=algos_found, save_folder=save_exp, auto_close=auto_close, place='hip')

                    if muscle_flag:
                        trials.compare_motion_pair(results=results, algos=algos_found, save_folder=save_exp, auto_close=auto_close, place='ankle')

                    # Perform velocity comparison
                    trials.compare_velocity(velocities=velocities, algos=algos_found, save_folder=save_exp,
                                            auto_close=auto_close)


                else:
                    print(f"Not enough results found for comparison. Expected at least 2 results.")
            else:
                print("Log Dir: ", log_dir)
                path, config, _ = Adaptive_RL.get_last_checkpoint(path=log_dir, best=(not last_checkpoint))

                if cpg_flag:
                    cpg_oscillators, cpg_neurons, cpg_tau_r, cpg_tau_a = trials.retrieve_cpg(config)
                    env = Adaptive_RL.wrap_cpg(env, env_name, cpg_oscillators, cpg_neurons, cpg_tau_r, cpg_tau_a, hh)
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
                right_ankle_movement = joints[2]
                left_ankle_movement = joints[5]

                right_hip_movement_clean = np.mean(right_hip_movement, axis=0)
                left_hip_movement_clean = np.mean(left_hip_movement, axis=0)
                right_hip_movement_clean = trials.cut_values_at_zero(right_hip_movement_clean)
                left_hip_movement_clean = trials.cut_values_at_zero(left_hip_movement_clean)

                right_ankle_movement_clean = np.mean(right_ankle_movement, axis=0)
                left_ankle_movement_clean = np.mean(left_ankle_movement, axis=0)
                right_ankle_movement_clean = trials.cut_values_at_zero(right_ankle_movement_clean)
                left_ankle_movement_clean = trials.cut_values_at_zero(left_ankle_movement_clean)

                trials.get_energy_per_meter(energies, distances, plot_fig=True, save_folder=save_folder)
                trials.statistical_analysis(data=velocities, y_axis_name="Velocity(m/s)", x_axis_name="Time",
                                            title="Velocity across time", mean_calc=True, save_folder=save_folder)
                # trials.plot_phase(right_hip_movement_clean, left_hip_movement_clean, algo=algorithm, name="Hip Joint Motion", save_folder=save_folder)
                trials.plot_phase(right_hip_movement_clean, algo=algorithm, name="Hip Right Joint Motion", save_folder=save_folder)
                trials.plot_phase(left_hip_movement_clean, algo=algorithm, name="Hip Left Joint Motion", save_folder=save_folder)
                # trials.plot_phase(right_ankle_movement_clean, left_ankle_movement_clean, algo=algorithm,
                #                   name="Ankle Joint Motion", save_folder=save_folder)
                trials.plot_phase(right_ankle_movement_clean, algo=algorithm, name="Ankle Right Joint Motion", save_folder=save_folder)
                trials.plot_phase(left_ankle_movement_clean, algo=algorithm, name="Ankle Left Joint Motion", save_folder=save_folder)

                lags_hip, crossCorr_hip = trials.perform_autocorrelation(right_hip_movement, left_hip_movement, "Hips Correlation", save_folder=save_folder)
                lags_ankle, crossCorr_ankle = trials.perform_autocorrelation(right_ankle_movement, left_ankle_movement, "Ankle Correlation", save_folder=save_folder)
                print("Mean Reward: ", np.mean(rewards), "\n Mean Distance: ", np.mean(distances))

                # Saving arrays:
                np.save(f'{save_folder}/right_ankle_movement.npy', right_ankle_movement_clean)
                np.save(f'{save_folder}/left_ankle_movement.npy', left_ankle_movement_clean)
                np.save(f'{save_folder}/right_hip_movement.npy', right_hip_movement_clean)
                np.save(f'{save_folder}/left_hip_movement.npy', left_hip_movement_clean)
                np.save(f'{save_folder}/crossCorrelation_ankle.npy', crossCorr_hip)
                np.save(f'{save_folder}/lags_ankle.npy', lags_ankle)
                np.save(f'{save_folder}/crossCorrelation_hip.npy', crossCorr_ankle)
                np.save(f'{save_folder}/lags_hip.npy', lags_hip)

        else:
            """ load network weights """
            path, config, _ = Adaptive_RL.get_last_checkpoint(path=log_dir, best=(not last_checkpoint))

            if cpg_flag:
                cpg_oscillators, cpg_neurons, cpg_tau_r, cpg_tau_a = trials.retrieve_cpg(config)
                env = Adaptive_RL.wrap_cpg(env, env_name, cpg_oscillators, cpg_neurons, cpg_tau_r, cpg_tau_a, hh)
                print(env.cpg_model.print_characteristics())
            agent, _ = Adaptive_RL.load_agent(config, path, env, muscle_flag)
            print("Loaded weights from {} algorithm, path: {}".format(algorithm, path))
            trials.evaluate(agent, env=env, algorithm=algorithm, num_episodes=num_episodes, max_episode_steps=1500, no_done=False)
    else:
        algorithm = "random"
        # if cpg_flag:
        #     env = Adaptive_RL.wrap_cpg(env, env_name, 2, 2, hh)
        # path_T, config_T, _ = Adaptive_RL.get_last_checkpoint(path="training/myoAmp1DoFWalk-v0-SAC-CPG/3/logs")
        # env_leg = Adaptive_RL.MyoSuite("myoAmp1DoFWalk-v0", reset_type='static', scaled_actions=False, max_episode_steps=1000)
        # env_leg = Adaptive_RL.apply_wrapper(env_leg, direct=True)
        # cpg_oscillators, cpg_neurons, cpg_tau_r, cpg_tau_a = trials.retrieve_cpg(config_T)
        # env_leg = Adaptive_RL.wrap_cpg(env_leg, "myoAmp1DoFWalk-v0", cpg_oscillators, cpg_neurons, cpg_tau_r, cpg_tau_a, hh)
        # agent_leg, _ = Adaptive_RL.load_agent(config_T, path_T, env_leg, muscle_flag)
        # agent_abled = utils_extra.load_mpo("training/myoLeg/logs/", env_leg)
        # trials.evaluate_envs(env=env, env2=env_leg, algorithm=algorithm, num_episodes=num_episodes, no_done=True, max_episode_steps=500, model2=agent_leg)
        print(env.action_space.shape)
        print(env.get_obs_dict(env.sim).keys())
        trials.evaluate(env=env, algorithm=algorithm, num_episodes=num_episodes, no_done=True, max_episode_steps=500, model=None)
    env.close()


if __name__ == '__main__':
    main_running()



"""
Walking Plain
The t-test statistic is -8.621269144974763 and the p-value is 2.1155092044111955e-15  for the  Energy per Meter  value
P-value < 0.05, showing significant difference between groups, rejecting null Hypothesis for the  Energy per Meter  value
The t-test statistic is 10.896673835042327 and the p-value is 5.796560877217538e-22  for the  Distance Travelled  value
P-value < 0.05, showing significant difference between groups, rejecting null Hypothesis for the  Distance Travelled  value
"""


"""
Hilly Walk

Falls in SAC algorithm 50
Average Speed and Distance over 50 episodes: 0.43 m/s with total energy: 315.74 Joules per meter, travelled 4.87 meters

Falls in SAC-CPG algorithm 50
Average Speed and Distance over 50 episodes: 0.37 m/s with total energy: 227.97 Joules per meter, travelled 4.23 meters

Results for SAC-CPG-HH not found. Skipping.
The t-test statistic is 11.97887279465855 and the p-value is 6.653619336536159e-21  for the  Energy per Meter  value
P-value < 0.05, showing significant difference between groups, rejecting null Hypothesis for the  Energy per Meter  value
The t-test statistic is 2.2466347611302524 and the p-value is 0.0269073714233313  for the  Distance Travelled  value
P-value < 0.05, showing significant difference between groups, rejecting null Hypothesis for the  Distance Travelled  value
The t-test statistic is 2.639123151609461 and the p-value is 0.009670769318163384  for the  Rewards  value
P-value < 0.05, showing significant difference between groups, rejecting null Hypothesis for the  Rewards  value

"""

"""
Rough Walk 
Falls in SAC algorithm 50
Average Speed and Distance over 50 episodes: 0.31 m/s with total energy: 294.87 Joules per meter, travelled 3.75 meters


Falls in SAC-CPG algorithm 50
Average Speed and Distance over 50 episodes: 0.33 m/s with total energy: 202.87 Joules per meter, travelled 3.60 meters

Results for SAC-CPG-HH not found. Skipping.
The t-test statistic is 9.598539852758954 and the p-value is 9.003514578821967e-16  for the  Energy per Meter  value
P-value < 0.05, showing significant difference between groups, rejecting null Hypothesis for the  Energy per Meter  value
The t-test statistic is 1.1782776433022768 and the p-value is 0.24153912155560178  for the  Distance Travelled  value
P-value >= 0.05, Fail to reject null Hypothesis for the  Distance Travelled  value
The t-test statistic is -6.4556939738211385 and the p-value is 4.142541384826433e-09  for the  Rewards  value
P-value < 0.05, showing significant difference between groups, rejecting null Hypothesis for the  Rewards  value


"""