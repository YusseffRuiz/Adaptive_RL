from myosuite.utils import gym

import Adaptive_RL
import Experiments.experiments_utils as trials
import warnings
import logging
import argparse


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
        num_episodes = 40
        env = gym.make(env_name, render_mode="rgb_array", max_episode_steps=1000)
    else:
        env = Adaptive_RL.Gym(env_name, render_mode="human")

    path = log_dir
    path, config = Adaptive_RL.get_last_checkpoint(path=path)

    if not random:
        agent = Adaptive_RL.load_agent(config, path, env)

        print("Loaded weights from {} algorithm, path: {}".format(algorithm, path))

        if video_record:
            video_folder = "videos/" + env_name
            Adaptive_RL.record_video(env_name, video_folder, algorithm, agent)
            print("Video Recorded")

        elif experiment:
            trials.evaluate_experiment(agent, env, algorithm, episodes_num=num_episodes, env_name=save_folder)

        else:
            """ load network weights """
            trials.evaluate(agent, env, algorithm, num_episodes)
    else:
        algorithm = "random"
        trials.evaluate(env=env, algorithm=algorithm, num_episodes=3, no_done=True, max_episode_steps=500)
    env.close()


if __name__ == '__main__':
    # plot.plot(paths="Walker2d-v4/logs/Walker2d-v4-DDPG", x_axis="train/seconds", x_label="Seconds", title=f"_training")
    main_running()

