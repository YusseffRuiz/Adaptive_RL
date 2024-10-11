from myosuite.utils import gym

import Adaptive_RL
import Experiments.experiments_utils as trials
import warnings
import logging


logger: logging.Logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)

def main_running():
    """ play a couple of showcase episodes """
    num_episodes = 5

    # env_name = "Ant-v4"
    env_walker = "Walker2d-v4"
    env_mujoco = "Humanoid-v4"
    env_myo = "myoLegWalk-v0"
    env_name = env_mujoco

    video_record = False
    experiment = False
    cpg_flag = True
    random = False
    algorithm_mpo = "MPO"
    algorithm_sac = "SAC"
    algorithm_ppo = "PPO"
    algorithm_ddpg = "DDPG"
    algorithm = algorithm_ppo

    env_name, save_folder, log_dir = trials.get_name_environment(env_name, cpg_flag=cpg_flag, algorithm=algorithm, experiment_number=0)

    if experiment:
        num_episodes = 40
        env = gym.make(env_name, render_mode="rgb_array", max_episode_steps=1000)
    else:
        env = Adaptive_RL.Gym(env_name, render_mode="human")

    path = f"{env_name}/logs/{save_folder}/"
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

