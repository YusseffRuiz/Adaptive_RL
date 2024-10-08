from myosuite.utils import gym

import Adaptive_RL
from Adaptive_RL import SAC, MPO, DDPG, PPO
import Experiments.experiments_utils as trials
import warnings
import logging
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv


logger: logging.Logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)


def main_running():
    """ play a couple of showcase episodes """
    num_episodes = 5

    # env_name = "Ant-v4"
    # env_name = "Walker2d-v4"
    env_mujoco = "Humanoid-v4"
    env_myo = "myoLegWalk-v0"
    env_name = env_mujoco

    video_record = False
    experiment = False
    cpg_flag = True
    random = False
    algorithm_mpo = "MPO"
    algorithm_a2c = "A2C"
    algorithm_sac = "SAC"
    algorithm_ppo = "PPO"
    algorithm = algorithm_ppo

    env_name, save_folder, log_dir = trials.get_name_environment(env_name, cpg_flag=cpg_flag, algorithm=algorithm, experiment_number=0)

    if experiment:
        num_episodes = 40
        env = gym.make(env_name, render_mode="rgb_array", max_episode_steps=1000)
    else:
        env = gym.make(env_name, render_mode="human", max_episode_steps=1000)

    path = f"{env_name}/logs/{save_folder}/"
    path = Adaptive_RL.get_last_checkpoint(path=path)

    if not random:
        if algorithm == "MPO":
            # agent = tonic.torch.agents.MPO()  # For walker2d no CPG
            agent = MPO(hidden_size=256)
            agent.initialize(observation_space=env.observation_space, action_space=env.action_space)
            # path_walker2d = f"{env_name}/tonic_train/0/checkpoints/step_4675008"
            path_walker2d_cpg = f"{env_name}/tonic_train/0/checkpoints/step_4125000"
            # path_walker2d_cpg = f"{env_name}/logs/{save_folder}/checkpoints/step_5000000.pt"
            path_ant2d_cpg = f"{env_name}/logs/{save_folder}/checkpoints/step_5000000.pt"
            path_humanoid_cpg = f"{log_dir}/checkpoints/step_10000000.pt"
            # path_temp_cpg = "Walker2d-v4-CPG/tonic_train/0/checkpoints/step_4225008"
            # if cpg_flag:
            #     path_chosen = path_temp_cpg
            # else:
            #     path_chosen = path_walker2d
            agent.load(path)
        elif algorithm == "SAC":
            agent = SAC(hidden_size=1024)
            agent.initialize(observation_space=env.observation_space, action_space=env.action_space)
            agent.load(path)
            print(f"model {save_folder} loaded")
        elif algorithm == "PPO":
            agent = PPO(hidden_size=1024, hidden_layers=2)
            agent.initialize(observation_space=env.observation_space, action_space=env.action_space)
            agent.load(path)
        else:
            agent = None

        print("Loaded weights from {} algorithm".format(algorithm))

        if video_record:
            video_folder = "videos/" + env_name
            record_video(env_name, video_folder, algorithm, agent)
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

def record_video(env_name, video_folder, alg, agent):
    video_length = 1000
    vec_env = DummyVecEnv([lambda: gym.make(env_name, render_mode="rgb_array", max_episode_steps=1000)])

    obs = vec_env.reset()
    # Record the video starting at the first step
    vec_env = VecVideoRecorder(vec_env, video_folder,
                               record_video_trigger=lambda x: x == 0, video_length=video_length,
                               name_prefix=f"{alg}-agent-{env_name}")
    vec_env.reset()
    for _ in range(video_length + 1):
        if alg == "mpo":
            action = agent.test_step(obs)
        elif alg == "sac":
            action, *_ = [agent.predict(obs, deterministic=True)]
            action = action[0]
        else:
            action, *_ = [agent.select_action((obs[None, :]))]
            action = action.cpu().numpy()[0]
        obs, _, _, _ = vec_env.step(action)
    # Save the video
    vec_env.close()


if __name__ == '__main__':
    main_running()

