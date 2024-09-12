from stable_baselines3 import PPO, SAC
import gymnasium as gym
import torch
import MPO_Algorithm
import Experiments.experiments_utils as trials
import tonic

import logging
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

from main import get_name_environment, evaluate

logger: logging.Logger = logging.getLogger(__name__)


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


def main_running():
    """ play a couple of showcase episodes """
    num_episodes = 3

    env_name = "Ant-v4"
    env_name, save_folder, log_dir = get_name_environment(env_name, cpg_flag=True, algorithm="MPO")

    video_record = False
    algorithm_mpo = "mpo"
    algorithm_a2c = "a2c"
    algorithm_sac = "sac"
    algorithm = algorithm_mpo

    env = gym.make(env_name, render_mode="human", max_episode_steps=1000)

    if algorithm == "mpo":
        #  agent = tonic.torch.agents.MPO()  # For walker2d no CPG
        agent = MPO_Algorithm.agents.MPO(lr_actor=3.53e-5, lr_critic=6.081e-5, lr_dual=0.00213, hidden_size=512)
        agent.initialize(observation_space=env.observation_space, action_space=env.action_space)
        path_walker2d = f"{env_name}/tonic_train/0/checkpoints/step_4675008"
        path_walker2d = f"{env_name}/tonic_train/0/checkpoints/step_4675008"
        path_ant2d_cpg = f"{env_name}/logs/{save_folder}/checkpoints/step_1400000.pt"
        agent.load(path_ant2d_cpg)
    elif algorithm == "sac":
        path_tmp = f"{save_folder}/logs/{env_name}/best_model.zip"
        path_final = f"{save_folder}/{env_name}-SAC-top"
        agent = SAC.load(path_tmp)
        print(f"model {save_folder} loaded")
    else:
        agent = None

    print("Loaded weights from {} algorithm".format(algorithm))

    if video_record:
        video_folder = "videos/" + env_name
        record_video(env_name, video_folder, algorithm, agent)
        print("Video Recorded")

    else:
        """ load network weights """
        evaluate(agent, env, algorithm, num_episodes)
    env.close()


def register_new_env():
    register(
        # unique identifier for the env `name-version`
        id="Hopper-CPG",
        # path to the class for creating the env
        # Note: entry_point also accept a class as input (and not only a string)
        entry_point="gymnasium.envs.mujoco:HopperCPG",
        # Max number of steps per episode, using a `TimeLimitWrapper`
        max_episode_steps=1000,
    )

    register(
        # unique identifier for the env `name-version`
        id="Walker2d-v4-CPG-MPO",
        # path to the class for creating the env
        # Note: entry_point also accept a class as input (and not only a string)
        entry_point="gymnasium.envs.mujoco:Walker2dCPGEnv",
        # Max number of steps per episode, using a `TimeLimitWrapper`
        max_episode_steps=1000,
    )


def evaluate_experiment(agent, env, alg):
    import numpy as np

    total_rewards = []
    range_episodes = 1500
    obs, *_ = env.reset()
    for i in range(range_episodes):
        with torch.no_grad():
            if alg == "mpo":
                action = agent.test_step(obs, _)
            elif alg == "sac":
                action, *_ = agent.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
        obs, reward, done, *_ = env.step(action)
        if i == 1000:
            vel_env_1 = trials.get_velocity(obs)
            print(vel_env_1, " m/s")

        total_rewards.append(reward)
    average_reward = np.mean(total_rewards)
    print(f"Average Reward over {range_episodes} episodes: {average_reward}")


def test_get_values():
    env_name = "Walker2d-v4"
    env_name, save_folder, log_dir = get_name_environment(env_name, cpg_flag=False)
    env = gym.make(env_name, render_mode="human", max_episode_steps=1000, autoreset=False)
    agent = MPO_Algorithm.agents.MPO()
    agent.initialize(observation_space=env.observation_space, action_space=env.action_space)
    path = f"{env_name}/tonic_train/0/checkpoints/step_4675008.pt"
    agent.load(path)
    evaluate_experiment(agent, env, "mpo")
    env.close()


if __name__ == '__main__':
    register_new_env()
    main_running()
    # test_get_values()

