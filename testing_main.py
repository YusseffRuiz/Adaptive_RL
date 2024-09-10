import tonic.torch.agents

from stable_baselines3 import PPO, SAC

import gymnasium as gym
from reinforce import A2C

import os
import torch

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

    env_name = "Walker2d-v4"
    env_name, save_folder, log_dir = get_name_environment(env_name, cpg_flag=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    video_record = False
    algorithm_mpo = "mpo"
    algorithm_a2c = "a2c"
    algorithm_sac = "sac"
    algorithm = algorithm_mpo

    env = gym.make(env_name, render_mode="human", max_episode_steps=1000)

    if algorithm == "mpo":
        agent = tonic.torch.agents.MPO()
        agent.initialize(observation_space=env.observation_space, action_space=env.action_space)
        path_tmp = f"{env_name}/tonic_train_1/0/checkpoints/step_2325008"
        agent.load(path_tmp)
    elif algorithm == "sac":
        path_tmp = f"{save_folder}/logs/{env_name}/best_model.zip"
        path_final = f"{save_folder}/{env_name}-SAC-top"
        agent = SAC.load(path_tmp)
        print(f"model {save_folder} loaded")
    else:
        # agent hyperparams
        actor_lr = 0.001
        critic_lr = 0.005
        obs_shape = env.observation_space.shape[0]
        action_shape = env.action_space.shape[0]
        actor_weights_path = "weights_1/FinalWeights/actor_weights.h5"
        critic_weights_path = "weights_1/FinalWeights/critic_weights.h5"
        agent = A2C(obs_shape, action_shape, torch.device(device), critic_lr, actor_lr, n_envs=1)

        agent.actor.load_state_dict(torch.load(actor_weights_path, weights_only=True))
        agent.critic.load_state_dict(torch.load(critic_weights_path, weights_only=True))
        agent.actor.eval()
        agent.critic.eval()

    print("Loaded weights from {} algorithm".format(algorithm))

    if video_record:
        video_folder = "videos/" + env_name
        record_video(env_name, video_folder, algorithm, agent)
        print("Video Recorded")

    else:
        """ load network weights """
        evaluate(agent, env, algorithm)
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
        id="Walker2d-v4-CPG",
        # path to the class for creating the env
        # Note: entry_point also accept a class as input (and not only a string)
        entry_point="gymnasium.envs.mujoco:Walker2dCPGEnv",
        # Max number of steps per episode, using a `TimeLimitWrapper`
        max_episode_steps=1000,
    )


if __name__ == '__main__':
    register_new_env()
    main_running()


