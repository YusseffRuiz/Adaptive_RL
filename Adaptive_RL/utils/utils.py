import os
from Adaptive_RL.agents import SAC, PPO, MPO, DDPG
from Adaptive_RL.utils import logger
from gymnasium.envs.registration import register
import yaml
import argparse
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv


"""
TODO:
Modify video recorder 
"""

def get_last_checkpoint(path):
    arguments_path = os.path.join(path, 'config.yaml')
    path = os.path.join(path, 'checkpoints')
    # List all the checkpoints.
    checkpoint_ids = []
    if os.path.exists(path):
        for file in os.listdir(path):
            if file[:5] == 'step_':
                checkpoint_id = file.split('.')[0]
                checkpoint_ids.append(int(checkpoint_id[5:]))

        if checkpoint_ids:
            checkpoint_id = max(checkpoint_ids)
            checkpoint_path = os.path.join(path, f'step_{checkpoint_id}')
        else:
            checkpoint_path = None
            print('No checkpoint found')
    else:
        checkpoint_path = None
        print("No checkpoint Found")

    # Load the experiment configuration.
    if os.path.exists(arguments_path):
        with open(arguments_path, 'r') as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
        config = argparse.Namespace(**config)
    else:
        config = None
    return checkpoint_path, config


def load_checkpoint(checkpoint, path):
    logger.log(f'Trying to load experiment from {path}')

    # Use no checkpoint, the agent is freshly created.
    if checkpoint == 'none':
        logger.log('Not loading any weights')
        checkpoint_path = None
    else:
        checkpoint_path = os.path.join(path, 'checkpoints')
        if not os.path.isdir(checkpoint_path):
            logger.error(f'{checkpoint_path} is not a directory, starting from scratch')
            checkpoint_path = None
        else:
            # List all the checkpoints.
            checkpoint_ids = []
            for file in os.listdir(checkpoint_path):
                if file[:5] == 'step_':
                    checkpoint_id = file.split('.')[0]
                    checkpoint_ids.append(int(checkpoint_id[5:]))

            if checkpoint_ids:
                # Use the last checkpoint.
                if checkpoint == 'last':
                    checkpoint_id = max(checkpoint_ids)
                    checkpoint_path = os.path.join(
                        checkpoint_path, f'step_{checkpoint_id}')

                # Use the specified checkpoint.
                else:
                    checkpoint_id = int(checkpoint)
                    if checkpoint_id in checkpoint_ids:
                        checkpoint_path = os.path.join(
                            checkpoint_path, f'step_{checkpoint_id}')
                    else:
                        logger.error(f'Checkpoint {checkpoint_id} '
                                           f'not found in {checkpoint_path}')
                        checkpoint_path = None

            else:
                logger.error(f'No checkpoint found in {checkpoint_path}, starting from scratch')
                checkpoint_path = None
    return checkpoint_path


def load_agent(config, path, env):
    if config.agent["agent"] == "DDPG":
        agent = DDPG(learning_rate=config.agent["learning_rate"], batch_size=config.agent["batch_size"],
                     learning_starts=config.agent["learning_starts"], noise_std=config.agent["noise_std"],
                     hidden_layers=config.agent["hidden_layers"], hidden_size=config.agent["hidden_size"])
    elif config.agent["agent"] == "MPO":
        agent = MPO(lr_actor=config.agent["lr_actor"], lr_critic=config.agent["lr_critic"], lr_dual=config.agent["lr_dual"],
                    hidden_size=config.agent["neuron_number"], discount_factor=config.agent["gamma"],
                    replay_buffer_size=config.agent["replay_buffer_size"], hidden_layers=config.agent["layers_number"])
    elif config.agent["agent"] == "SAC":
        agent = SAC(learning_rate=config.agent["learning_rate"], hidden_size=config.agent["neuron_number"],
                    discount_factor=config.agent["gamma"], hidden_layers=config.agent["layers_number"],)
    elif config.agent["agent"] == "PPO":
        agent = PPO(learning_rate=config.agent["learning_rate"], hidden_size=config.agent["hidden_size"],
                    hidden_layers=config.agent["hidden_layers"], discount_factor=config.agent["discount_factor"],
                    batch_size=config.agent["batch_size"], entropy_coeff=config.agent["entropy_coeff"],
                    clip_range=config.agent["clip_range"], replay_buffer_size=config.agent["replay_buffer_size"])
    else:
        agent = None
    agent.initialize(observation_space=env.observation_space, action_space=env.action_space)
    agent.load(path)
    agent.get_config(print_conf=True)
    return agent

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


def register_new_env():

    register(
        # unique identifier for the env `name-version`
        id="Walker2d-v4-CPG",
        # path to the class for creating the env
        # Note: entry_point also accept a class as input (and not only a string)
        entry_point="gymnasium.envs.mujoco:Walker2dCPGEnv",
        # Max number of steps per episode, using a `TimeLimitWrapper`
        max_episode_steps=1000,
    )
    print("Registered new env Walker2d-v4-CPG")

    register(
        # unique identifier for the env `name-version`
        id="Ant-v4-CPG",
        # path to the class for creating the env
        # Note: entry_point also accept a class as input (and not only a string)
        entry_point="gymnasium.envs.mujoco:AntCPGEnv",
        # Max number of steps per episode, using a `TimeLimitWrapper`
        max_episode_steps=1000,
    )
    print("Registered new env Ant-v4-CPG")

    register(
        # unique identifier for the env `name-version`
        id="Humanoid-v4-CPG",
        # path to the class for creating the env
        # Note: entry_point also accept a class as input (and not only a string)
        entry_point="gymnasium.envs.mujoco:HumanoidEnvCPG",
        # Max number of steps per episode, using a `TimeLimitWrapper`
        max_episode_steps=1000,
    )
    print("Registered new env Humanoid-v4-CPG")
