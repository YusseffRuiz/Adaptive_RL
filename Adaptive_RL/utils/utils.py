import os
from Adaptive_RL.agents import SAC, PPO, MPO, DDPG
from Adaptive_RL import CPGWrapper, MyoSuite, Gym
from MatsuokaOscillator import MatsuokaNetworkWithNN
from Adaptive_RL.utils import logger
from Adaptive_RL.dep_search.dep_agents import dep_factory
from gymnasium.envs.registration import register
import yaml
import argparse
from gymnasium.wrappers import RecordVideo
import torch
from moviepy.editor import VideoFileClip, concatenate_videoclips


def get_last_checkpoint(path, best=True):
    arguments_path = os.path.join(path, 'config.yaml')
    path = os.path.join(path, 'checkpoints')
    checkpoint_folder = path
    # List all the checkpoints.
    checkpoint_ids = []
    if os.path.exists(path):
        if best:
            checkpoint_path = os.path.join(path, "best_model")
        else:
            for file in os.listdir(path):
                if file[:5] == 'step_':
                    checkpoint_id = file.split('.')[0]
                    checkpoint_ids.append(int(checkpoint_id[5:]))

            if checkpoint_ids:
                checkpoint_id = max(checkpoint_ids)
                checkpoint_path = os.path.join(path, f'step_{checkpoint_id}')
            else:
                checkpoint_path = None
    else:
        checkpoint_path = None

    # Load the experiment configuration.
    if os.path.exists(arguments_path):
        with open(arguments_path, 'r') as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
        config = argparse.Namespace(**config)
        print(f"Load from {arguments_path}")
    else:
        config = None
    if checkpoint_path is None:
        logger.error("No checkpoints found")
    return checkpoint_path, config, checkpoint_folder


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


def load_agent(config, path, env, muscle_flag=False):
    if config.agent["agent"] == "DDPG":
        agent = DDPG(learning_rate=config.agent["learning_rate"], batch_size=config.agent["batch_size"],
                     learning_starts=config.agent["learning_starts"], noise_std=config.agent["noise_std"],
                     hidden_layers=config.agent["hidden_layers"], hidden_size=config.agent["hidden_size"],
                     replay_buffer_size=config.agent["replay_buffer_size"], )
    elif config.agent["agent"] == "MPO":
        agent = MPO(lr_actor=config.agent["lr_actor"], lr_critic=config.agent["lr_critic"],
                    lr_dual=config.agent["lr_dual"],
                    hidden_size=config.agent["hidden_size"], discount_factor=config.agent["gamma"],
                    replay_buffer_size=config.agent["replay_buffer_size"], hidden_layers=config.agent["hidden_layers"])
    elif config.agent["agent"] == "SAC":
        agent = SAC(learning_rate=config.agent["learning_rate"], batch_size=config.agent["batch_size"],
                    learning_starts=config.agent["learning_starts"], noise_std=config.agent["noise_std"],
                    hidden_layers=config.agent["hidden_layers"], hidden_size=config.agent["hidden_size"],
                    replay_buffer_size=config.agent["replay_buffer_size"],
                    discount_factor=config.agent["discount_factor"], )
    elif config.agent["agent"] == "PPO":
        agent = PPO(learning_rate=config.agent["learning_rate"], hidden_size=config.agent["hidden_size"],
                    hidden_layers=config.agent["hidden_layers"], discount_factor=config.agent["discount_factor"],
                    batch_size=config.agent["batch_size"], entropy_coeff=config.agent["entropy_coeff"],
                    clip_range=config.agent["clip_range"], replay_buffer_size=config.agent["replay_buffer_size"],
                    normalizer=config.agent["normalizer"], decay_lr=config.agent["discount_factor"])
    else:
        agent = None
    if muscle_flag:  #Modify if using MPO
        agent = dep_factory(3, agent)(learning_rate=config.agent["learning_rate"], batch_size=config.agent["batch_size"],
                    learning_starts=config.agent["learning_starts"], noise_std=config.agent["noise_std"],
                    hidden_layers=config.agent["hidden_layers"], hidden_size=config.agent["hidden_size"],
                    replay_buffer_size=config.agent["replay_buffer_size"],
                    discount_factor=config.agent["discount_factor"], )
    agent.initialize(observation_space=env.observation_space, action_space=env.action_space)
    step = agent.load(path)
    agent.get_config(print_conf=True)
    return agent, step


def wrap_cpg(env, env_name, cpg_oscillators=2, cpg_neurons=2, cpg_tau_r=2, cpg_tau_a=12, hh=False):
    amplitude = max(env.action_space.high)
    if 'myo' in env_name:
        min_value = 0
    else:
        min_value = min(env.action_space.low)
    cpg_model = MatsuokaNetworkWithNN(num_oscillators=cpg_oscillators,
                                      da=env.action_space.shape[0],
                                      neuron_number=cpg_neurons, tau_r=cpg_tau_r,
                                      tau_a=cpg_tau_a, hh=hh, max_value=amplitude, min_value=min_value)
    env = CPGWrapper(env, cpg_model=cpg_model, use_cpg=True)
    return env

def get_cpg_model(env_name, cpg_oscillators=2, cpg_neurons=2, cpg_tau_r=2, cpg_tau_a=12, hh=False):
    if 'myo' in env_name:
        dummy_env = MyoSuite(env_name, reset_type='random', scaled_actions=False)
        min_value = 0
    else:
        dummy_env = Gym(env_name)
        min_value = min(dummy_env.action_space.low)
    amplitude = max(dummy_env.action_space.high)
    cpg_model = MatsuokaNetworkWithNN(num_oscillators=cpg_oscillators,
                                      da=dummy_env.action_space.shape[0],
                                      neuron_number=cpg_neurons, tau_r=cpg_tau_r,
                                      tau_a=cpg_tau_a, hh=hh, max_value=amplitude, min_value=min_value)
    del dummy_env
    return cpg_model


def record_video(env, video_folder, alg, agent, env_name):
    video_length = 1500
    # Record the video starting at the first step
    env.reset()
    env_v = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: x % 2 == 0,
                        video_length=0, name_prefix=f"{alg}-agent-{env_name}")

    obs, *_ = env_v.reset()
    env_v.start_video_recorder()

    for _ in range(video_length):
        if alg != "random":
            action = agent.test_step(obs)
        else:
            action = env.action_space.sample()

        # Take a step in the environment and record it
        obs, reward, terminated, truncated, *_ = env_v.step(action)
        env_v.render()

        # If episode ends, reset but keep recording
        if terminated or truncated:
            obs, *_ = env_v.reset()
            terminated = False
            truncated = False

    # Save the video
    env_v.close_video_recorder()
    env_v.close()

    output_path = os.path.join(video_folder, f"{alg}-agent-{env_name}.mp4")
    # List all video files in the folder and sort by name or creation time if needed
    video_files = sorted([os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith(".mp4")])
    extra_files = sorted([os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith(".json")])
    print(video_files)
    # Load each video file with moviepy
    clips = [VideoFileClip(f) for f in video_files]
    # Concatenate all video clips
    final_clip = concatenate_videoclips(clips, method="compose")

    # Write the output to a single file
    final_clip.write_videofile(output_path, codec="libx264", fps=24)

    # Close each clip to free memory
    for clip in clips:
        clip.close()

    # Delete the individual video files after concatenation
    for video_file in video_files:
        os.remove(video_file)
    for extra_file in extra_files:
        os.remove(extra_file)


def to_tensor(data):
    """
    Check if the input data is already a tensor. If not, convert it to a tensor.
    Parameters:
        - data: The input data, which may or may not be a tensor.
    Returns:
        - Tensor: A PyTorch tensor, either the original data if it was a tensor or the converted tensor.
    """
    if torch.is_tensor(data):
        return data
    return torch.as_tensor(data, dtype=torch.float32)


def file_to_hyperparameters(file_path, env, algorithm):
    """
    Loads the hyperparameters from a YAML file.
    """
    if not os.path.isfile(file_path):
        logger.error("Failed to find file in {}".format(file_path))
        return None
    with open(file_path, 'r') as file:
        hyperparameters = yaml.safe_load(file)
    if 'cpg' in hyperparameters[env]:
        cpg = hyperparameters[env]['cpg']
    else:
        cpg = None
    if algorithm in hyperparameters[env]:
        hyperparams = hyperparameters[env][algorithm]
    else:
        hyperparams = None
        logger.error(f"Hyperparameters not found in {file_path}")
    return hyperparams, cpg


def register_new_env():
    """
    Used when we wanted to add new environments
    """
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
