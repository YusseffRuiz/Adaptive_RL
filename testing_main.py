import gymnasium as gym
from Adaptive_RL import SAC, MPO, DDPG, PPO
import Experiments.experiments_utils as trials
import warnings
import logging
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

from main import evaluate

logger: logging.Logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)


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
    num_episodes = 5

    # env_name = "Ant-v4"
    # env_name = "Walker2d-v4"
    env_name = "Humanoid-v4"

    video_record = False
    experiment = False
    cpg_flag = True
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

    if algorithm == "MPO":
        # agent = tonic.torch.agents.MPO()  # For walker2d no CPG
        agent = MPO(hidden_size=1024)
        agent.initialize(observation_space=env.observation_space, action_space=env.action_space)
        path_walker2d = f"{env_name}/tonic_train/0/checkpoints/step_4675008"
        path_walker2d_cpg = f"{env_name}/tonic_train/0/checkpoints/step_4125000"
        # path_walker2d_cpg = f"{env_name}/logs/{save_folder}/checkpoints/step_5000000.pt"
        path_ant2d_cpg = f"{env_name}/logs/{save_folder}/checkpoints/step_5000000.pt"
        path_humanoid_cpg = f"{log_dir}/checkpoints/step_10000000.pt"
        path_temp_cpg = "Walker2d-v4-CPG/tonic_train/0/checkpoints/step_4225008"
        if cpg_flag:
            path_chosen = path_temp_cpg
        else:
            path_chosen = path_walker2d
        agent.load(path_chosen)
    elif algorithm == "SAC":
        if cpg_flag:
            path_tmp = f"{env_name}/logs/{save_folder}/best_model"
        else:
            path_tmp = "Walker2d-v4-SAC-1/logs/Walker2d-v4/best_model"
        agent = SAC(hidden_size=256)
        agent.initialize(observation_space=env.observation_space, action_space=env.action_space)
        agent.load(path_tmp)
        print(f"model {save_folder} loaded")
    if algorithm == "PPO":
        lr_actor = 3e-4
        lr_critic = 1e-4
        gamma = 0.99
        neuron_number = 1024
        path_humanoid_cpg = "Humanoid-v4-CPG/logs/Humanoid-v4-CPG-PPO/checkpoints/step_10000.pt"
        agent = PPO(lr_actor=lr_actor, lr_critic=lr_critic, hidden_size=neuron_number, discount_factor=gamma)
        agent.initialize(observation_space=env.observation_space, action_space=env.action_space)
        agent.load(path_humanoid_cpg)
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


if __name__ == '__main__':
    register_new_env()
    main_running()
    # test_get_values()

