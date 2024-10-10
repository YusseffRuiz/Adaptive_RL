from myosuite.utils import gym

import Adaptive_RL
from Adaptive_RL import SAC, MPO, DDPG, PPO, plot
import Experiments.experiments_utils as trials
import warnings
import logging
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from rl_zoo3 import ALGOS


logger: logging.Logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)

def main_running():
    """ play a couple of showcase episodes """
    num_episodes = 5

    # env_name = "Ant-v4"
    env_walker = "Walker2d-v4"
    env_mujoco = "Humanoid-v4"
    env_myo = "myoLegWalk-v0"
    env_name = env_walker

    video_record = False
    experiment = False
    cpg_flag = False
    random = False
    algorithm_mpo = "MPO"
    algorithm_sac = "SAC"
    algorithm_ppo = "PPO"
    algorithm_ddpg = "DDPG"
    algorithm = algorithm_ddpg

    env_name, save_folder, log_dir = trials.get_name_environment(env_name, cpg_flag=cpg_flag, algorithm=algorithm, experiment_number=0)

    if experiment:
        num_episodes = 40
        env = gym.make(env_name, render_mode="rgb_array", max_episode_steps=1000)
    else:
        env = Adaptive_RL.Gym(env_name, render_mode="human")

    path = f"{env_name}/logs/{save_folder}/"
    path, config = Adaptive_RL.get_last_checkpoint(path=path)

    if not random:
        agent = load_agent(config, path)
        agent.initialize(observation_space=env.observation_space, action_space=env.action_space)

        print("Loaded weights from {} algorithm, path: {}".format(algorithm, path))

        if video_record:
            video_folder = "videos/" + env_name
            record_video(env_name, video_folder, algorithm, agent)
            print("Video Recorded")

        elif experiment:
            trials.evaluate_experiment(agent, env, algorithm, episodes_num=num_episodes, env_name=save_folder)

        else:
            """ load network weights """
            # agent.model.eval()
            print(agent.get_config())
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

def load_agent(config, path):
    if config.agent["agent"] == "DDPG":
        agent = DDPG(learning_rate=config.agent["learning_rate"], batch_size=config.agent["batch_size"],
                     learning_starts=config.agent["learning_starts"], noise_std=config.agent["noise_std"],
                     hidden_layers=config.agent["hidden_layers"], hidden_size=config.agent["hidden_size"])
    elif config.agent["agent"] == "MPO":
        agent = MPO(lr_actor=config.agent["lr_actor"], lr_critic=config.agent["lr_critic"], lr_dual=config.agent["lr_dual"],
                    hidden_size=config.agent["neuron_number"], discount_factor=config.agent["gamma"],
                    replay_buffer_size=config.agent["replay_buffer_size"], hidden_layers=config.agent["layers_number"])
    elif config.agent["agent"] == "SAC":
        agent = SAC(lr_actor=config.agent["lr_actor"], lr_critic=config.agent["lr_critic"], hidden_size=config.agent["neuron_number"],
                    discount_factor=config.agent["gamma"], hidden_layers=config.agent["layers_number"],)
    elif config.agent["agent"] == "PPO":
        agent = PPO(lr_actor=config.agent["lr_actor"], lr_critic=config.agent["lr_critic"], hidden_size=config.agent["neuron_number"],
                    hidden_layers=config.agent["layers_number"], discount_factor=config.agent["gamma"],
                    batch_size=config.agent["batch_size"], entropy_coeff=config.agent["ent_coeff"], clip_range=config.agent["clip_range"])
    else:
        agent = None

    agent.load(path)
    return agent


if __name__ == '__main__':
    # plot.plot(paths="Walker2d-v4/logs/Walker2d-v4-DDPG", x_axis="train/seconds", x_label="Seconds", title=f"_training")
    main_running()

