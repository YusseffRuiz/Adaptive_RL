import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from MPO_Algorithm.replay_buffer import ReplayBuffer
from MPO_Algorithm import MPOTrainer
from MatsuokaOscillator import MatsuokaNetworkWithNN, MatsuokaAgent
import torch.multiprocessing as mp

lock = mp.Lock()  # Lock to synchronize access to shared_best_reward


class MpoMatsuokaTrainer(MPOTrainer):
    def __init__(self, env, agent, n_envs=4, n_steps_per_update=2048, gamma=0.99, lam=0.95, device="cpu",
                 weights_path=None, log_dir="runs/", replay_buffer_capacity=100000, batch_size=64, shared_best_reward=None):
        super().__init__(env, agent, n_envs, n_steps_per_update, gamma, lam, device, weights_path, log_dir,
                         replay_buffer_capacity, batch_size)

        # Definition of the Matsuoka Parameters, the network is the one controlling the action space of the environment.
        self.input_values = self.envs.observation_space.shape[1]
        self.output_values = self.envs.action_space.shape[1]
        self.action_space = self.agent.action_space
        self.neuron_number = 2
        self.shared_best_reward = shared_best_reward
        self.shared_best_reward = -1000

        num_oscillators = 2  # One for each leg, then we can try using one per DoF.

        # Definition of the Matsuoka Network using the actor agent.
        self.matsuoka_network = MatsuokaNetworkWithNN(num_oscillators=num_oscillators,
                                                      observation_space=self.input_values,
                                                      action_dim=self.output_values, action_space=self.action_space,
                                                      n_envs=self.n_envs, neuron_number=self.neuron_number)

        self.param_dim = self.matsuoka_network.parameters_dimension

        # Redefinition of the ReplayBuffer
        self.replay_buffer = ReplayBuffer.from_matsuoka(replay_buffer_capacity, self.envs.observation_space.shape[1],
                                                        self.envs.action_space.shape[1], param_dim=self.param_dim)

        self.matsuoka_nn = MatsuokaAgent(self.input_values, 128, num_oscillators, self.neuron_number,
                                         self.output_values, self.action_space, device="cuda")

    def collect_trajectories(self):
        """
            Collects trajectories, now integrating the Matsuoka oscillator with the action selection process.
        """
        current_states = self.reset_envs()

        for _ in range(self.n_steps_per_update):
            state_tensors = torch.tensor(current_states, dtype=torch.float32, device=self.device)

            # Use the integrated Matsuoka oscillator to select actions
            with torch.no_grad():
                optimized_params, log_probs, entropies = self.matsuoka_nn.select_action(state_tensors)
                actions = self.matsuoka_network.step(optimized_params)
                values = self.agent.critic(state_tensors)

            # Perform the actions in all environments at once
            next_states, rewards, dones, *_ = self.envs.step(actions.detach().cpu().numpy())
            dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

            # Store the experiences for each environment
            for i in range(self.n_envs):
                self.replay_buffer.push(state_tensors[i], actions[i], rewards[i], dones[i],
                                        log_probs[i], values[i], entropies[i], optimized_params[i])

            current_states = next_states

    def train(self, n_updates=1000, save_interval=100, save=True):
        """
            The main training loop for the MPO agent.

            Args:
                n_updates (int): Number of updates (training iterations).
                save_interval (int): Number of steps between saving checkpoints.
                save (bool): Flag indicating whether to save checkpoints.
        """
        # Check and load existing weights if available
        weights_path = "weights" if self.weights_path is None else self.weights_path
        start_iteration = self.load_checkpoint(weights_path)

        for update in tqdm(range(start_iteration, n_updates)):

            # Collect trajectories from the environment 1st Step, gather state and actions based on the current policy.
            self.collect_trajectories()

            if len(self.replay_buffer) < self.batch_size:
                continue  # Skip training if the buffer isn't full enough

            # Sample a batch from the replay buffer
            state, action, reward, done, log_probs, values, entropy, optim_parameters = self.replay_buffer.sample(
                self.batch_size)

            # Compute returns and advantages
            # Utilizing GAE (Generalized Advantage Estimation) get advantages
            returns, advantages = self.compute_returns_and_advantages(reward, done, values)
            # Update the agent's policy E-Step
            # Compute a new policy minimizing KL divergence, while maximizing return
            self.matsuoka_nn.update_policy(state, optim_parameters, log_probs,
                                           values.mean(dim=0), values.std(dim=0))

            # Update the value network
            self.agent.update_value_network(state, returns)

            # Log data
            avg_reward = reward.mean()
            avg_entropy = entropy.mean()

            lock.acquire()
            if ((update + 1) % save_interval == 0) and save:
                if avg_reward > self.shared_best_reward:
                    self.shared_best_reward = avg_reward
                self.writer.add_scalar('Average Reward', avg_reward, update)
                self.writer.add_scalar('Average Entropy', avg_entropy, update)
                self.save_weights(update, n_updates, self.shared_best_reward, weights_path)
            lock.release()
