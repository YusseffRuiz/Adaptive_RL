"""

This file implements the Maximum a Posteriori Policy Optimization (MPO) algorithm for continuous action spaces
in reinforcement learning tasks.
Based on MPO algorithm: https://arxiv.org/pdf/1806.06920v1 and https://arxiv.org/pdf/1812.02256
The file is designed to facilitate efficient and stable training of agents in continuous control environments.

Owner: Adan Yusseff Domínguez Ruiz
Institution: Instituto Tecnológico de Monterrey

This implementation supports saving and loading model checkpoints, including optimizer states, for
resumable training. It also integrates TensorBoard for monitoring key training metrics.

It still requires implementation for discrete control environments and multi-GPU training

"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .replay_buffer import ReplayBuffer
from MatsuokaOscillator import MatsuokaOscillator, MatsuokaNetworkWithNN, MatsuokaNetwork, NeuralNetwork

from torch.nn.parallel import DistributedDataParallel as DDP


class MPOAgent:
    def __init__(self, state_dim, action_dim, action_space, hidden_dim=128, kl_epsilon=0.01, actor_lr=3e-4,
                 critic_lr=3e-4, device=torch.device("cpu")):
        """
        Initializes the MPOAgent with the policy and value networks, optimizers, and other necessary parameters.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space (continuous).
            action_space (gym.spaces.Continuous): Continuous action space.
            hidden_dim (int): Number of hidden units in each layer of the networks.
            kl_epsilon (float): KL-divergence constraint threshold.
            actor_lr and critic_lr (float): Learning rate for the optimizers.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.device = device
        print(f"\nUsing {device} device")
        self.kl_epsilon = kl_epsilon
        self.action_space = action_space

        # Define the policy network (actor)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Batch Normalization
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Layer Normalization
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # Outputs mean and log_std for each action dimension
        ).to(self.device)

        # Define the value network (critic) with similar architecture
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Batch Normalization
            nn.SiLU(),  # Advanced Activation Function
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Layer Normalization
            nn.SiLU(),  # Advanced Activation Function
            nn.Linear(hidden_dim, 1)
        ).to(self.device)

        # Optimizers for both networks
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=critic_lr)

        # Define Matsuoka NN
        num_oscillators = 4
        output_size = 3  # tau_r, weights, and beta for each oscillator

        nn_mat_model = NeuralNetwork(input_size=num_oscillators, hidden_size=hidden_dim, output_size=output_size, device=self.device)
        self.matsuoka_network = MatsuokaNetworkWithNN(num_oscillators, self.actor, neuron_number=2)

    def select_action(self, state):
        """
        Selects an action based on the current policy.

        Args:
            state (torch.Tensor): The current state.

        Returns:
            action (torch.Tensor): The selected continuous action.
            log_prob (torch.Tensor): Log probability of the selected action.
        """
        params = self.actor(state)

        # Split the output into mean and log_std for each action dimension
        mean, log_std = params[:, :params.shape[1] // 2], params[:, params.shape[1] // 2:]
        if torch.isnan(log_std).any():
            log_std = torch.zeros_like(log_std)
        if torch.isnan(mean).any():
            mean = torch.zeros_like(mean)
        std = torch.exp(log_std)

        # Create a normal distribution and sample a continuous action
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()

        # Clamp the actions to the valid range
        action = torch.clamp(action, self.action_space.low[0], self.action_space.high[0])
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy

    @staticmethod
    def compute_kl_divergence(old_mean, old_log_std, new_mean, new_log_std):
        """
        Computes the KL divergence between the old and new policy distributions.

        Args:
            old_mean (torch.Tensor): Mean of the old policy distribution.
            old_log_std (torch.Tensor): Log std of the old policy distribution.
            new_mean (torch.Tensor): Mean of the new policy distribution.
            new_log_std (torch.Tensor): Log std of the new policy distribution.

        Returns:
            kl_divergence (torch.Tensor): KL divergence between the old and new policies.
        """
        old_std = torch.exp(old_log_std)
        new_std = torch.exp(new_log_std)
        kl_div = torch.log(new_std / old_std) + (old_std ** 2 + (old_mean - new_mean) ** 2) / (2 * new_std ** 2) - 0.5
        return kl_div.sum(dim=-1)

    def update_policy(self, states, actions, old_log_probs, old_mean, old_log_std):
        """
        Updates the policy network using the MPO algorithm.

        Args:
            states (torch.Tensor): Batch of states.
            actions (torch.Tensor): Batch of continuous actions taken.
            old_log_probs (torch.Tensor): Log probabilities of actions under the old policy.
            old_mean (torch.Tensor): Means of the old policy.
            old_log_std (torch.Tensor): Log standard deviations of the old policy.
        """

        # Forward pass through policy network to get new mean and log_std
        # E-Step = compute policy
        params = self.actor(states)
        new_mean, new_log_std = params[:, :params.shape[1] // 2], params[:, params.shape[1] // 2:]

        # Calculate log probabilities under new policy
        if torch.isnan(new_mean).any():
            new_mean = torch.zeros_like(new_mean)
            new_log_std = torch.zeros_like(new_log_std)
        new_std = torch.exp(new_log_std)
        dist = torch.distributions.Normal(new_mean, new_std)
        new_log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)

        # Compute the KL divergence
        kl_div = self.compute_kl_divergence(old_mean=old_mean, old_log_std=old_log_std, new_mean=new_mean,
                                            new_log_std=new_log_std)

        # Ensure KL divergence is within the allowable threshold
        kl_penalty = torch.clamp(kl_div - self.kl_epsilon, min=0).mean()

        # Compute loss (M-step)
        old_log_probs.requires_grad_(True)
        policy_loss = (new_log_probs - old_log_probs.view(-1)).mean() + kl_penalty - 0.001 * entropy.mean()
        # Update the policy network
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

    def compute_value_loss(self, states, returns):
        """
        Computes the loss for the value network.

        Args:
            states (torch.Tensor): Batch of states.
            returns (torch.Tensor): Computed returns from the environment.

        Returns:
            value_loss (torch.Tensor): The loss for the value network.
        """
        value_preds = self.critic(states)
        value_loss = nn.MSELoss()(value_preds, returns)
        return value_loss

    def update_value_network(self, states, returns):
        """
        Updates the value network based on the value loss.

        Args:
            states (torch.Tensor): Batch of states.
            returns (torch.Tensor): Computed returns from the environment.
        """
        value_loss = self.compute_value_loss(states, returns)
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

    def load_checkpoint(self, path=None):
        """
        Loads the weights at the specified path, if available.
        :param path: path of the weights to be loaded.
        :return: 0 if no weights were loaded, 1 otherwise.
        """
        checkpoint_files = [f for f in os.listdir(path) if f.endswith('MPO.h5')]  # If found return True
        if not checkpoint_files:
            print(f"\nNo checkpoint found at {path}, we can't load a model")
            return 0
        actor_weights_path = "actor_weights_MPO.h5"
        critic_weights_path = "critic_weights_MPO.h5"
        path_actor = os.path.join(path, actor_weights_path)
        path_critic = os.path.join(path, critic_weights_path)
        checkpoint_actor = torch.load(path_actor, map_location=self.device, weights_only=True)
        checkpoint_critic = torch.load(path_critic, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(checkpoint_actor)
        self.critic.load_state_dict(checkpoint_critic)
        self.actor.eval()
        self.critic.eval()
        print(f"\nLoaded weights at {path_actor} and {path_critic}, using checkpoint.\n")


class MPOTrainer:
    def __init__(self, env, agent, n_envs=4, n_steps_per_update=2048, gamma=0.99, lam=0.95, device="cpu",
                 weights_path=None, log_dir="runs/", replay_buffer_capacity=100000, batch_size=64):
        """
        Initializes the MPOTrainer with the environment and agent.

        Args:
            env (str): Name of the Gym environment.
            agent (MPOAgent): The MPO agent to be trained.
            n_envs (int): Number of parallel environments.
            n_steps_per_update (int): Number of steps to collect before each policy update.
            gamma (float): Discount factor for rewards.
            lam (float): GAE (Generalized Advantage Estimation) parameter.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        print(f"\nUsing {device} device, and working with {n_envs} environments")
        self.envs = gym.vector.AsyncVectorEnv([
            lambda: gym.make(env) for _ in range(n_envs)
        ])
        self.n_envs = n_envs
        self.n_steps_per_update = n_steps_per_update
        self.gamma = gamma
        self.lam = lam

        # Initialize environment and agent
        torch.set_num_threads(n_envs)
        self.agent = agent
        self.agent.actor = (self.agent.actor.to(self.device))
        self.agent.critic = (self.agent.critic.to(self.device))
        self.weights_path = weights_path

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)

        # Replay Buffer implementation
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity, self.envs.observation_space.shape[1],
                                          self.envs.action_space.shape[1])
        self.batch_size = batch_size

    def reset_envs(self):
        states, *_ = self.envs.reset()
        return states

    def collect_trajectories(self):
        """
        Collects trajectories from the environment using the current policy.

        Returns:
            states (torch.Tensor): Collected states.
            actions (torch.Tensor): Collected actions.
            rewards (torch.Tensor): Collected rewards.
            dones (torch.Tensor): Boolean flags indicating episode termination.
            log_probs (torch.Tensor): Log probabilities of actions taken.
            values (torch.Tensor): Value predictions for the collected states.
            Total number of steps collected (for later sampling).
        """

        current_states = self.reset_envs()

        for _ in range(self.n_steps_per_update):
            state_tensors = torch.tensor(current_states, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                actions, log_probs, entropies = self.agent.select_action(state_tensors)
                values = self.agent.critic(state_tensors)
                # Perform the actions in all environments at once
            next_states, rewards, dones, *_ = self.envs.step(actions.cpu().numpy())

            dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

            # Store the experiences for each environment
            for i in range(self.n_envs):
                self.replay_buffer.push(state_tensors[i], actions[i], rewards[i], dones[i],
                                        log_probs[i], values[i], entropies[i])

            current_states = next_states

    def compute_returns_and_advantages(self, rewards, dones, values):
        """
        Computes the returns and advantages for each step in the trajectory.

        Args:
            rewards (torch.Tensor): Rewards collected during the trajectory.
            dones (torch.Tensor): Boolean flags indicating episode termination.
            values (torch.Tensor): Value predictions for the states.

        Returns:
            returns (torch.Tensor): Computed returns.
            advantages (torch.Tensor): Computed advantages.
        """
        returns = torch.zeros((values.shape[0], values.shape[1]), dtype=torch.float32, device=self.device)
        advantages = torch.zeros((values.shape[0], values.shape[1]), dtype=torch.float32, device=self.device)

        gae = 0
        next_value = 0

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            next_value = values[step]

            returns[step] = gae + values[step]
            advantages[step] = gae
        return returns, advantages

    def save_weights(self, it, tot_it, avg_reward, path=None):
        """
        Saves the weights at the specified path.
        :param tot_it: total iterations
        :param it: iteration number to save the number of the weights
        :param avg_reward: reward gathered at that iteration
        :param path: path to be saved, if none exist, it will be created.
        """
        if not os.path.exists(path):
            os.mkdir(path)
        try:
            torch.save({
                'actor_net': self.agent.actor.module.state_dict(),
                'critic_net': self.agent.critic.module.state_dict(),
                'actor_optimizer': self.agent.actor_optimizer.state_dict(),
                'critic_optimizer': self.agent.critic_optimizer.state_dict(),
            }, f'{path}/mpo_agent_update_{it + 1}.pth')
        except AttributeError:
            torch.save({
                'actor_net': self.agent.actor.state_dict(),
                'critic_net': self.agent.critic.state_dict(),
                'actor_optimizer': self.agent.actor_optimizer.state_dict(),
                'critic_optimizer': self.agent.critic_optimizer.state_dict(),
            }, f'{path}/mpo_agent_update_{it + 1}.pth')

        print(f"\nSaved model at update {it + 1}")
        print(f"Update {it + 1}/{tot_it} completed with avg reward: {avg_reward}")

    def load_checkpoint(self, path=None):
        """
        Loads the weights at the specified path, if available.
        :param path: path of the weights to be loaded.
        :return: 0 if no weights were loaded, 1 otherwise.
        """
        if not os.path.exists(path):
            print(f"\nNo checkpoint found at {path}, starting from scratch.")
            return 0
        checkpoint_files = [f for f in os.listdir(path) if f.endswith('.pth')]  # If found return True
        latest_checkpoint = max(checkpoint_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
        iteration = int(latest_checkpoint.split('_')[-1].split('.')[0])
        latest_checkpoint_path = os.path.join(path, latest_checkpoint)
        checkpoint = torch.load(latest_checkpoint_path, map_location=self.device, weights_only=True)

        # Load model states
        try:
            self.agent.actor.module.load_state_dict(checkpoint['actor_net'])
            self.agent.critic.module.load_state_dict(checkpoint['critic_net'])
        except AttributeError:
            self.agent.actor.load_state_dict(checkpoint['actor_net'])
            self.agent.critic.load_state_dict(checkpoint['critic_net'])

        # Verify and load optimizer states if they exist
        if 'actor_optimizer' in checkpoint:
            self.agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        else:
            print("No actor optimizer state found. Initializing to default.")

        if 'critic_optimizer' in checkpoint:
            self.agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        else:
            print("No critic optimizer state found. Initializing to default.")

        self.agent.critic.eval()
        self.agent.actor.eval()
        print(
            f"\nLoaded weights at {path}, starting at iteration {iteration}. \n Process bar will show missing steps \n")
        return iteration

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
            state, action, reward, done, log_probs, values, entropy = self.replay_buffer.sample(self.batch_size)

            # Compute returns and advantages
            # Utilizing GAE (Generalized Advantage Estimation) get advantages
            returns, advantages = self.compute_returns_and_advantages(reward, done, values)
            # Update the agent's policy E-Step
            # Compute a new policy minimizing KL divergence, while maximizing return
            self.agent.update_policy(state, action, log_probs,
                                     values.mean(dim=0), values.std(dim=0))

            # Update the value network
            self.agent.update_value_network(state, returns)

            # Log data
            avg_reward = reward.mean()
            avg_entropy = entropy.mean()

            if ((update + 1) % save_interval == 0) and save:
                self.writer.add_scalar('Average Reward', avg_reward, update)
                self.writer.add_scalar('Average Entropy', avg_entropy, update)
                self.save_weights(update, n_updates, avg_reward, weights_path)



