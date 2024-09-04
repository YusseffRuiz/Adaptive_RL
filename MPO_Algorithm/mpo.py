"""
Modified MPO Algorithm

This code is a modified version of the Maximum a Posteriori Policy Optimization (MPO) algorithm,
originally implemented by daisatojp and available at: https://github.com/daisatojp/mpo

The original implementation has been adapted to integrate with a Matsuoka Oscillator (CPG) control
framework and customized for specific reinforcement learning tasks. All modifications to the original
code are marked and documented within this file.

Original Repository:
- Author: daisatojp
- Repository: https://github.com/daisatojp/mpo

LICENSE: The original code is licensed under the MIT License. For more information on the terms and
conditions, refer to the LICENSE file in the original repository.

Modifications made by Adan Dominguez, 2024.

"""

import os
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.distributions import MultivariateNormal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
from .actor import Actor
from .critic import Critic
from .replay_buffer import ReplayBuffer


def bt(m):
    return m.transpose(dim0=-2, dim1=-1)


def btr(m):
    return m.diagonal(dim1=-2, dim2=-1).sum(-1)


def gaussian_kl(μi, μ, Ai, A):
    """
    decoupled KL between two multivariate gaussian distribution
    C_μ = KL(f(x|μi,Σi)||f(x|μ,Σi))
    C_Σ = KL(f(x|μi,Σi)||f(x|μi,Σ))
    :param μi: (B, n)
    :param μ: (B, n)
    :param Ai: (B, n, n)
    :param A: (B, n, n)
    :return: C_μ, C_Σ: scalar
        mean and covariance terms of the KL
    :return: mean of determinanats of Σi, Σ
    ref : https://stanford.edu/~jduchi/projects/general_notes.pdf page.13
    """
    n = A.size(-1)
    μi = μi.unsqueeze(-1)  # (B, n, 1)
    μ = μ.unsqueeze(-1)  # (B, n, 1)
    Σi = Ai @ bt(Ai)  # (B, n, n)
    Σ = A @ bt(A)  # (B, n, n)
    Σi_det = Σi.det()  # (B,)
    Σ_det = Σ.det()  # (B,)
    # determinant can be minus due to numerical calculation error
    # https://github.com/daisatojp/mpo/issues/11
    Σi_det = torch.clamp_min(Σi_det, 1e-6)
    Σ_det = torch.clamp_min(Σ_det, 1e-6)
    Σi_inv = Σi.inverse()  # (B, n, n)
    Σ_inv = Σ.inverse()  # (B, n, n)
    inner_μ = ((μ - μi).transpose(-2, -1) @ Σi_inv @ (μ - μi)).squeeze()  # (B,)
    inner_Σ = torch.log(Σ_det / Σi_det) - n + btr(Σ_inv @ Σi)  # (B,)
    C_μ = 0.5 * torch.mean(inner_μ)
    C_Σ = 0.5 * torch.mean(inner_Σ)
    return C_μ, C_Σ, torch.mean(Σi_det), torch.mean(Σ_det)


class MPO(object):
    """
    Maximum A Posteriori Policy Optimization (MPO)
    :param device: cuda or cpu
    :param env: gym environment
    :param dual_constraint:
        (float) hard constraint of the dual formulation in the E-step
        correspond to [2] p.4 ε
    :param kl_mean_constraint:
        (float) hard constraint of the mean in the M-step
        correspond to [2] p.6 ε_μ for continuous action space
    :param kl_var_constraint:
        (float) hard constraint of the covariance in the M-step
        correspond to [2] p.6 ε_Σ for continuous action space
    :param kl_constraint:
        (float) hard constraint in the M-step
        correspond to [2] p.6 ε_π for discrete action space
    :param discount_factor: (float) discount factor used in Policy Evaluation
    :param alpha_scale: (float) scaling factor of the lagrangian multiplier in the M-step
    :param sample_episode_num: the number of sampled episodes
    :param sample_episode_maxstep: maximum sample steps of an episode
    :param sample_action_num:
    :param batch_size: (int) size of the sampled mini-batch
    :param episode_rerun_num:
    :param mstep_iteration_num: (int) the number of iterations of the M-step
    :param evaluate_episode_maxstep: maximum evaluate steps of an episode
    [1] https://arxiv.org/pdf/1806.06920.pdf
    [2] https://arxiv.org/pdf/1812.02256.pdf
    """
    def __init__(self,
                 device,
                 env,
                 dual_constraint=0.1,
                 kl_mean_constraint=0.01,
                 kl_var_constraint=0.0001,
                 kl_constraint=0.01,
                 discount_factor=0.995,
                 alpha_mean_scale=0.1,
                 alpha_var_scale=0.1,
                 alpha_scale=10.0,
                 alpha_mean_max=0.1,
                 alpha_var_max=10.0,
                 alpha_max=1.0,
                 sample_episode_num=30,
                 sample_episode_maxstep=200,
                 sample_action_num=64,
                 batch_size=256,
                 episode_rerun_num=3,
                 mstep_iteration_num=5,
                 evaluate_period=10,
                 evaluate_episode_num=100,
                 evaluate_episode_maxstep=200):
        self.device = device
        print(f"Working with {device} processor")
        self.env = env

        # the number of dimensions of state space
        self.ds = env.observation_space.shape[0]
        # the number of dimensions of action space
        self.da = env.action_space.shape[0]

        self.ε_dual = dual_constraint
        self.ε_kl_μ = kl_mean_constraint
        self.ε_kl_Σ = kl_var_constraint
        self.ε_kl = kl_constraint
        self.γ = discount_factor
        self.α_μ_scale = alpha_mean_scale
        self.α_Σ_scale = alpha_var_scale
        self.α_scale = alpha_scale
        self.α_μ_max = alpha_mean_max
        self.α_Σ_max = alpha_var_max
        self.α_max = alpha_max
        self.sample_episode_num = sample_episode_num
        self.sample_episode_maxstep = sample_episode_maxstep
        self.sample_action_num = sample_action_num
        self.batch_size = batch_size
        self.episode_rerun_num = episode_rerun_num
        self.mstep_iteration_num = mstep_iteration_num
        self.evaluate_period = evaluate_period
        self.evaluate_episode_num = evaluate_episode_num
        self.evaluate_episode_maxstep = evaluate_episode_maxstep

        self.actor = Actor(env).to(self.device)
        self.critic = Critic(env).to(self.device)
        self.target_actor = Actor(env).to(self.device)
        self.target_critic = Critic(env).to(self.device)

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False

        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=3e-4)
        self.norm_loss_q = nn.SmoothL1Loss()

        self.η = np.random.rand()
        self.α_μ = 0.0  # lagrangian multiplier for continuous action space in the M-step
        self.α_Σ = 0.0  # lagrangian multiplier for continuous action space in the M-step
        self.α = 0.0  # lagrangian multiplier for discrete action space in the M-step

        self.replay_buffer = ReplayBuffer()

        self.max_return_eval = -np.inf
        self.iteration = 1

    def dual(self, n, target_q_np):
        """
        dual function of the non-parametric variational
        Q = target_q_np  (K, N)
        g(η) = η*ε + η*mean(log(mean(exp(Q(s, a)/η), along=a)), along=s)
        For numerical stabilization, this can be modified to
        Qj = max(Q(s, a), along=a)
        g(η) = η*ε + mean(Qj, along=j) + η*mean(log(mean(exp((Q(s, a)-Qj)/η), along=a)), along=s)
        """
        max_q = np.max(target_q_np, 1)
        output = (n * self.ε_dual + np.mean(max_q) + n *
                  np.mean(np.log(np.mean(np.exp((target_q_np - max_q[:, None]) / n), axis=1))))
        return output

    def train(self,
              iteration_num=1000,
              log_dir='log',
              model_save_period=10):
        """
        :param iteration_num: max numbers of iterations to complete
        :param log_dir: where the weights are being saved
        :param model_save_period: how many iterations to save the weights
        """

        model_save_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        writer = SummaryWriter(os.path.join(log_dir, 'tb'))

        for it in range(self.iteration, iteration_num + 1):
            self.__sample_trajectory(self.sample_episode_num)
            buff_sz = len(self.replay_buffer)

            mean_reward = self.replay_buffer.mean_reward()
            mean_return = self.replay_buffer.mean_return()
            mean_loss_q = []
            mean_loss_p = []
            mean_loss_l = []
            mean_est_q = []
            max_kl_μ = []
            max_kl_Σ = []
            mean_Σ_det = []

            for r in range(self.episode_rerun_num):
                for indices in tqdm(
                        BatchSampler(
                            SubsetRandomSampler(range(buff_sz)), self.batch_size, drop_last=True),
                        desc='training {}/{}'.format(r+1, self.episode_rerun_num)):
                    K = len(indices)  # the sample number of states
                    N = self.sample_action_num  # the sample number of actions per state
                    ds = self.ds  # the number of state space dimensions
                    da = self.da  # the number of action space dimensions

                    state_batch, action_batch, next_state_batch, reward_batch = zip(
                        *[self.replay_buffer[index] for index in indices])

                    state_batch = torch.from_numpy(np.stack(state_batch)).type(torch.float32).to(self.device)  # (K, ds)
                    # (K, da) or (K,)
                    action_batch = torch.from_numpy(np.stack(action_batch)).type(torch.float32).to(self.device)
                    # (K, ds)
                    next_state_batch = torch.from_numpy(np.stack(next_state_batch)).type(torch.float32).to(self.device)
                    reward_batch = torch.from_numpy(np.stack(reward_batch)).type(torch.float32).to(self.device)  # (K,)

                    # Policy Evaluation
                    # [2] 3 Policy Evaluation (Step 1)
                    loss_q, q = self.__update_critic_td(
                        state_batch=state_batch,
                        action_batch=action_batch,
                        next_state_batch=next_state_batch,
                        reward_batch=reward_batch,
                        sample_num=self.sample_action_num
                    )
                    mean_loss_q.append(loss_q.item())
                    mean_est_q.append(q.abs().mean().item())

                    # E-Step of Policy Improvement
                    # [2] 4.1 Finding action weights (Step 2)
                    with torch.no_grad():
                        # sample N actions per state
                        b_μ, b_A = self.target_actor.forward(state_batch)  # (K,)
                        b = MultivariateNormal(b_μ, scale_tril=b_A)  # (K,)
                        sampled_actions = b.sample((N,))  # (N, K, da)
                        expanded_states = state_batch[None, ...].expand(N, -1, -1)  # (N, K, ds)
                        target_q = self.target_critic.forward(
                            expanded_states.reshape(-1, ds),  # (N * K, ds)
                            sampled_actions.reshape(-1, da)  # (N * K, da)
                        ).reshape(N, K)  # (N, K)
                        target_q_np = target_q.cpu().transpose(0, 1).numpy()  # (K, N)

                    # https://arxiv.org/pdf/1812.02256.pdf
                    # [2] 4.1 Finding action weights (Step 2)
                    #   Using an exponential transformation of the Q-values
                    bounds = [(1e-6, None)]
                    res = minimize(self.dual, np.array([self.η]), target_q_np, method='SLSQP', bounds=bounds)
                    self.η = res.x[0]

                    qij = torch.softmax(target_q / self.η, dim=0)  # (N, K) or (da, K)

                    # M-Step of Policy Improvement
                    # [2] 4.2 Fitting an improved policy (Step 3)
                    for _ in range(self.mstep_iteration_num):
                        μ, A = self.actor.forward(state_batch)
                        # First term of last eq of [2] p.5
                        # see also [2] 4.2.1 Fitting an improved Gaussian policy
                        π1 = MultivariateNormal(loc=μ, scale_tril=b_A)  # (K,)
                        π2 = MultivariateNormal(loc=b_μ, scale_tril=A)  # (K,)
                        loss_p = torch.mean(
                            qij * (
                                    π1.expand((N, K)).log_prob(sampled_actions)  # (N, K)
                                    + π2.expand((N, K)).log_prob(sampled_actions)  # (N, K)
                            )
                        )
                        mean_loss_p.append((-loss_p).item())

                        kl_μ, kl_Σ, Σi_det, Σ_det = gaussian_kl(μi=b_μ, μ=μ, Ai=b_A, A=A)
                        max_kl_μ.append(kl_μ.item())
                        max_kl_Σ.append(kl_Σ.item())
                        mean_Σ_det.append(Σ_det.item())

                        if np.isnan(kl_μ.item()):  # This should not happen
                            raise RuntimeError('kl_μ is nan')
                        if np.isnan(kl_Σ.item()):  # This should not happen
                            raise RuntimeError('kl_Σ is nan')

                        # Update lagrange multipliers by gradient descent
                        # this equation is derived from last eq of [2] p.5,
                        # just differentiate with respect to α
                        # and update α so that the equation is to be minimized.
                        self.α_μ -= self.α_μ_scale * (self.ε_kl_μ - kl_μ).detach().item()
                        self.α_Σ -= self.α_Σ_scale * (self.ε_kl_Σ - kl_Σ).detach().item()

                        self.α_μ = np.clip(0.0, self.α_μ, self.α_μ_max)
                        self.α_Σ = np.clip(0.0, self.α_Σ, self.α_Σ_max)

                        self.actor_optimizer.zero_grad()
                        # last eq of [2] p.5
                        loss_l = -(
                                loss_p
                                + self.α_μ * (self.ε_kl_μ - kl_μ)
                                + self.α_Σ * (self.ε_kl_Σ - kl_Σ)
                        )
                        mean_loss_l.append(loss_l.item())
                        loss_l.backward()
                        clip_grad_norm_(self.actor.parameters(), 0.1)
                        self.actor_optimizer.step()

            self.__update_param()

            return_eval = None
            if it % self.evaluate_period == 0:
                self.actor.eval()
                return_eval = self._evaluate()
                self.actor.train()
                if return_eval is None:
                    return_eval = self.max_return_eval
                self.max_return_eval = max(self.max_return_eval, return_eval)

            mean_loss_q = np.mean(mean_loss_q)
            mean_loss_p = np.mean(mean_loss_p)
            mean_loss_l = np.mean(mean_loss_l)
            mean_est_q = np.mean(mean_est_q)
            max_kl_μ = np.max(max_kl_μ)
            max_kl_Σ = np.max(max_kl_Σ)
            mean_Σ_det = np.mean(mean_Σ_det)

            print('iteration :', it)
            if it % self.evaluate_period == 0:
                print('  max_return_eval :', self.max_return_eval)
                print('  return_eval :', return_eval)
            print('  mean return :', mean_return)
            print('  mean reward :', mean_reward)
            print('  mean loss_q :', mean_loss_q)
            print('  mean loss_p :', mean_loss_p)
            print('  mean loss_l :', mean_loss_l)
            print('  mean est_q :', mean_est_q)
            print('  η :', self.η)
            print('  max_kl_μ :', max_kl_μ)
            print('  max_kl_Σ :', max_kl_Σ)
            print('  mean_Σ_det :', mean_Σ_det)
            print('  α_μ :', self.α_μ)
            print('  α_Σ :', self.α_Σ)

            self.save_model(os.path.join(model_save_dir, 'model_latest.pt'))
            if it % model_save_period == 0:
                self.save_model(os.path.join(model_save_dir, 'model_{}.pt'.format(it)))

            if it % self.evaluate_period == 0:
                writer.add_scalar('max_return_eval', self.max_return_eval, it)
                writer.add_scalar('return_eval', return_eval, it)
            writer.add_scalar('return', mean_return, it)
            writer.add_scalar('reward', mean_reward, it)
            writer.add_scalar('loss_q', mean_loss_q, it)
            writer.add_scalar('loss_p', mean_loss_p, it)
            writer.add_scalar('loss_l', mean_loss_l, it)
            writer.add_scalar('mean_q', mean_est_q, it)
            writer.add_scalar('η', self.η, it)
            writer.add_scalar('max_kl_μ', max_kl_μ, it)
            writer.add_scalar('max_kl_Σ', max_kl_Σ, it)
            writer.add_scalar('mean_Σ_det', mean_Σ_det, it)
            writer.add_scalar('α_μ', self.α_μ, it)
            writer.add_scalar('α_Σ', self.α_Σ, it)

            writer.flush()

        # end training
        if writer is not None:
            writer.close()

    def load_model(self, path=None):
        """
        loads a model from a given path
        :param path: (str) file path (.pt file)
        """
        load_path = path if path is not None else "weights"
        checkpoint = torch.load(load_path, weights_only=True)
        self.iteration = checkpoint['iteration']
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optim_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optim_state_dict'])
        self.critic.train()
        self.target_critic.train()
        self.actor.train()
        self.target_actor.train()

    def save_model(self, path=None):
        """
        saves a model to a given path
        :param path: (str) file path (.pt file)
        """
        data = {
            'iteration': self.iteration,
            'actor_state_dict': self.actor.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optim_state_dict': self.actor_optimizer.state_dict(),
            'critic_optim_state_dict': self.critic_optimizer.state_dict()
        }
        torch.save(data, path)

    def _sample_trajectory_worker(self, i):
        buff = []
        state, _ = self.env.reset()
        for steps in range(self.sample_episode_maxstep):
            action = self.target_actor.select_action(
                torch.from_numpy(state).type(torch.float32).to(self.device)
            ).cpu().numpy()
            next_state, reward, termination, _, _ = self.env.step(action)
            buff.append((state, action, next_state, reward))
            if termination:
                break
            else:
                state = next_state
        return buff

    def __sample_trajectory(self, sample_episode_num):
        self.replay_buffer.clear()
        episodes = [self._sample_trajectory_worker(i)
                    for i in tqdm(range(sample_episode_num), desc='sample_trajectory')]
        self.replay_buffer.store_episodes(episodes)

    def sample_trajectory_public(self, sample_episode_num):
        self.__sample_trajectory(sample_episode_num)

    def _evaluate(self):
        """
        :return: average return over 100 consecutive episodes
        """
        with torch.no_grad():
            total_rewards = []
            for e in tqdm(range(self.evaluate_episode_num), desc='evaluating'):
                total_reward = 0.0
                state, _ = self.env.reset()
                for s in range(self.evaluate_episode_maxstep):
                    action = self.actor.select_action(
                        torch.from_numpy(state).type(torch.float32).to(self.device)
                    ).cpu().numpy()
                    state, reward, termination, _, _ = self.env.step(action)
                    total_reward += reward
                    if termination:
                        break
                total_rewards.append(total_reward)
            return np.mean(total_rewards)

    def __update_critic_td(self,
                           state_batch,
                           action_batch,
                           next_state_batch,
                           reward_batch,
                           sample_num=64):
        """
        :param state_batch: (B, ds)
        :param action_batch: (B, da) or (B,)
        :param next_state_batch: (B, ds)
        :param reward_batch: (B,)
        :param sample_num:
        :return:
        """
        B = state_batch.size(0)
        ds = self.ds
        da = self.da
        with torch.no_grad():
            r = reward_batch  # (B,)
            π_μ, π_A = self.target_actor.forward(next_state_batch)  # (B,)
            π = MultivariateNormal(π_μ, scale_tril=π_A)  # (B,)
            sampled_next_actions = π.sample((sample_num,)).transpose(0, 1)  # (B, sample_num, da)
            expanded_next_states = next_state_batch[:, None, :].expand(-1, sample_num, -1)  # (B, sample_num, ds)
            expected_next_q = self.target_critic.forward(
                expanded_next_states.reshape(-1, ds),  # (B * sample_num, ds)
                sampled_next_actions.reshape(-1, da)  # (B * sample_num, da)
            ).reshape(B, sample_num).mean(dim=1)  # (B,)
            y = r + self.γ * expected_next_q
        self.critic_optimizer.zero_grad()
        t = self.critic(state_batch, action_batch).squeeze()
        loss = self.norm_loss_q(y, t)
        loss.backward()
        self.critic_optimizer.step()
        return loss, y

    def __update_param(self):
        """
        Sets target parameters to trained parameter
        """
        # Update policy parameters
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        # Update critic parameters
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

    def update_param_public(self):
        self.__update_param()
