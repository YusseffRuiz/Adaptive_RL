import os

import torch
import torch.nn as nn
from scipy.optimize import minimize
from torch.distributions import MultivariateNormal
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from MPO_Algorithm.replay_buffer import ReplayBuffer
from MPO_Algorithm import MPO, gaussian_kl, btr, bt
from MatsuokaOscillator import MatsuokaNetworkWithNN, MatsuokaActor


class MpoMatsuokaTrainer(MPO):
    """
    TODO:
    Train:
    Cambiar la lógica del target_actor forward, agregar el matsuoka step
    __update_critic en vez de meter el action_batch, meter params_batch
    """
    def __init__(self,
                 device,
                 env,
                 dual_constraint=0.1,
                 kl_mean_constraint=0.02,
                 kl_var_constraint=0.0001,
                 kl_constraint=0.02,
                 discount_factor=0.995,
                 alpha_mean_scale=0.1,
                 alpha_var_scale=0.1,
                 alpha_scale=10.0,
                 alpha_mean_max=0.1,
                 alpha_var_max=10.0,
                 alpha_max=1.0,
                 sample_episode_num=100,
                 sample_episode_maxstep=250,
                 sample_action_num=20,
                 batch_size=256,
                 episode_rerun_num=3,
                 mstep_iteration_num=15,
                 evaluate_period=10,
                 evaluate_episode_num=100,
                 evaluate_episode_maxstep=250, neuron_number=4, num_oscillators=4):
        super().__init__(device, env,
                         dual_constraint,
                         kl_mean_constraint,
                         kl_var_constraint,
                         kl_constraint,
                         discount_factor,
                         alpha_mean_scale,
                         alpha_var_scale,
                         alpha_scale,
                         alpha_mean_max,
                         alpha_var_max,
                         alpha_max,
                         sample_episode_num,
                         sample_episode_maxstep,
                         sample_action_num,
                         batch_size,
                         episode_rerun_num,
                         mstep_iteration_num,
                         evaluate_period,
                         evaluate_episode_num,
                         evaluate_episode_maxstep)

        # Definition of the Matsuoka Parameters, the network is the one controlling the action space of the environment.
        self.neuron_number = neuron_number
        self.num_oscillators = num_oscillators  # One for each leg, then we can try using one per DoF.
        print(f"Training with {neuron_number} neurons and {num_oscillators} oscillators.")

        # Definition of the Matsuoka Network using the actor agent.
        self.matsuoka_network = MatsuokaNetworkWithNN(num_oscillators=self.num_oscillators,
                                                      env=self.env, n_envs=1, neuron_number=self.neuron_number, tau_r=1, tau_a=12)

        self.param_dim = self.matsuoka_network.parameters_dimension

        self.actor = MatsuokaActor(env, neuron_number=self.neuron_number, num_oscillators=self.num_oscillators).to(device)
        self.target_actor = MatsuokaActor(env, neuron_number=self.neuron_number, num_oscillators=self.num_oscillators).to(self.device)

        # Redefinition of the Replay Buffer
        self.replay_buffer = ReplayBuffer(matsuoka=True)

    def _sample_trajectory_worker(self, i):
        """
            Collects trajectories, now integrating the Matsuoka oscillator with the action selection process.
        """
        buff = []
        state, _ = self.env.reset()
        for steps in range(self.sample_episode_maxstep):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            params = self.target_actor.select_action(state_tensor)
            action = self.matsuoka_network.step(params).detach().cpu().numpy()
            next_state, reward, done, _, _ = self.env.step(action)
            buff.append((state, action, next_state, reward, params.detach().cpu().numpy()))
            if done:
                break
            else:
                state = next_state
        return buff

    def __update_critic_td(self,
                           state_batch,
                           action_batch,
                           next_state_batch,
                           reward_batch,
                           sample_num=20):
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
            sampled_next_params = π.sample((sample_num,)).transpose(0, 1)  # (B, sample_num, dp)
            sampled_next_actions = self.matsuoka_network.step(sampled_next_params)  # (B, sample_num, da)

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

    def _evaluate(self):
        """
        :return: average return over 100 consecutive episodes
        """
        with torch.no_grad():
            total_rewards = []
            for _ in tqdm(range(self.evaluate_episode_num), desc='evaluating'):
                total_reward = 0.0
                state, _ = self.env.reset()
                for s in range(self.evaluate_episode_maxstep):
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
                    params = self.actor.select_action(state_tensor)
                    action = self.matsuoka_network.step(params)
                    state, reward, termination, _, _ = self.env.step(action.detach().cpu().numpy())
                    total_reward += reward
                    if termination:
                        break
                total_rewards.append(total_reward)
            return np.mean(total_rewards)

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
            self.sample_trajectory_public(self.sample_episode_num)
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
                        desc='training {}/{}'.format(r + 1, self.episode_rerun_num)):
                    K = len(indices)  # the sample number of states
                    N = self.sample_action_num  # the sample number of actions per state
                    ds = self.ds  # the number of state space dimensions
                    da = self.da  # the number of action space dimensions
                    dp = self.param_dim

                    state_batch, action_batch, next_state_batch, reward_batch, params_batch = zip(
                        *[self.replay_buffer[index] for index in indices])

                    state_batch = torch.from_numpy(np.stack(state_batch)).type(torch.float32).to(self.device)  # (K, ds)
                    # (K, da) or (K,)
                    action_batch = torch.from_numpy(np.stack(action_batch)).type(torch.float32).to(self.device)
                    # (K, ds)
                    next_state_batch = torch.from_numpy(np.stack(next_state_batch)).type(torch.float32).to(self.device)
                    reward_batch = torch.from_numpy(np.stack(reward_batch)).type(torch.float32).to(self.device)  # (K,)
                    params_batch =torch.from_numpy(np.stack(params_batch)).type(torch.float32).to(self.device)  # (K, dp)

                    # Policy Evaluation
                    # [2] 3 Policy Evaluation (Step 1)
                    loss_q, q = self.__update_critic_td(
                        state_batch=state_batch,
                        action_batch=action_batch,  # Instead of updating with actions, update with params
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
                        sampled_params = b.sample((N,))  # (N, K, dp)
                        sampled_actions = self.matsuoka_network.step(sampled_params)  # (N, K, da)
                        expanded_states = state_batch[None, ...].expand(N, -1, -1)  # (N, K, ds)
                        target_q = self.target_critic.forward(
                            expanded_states.reshape(-1, ds),  # (N * K, ds)
                            sampled_actions.reshape(-1, da)  # (N * K, da)  # Modification for params_dim
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
                                    π1.expand((N, K)).log_prob(sampled_params)  # (N, K)
                                    + π2.expand((N, K)).log_prob(sampled_params)  # (N, K)
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

                        self.α_μ = np.clip(self.α_μ, 0.0, self.α_μ_max)
                        self.α_Σ = np.clip(self.α_Σ, 0.0, self.α_Σ_max)

                        self.actor_optimizer.zero_grad()
                        # last eq of [2] p.5
                        loss_l = -(
                                loss_p
                                + self.α_μ * (self.ε_kl_μ - kl_μ)
                                + self.α_Σ * (self.ε_kl_Σ - kl_Σ)
                        )
                        mean_loss_l.append(loss_l.item())
                        loss_l.backward()
                        clip_grad_norm_(self.actor.parameters(), 0.5)
                        self.actor_optimizer.step()

            self.update_param_public()

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
