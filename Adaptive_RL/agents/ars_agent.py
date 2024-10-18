import torch
import numpy as np
from Adaptive_RL import neural_networks, logger
from Adaptive_RL.agents import base_agent
from Adaptive_RL.utils import Segment

class ARS(base_agent.BaseAgent):
    """
    Augmented Random Search (ARS) Agent

    Implements ARS which perturbs the policy in random directions and evaluates its performance.
    """

    def __init__(self, environment, hidden_size=256, hidden_layers=2, learning_rate = 0.02, num_directions=8, delta_std=0.05, num_top_directions=None,
                 alive_bonus_offset=0.0, epsilon=1e-6):
        super().__init__()
        self.num_directions = num_directions
        self.delta_std = delta_std
        self.step_size = learning_rate * 0.5 # Step size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.num_top_directions = int(num_top_directions) or num_directions
        self.num_top_directions = min(self.num_top_directions, self.num_directions)
        self.model = neural_networks.ARSModelNetwork(hidden_size=hidden_size, hidden_layers=hidden_layers).get_model()
        self.replay_buffer = Segment()
        # self.actor_updater = StochasticPolicyGradient(lr_actor=learning_rate)  # To match General Structure, but not used
        self.actor_updater = None  # To match General Structure, but not used
        self.environment = environment # For ARS we need interaction with the environment
        self.environment.reset()
        self.alive_bonus_offset = alive_bonus_offset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize config for logging
        self.config = {
            "agent": "ARS",
            "learning_rate": learning_rate,
            "delta_std": delta_std,
            "num_directions": num_directions,
            "num_top_directions": num_top_directions,
            "hidden_size": hidden_size,
            "hidden_layers": hidden_layers,
        }

    def initialize(self, observation_space, action_space, seed=None):
        self.observation_space = observation_space
        self.action_space = action_space

        # Initialize the model (policy) weights randomly.
        self.model.initialize(observation_space, action_space)
        # self.actor_updater.initialize(self.model)

    def step(self, observations, steps=None):
        """Select actions using the policy."""
        actions = self._step(observations)
        actions = actions.cpu().numpy()
        # Keep some values for the next update.
        self.last_observations = observations.copy()
        self.last_actions = actions.copy()

        return actions

    def test_step(self, observations):
        # Sample actions for testing.
        return self._test_step(observations).cpu().numpy()

    def _test_step(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            return self.model.actor(observations).loc

    def update(self, observations, rewards, resets, terminations, steps):
        """
        Store the last transitions in the replay buffer.
        """

        # Prepare to update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.record(self.last_observations)
        if self.model.return_normalizer:
            self.model.return_normalizer.record(rewards)

        self._update(observations)

    def _step(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            actions = self.model.actor(observations).sample()
        return actions

    def _update(self, observations):
        """
        Perform the ARS policy update using deltas.
        1. Generate deltas for the model's parameters.
        2. Evaluate both positive and negative perturbations.
        3. Select top-performing directions.
        4. Apply the policy update using the selected top directions.
        """
        deltas = []
        reward_deltas = torch.zeros(self.num_directions, dtype=torch.float32, device=self.device)

        # Generate deltas and evaluate their performance
        for i in range(self.num_directions):
            delta = self._get_deltas()
            deltas.append(delta)

            original_parameters = [param.clone() for param in self.model.parameters()]

            # Apply positive delta and evaluate
            self._apply_deltas(delta, direction=1)
            reward_pos = self._evaluate_rewards(observations)

            # Reset model parameters to original state
            for param, original_param in zip(self.model.parameters(), original_parameters):
                param.data.copy_(original_param.data)

            # Apply negative delta and evaluate
            self._apply_deltas(delta, direction=-1)  # Reverse deltas
            reward_neg = self._evaluate_rewards(observations)

            # Store the reward difference
            reward_deltas[i] = reward_pos - reward_neg

            # Reset model parameters to original state
            for param, original_param in zip(self.model.parameters(), original_parameters):
                param.data.copy_(original_param.data)

        # Select top-performing deltas and apply policy update
        top_deltas = self._select_top_directions(reward_deltas)
        self._apply_policy_update(deltas, reward_deltas, top_deltas)
        # actor_infos = self.actor_updater(observations)
        logger.store('top_deltas: ', top_deltas)
        # logger.store('actor_infos: ', actor_infos)

        # Update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.update()
        if self.model.return_normalizer:
            self.model.return_normalizer.update()


    def _evaluate_rewards(self, observations):
        """
        Evaluate the reward by interacting with the environment.
        This function takes a step in the environment and collects rewards.
        """
        actions = self.test_step(observations)[0]
        # Take a step in the environments.
        obs, reward, done, *infos = self.environment.step(actions)
        return reward.sum().item()


    def _select_top_directions(self, reward_deltas):
        """
        Select the top-performing directions based on reward deltas.
        """
        if self.num_directions <= len(reward_deltas):
            top_indices = torch.topk(torch.abs(reward_deltas[:self.num_directions]), self.num_top_directions).indices
        else:
            raise ValueError(
                f"num_directions ({self.num_directions}) exceeds available reward deltas ({len(reward_deltas)}).")

        return top_indices

    def _apply_policy_update(self, deltas, reward_deltas, top_deltas):
        """
        Update the model's parameters based on the top-performing deltas.
        Optimized version to reduce computation time.
        """
        # Initialize step with zeros for each parameter
        step = [torch.zeros_like(param, device=self.device) for param in self.model.parameters()]


        std_reward_deltas = torch.std(reward_deltas[:self.num_directions]) + self.epsilon

        for idx in top_deltas:
            idx = idx.item()  # Convert tensor index to scalar

            # Fetch the reward difference and delta
            reward_pos_neg_diff = reward_deltas[idx].item()  # Convert to scalar value
            delta = deltas[idx]  # Fetch the list of parameter-wise deltas for this direction
            # Move tensors to CPU before converting to NumPy
            delta = [np.sum(tensor.cpu().numpy()) for tensor in delta]
            step_res = reward_pos_neg_diff * torch.tensor(delta)
            for i in range(len(step)):
                step[i] += step_res[i]  # Update the step for each parameter slot
                # Normalize the step by the number of top directions and the standard deviation
                step[i] /= (self.num_top_directions * std_reward_deltas)

        # Apply the step to update the policy parameters
        with torch.no_grad():
            for param, param_step in zip(self.model.parameters(), step):
                param += self.step_size * param_step

        logger.store('step: ', step)

    def _get_deltas(self):
        """
        Generate random perturbations (deltas) for the model parameters.
        """
        deltas = []
        for param in self.model.parameters():
            delta = torch.normal(0, self.delta_std, size=param.shape).to(self.device)
            deltas.append(delta)
        return deltas

    def _apply_deltas(self, deltas, direction):
        """
        Apply perturbations (deltas) to the model's parameters in the given direction.
        direction == 1 applies positive delta, direction == -1 applies negative delta.
        """
        with torch.no_grad():
            for param, delta in zip(self.model.parameters(), deltas):
                param += direction * delta