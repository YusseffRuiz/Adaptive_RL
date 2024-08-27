import numpy as np
import math
import torch
from torch import nn
from torch import optim

import MPO_Algorithm


class MatsuokaAgent(nn.Module):
    def __init__(self, input_size, hidden_size, num_oscillators, neuron_number, action_dim, action_space, device):
        """
        :param input_size: size of input data, number of sensory inputs influencing oscillators
        :param hidden_size:
        :param output_size: number of parameters the NN will control (weights, decay)
        """
        super(MatsuokaAgent, self).__init__()

        self.kl_epsilon = 0.01  # For the decay

        self.neuron_number = neuron_number
        self.num_oscillators = num_oscillators
        self.action_space = action_space
        self.mpo_agent = MPO_Algorithm.MPOAgent

        self.input_neuron = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # Input to hidden layer
            nn.LayerNorm(hidden_size),
            nn.SiLU(),  # Activation function
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),  # Activation Function
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, num_oscillators * neuron_number * 2)
            # Hidden to output layer, output size y the neuron number
        ).to(device)

        self.output_neuron = nn.Sequential(
            nn.Linear(neuron_number, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, action_dim)
        ).to(device)

        self.input_optimizer = optim.AdamW(self.input_neuron.parameters(), lr=3e-4)
        self.output_optimizer = optim.AdamW(self.output_neuron.parameters(), lr=3e-4)

    def select_action(self, state):
        """
        Selects an action based on the current policy.

        Args:
            state (torch.Tensor): The current state.

        Returns:
            action (torch.Tensor): The selected continuous action.
            log_prob (torch.Tensor): Log probability of the selected action.
        """
        params = self.input_neuron(state)
        params = params.reshape(params.shape[0], self.num_oscillators, self.neuron_number*2)
        # Split the output into mean and log_std for each action dimension
        mean, log_std = params[:, :, :params.shape[2] // 2], params[:, :, params.shape[2] // 2:]

        if torch.isnan(log_std).any():
            log_std = torch.zeros_like(log_std)
        if torch.isnan(mean).any():
            mean = torch.zeros_like(mean)
        std = torch.exp(log_std)

        # Create a normal distribution and sample a continuous action
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()

        # Clamp the actions to the valid range
        if self.action_space is not None:
            action = torch.clamp(action, self.action_space.low[0], self.action_space.high[0])

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy

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
        params = self.input_neuron(states)

        params = params.reshape(params.shape[0], self.num_oscillators, self.neuron_number*2)
        new_mean, new_log_std = params[:, :, :params.shape[2] // 2], params[:, :, params.shape[2] // 2:]

        # Calculate log probabilities under new policy
        if torch.isnan(new_mean).any():
            new_mean = torch.zeros_like(new_mean)
            new_log_std = torch.zeros_like(new_log_std)
        new_std = torch.exp(new_log_std)
        dist = torch.distributions.Normal(new_mean, new_std)
        new_log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)

        # Compute the KL divergence
        kl_div = self.mpo_agent.compute_kl_divergence(old_mean=old_mean, old_log_std=old_log_std, new_mean=new_mean,
                                            new_log_std=new_log_std)

        # Ensure KL divergence is within the allowable threshold
        kl_penalty = torch.clamp(kl_div - self.kl_epsilon, min=0).mean()

        # Compute loss (M-step)
        old_log_probs.requires_grad_(True)
        policy_loss = (new_log_probs.view(-1) - old_log_probs.view(-1)).mean() + kl_penalty - 0.001 * entropy.mean()
        # Update the policy network
        self.input_optimizer.zero_grad()
        self.output_optimizer.zero_grad()
        policy_loss.backward()
        self.input_optimizer.step()
        self.output_optimizer.step()


class MatsuokaNetwork:
    def __init__(self, num_oscillators, action_space=None, neuron_number=2, tau_r=1.0, tau_a=12.0, weights=None, u=None, beta=2.5, dt=0.01):
        """
        Initialize the Coupled Oscillators system.

        Parameters:
        - num_oscillators (int): Number of oscillators to connect.
        - tau_r (float): Time constant for neuron recovery.
        - tau_a (float): Time constant for neuron adaptation.
        - weights (array): Weight matrix for neuron interactions (optional).
        - u (array): External input (firing rate) to the neurons (optional).
        - beta (float): Adaptation coefficient.
        - dt (float): Time step for integration.
        """
        self.neuron_number = neuron_number
        self.oscillators = [MatsuokaOscillator(neuron_number=neuron_number, tau_r=tau_r, tau_a=tau_a, weights=weights, u=u, beta=beta, dt=dt, action_space=action_space) for _ in range(num_oscillators)]
        self.num_oscillators = num_oscillators

    def step(self):
        """
        Perform a single step for all coupled oscillators.
        Each oscillator's output is fed as input to the next oscillator.
        """
        outputs = [oscillator.y for oscillator in self.oscillators]

        # Update each oscillator with the output of the previous one
        for i in range(self.num_oscillators):
            if i == 0:
                # For the first oscillator, input is from the last oscillator (wrap-around)
                self.oscillators[i].step(weights=outputs[-1])
            else:
                self.oscillators[i].step(weights=outputs[i - 1])

    def run(self, steps=1000):
        """
        Run the coupled oscillators for a given number of steps.

        Parameters:
        - steps (int): Number of simulation steps.

        Returns:
        - y_outputs (list of arrays): Outputs of all oscillators over time.
        """
        y_outputs = [np.zeros((steps, self.neuron_number)) for _ in range(self.num_oscillators)]

        for step_ in range(steps):
            self.step()
            for i in range(self.num_oscillators):
                y_outputs[i][step_, :] = self.oscillators[i].y.detach().cpu().numpy

        return y_outputs


class MatsuokaOscillator:
    def __init__(self, action_space, neuron_number=2, tau_r=1.0, tau_a=12.0, weights=None, u=None, beta=2.5, dt=0.01):
        """
                Initialize the Matsuoka Oscillator.

                Parameters:
                - neuron_number (int): Number of neurons in the oscillator network.
                - tau_r (float): Time constant for neuron recovery.
                - tau_a (float): Time constant for neuron adaptation.
                - weights (array): Weight matrix for neuron interactions (default: None, initialized as identity matrix).
                - u (array): External input (firing rate) to the neurons (default: None, initialized as ones).
                - beta (float): Adaptation coefficient.
                - dt (float): Time step for integration.
                """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.action_space = action_space
        self.neuron_number = neuron_number
        self.tau_r = tau_r
        self.tau_a = tau_a
        self.beta = beta
        self.dt = dt
        self.x = torch.arange(0, self.neuron_number, 1, dtype=torch.float32).to(self.device)
        # Neuron initial membrane potential
        self.y = torch.zeros(neuron_number, dtype=torch.float32, device=self.device)
        # Output, it is the neurons update, which is mapped via NN to the action space.
        self.z = torch.zeros(neuron_number, dtype=torch.float32, device=self.device)  # Correction value

        if weights is None:
            self.weights = torch.ones(neuron_number, dtype=torch.float32, device=self.device)
        else:
            assert len(weights) == neuron_number, \
                "Weights must be a square matrix with size equal to the number of neurons."
            self.weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # Initialize external input (ones by default)
        if u is None:
            self.u = torch.ones(neuron_number, dtype=torch.float32, device=self.device)
        else:
            assert len(u) == neuron_number, "Input array u - (fire rate) must match the number of neurons."
            self.u = torch.tensor(u, dtype=torch.float32, device=self.device)

    def step(self, weights=None):
        """
               Perform a single update step for the Matsuoka oscillator network.

               Parameters:
               - tau_r (float): Optional. Time constant for neuron recovery.
               - weights (array): Optional. Weight matrix for neuron interactions.
               - beta (float): Optional. Adaptation coefficient.
               """
        # Update parameters if provided
        if weights is not None:
            assert len(weights) == self.neuron_number, \
                "Weights must be a matrix with size equal to the number of neurons."
            self.weights = weights

        # Update membrane potentials
        # Store previous output
        y_prev = torch.roll(input=self.y, shifts=1)
        y_prev[0] = self.y[0]

        dx = (-self.x - self.weights*y_prev + self.u - self.beta*self.z)*self.dt/self.tau_r

        self.x += dx

        # Update outputs
        self.y = torch.maximum(torch.tensor(0.0), self.x)

        if self.action_space is not None:
            self.y = torch.clamp(self.y, self.action_space.low[0], self.action_space.high[0])

        # Update adaptation variables
        dz = [((math.pow(self.y[i], 1) - self.z[i]) * self.dt / self.tau_a) for i in range(self.neuron_number)]
        for i in range(self.neuron_number):
            self.z[i] += dz[i]
        right_output = self.y[0:self.neuron_number//2]
        left_output = self.y[0:self.neuron_number//2:self.neuron_number]  # Opposite Phase
        return right_output, left_output

    def run(self, steps=1000, weights_seq=None):
        """
        Method implemented to be used by itself.
        :param steps:
        :param weights_seq:
        :return:
        """
        y_output = torch.zeros(steps, self.neuron_number, dtype=torch.float32, device=self.device)
        for i in range(steps):
            weights = weights_seq[i] if weights_seq is not None else None
            self.step(weights=weights)
            y_output[i, :] = self.y
        return y_output


class MatsuokaNetworkWithNN:
    """
    A class that represents a network of Matsuoka oscillators controlled by a neural network model.

    Attributes:
    -----------
    num_oscillators : int
        The number of Matsuoka oscillators in the network.
    nn_model : torch.nn.Module
        The neural network model that controls the parameters of the oscillators.
    neuron_number : int, optional
        The number of neurons in each Matsuoka oscillator (default is 2).
    action_space : int
        The dimension of the action space, which represents the number of control outputs (default is 6).
    oscillators : list
        A list of MatsuokaOscillator objects, one for each oscillator in the network.

    Methods:
    --------
    step(sensory_input):
        Takes a single step in the simulation, updating the state of each oscillator based on the sensory input and
        the neural network output. Returns the control actions as a torch tensor.

    run(steps=1000, sensory_input_seq=None):
        Runs the simulation for a given number of steps, returning the outputs of the oscillators over time.
    """

    def __init__(self, num_oscillators, observation_space, action_dim, action_space, n_envs, neuron_number=2,
                 tau_r=2.0, tau_a=12.0, dt=0.1):
        """
        Initializes the MatsuokaNetworkWithNN with a specified number of oscillators and a neural network model.

        Parameters:
        -----------
        num_oscillators : int
            The number of Matsuoka oscillators in the network.
        nn_model : torch.nn.Module
            The neural network model used to control the parameters of the oscillators.
        neuron_number : int, optional
            The number of neurons in each Matsuoka oscillator (default is 2).
        tau_r : float, optional
            The time constant for the rise time of the oscillator neurons (default is 2.0).
        tau_a : float, optional
            The time constant for the adaptation time of the oscillator neurons (default is 12.0).
        dt : float, optional
            The time step for the simulation (default is 0.1).
        """

        self.oscillators = [MatsuokaOscillator(action_space=action_space, neuron_number=neuron_number, tau_r=tau_r,
                                               tau_a=tau_a, dt=dt) for _ in range(num_oscillators)]
        self.num_oscillators = num_oscillators
        self.observation_space = observation_space
        self.neuron_number = neuron_number
        self.action_dim = action_dim
        self.n_envs = n_envs
        self.parameters_dimension = self.num_oscillators * self.neuron_number
        self.nn_model = MatsuokaAgent(observation_space, 128, num_oscillators, neuron_number, action_dim,
                                      action_space, device="cuda")

    def step(self, sensory_input):
        """
                Takes a single step in the simulation, updating the state of each oscillator based on the sensory input and
                the neural network output.

                Parameters:
                -----------
                sensory_input : torch.Tensor
                    The input to the neural network, which influences the oscillator parameters.

                Returns:
                --------
                output_actions : torch.Tensor
                    A tensor representing the control actions for the environment, with a dimension matching the action space.
                """

        # nn_output = self.nn_model.input_forward(sensory_input)  # Output is a matrix size [oscillators, neurons]

        output_y = torch.zeros(self.n_envs, self.neuron_number, dtype=torch.float32, device="cuda")

        for env in range(self.n_envs):
            for i in range(self.num_oscillators):
                right_output, left_output = self.oscillators[i].step(weights=sensory_input[env, i, :])
                output_y[env, 0:self.neuron_number//2] += right_output/self.num_oscillators
                output_y[env, self.neuron_number//2:self.neuron_number] += left_output/self.num_oscillators
        output_actions = self.nn_model.output_neuron(output_y)
        return output_actions

    def run(self, steps=1000, sensory_input_seq=None):
        """
                Runs the simulation for a specified number of steps, returning the outputs of the oscillators over time.

                Parameters:
                -----------
                steps : int, optional
                    The number of time steps to run the simulation (default is 1000).
                sensory_input_seq : numpy.ndarray, optional
                    A sequence of sensory inputs to be fed into the network at each time step (default is None, which uses ones).

                Returns:
                --------
                y_outputs : numpy.ndarray
                    A 3D array containing the outputs of the oscillators at each time step. The shape of the array is
                    (steps, num_oscillators, neuron_number).
                """
        y_outputs = np.zeros((steps, self.num_oscillators, self.neuron_number))

        for t in range(steps):
            sensory_input = sensory_input_seq[t] if sensory_input_seq is not None else np.ones(self.num_oscillators)
            self.step(sensory_input)
            for i in range(self.num_oscillators):
                y_outputs[t, i, :] = self.oscillators[i].y.detach().cpu().numpy()
        return y_outputs
