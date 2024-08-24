import numpy as np
import math
import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        """
        :param input_size: size of input data, number of sensory inputs influencing oscillators
        :param hidden_size:
        :param output_size: number of parameters the NN will control (weights, decay)
        """
        super(NeuralNetwork,self).__init__()
        self.neuron = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # Input to hidden layer
            nn.LayerNorm(hidden_size),
            nn.SiLU(),  # Activation function
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, output_size)  # Hidden to output layer
        ).to(device)

    def forward(self, x):
        out = self.neuron(x)
        return out


class MatsuokaNetwork:
    def __init__(self, num_oscillators, neuron_number=2, tau_r=1.0, tau_a=12.0, weights=None, u=None, beta=2.5, dt=0.01):
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
        self.oscillators = [MatsuokaOscillator(neuron_number=neuron_number, tau_r=tau_r, tau_a=tau_a, weights=weights, u=u, beta=beta, dt=dt) for _ in range(num_oscillators)]
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

        for step in range(steps):
            self.step()
            for i in range(self.num_oscillators):
                y_outputs[i][step, :] = self.oscillators[i].y

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
        self.action_space = action_space
        self.neuron_number = neuron_number
        self.tau_r = tau_r
        self.tau_a = tau_a
        self.beta = beta
        self.dt = dt
        self.x = np.linspace(0, self.neuron_number, self.neuron_number)  # Neuron initial membrane potential
        self.y = np.zeros(neuron_number)  # Output, it should be the regulated Action Space.
        self.z = np.zeros(neuron_number)

        if weights is None:
            self.weights = np.ones(neuron_number)
        else:
            assert len(weights) == neuron_number, \
                "Weights must be a square matrix with size equal to the number of neurons."
            self.weights = np.array(weights)

        # Initialize external input (ones by default)
        if u is None:
            self.u = np.ones(neuron_number)
        else:
            assert len(u) == neuron_number, "Input array u - (fire rate) must match the number of neurons."
            self.u = np.array(u)

    def step(self, tau_r=None, weights=None, beta=None):
        """
               Perform a single update step for the Matsuoka oscillator network.

               Parameters:
               - tau_r (float): Optional. Time constant for neuron recovery.
               - weights (array): Optional. Weight matrix for neuron interactions.
               - beta (float): Optional. Adaptation coefficient.
               """
        # Update parameters if provided
        if tau_r is not None:
            self.tau_r = tau_r
        if weights is not None:
            assert len(weights) == self.neuron_number, \
                "Weights must be a square matrix with size equal to the number of neurons."
            self.weights = weights
        if beta is not None:
            self.beta = beta

        # Update membrane potentials
        dx = [0.0 for _ in range(self.neuron_number)]
        for i in range(self.neuron_number):
            if i == self.neuron_number-1:
                dx[i] = (-self.x[i] - self.weights[i] * self.y[0] + self.u[i] - self.beta*self.z[i]) * self.dt/self.tau_r
            else:
                dx[i] = (-self.x[i] - self.weights[i] * self.y[i+1] + self.u[i] - self.beta*self.z[i]) * self.dt/self.tau_r
        for i in range(self.neuron_number):
            self.x[i] += dx[i]

        # Update outputs
        self.y = np.maximum(0.0, self.x)
        action = torch.tensor(self.y, dtype=torch.float32, device="cuda")

        action = torch.clamp(action, self.action_space.low[0], self.action_space.high[0])

        self.y = action.cpu().numpy()

        # Update adaptation variables
        dz = [((math.pow(self.y[i], 1) - self.z[i]) * self.dt / self.tau_a) for i in range(self.neuron_number)]
        for i in range(self.neuron_number):
            self.z[i] += dz[i]

        right_output = action[0]
        left_output = action[1]  # Opposite Phase
        return right_output, left_output

    def run(self, steps=1000, tau_r_seq=None, weights_seq=None, beta_seq=None):
        """
        Method implemented to be used by itself.
        :param steps:
        :param tau_r_seq:
        :param weights_seq:
        :param beta_seq:
        :return:
        """
        y_output = np.zeros((steps, self.neuron_number))
        for i in range(steps):
            tau_r = tau_r_seq[i] if tau_r_seq is not None else None
            weights = weights_seq[i] if weights_seq is not None else None
            beta = beta_seq[i] if beta_seq is not None else None
            self.step(tau_r=tau_r, weights=weights, beta=beta)
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

    def __init__(self, num_oscillators, nn_model, action_dim, action_space, neuron_number=2, tau_r=2.0, tau_a=12.0, dt=0.1):
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
            TODO
            update the state of the oscillators based on the sensory input and the
            neural network output. Returns the control actions as a torch tensor.
        """

        self.oscillators = [MatsuokaOscillator(action_space=action_space, neuron_number=neuron_number, tau_r=tau_r, tau_a=tau_a, dt=dt) for _ in
                            range(num_oscillators)]
        self.num_oscillators = num_oscillators
        self.nn_model = nn_model
        self.neuron_number = neuron_number
        self.action_space = action_dim

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

        nn_output = self.nn_model(sensory_input)

        tau_r_seq = nn_output[:, 0]
        weights_seq = nn_output[:, 1:1 + self.oscillators[0].neuron_number]
        beta_seq = nn_output[:, 2]

        output_actions = torch.zeros(self.action_space, dtype=torch.float32, device="cuda")

        for i in range(self.num_oscillators):
            right_output, left_output = self.oscillators[i].step(tau_r=tau_r_seq[i], weights=weights_seq[i],
                                                                 beta=beta_seq[i])
            output_actions[0:3] = right_output
            output_actions[3:6] = left_output
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
                y_outputs[t, i, :] = self.oscillators[i].y
        return y_outputs
