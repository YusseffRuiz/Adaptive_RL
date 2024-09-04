import numpy as np
import math
import torch
from .matsuoka_actor import MatsuokaActor


class MatsuokaNetwork:
    def __init__(self, num_oscillators, action_space=None, neuron_number=2, tau_r=1.0, tau_a=12.0, weights=None, u=None,
                 beta=2.5, dt=0.01):
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
        self.oscillators = [
            MatsuokaOscillator(neuron_number=neuron_number, tau_r=tau_r, tau_a=tau_a, weights=weights, u=u, beta=beta,
                               dt=dt, action_space=action_space) for _ in range(num_oscillators)]
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
    def __init__(self, action_space, num_oscillators, neuron_number=2, tau_r=1.0, tau_a=12.0, weights=None, u=None,
                 beta=2.5, dt=0.01):
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
        #self.action_space = action_space
        self.action_dim = action_space
        self.neuron_number = neuron_number
        self.param_dim = neuron_number * num_oscillators
        self.tau_r = tau_r
        self.tau_a = tau_a
        self.beta = beta
        self.dt = dt
        x = torch.arange(0, num_oscillators * neuron_number, 1, dtype=torch.float32).to(self.device)
        self.x = x.view(num_oscillators, neuron_number)
        # Neuron initial membrane potential
        self.y = torch.zeros((num_oscillators, neuron_number), dtype=torch.float32, device=self.device)
        # Output, it is the neurons update, which is mapped via NN to the action space.
        self.z = torch.zeros((num_oscillators, neuron_number), dtype=torch.float32,
                             device=self.device)  # Correction value

        if weights is None:
            self.weights = torch.ones(neuron_number, dtype=torch.float32, device=self.device)
        else:
            assert len(weights) == neuron_number, \
                "Weights must be a square matrix with size equal to the number of neurons."
            self.weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # Initialize external input (ones by default)
        if u is None:
            self.u = torch.full((num_oscillators, neuron_number), 2.5, dtype=torch.float32).to(self.device)
        else:
            assert len(u) == neuron_number, "Input array u - (fire rate) must match the number of neurons."
            self.u = torch.tensor(u, dtype=torch.float32, device=self.device)

    def step(self, weights=None, num_oscillators=1):
        """
               Perform a single update step for the Matsuoka oscillator network.

               Parameters:
               - tau_r (float): Optional. Time constant for neuron recovery.
               - weights (array): Optional. Weight matrix for neuron interactions.
               - beta (float): Optional. Adaptation coefficient.
               """
        # Update parameters if provided

        if weights is not None:
            self.weights = weights
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        output = weights[self.neuron_number:]
        weights = weights[:self.neuron_number]
        # Reshape params if necessary
        if weights.dim() > 2:
            # Shape comes in shape [N, K, paramsDim]
            assert weights.size(2) == self.param_dim, \
                    "Weights must be a matrix with size equal to the number of neurons."
            batch_size = weights.size(0) * weights.size(1)
            oscillator_number = num_oscillators
            params_input = weights.reshape(batch_size, oscillator_number, self.neuron_number)
            # Update membrane potentials
            # Store previous output
            # Modify original oscillator values to match batch_size
            local_x = self.x.unsqueeze(0).repeat(batch_size, 1, 1)
            local_y = self.y.unsqueeze(0).repeat(batch_size, 1, 1)
            local_z = self.z.unsqueeze(0).repeat(batch_size, 1, 1)

            y_prev = torch.roll(local_y, shifts=1, dims=1)
            y_prev[:, 0, :] = local_y[:, 0, :]

            dx = ((-local_x - params_input * y_prev + self.u.unsqueeze(0).repeat(batch_size, 1, 1) -
                   self.beta * local_z) *
                  self.dt / self.tau_r)

            local_x += dx

            # Update membrane potentials and output
            #local_y = torch.clamp(torch.relu(local_x), min=self.action_space.low[0],
            #                      max=self.action_space.high[0])

            # Update adaptation variables
            dz = (local_y - local_z) * self.dt / self.tau_a
            local_z += dz

            # Each oscillator control each joint. 1st - thigh, 2nd - knee, 3rd - ankle
            # Generalized mapping using PyTorch's advanced indexing and reshaping
            osc_indices = torch.arange(num_oscillators).repeat_interleave(self.neuron_number)  # [0, 0, 1, 1, 2, 2, ...]
            neuron_indices = torch.arange(self.neuron_number).repeat(num_oscillators)  # [0, 1, 0, 1, 0, 1, ...]

            # Use advanced indexing to fill the output tensor
            output_tensor = local_y[:, osc_indices, neuron_indices]

            right_output = output_tensor[:, :self.action_dim//2]
            left_output = output_tensor[:, self.action_dim//2:]
            self.x = local_x[0, :, :]
            self.y = local_y[0, :, :]
            self.z = local_z[0, :, :]

            # right_output = local_y[:, :, :self.param_dim // 2].reshape(batch_size, -1) # output for NN, not used
            # left_output = local_y[:, :, self.param_dim // 2:].reshape(batch_size, -1) # output for NN, not used
        else:
            assert len(weights) == self.param_dim, \
                    "Weights must be a matrix with size equal to the number of neurons, right now is {len(weights)}."
            weights = weights.reshape(num_oscillators, self.neuron_number)
            y_prev = torch.roll(self.y, shifts=1, dims=1)
            y_prev[0] = self.y[0]

            dx = (-self.x - weights * y_prev + self.u - self.beta * self.z) * self.dt / self.tau_r

            self.x += dx

            # Update membrane potentials and output
            #self.y = torch.clamp(self.x, min=self.action_space.low[0],
            #                     max=self.action_space.high[0])

            # Update adaptation variables
            dz = (self.y - self.z) * self.dt / self.tau_a
            self.z += dz

            # Generalized mapping using PyTorch's advanced indexing and reshaping
            osc_indices = torch.arange(num_oscillators).repeat_interleave(self.neuron_number)  # [0, 0, 1, 1, 2, 2, ...]
            neuron_indices = torch.arange(self.neuron_number).repeat(num_oscillators)  # [0, 1, 0, 1, 0, 1, ...]

            # Use advanced indexing to fill the output tensor
            output_tensor = torch.cat((self.y[osc_indices, neuron_indices], output), dim=-1)

            right_output = output_tensor[:self.action_dim // 2]
            left_output = output_tensor[self.action_dim // 2:]
            # right_output = self.y[:, :self.param_dim // 2].reshape(-1) # Output for NN, not used
            # left_output = self.y[:, self.param_dim // 2:].reshape(-1)  # Output for NN, not used

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

    def __init__(self, num_oscillators, env, n_envs=1, neuron_number=2,
                 tau_r=1.0, tau_a=6.0, dt=0.1):
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
        #self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_oscillators = num_oscillators
        self.observation_space = env[0]#env.observation_space
        self.action_space = None#env.action_space
        self.neuron_number = neuron_number
        self.action_dim = env[1]#env.action_space.shape[0]
        self.n_envs = n_envs
        self.parameters_dimension = self.num_oscillators * self.neuron_number
        self.oscillators = MatsuokaOscillator(action_space=self.action_dim, num_oscillators=num_oscillators,
                                              neuron_number=neuron_number, tau_r=tau_r,
                                              tau_a=tau_a, dt=dt)
        #self.nn_model = MatsuokaActor(env, neuron_number=self.neuron_number, num_oscillators=self.num_oscillators).to(
        #    self.device)

    def step(self, params_input):
        """
                Takes a single step in the simulation, updating the state of each oscillator based on the sensory input and
                the neural network output.

                Parameters:
                -----------
                sensory_input : torch.Tensor
                    The input to the neural network, which influences the oscillator parameters.
                    Instead of the input neural network, in this case we are receiving the optimized parameters of.
                    the MPO trainer.

                Returns:
                --------
                output_actions : torch.Tensor
                    A tensor representing the control actions for the environment, with a dimension matching the action space.
                """

        # nn_output = self.nn_model.input_forward(sensory_input)  # Output is a matrix size [oscillators, neurons]

        # Original inputs are in the shape [B, sample, num_oscillators*neuron_number
        # Rearrange inputs in the form num_oscillators*neuron_numer
        right_output, left_output = self.oscillators.step(weights=params_input, num_oscillators=self.num_oscillators)

        # Combine the right and left outputs as needed
        output_actions = torch.cat((right_output, left_output), dim=-1)
        # output_actions = self.nn_model.output_neuron(output_y) # Possible not working
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
            y_outputs[t, :, :] = self.oscillators.y
        return y_outputs
