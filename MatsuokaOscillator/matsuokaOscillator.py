import math
import numpy as np
import torch
from MatsuokaOscillator import oscillators_helper
from MatsuokaOscillator.hudgkin_huxley import HHNeuron


class MatsuokaNetwork:
    def __init__(self, num_oscillators, neuron_number=2, tau_r=1.0, tau_a=12.0, weights=None, u=None,
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
        self.num_oscillators = num_oscillators
        self.oscillators = MatsuokaOscillator(neuron_number=neuron_number, tau_r=tau_r, tau_a=tau_a, weights=weights,
                                              u=u, beta=beta,
                                              dt=dt, num_oscillators=num_oscillators)

    def step(self):
        """
        Perform a single step for all coupled oscillators.
        Each oscillator's output is fed as input to the next oscillator.
        """
        # Update each oscillator with the output of the previous one
        self.oscillators.step()

    def run(self, steps=1000):
        """
        Run the coupled oscillators for a given number of steps.

        Parameters:
        - steps (int): Number of simulation steps.

        Returns:
        - y_outputs (list of arrays): Outputs of all oscillators over time.
        """
        y_outputs = [torch.zeros((steps, self.neuron_number)) for _ in range(self.num_oscillators)]

        for step_ in range(steps):
            self.step()
            for i in range(self.num_oscillators):
                y_outputs[i][step_, :] = self.oscillators.y[i]

        return y_outputs


class MatsuokaOscillator:
    def __init__(self, num_oscillators=1, amplitude=2.5, frequency=1.0, initial_phase=0.0, neuron_number=2, tau_r=1.0, tau_a=12.0, weights=None, u=None,
                 beta=2.5, dt=0.01):
        """
                Initialize the Matsuoka Oscillator.

                Parameters:
                - amplitude:
                - frequency:
                - initial_phase:
                - action_space (array): Action space of all oscillators. - to be removed
                - neuron_number (int): Number of neurons in the oscillator network.
                - num_oscillators (int): Number of oscillators to connect.
                - tau_r (float): Time constant for neuron recovery.
                - tau_a (float): Time constant for neuron adaptation.
                - weights (array): Weight matrix for neuron interactions
                    (default: None, initialized as neurons*osc matrix)
                - u (array): External input (firing rate) to the neurons (default: None, initialized as 2.5).
                - beta (float): Adaptation coefficient.
                - dt (float): Time step for integration.
                """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = initial_phase
        self.neuron_number = neuron_number
        self.param_dim = neuron_number * num_oscillators
        self.num_oscillators = num_oscillators
        self.tau_r = tau_r
        self.tau_a = tau_a
        self.beta = beta
        self.dt = dt
        escalated_number = 0.1
        excitation_signal = 2.5
        self.x = torch.arange(0, num_oscillators * neuron_number, 1, dtype=torch.float32).to(self.device)
        if num_oscillators > 1:
            self.x = self.x.view(num_oscillators, neuron_number)
            # Neuron initial membrane potential
            oscillator_numbers = torch.arange(1, num_oscillators + 1, dtype=torch.float32,
                                              device=self.device).unsqueeze(1)
            self.y = torch.ones((num_oscillators, neuron_number), dtype=torch.float32,
                                 device=self.device)*0.1 * oscillator_numbers * frequency
            # Output, it is the neurons update, which is mapped via NN to the action space.
            self.z = torch.zeros((num_oscillators, neuron_number), dtype=torch.float32,
                                 device=self.device)  # Correction value
            if weights is None:
                self.weights = torch.ones((num_oscillators,neuron_number), dtype=torch.float32, device=self.device)
                self.weights *= excitation_signal
            else:
                assert len(self.weights) == self.param_dim, \
                    f"Weights must be a matrix with size equal to the number of neurons, right now is {self.weights.shape}."
                self.weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        else:
            self.y = torch.randn(neuron_number, dtype=torch.float32, device=self.device) * escalated_number * frequency
            self.z = torch.zeros(neuron_number, dtype=torch.float32, device=self.device)

            if weights is None:
                self.weights = torch.ones(neuron_number, dtype=torch.float32, device=self.device)
                self.weights *= excitation_signal
            else:
                assert len(weights) == neuron_number, \
                    "Weights must be a square matrix with size equal to the number of neurons."
                self.weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # Initialize external input (ones by default)
        if u is None:
            if num_oscillators > 1:
                self.u = torch.ones((num_oscillators,neuron_number), dtype=torch.float32, device=self.device)
                self.u *= excitation_signal
            else:
                self.u = torch.ones(neuron_number, dtype=torch.float32, device=self.device) * excitation_signal
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
            # Verify input comes in tensor form
            if not torch.is_tensor(weights):
                weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
            self.weights = weights
        else:
            weights = self.weights

        weights_tmp = weights

        if self.num_oscillators > 1:
            assert len(weights_tmp) * len(weights_tmp[0]) == self.param_dim, \
                f"Weights must be a matrix with size equal to the number of neurons, right now is {len(weights_tmp)}."
            weights_tmp = weights_tmp.reshape(self.num_oscillators, self.neuron_number)
            y_prev = torch.roll(self.y, shifts=1, dims=1)
        else:
            assert len(weights_tmp) == self.param_dim, \
                f"Weights must be a matrix with size equal to the number of neurons, right now is {self.weights.shape}."
            y_prev = torch.roll(self.y, shifts=1)

        dx = (-self.x - weights_tmp * y_prev + self.amplitude*self.u - self.beta * self.z) * self.dt / self.tau_r

        self.x += dx

        # Update adaptation variables
        dz = (self.y - self.z) * self.dt / self.tau_a
        self.z += dz

        # Update phase and frequency
        self.phase += self.frequency * self.dt
        self.y = torch.relu(self.x)

        return self.y

    def run(self, steps=1000, weights_seq=None):
        """
        Method implemented to be used by itself. Showing the behaviour of the oscillators
        :param steps: Steps to run the environment
        :param weights_seq: Weights applied into the oscillators.
        :return: output of the oscillators funcion, with the number of neurons
        """
        y_output = torch.zeros(steps, self.neuron_number, dtype=torch.float32, device=self.device)

        for i in range(steps):
            weights = weights_seq[i] if weights_seq is not None else None
            self.step(weights=weights)
            y_output[i, :] = self.y
        return y_output

    def apply_control_signal(self, control_signal):
        self.y += control_signal
        return self.y


class MatsuokaNetworkWithNN:
    """
    A class that represents a network of Matsuoka oscillators controlled by a neural network , which can be DRL.

    Attributes:
    -----------
    num_oscillators : int
        The number of Matsuoka oscillators in the network, these are the number of joints to be controlled.
    neuron_number : int, optional
        The number of neurons in each Matsuoka oscillator (default is 2), number of limbs to be controlled.

    Methods:
    --------
    step(sensory_input):
        Takes a single step in the simulation, updating the state of each oscillator based on the sensory input.
         Returns the control actions as a torch tensor.

    run(steps=1000, sensory_input_seq=None):
        Runs the simulation for a given number of steps, returning the outputs of the oscillators over time.
    """

    def __init__(self, num_oscillators, da=None, neuron_number=2,
                 tau_r=1.0, tau_a=6.0, dt=0.1, amplitude=1.75, hh=False):
        """
        Initializes the MatsuokaNetworkWithNN with a specified number of oscillators and a neural network model.

        Parameters:
        -----------
        num_oscillators : int
            The number of Matsuoka oscillators in the network.
        neuron_number : int, optional
            The number of neurons in each Matsuoka oscillator (default is 2).
        tau_r : float, optional
            The time constant for the rise time of the oscillator neurons (default is 2.0).
        tau_a : float, optional
            The time constant for the adaptation time of the oscillator neurons (default is 12.0).
        dt : float, optional
            The time step for the simulation (default is 0.1).
        """
        # self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_oscillators = num_oscillators
        self.neuron_number = neuron_number
        self.action_dim = da
        self.parameters_dimension = self.num_oscillators * self.neuron_number
        error_margin = math.pi*0.05
        self.desired_phase_difference = 0.0
        self.phase_error = 0.0
        self.pid_controller = oscillators_helper.PIDController(Kp=1.0, Ki=0.05, Kd=0.01, dt=0.01, margin=error_margin)
        """ You can create similar characteristics oscillators or different"""

        self.oscillator_right, self.oscillator_left = self.initialize_oscillator(action_dim=self.action_dim,
                                                                                 neuron_number=neuron_number, num_oscillators=num_oscillators,
                                                                                 amplitude=amplitude,
                                                                                 tau_r=tau_r, tau_a=tau_a, dt=dt, hh=hh)

        # self.nn_model = MatsuokaActor(env, neuron_number=self.neuron_number, num_oscillators=self.num_oscillators).to(
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

        # oscillators_input = oscillators_helper.env_selection(weights=params_input, action_dim=self.action_dim, device=self.device)
        oscillators_input = params_input
        if self.action_dim is None:
            output_actions = self.oscillator_right.step(weights=oscillators_input)
            return output_actions

        if self.action_dim == 70:
            quadriceps_input = oscillators_input[0:4]  # Assuming 4 quadriceps-related muscles
            hamstrings_input = oscillators_input[4:6]
            hip_flexors_input = oscillators_input[6:9]
            hip_extensors_input = oscillators_input[9:14]
            dorsiflexors_input = oscillators_input[14:17]
            plantarflexors_input = oscillators_input[17:22]
            motor_input = oscillators_input[22]  # Motor actuator input for the right leg
            # Update oscillators for the right leg
            self.oscillator_right['quadriceps'].step(quadriceps_input)
            self.oscillator_right['hamstrings'].step(hamstrings_input)
            self.oscillator_right['hip_flexors'].step(hip_flexors_input)
            self.oscillator_right['hip_extensors'].step(hip_extensors_input)
            self.oscillator_right['ankle_motor'].step(motor_input)  # Control motor directly for right leg

            # Update oscillators for the left leg
            self.oscillator_left['quadriceps'].step(quadriceps_input)
            self.oscillator_left['hamstrings'].step(hamstrings_input)
            self.oscillator_left['hip_flexors'].step(hip_flexors_input)
            self.oscillator_left['hip_extensors'].step(hip_extensors_input)
            self.oscillator_left['ankle_dorsiflexors'].step(dorsiflexors_input)
            self.oscillator_left['ankle_plantarflexors'].step(plantarflexors_input)
        else:
            self.oscillator_right.step(weights=oscillators_input[0:self.neuron_number])
            self.oscillator_left.step(weights=oscillators_input[self.neuron_number:])

        actual_phase_difference = self.oscillator_right.phase - self.oscillator_left.phase
        error = self.desired_phase_difference - actual_phase_difference
        self.phase_error = error
        control_signal = self.pid_controller.step(error)

        # Phase synchronization with PID
        right_output = self.oscillator_right.apply_control_signal(-control_signal)
        left_output = self.oscillator_left.apply_control_signal(control_signal)

        oscillators_output = torch.stack([right_output, left_output], dim=0)
        output_actions = oscillators_helper.env_selection(weights=params_input, action_dim=self.action_dim, output=oscillators_output,
                                       device=self.device)

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
        y_outputs = torch.zeros(steps, self.num_oscillators, self.neuron_number, dtype=torch.float32, device=self.device)

        for t in range(steps):
            sensory_input = sensory_input_seq[t] if sensory_input_seq is not None else np.ones(self.num_oscillators)
            self.step(sensory_input)
            y_outputs[t, :, :] = self.oscillator_right.y
        return y_outputs

    def initialize_oscillator(self, action_dim, initial_phase=0.0, amplitude=1.5,
                              neuron_number=2, num_oscillators=None, tau_r=1.0, tau_a=6.0, dt=0.01, hh=False):
        """ Initialize multiple oscillators with different frequencies.
        Must be modified and improved to match higher size Action spaces, multiple oscillators with different muscle groups.
        """
        OscillatorClass = HHMatsuokaOscillator if hh else MatsuokaOscillator
        if action_dim is None:
            oscillators = OscillatorClass(num_oscillators=num_oscillators,
                                                  initial_phase=initial_phase,
                                                  amplitude=amplitude,
                                                  neuron_number=neuron_number, tau_r=tau_r,
                                                  tau_a=tau_a, dt=dt)
            return oscillators, None
        if action_dim == 70:  # Creation of different size oscillators
            # Define the frequencies for different muscle groups (example values, can be tuned)
            frequency_right = {
                'quadriceps': 1.0,
                'hamstrings': 0.8,
                'hip_flexors': 1.2,
                'hip_extensors': 1.1,
                'ankle_motor': 0.9  # This is for the prosthetic motor
            }

            frequency_left = {
                'quadriceps': 1.0,
                'hamstrings': 0.8,
                'hip_flexors': 1.2,
                'hip_extensors': 1.1,
                'ankle_dorsiflexors': 1.0,
                'ankle_plantarflexors': 1.0,
            }

            # Initialize the oscillators for the right leg
            right_oscillators = {
                'quadriceps': OscillatorClass(num_oscillators=1,
                                                 initial_phase=initial_phase, frequency=frequency_right['quadriceps'],
                                                 amplitude=amplitude, neuron_number=neuron_number,
                                                 tau_r=tau_r, tau_a=tau_a, dt=dt),
                'hamstrings': OscillatorClass(num_oscillators=1,
                                                 initial_phase=initial_phase, frequency=frequency_right['hamstrings'],
                                                 amplitude=amplitude, neuron_number=neuron_number,
                                                 tau_r=tau_r, tau_a=tau_a, dt=dt),
                'hip_flexors': OscillatorClass(num_oscillators=1,
                                                  initial_phase=initial_phase, frequency=frequency_right['hip_flexors'],
                                                  amplitude=amplitude, neuron_number=neuron_number,
                                                  tau_r=tau_r, tau_a=tau_a, dt=dt),
                'hip_extensors': OscillatorClass(num_oscillators=1,
                                                    initial_phase=initial_phase,
                                                    frequency=frequency_right['hip_extensors'],
                                                    amplitude=amplitude, neuron_number=neuron_number,
                                                    tau_r=tau_r, tau_a=tau_a, dt=dt),
                'ankle_motor': OscillatorClass(num_oscillators=1,
                                                  initial_phase=initial_phase, frequency=frequency_right['ankle_motor'],
                                                  amplitude=amplitude, neuron_number=neuron_number,
                                                  tau_r=tau_r, tau_a=tau_a, dt=dt)
            }

            # Initialize the oscillators for the left leg
            left_oscillators = {
                'quadriceps': OscillatorClass(num_oscillators=1,
                                                 initial_phase=initial_phase, frequency=frequency_left['quadriceps'],
                                                 amplitude=amplitude, neuron_number=neuron_number,
                                                 tau_r=tau_r, tau_a=tau_a, dt=dt),
                'hamstrings': OscillatorClass(num_oscillators=1,
                                                 initial_phase=initial_phase, frequency=frequency_left['hamstrings'],
                                                 amplitude=amplitude, neuron_number=neuron_number,
                                                 tau_r=tau_r, tau_a=tau_a, dt=dt),
                'hip_flexors': OscillatorClass(num_oscillators=1,
                                                  initial_phase=initial_phase, frequency=frequency_left['hip_flexors'],
                                                  amplitude=amplitude, neuron_number=neuron_number,
                                                  tau_r=tau_r, tau_a=tau_a, dt=dt),
                'hip_extensors': OscillatorClass(num_oscillators=1,
                                                    initial_phase=initial_phase,
                                                    frequency=frequency_left['hip_extensors'],
                                                    amplitude=amplitude, neuron_number=neuron_number,
                                                    tau_r=tau_r, tau_a=tau_a, dt=dt),
                'ankle_dorsiflexors': OscillatorClass(num_oscillators=1,
                                                         initial_phase=initial_phase,
                                                         frequency=frequency_left['ankle_dorsiflexors'],
                                                         amplitude=amplitude, neuron_number=neuron_number,
                                                         tau_r=tau_r, tau_a=tau_a, dt=dt),
                'ankle_plantarflexors': OscillatorClass(num_oscillators=1,
                                                           initial_phase=initial_phase,
                                                           frequency=frequency_left['ankle_plantarflexors'],
                                                           amplitude=amplitude, neuron_number=neuron_number,
                                                           tau_r=tau_r, tau_a=tau_a, dt=dt)
            }

            return right_oscillators, left_oscillators
        else:  # Regular same size oscillator
            frequency_right = 1.0
            frequency_left = 1.5
            right_oscillator = OscillatorClass(num_oscillators=1,
                                                  initial_phase=initial_phase, frequency=frequency_right,
                                                  amplitude=amplitude,
                                                  neuron_number=neuron_number, tau_r=tau_r,
                                                  tau_a=tau_a, dt=dt)
            left_oscillator = OscillatorClass(num_oscillators=1,
                                                 initial_phase=initial_phase, frequency=frequency_left, amplitude=amplitude,
                                                 neuron_number=neuron_number, tau_r=tau_r,
                                                 tau_a=tau_a, dt=dt)
        return right_oscillator, left_oscillator


class HHMatsuokaOscillator(MatsuokaOscillator):
    def __init__(self, num_oscillators, amplitude=2.5, frequency=1.0, initial_phase=0.0, neuron_number=2, tau_r=1.0,
                 tau_a=12.0, weights=None, u=None, beta=2.5, dt=0.01):
        super(HHMatsuokaOscillator, self).__init__(num_oscillators, amplitude, frequency, initial_phase,
                                                         neuron_number, tau_r, tau_a, weights, u, beta, dt)
        # Create an array of HH Neurons for each oscillator
        if num_oscillators==1:
            self.neurons = torch.nn.ModuleList([HHNeuron() for _ in range(neuron_number)])
        else:
            self.neurons = [[HHNeuron() for _ in range(neuron_number)] for _ in range(num_oscillators)]


    def step(self, weights=None):
        """
            Perform a single update step using Hodgkin-Huxley neurons.
            """
        m = 0.05
        h = 0.6
        n = 0.32
        if self.num_oscillators == 1:
            for i, neuron in enumerate(self.neurons):
                # Each neuron updates its membrane potential and other variables
                total_input = self.u[i] + torch.sum(self.weights[i] + self.y[i])  # Neurons influence each other
                dVdt, dmdt, dhdt, dndt = neuron.forward(V=self.y[i],m=m, h=h, n=n, I_ext=total_input, dt=self.dt)

                # Update the gating variables for each neuron
                if i+1 % 2 == 0:
                    self.y[i] = torch.sin(self.y[i] + dVdt*self.dt)
                else:
                    self.y[i] = torch.sin(-(self.y[i] + dVdt*self.dt))
                m += dmdt
                h += dhdt
                n += dndt
        else:
            for i in range(self.num_oscillators):
                for j in range(self.neuron_number):
                    # Each neuron updates its membrane potential and other variables
                    total_input = self.u[i][j] + torch.sum(self.weights[i][j] * self.y[i][j])  # Neurons influence each other
                    dVdt, dmdt, dhdt, dndt = self.neurons[i][j].forward(V=self.y[i, j], m=m, h=h,
                                                                    n=n, I_ext=total_input, dt=self.dt)

                    # Update the gating variables for each neuron
                    if i + 1 % 2 == 0:
                        self.y[i][j] = torch.sin(self.y[i][j] + dVdt * self.dt)
                    else:
                        self.y[i][j] = torch.sin(-(self.y[i][j] + dVdt * self.dt))
                    m = dmdt
                    h = dhdt
                    n = dndt
        # **Baseline correction step by step** (compute mean and subtract from the output)
        baseline_correction = torch.mean(self.y)  # Compute mean across neurons
        self.y -= baseline_correction  # Subtract baseline correction from outputs

        return self.y


class HHMatsuokaNetwork(MatsuokaNetwork):
    def __init__(self, num_oscillators=2, neuron_number=2, tau_r=1.0, tau_a=12.0, weights=None, u=None,
                 beta=2.5, dt=0.01):
        super(HHMatsuokaNetwork, self).__init__(num_oscillators, neuron_number, tau_r, tau_a, weights, u, beta, dt)
        self.oscillators = HHMatsuokaOscillator(neuron_number=neuron_number, tau_r=tau_r, tau_a=tau_a, weights=weights,
                                              u=u, beta=beta,
                                              dt=dt, num_oscillators=num_oscillators)