import math
import numpy as np
import torch
from MatsuokaOscillator import oscillators_helper
from MatsuokaOscillator.hudgkin_huxley import HHNeuron
from concurrent.futures import ThreadPoolExecutor


class MatsuokaNetwork:
    def __init__(self, num_oscillators, neuron_number=2, tau_r=32.0, tau_a=96.0, weights=None, u=None,
                 beta=2.5, dt=2.0, amplitude=1.0):
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
        self.feedback_strength = 0.1
        self.oscillators = MatsuokaOscillator(neuron_number=neuron_number, tau_r=tau_r, tau_a=tau_a, weights=weights,
                                              u=u, beta=beta,
                                              dt=dt, num_oscillators=num_oscillators, amplitude=amplitude)

    def step(self, weights):
        """
        Perform a single step for all coupled oscillators.
        Each oscillator's output is fed as input to the next oscillator.
        """
        outputs = torch.clone(self.oscillators.y)
        outputs = torch.roll(outputs, shifts=1, dims=1)
        # Update each oscillator with the output of the previous one
        # Feedback mechanism
        adaptive_feedback_strength = 0.5*torch.sigmoid(torch.mean(outputs))
        feedback = adaptive_feedback_strength * torch.sum(outputs, dim=0, keepdim=True) - adaptive_feedback_strength * outputs
        # print(feedback)
        self.oscillators.u += feedback
        self.oscillators.u = 5*torch.sigmoid(self.oscillators.u)
        self.oscillators.y += feedback
        self.oscillators.step(weights=weights)


    def run(self, steps=1000, weights_seq=None):
        """
        Run the coupled oscillators for a given number of steps.

        Parameters:
        - steps (int): Number of simulation steps.

        Returns:
        - y_outputs (list of arrays): Outputs of all oscillators over time.
        """
        y_outputs = [torch.zeros((steps, self.neuron_number)) for _ in range(self.num_oscillators)]

        for step_ in range(steps):
            weights = weights_seq[step_] if weights_seq is not None else None
            self.step(weights)
            for i in range(self.num_oscillators):
                y_outputs[i][step_, :] = self.oscillators.y[i]

        return y_outputs


class MatsuokaOscillator:
    def __init__(self, num_oscillators=1, amplitude=1.0, frequency=1.0, initial_phase=0.0, neuron_number=2, tau_r=1.0,
                 tau_a=6.0, weights=None, u=None,
                 beta=2.5, dt=2.0, device="cuda"):
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
        self.device = device
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = initial_phase
        self.neuron_number = neuron_number
        self.param_dim = neuron_number * num_oscillators
        self.num_oscillators = num_oscillators
        self.tau_r = tau_r
        self.or_tau_r = tau_r
        self.tau_a = tau_a
        self.or_tau_a = tau_a
        self.beta = beta
        self.dt = dt
        excitation_signal = 2.5
        escalated_number = 0.1
        self.x = torch.arange(-self.param_dim/2, self.param_dim/2, 1, dtype=torch.float32).to(self.device)

        if num_oscillators > 1:
            self.x = self.x.view(num_oscillators, neuron_number)
            # Neuron initial membrane potential
            oscillator_numbers = torch.arange(1, num_oscillators + 1, dtype=torch.float32,
                                              device=self.device).unsqueeze(1)
            self.y = torch.randn((num_oscillators, neuron_number), dtype=torch.float32,
                                 device=self.device)* frequency
            # Output, it is the neurons update, which is mapped via NN to the action space.
            self.z = torch.ones((num_oscillators, neuron_number), dtype=torch.float32,
                                 device=self.device)*escalated_number  # Correction value
            signs = [1 if i % 2 == 0 else -1 for i in range(neuron_number)]
            self.sign_tensor = torch.tensor([signs] * self.num_oscillators,
                                            device=self.device)
            self.y *= self.sign_tensor
            if weights is None:
                self.weights = torch.ones((num_oscillators,neuron_number), dtype=torch.float32, device=self.device)
                self.weights *= excitation_signal
            else:
                assert len(self.weights) == self.param_dim, \
                    f"Weights must be a matrix with size equal to the number of neurons, right now is {self.weights.shape}."
                self.weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        else:
            self.y = torch.ones(neuron_number, dtype=torch.float32, device=self.device) * frequency * escalated_number
            self.z = torch.ones(neuron_number, dtype=torch.float32, device=self.device)*escalated_number
            signs = [1 if i % 2 == 0 else -1 for i in range(neuron_number)]
            self.sign_tensor = torch.tensor(signs, dtype=torch.float32, device=self.device)
            self.y *= self.sign_tensor

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
        if weights_tmp.device != self.device:
            weights_tmp = weights_tmp.to(dtype=torch.float32, device=self.device)

        if self.num_oscillators > 1:
            assert weights_tmp.shape[0]*weights_tmp.shape[1] == self.param_dim, \
                f"Weights must be a matrix with size equal to the number of neurons, right now is {len(weights_tmp)}."
            weights_tmp = weights_tmp.reshape(self.num_oscillators, self.neuron_number)
            y_prev = torch.roll(self.y, shifts=1, dims=1)
        else:
            assert len(weights_tmp) == self.param_dim, \
                f"Weights must be a matrix with size equal to the number of neurons, right now is {self.weights.shape}."
            y_prev = torch.roll(self.y, shifts=1)
        dx = (-self.x - weights_tmp * y_prev + self.u - self.beta * self.z) * self.dt / self.tau_r

        self.x += dx
        torch.clamp(self.x, min=-self.amplitude, max=self.amplitude)

        # Update adaptation variables
        dz = (self.y - self.z) * self.dt / self.tau_a
        self.z += dz
        torch.clamp(self.z, min=-10, max=10)

        self.y = torch.tanh(self.x- torch.mean(self.x))
        # self.phase = self.value_to_phase(self.y)
        return self.amplitude*self.y

    def run(self, steps=1000, weights_seq=None, tau_seq=None):
        """
        Method implemented to be used by itself. Showing the behaviour of the oscillators
        :param steps: Steps to run the environment
        :param weights_seq: Weights applied into the oscillators.
        :param right_form: If the weights are in the correct scale, most of the time, False, only for testing purposes.
        :return: output of the oscillators funcion, with the number of neurons
        """

        y_output = torch.zeros(steps, self.neuron_number, dtype=torch.float32, device=self.device)

        for i in range(steps):
            if tau_seq is not None:
                self.tau_a = int(tau_seq[i][1]/6)
                self.tau_r = tau_seq[i][1]
            weights = weights_seq[i] if weights_seq is not None else None
            y = self.step(weights=weights)
            y_output[i, :] = y
        return y_output

    def apply_control_signal(self, control_signal):
        self.y[0] += control_signal

    def modify_tau(self, scaling_factor=1):
        """
        Function to modify tau values
        """
        self.tau_r = self.tau_a // 6
        self.tau_a = self.or_tau_a * scaling_factor


    @staticmethod
    def value_to_phase(output):
        # Define target range in radians
        min_angle = -math.pi / 4  # -π/4
        max_angle = math.pi / 2  # π/2
        # Map `output` from [-1, 1] to [-45, 90]
        return output * ((max_angle - min_angle) / 2) + ((max_angle + min_angle) / 2)


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
                 tau_r=2.0, tau_a=12.0, max_value=1.5, min_value=0.0, hh=False, error_margin=10):
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
        amplitude : float, optional
        The amplitude of the oscillators (default is 1.75).
        hh : bool, optional
        if we are using HH neurons
        error_margin : float, optional
        Error in percentage for the PID controller.
        """
        # self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_oscillators = num_oscillators
        self.neuron_number = neuron_number
        self.action_dim = da
        self.feedback_strength = 0.5
        self.parameters_dimension = self.num_oscillators * self.neuron_number
        self.min_value = torch.tensor(min_value, dtype=torch.float32, device=self.device)
        self.max_value = torch.tensor(max_value, dtype=torch.float32, device=self.device)
        amplitude = np.max(max_value)
        error_margin = (error_margin/100)*math.pi # convert the percentage to the actual cycle

        if self.action_dim != 70:
            self.desired_phase_difference = math.pi
            self.phase_error = 0.0
            self.phase_1 = 0.0
            self.phase_2 = 0.0
            self.control_signal = 0.0
            self.isMuscular = False
        else:
            self.desired_phase_difference = torch.ones(3, dtype=torch.float).to(self.device) * math.pi
            self.phase_error = torch.zeros(3, dtype=torch.float).to(self.device)
            self.isMuscular = True
        self.pid_controller = oscillators_helper.PIDController(Kp=1.0, Ki=0.05, Kd=0.01, dt=0.01, margin=error_margin)
        """ You can create similar characteristics oscillators or different"""

        self.oscillator_right, self.oscillator_left = self.initialize_oscillator(
            action_dim=self.action_dim,neuron_number=neuron_number, num_oscillators=num_oscillators,
            amplitude=amplitude, tau_r=tau_r, tau_a=tau_a, hh=hh)
        self.characteristics = {
            "num_oscillators": num_oscillators,
            "neuron_number": neuron_number,
            "tau_r": tau_r,
            "tau_a": tau_a,
            "amplitude": amplitude,
            "hh neuron": hh
        }

        # self.nn_model = MatsuokaActor(env, neuron_number=self.neuron_number, num_oscillators=self.num_oscillators).to(
        #    self.device)

    def print_characteristics(self):
        return self.characteristics

    def step(self, params_input, modifiers):
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

        oscillators_input = oscillators_helper.scale_osc_weights(modifiers)
        if self.action_dim is None:
            output_actions = self.oscillator_right.step(weights=oscillators_input)
            return output_actions.cpu().numpy()

        if self.isMuscular:
            with ThreadPoolExecutor(max_workers=12) as executor:
                # Right leg oscillators
                futures = [
                    executor.submit(self.execute_step, self.oscillator_right['quadriceps_right'], oscillators_input[0]),
                    executor.submit(self.execute_step, self.oscillator_right['hamstrings_right'], oscillators_input[1]),
                    executor.submit(self.execute_step, self.oscillator_right['hip_flexors_right'],
                                    oscillators_input[2]),
                    executor.submit(self.execute_step, self.oscillator_right['hip_extensors_right'],
                                    oscillators_input[3]),
                    executor.submit(self.execute_step, self.oscillator_right['ankle_motor_right'],
                                    oscillators_input[4]),
                    executor.submit(self.execute_step, self.oscillator_left['quadriceps_left'], oscillators_input[5]),
                    executor.submit(self.execute_step, self.oscillator_left['hamstrings_left'], oscillators_input[6]),
                    executor.submit(self.execute_step, self.oscillator_left['hip_flexors_left'], oscillators_input[7]),
                    executor.submit(self.execute_step, self.oscillator_left['hip_extensors_left'],
                                    oscillators_input[8]),
                    executor.submit(self.execute_step, self.oscillator_left['ankle_dorsiflexors_left'],
                                    oscillators_input[9]),
                    executor.submit(self.execute_step, self.oscillator_left['ankle_plantarflexors_left'],
                                    oscillators_input[10])]
                # Wait for all threads to complete and collect results
                oscillators_output = [future.result() for future in futures]

            # actual_phase_difference = torch.zeros(3, dtype=torch.float, device=self.device)
            # actual_phase_difference[0] = self.oscillator_right['hip_flexors'].phase - self.oscillator_left[
            #     'hip_flexors'].phase
            # actual_phase_difference[1] = self.oscillator_right['ankle_dorsiflexors'].phase - self.oscillator_left[
            #     'ankle_dorsiflexors'].phase
            # actual_phase_difference[2] = self.oscillator_right['ankle_plantarflexors'].phase - self.oscillator_left[
            #     'ankle_plantarflexors'].phase
            oscillator_left = None


        else:
            oscillator_right = self.oscillator_right.step(weights=oscillators_input[0:self.neuron_number])
            feedback = self.oscillators_feedback(oscillator_right)
            # print(feedback)
            self.oscillator_left.u = 5 * torch.sigmoid(self.oscillator_left.u + feedback)
            self.oscillator_left.y += feedback
            oscillator_left = self.oscillator_left.step(weights=oscillators_input[self.neuron_number:])
            feedback = self.oscillators_feedback(oscillator_left)
            self.oscillator_right.u = 5 * torch.sigmoid(self.oscillator_right.u + feedback)
            self.oscillator_right.y += feedback
            # oscillator_right = self.oscillator_right.step(oscillators_input)
            # oscillators_output = oscillator_right
            # oscillator_left = self.oscillator_left.step()
            oscillators_output = torch.stack([oscillator_right, oscillator_left], dim=0)
            # self.hip_oscillator.step(weights=[params_input[0], params_input[3]])

        # Outputs assignation
        # print("hip: ", hip_output)
        # [self.phase_1, self.phase_2] = hip_output
        # print("knees: ", knee_output)
        output_actions = oscillators_helper.env_selection(weights=params_input, action_dim=self.action_dim, output=oscillators_output,
                                       device=self.device)
        output_actions = torch.clamp(output_actions, min=self.min_value, max=self.max_value)

        return output_actions.cpu().numpy()

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
            sensory_input = sensory_input_seq[t] if sensory_input_seq is not None else np.ones(self.num_oscillators)*1.5
            y_outputs[t, :, :] = self.step(sensory_input)
        return y_outputs

    @staticmethod
    def initialize_oscillator(action_dim, initial_phase=0.0, amplitude=1.0,
                              neuron_number=2, num_oscillators=None, tau_r=16.0, tau_a=48.0, hh=False):
        """ Initialize multiple oscillators with different frequencies.
        Must be modified and improved to match higher size Action spaces, multiple oscillators with different muscle groups.
        """
        OscillatorClass = HHMatsuokaOscillator if hh else MatsuokaOscillator

        if action_dim is None:
            oscillators = OscillatorClass(num_oscillators=num_oscillators,
                                                  initial_phase=initial_phase,
                                                  amplitude=amplitude,
                                                  neuron_number=neuron_number, tau_r=tau_r,
                                                  tau_a=tau_a)
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

            hip_frequency_1 = frequency_right['hip_flexors']
            hip_frequency_2 = frequency_right['hip_extensors']

            # Initialize the oscillators for the right leg
            right_oscillators = {
                'quadriceps_right': OscillatorClass(num_oscillators=1,
                                                 initial_phase=initial_phase, frequency=frequency_right['quadriceps'],
                                                 amplitude=amplitude, neuron_number=4,
                                                 tau_r=tau_r, tau_a=tau_a),
                'hamstrings_right': OscillatorClass(num_oscillators=1,
                                                 initial_phase=initial_phase, frequency=frequency_right['hamstrings'],
                                                 amplitude=amplitude, neuron_number=2,
                                                 tau_r=tau_r, tau_a=tau_a),
                'hip_flexors_right': OscillatorClass(num_oscillators=1,
                                                  initial_phase=initial_phase, frequency=frequency_right['hip_flexors'],
                                                  amplitude=amplitude, neuron_number=3,
                                                  tau_r=tau_r, tau_a=tau_a),
                'hip_extensors_right': OscillatorClass(num_oscillators=1,
                                                    initial_phase=initial_phase,
                                                    frequency=frequency_right['hip_extensors'],
                                                    amplitude=amplitude, neuron_number=5,
                                                    tau_r=tau_r, tau_a=tau_a),
                'ankle_motor_right': OscillatorClass(num_oscillators=1,
                                                  initial_phase=initial_phase, frequency=frequency_right['ankle_motor'],
                                                  amplitude=amplitude, neuron_number=1,
                                                  tau_r=tau_r, tau_a=tau_a)
            }

            # Initialize the oscillators for the left leg
            left_oscillators = {
                'quadriceps_left': OscillatorClass(num_oscillators=1,
                                                 initial_phase=initial_phase+math.pi, frequency=frequency_left['quadriceps'],
                                                 amplitude=amplitude, neuron_number=4,
                                                 tau_r=tau_r, tau_a=tau_a),
                'hamstrings_left': OscillatorClass(num_oscillators=1,
                                                 initial_phase=initial_phase+math.pi, frequency=frequency_left['hamstrings'],
                                                 amplitude=amplitude, neuron_number=4,
                                                 tau_r=tau_r, tau_a=tau_a),
                'hip_flexors_left': OscillatorClass(num_oscillators=1,
                                                  initial_phase=initial_phase+math.pi, frequency=frequency_left['hip_flexors'],
                                                  amplitude=amplitude, neuron_number=3,
                                                  tau_r=tau_r, tau_a=tau_a),
                'hip_extensors_left': OscillatorClass(num_oscillators=1,
                                                    initial_phase=initial_phase+math.pi,
                                                    frequency=frequency_left['hip_extensors'],
                                                    amplitude=amplitude, neuron_number=5,
                                                    tau_r=tau_r, tau_a=tau_a),
                'ankle_dorsiflexors_left': OscillatorClass(num_oscillators=1,
                                                         initial_phase=initial_phase+math.pi,
                                                         frequency=frequency_left['ankle_dorsiflexors'],
                                                         amplitude=amplitude, neuron_number=3,
                                                         tau_r=tau_r, tau_a=tau_a),
                'ankle_plantarflexors_left': OscillatorClass(num_oscillators=1,
                                                           initial_phase=initial_phase+math.pi,
                                                           frequency=frequency_left['ankle_plantarflexors'],
                                                           amplitude=amplitude, neuron_number=6,
                                                           tau_r=tau_r, tau_a=tau_a)
            }
            # right_oscillators = [
            #     right_oscillators['quadriceps_right'],
            #     right_oscillators['hamstrings_right'],
            #     right_oscillators['hip_flexors_right'],
            #     right_oscillators['hip_extensors_right'],
            #     right_oscillators['ankle_motor_right']
            # ]
            # left_oscillators = [
            #     left_oscillators['quadriceps_left'],
            #     left_oscillators['hamstrings_left'],
            #     left_oscillators['hip_flexors_left'],
            #     left_oscillators['hip_extensors_left'],
            #     left_oscillators['ankle_dorsiflexors_left'],
            #     left_oscillators['ankle_plantarflexors_left']
            # ]
            # hip_oscillator = [] * 2  # Oscillator Synchronization for hip flexors and extensors
            # hip_oscillator[0] = OscillatorClass(num_oscillators=1, frequency=0.9 * hip_frequency_1, amplitude=0.1,
            #                                  neuron_number=3)
            # hip_oscillator[1] = OscillatorClass(num_oscillators=1, frequency=0.9 * hip_frequency_2, amplitude=0.1,
            #                                     neuron_number=5)

            return right_oscillators, left_oscillators
        else:  # Regular same size oscillator
            frequency_right = 0.5
            frequency_left = 1.5
            hip_frequency = frequency_right

            right_oscillator = OscillatorClass(num_oscillators=1,
                                                  initial_phase=initial_phase, frequency=frequency_right,
                                                  amplitude=amplitude,
                                                  neuron_number=neuron_number, tau_r=tau_r,
                                                  tau_a=tau_a)
            left_oscillator = OscillatorClass(num_oscillators=1,
                                                 initial_phase=initial_phase+math.pi, frequency=frequency_left, amplitude=amplitude,
                                                 neuron_number=neuron_number, tau_r=tau_r,
                                                 tau_a=tau_a)

            return right_oscillator, left_oscillator

    def oscillators_feedback(self,outputs_prev):
        """
        Function to produce the feedback into the oscillators, between each other
        """
        outputs = torch.clone(outputs_prev)
        outputs = torch.roll(outputs, shifts=1)
        # Update each oscillator with the output of the previous one
        # Feedback mechanism
        adaptive_feedback_strength = self.feedback_strength * torch.sigmoid(torch.mean(outputs))
        feedback = adaptive_feedback_strength * torch.sum(outputs, dim=0,
                                                          keepdim=True) - adaptive_feedback_strength * outputs
        return feedback


class HHMatsuokaOscillator(MatsuokaOscillator):
    def __init__(self, num_oscillators, amplitude=1.0, frequency=1.0, initial_phase=0.0, neuron_number=2, tau_r=16.0,
                 tau_a=48.0, weights=None, u=None, beta=2.5, dt=0.5):
        super(HHMatsuokaOscillator, self).__init__(num_oscillators, amplitude, frequency, initial_phase,
                                                         neuron_number, tau_r, tau_a, weights, u, beta, dt)

        self.neurons = HHNeuron()
        # print(self.neurons)
        self.v = self.init_output_tensor(neuron_number, num_oscillators, -67.0, 10, device=self.device)
        self.u = torch.ones(neuron_number, dtype=torch.float32, device=self.device) * 2.0
        self.u = self.u.unsqueeze(0)

        self.m = 0.05
        self.h = 0.6
        self.n = 0.32
        # Stack parameters for parallelized neuron forward pass
        self.m_tensor = self.init_output_tensor(neuron_number=neuron_number, num_oscillators=num_oscillators,
                                                base_value=self.m, noise_level=0.03, device=self.device)
        self.h_tensor = self.init_output_tensor(neuron_number=neuron_number, num_oscillators=num_oscillators,
                                                base_value=self.h, noise_level=0.1, device=self.device)
        self.n_tensor = self.init_output_tensor(neuron_number=neuron_number, num_oscillators=num_oscillators,
                                                base_value=self.n, noise_level=0.05, device=self.device)


        self.decay_counter_m = 0
        self.decay_counter_h = 0
        self.decay_counter_n = 0


    def step(self, weights=None, right_form=False):
        """
            Perform a single update step using Hodgkin-Huxley neurons.
            """
        # Calculate total input for each neuron
        # self.cont+=1
        if weights is not None:
            # Verify input comes in tensor form
            if not torch.is_tensor(weights):
                self.weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        if not right_form:
            weights_tmp = oscillators_helper.scale_osc_weights(self.weights, new_min=-20, new_max=20)
        else:
            weights_tmp = self.weights

        # Adjust weights dimension if num_oscillators == 1
        if self.num_oscillators == 1 and self.weights.dim() == 1:
            weights_tmp = weights_tmp.unsqueeze(0)  # Shape becomes (1, neuron_number)

        # Compute I_ext
        I_ext = self.u + torch.einsum("ij,ij->ij", weights_tmp, (self.v/1000))
        # I_ext = 10*torch.tanh(I_ext)

        # Update all neurons in parallel
        dV, dm, dh, dn = self.neurons.forward(V=self.v, m=self.m_tensor, h=self.h_tensor, n=self.n_tensor, I_ext=I_ext)


        self.v += dV
        self.m_tensor += dm
        self.h_tensor += dh
        self.n_tensor += dn
        self.v = torch.clamp(self.v, -80.0, 60.0)
        self.m_tensor = torch.clamp(self.m_tensor, 0.01, 0.99)
        self.h_tensor = torch.clamp(self.h_tensor, 0.01, 0.99)
        self.n_tensor = torch.clamp(self.n_tensor, 0.01, 0.99)
        # print("I: ", I_ext, "m: ", self.m_tensor, "h: ", self.h_tensor, "n: ", self.n_tensor, "V: ", self.v)
        # Generate output
        self.y = 2 * (self.v - (-80)) / (60 - (-80)) - 1
        if self.num_oscillators == 1:
            self.y = self.y.squeeze(0)

        # Verify tensors are not on the extreme values
        self.m_tensor, self.decay_counter_m = self.check_decay(tensor=self.m_tensor, counter=self.decay_counter_m, original_value=self.m)
        self.h_tensor, self.decay_counter_h = self.check_decay(tensor=self.h_tensor, counter=self.decay_counter_h, original_value=self.h)
        self.n_tensor, self.decay_counter_n = self.check_decay(tensor=self.n_tensor, counter=self.decay_counter_n, original_value=self.n)

        return self.y

    @staticmethod
    def check_decay(tensor, counter, original_value):
        """
        :param tensor: tensor to be modified and verified
        :param counter: counter which determines the decay rate
        :param original_value: original value to return the tensor
        """
        # Check if any value in the tensor exceeds the thresholds
        if (tensor >= 0.98).any() or (tensor <= 0.02).any():
            counter += 1

        if counter >= 10:
            tensor.fill_(original_value)
            counter = 0
        return tensor, counter



    @staticmethod
    def init_output_tensor(neuron_number=2, num_oscillators=1, base_value=-65.0, noise_level=3.0, device="cpu"):
        """
        Create a tensor of shape (neuron_number,) with random values near a base value.

        Args:
        - neuron_number (int): Total number of neurons.
        - base_value (float): The central value.
        - noise_level (float): Maximum random deviation from the base value.
        - device (str): Device to create the tensor on ("cpu" or "cuda").

        Returns:
        - tensor (torch.Tensor): Tensor with random values near `base_value`.
        """
        random_offsets = (torch.rand(neuron_number, device=device) - 0.5) * 2 * noise_level
        return (torch.ones((num_oscillators, neuron_number), dtype=torch.float32, device=device) * base_value
                    + random_offsets)


class HHMatsuokaNetwork(MatsuokaNetwork):
    def __init__(self, num_oscillators=2, neuron_number=2, tau_r=16.0, tau_a=48.0, weights=None, u=None,
                 beta=2.5, dt=0.5, amplitude=1.0):
        super(HHMatsuokaNetwork, self).__init__(num_oscillators, neuron_number, tau_r, tau_a, weights, u, beta, dt, amplitude)
        self.oscillators = HHMatsuokaOscillator(neuron_number=neuron_number, tau_r=tau_r, tau_a=tau_a, weights=weights,
                                              u=u, beta=beta,
                                              dt=dt, num_oscillators=num_oscillators, amplitude=amplitude)