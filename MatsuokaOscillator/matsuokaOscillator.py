import math
import numpy as np
import torch

from MatsuokaOscillator import oscillators_helper
from MatsuokaOscillator.hudgkin_huxley import HHNeuron


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
        outputs = torch.roll(outputs, shifts=1, dims=0)
        # Update each oscillator with the output of the previous one
        # Feedback mechanism
        adaptive_feedback_strength = self.feedback_strength*torch.sigmoid(torch.mean(outputs))
        feedback = adaptive_feedback_strength * torch.sum(outputs, dim=1, keepdim=True) - adaptive_feedback_strength * outputs
        self.oscillators.u += outputs*feedback
        # self.oscillators.y += feedback
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
                 beta=2.5, dt=2.0, isMuscular=False, device="cuda"):
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
        self.tau_r = torch.tensor(tau_r, device=self.device)
        self.tau_a = torch.tensor(tau_a, device=self.device)
        self.beta = beta
        self.dt = dt
        self.excitation_signal = 2.5
        self.escalated_number = 0.1
        self.isMuscular = isMuscular

        self.x = torch.arange(-self.param_dim/2, self.param_dim/2, 1, dtype=torch.float32).to(self.device)
        if isMuscular:
            self.activation_function = torch.tanh
            oscillators_helper.W_osc = 0.3
        else:
            self.activation_function = torch.tanh
            oscillators_helper.W_osc = 0.3

        oscillators_helper.W_drl = 1-oscillators_helper.W_osc
        self.w = oscillators_helper.W_drl  # Relevance of direct weights
        self.o = oscillators_helper.W_osc  # Relevance of Oscillator

        if num_oscillators > 1:
            self.x = self.x.view(num_oscillators, neuron_number)
            # Neuron initial membrane potential
            self.y = torch.randn((num_oscillators, neuron_number), dtype=torch.float32,
                                 device=self.device) * frequency
            # Output, it is the neurons update, which is mapped via NN to the action space.
            self.z = torch.ones((num_oscillators, neuron_number), dtype=torch.float32,
                                 device=self.device)*self.escalated_number  # Correction value
            signs = [1 if i % 2 == 0 else -1 for i in range(neuron_number)]
            self.sign_tensor = torch.tensor([signs] * self.num_oscillators,
                                            device=self.device)
            self.y *= self.sign_tensor
            if weights is None:
                self.weights = torch.ones((num_oscillators,neuron_number), dtype=torch.float32, device=self.device)
                self.weights *= self.excitation_signal
            else:
                assert len(self.weights) == self.param_dim, \
                    f"Weights must be a matrix with size equal to the number of neurons, right now is {self.weights.shape}."
                self.weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        else:
            self.y = torch.ones(neuron_number, dtype=torch.float32, device=self.device) * frequency * self.escalated_number
            self.z = torch.ones(neuron_number, dtype=torch.float32, device=self.device)*self.escalated_number
            signs = [1 if i % 2 == 0 else -1 for i in range(neuron_number)]
            self.sign_tensor = torch.tensor(signs, dtype=torch.float32, device=self.device)
            self.y *= self.sign_tensor

            if weights is None:
                self.weights = torch.ones(neuron_number, dtype=torch.float32, device=self.device)
                self.weights *= self.excitation_signal
            else:
                assert len(weights) == neuron_number, \
                    "Weights must be a square matrix with size equal to the number of neurons."
                self.weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # Initialize external input (ones by default)
        if u is None:
            if num_oscillators > 1:
                self.u = torch.ones((num_oscillators,neuron_number), dtype=torch.float32, device=self.device)
                self.u *= self.excitation_signal
            else:
                self.u = torch.ones(neuron_number, dtype=torch.float32, device=self.device) * self.excitation_signal
        else:
            assert len(u) == neuron_number, "Input array u - (fire rate) must match the number of neurons."
            self.u = torch.tensor(u, dtype=torch.float32, device=self.device)
        self.output = self.amplitude * self.y

    def step(self, weights=None, weights_origin=None):
        """
          Perform a single update step for the Matsuoka oscillator network.

           Parameters:
           - weights (array): Optional. Weight matrix for neuron interactions.
           - weights_origin:
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
        if weights_origin is not None:
            if not torch.is_tensor(weights_origin):
                weights_origin = torch.tensor(weights_origin, dtype=torch.float32, device=self.device)
            weights_origin = weights_origin
        else:
            weights_origin = torch.zeros_like(weights_tmp)

        if self.num_oscillators > 1:
            weights_tmp = weights_tmp.reshape(self.num_oscillators, self.neuron_number)
            y_prev = torch.roll(self.y, shifts=1, dims=1)
        else:
            assert len(weights_tmp) == self.param_dim, \
                f"Weights must be a matrix with size equal to the number of neurons, right now is {self.weights.shape}."
            y_prev = torch.roll(self.y, shifts=1, dims=0)
        dx = (-self.x - weights_tmp * y_prev + self.u - self.beta * self.z) * self.dt / self.tau_r
        self.x += dx
        # torch.clamp(self.x, min=-self.amplitude, max=self.amplitude)

        # Update adaptation variables
        dz = (self.y - self.z) * self.dt / self.tau_a
        self.z += dz
        torch.clamp(self.z, min=-10, max=10)
        tmp_output = self.activation_function(self.x - torch.mean(self.x))

        if self.isMuscular:
            tmp_output[1] = tmp_output[1] * 2.44  ## Experimento 2 con 2.88
            self.output = tmp_output
            self.y = torch.clamp(self.output, min=-2.88, max=2.88)
        else:
            self.output = self.amplitude * tmp_output
            self.y = torch.clamp(self.output, min=-self.amplitude, max=self.amplitude)

        if self.w != 0: # Just to update
            self.y = self.o * self.output + self.w * weights_origin
        else:
            self.y = tmp_output



        return self.output

    def run(self, steps=1000, weights_seq=None):
        """
        Method implemented to be used by itself. Showing the behaviour of the oscillators
        :param steps: Steps to run the environment
        :param weights_seq: Weights applied into the oscillators.
        :return: output of the oscillators function, with the number of neurons
        """

        y_output = torch.zeros(steps, self.neuron_number, dtype=torch.float32, device=self.device)
        y = weights_seq[0]
        for i in range(steps):
            weights = weights_seq[i] if weights_seq is not None else None
            y = self.step(weights=weights)
            y_output[i, :] = y
        return y_output

    def apply_control_signal(self, control_signal):
        self.y[0] += control_signal


    def reset_oscillator(self):
        self.x = torch.arange(-self.param_dim / 2, self.param_dim / 2, 1, dtype=torch.float32).to(self.device)
        self.phase *= 0
        self.z = self.z*0 + self.escalated_number
        self.weights  = self.weights*0 + self.excitation_signal
        self.u = self.u*0 + self.excitation_signal
        if self.num_oscillators > 1:
            self.x = self.x.view(self.num_oscillators, self.neuron_number)
            self.y = torch.randn((self.num_oscillators, self.neuron_number), dtype=torch.float32,
                                 device=self.device) * self.frequency
            self.y *= self.sign_tensor
        else:
            self.y = torch.ones(self.neuron_number, dtype=torch.float32, device=self.device) * self.frequency * self.escalated_number
            self.y *= self.sign_tensor


    @staticmethod
    def value_to_phase(output):
        # Define target range in radians
        min_angle = -20  # -π/4
        max_angle = 90  # π/2
        # Map `output` from [-1, 1] to [-45, 90]
        return -output * ((max_angle - min_angle) / 2) + ((max_angle + min_angle) / 2)


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
        self.feedback_strength = 0.3
        self.parameters_dimension = self.num_oscillators * self.neuron_number
        self.min_value = np.min(min_value)
        self.max_value = np.max(max_value)
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

        self.oscillator, self.oscillator_2 = self.initialize_oscillator(
            action_dim=self.action_dim,neuron_number=neuron_number, num_oscillators=num_oscillators,
            amplitude=amplitude, tau_r=tau_r, tau_a=tau_a, hh=hh, isMuscular=self.isMuscular, device=self.device)

        self.osc_output = self.oscillator.output
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

    def step(self, params_input, modifiers, feed_u=None):
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
        y_prev, feedback_y = self.oscillators_feedback(self.oscillator.y)
        self.oscillator.u = self.update_u(feed_u, feedback_y, y_prev)

        weights_input = oscillators_helper.env_selection(weights=params_input, action_dim=self.action_dim,
                                                         device=self.device)
        oscillators_output = self.oscillator.step(weights=oscillators_input, weights_origin=weights_input)


        self.osc_output = oscillators_output
        # print(self.osc_output)

        output_actions = oscillators_helper.env_selection(weights=params_input, action_dim=self.action_dim, output=oscillators_output,
                                       device=self.device)
        tmp_value = output_actions[-1]
        output_actions = np.clip(output_actions, a_min=self.min_value, a_max=self.max_value)  #Except Ankle Motor
        if self.isMuscular:  # Ankle motor exception
            output_actions[-1] = tmp_value
            output_actions = np.clip(output_actions, a_min=-2.88, a_max=2.88)
        # Assert to check for NaNs in output_actions
        assert not np.any(np.isnan(output_actions)), f"Error: output_actions contains NaN values! {output_actions}"

        return output_actions


    @staticmethod
    def initialize_oscillator(action_dim, initial_phase=0.0, amplitude=1.0,
                              neuron_number=2, num_oscillators=None, tau_r=16.0, tau_a=48.0, hh=False, isMuscular=False,
                              device='cpu'):
        """ Initialize multiple oscillators with different frequencies.
        Must be modified and improved to match higher size Action spaces, multiple oscillators with different muscle groups.
        """
        OscillatorClass = HHMatsuokaOscillator if hh else MatsuokaOscillator

        if action_dim is None:
            oscillators = OscillatorClass(num_oscillators=num_oscillators,
                                                  initial_phase=initial_phase,
                                                  amplitude=amplitude,
                                                  neuron_number=neuron_number, tau_r=tau_r,
                                                  tau_a=tau_a, device=device)
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
            oscillators = OscillatorClass(num_oscillators=num_oscillators,
                                          initial_phase=initial_phase, frequency=1.0,
                                          amplitude=amplitude,
                                          neuron_number=neuron_number, tau_r=[[tau_r],[tau_r/2]],
                                          tau_a=[[tau_a],[tau_a/2]], isMuscular=isMuscular, device=device)
            # oscillators.w = 0 # Making sure, it outputs literally the value of oscillators
            # oscillators.o = 1
            return oscillators, None

        else:  # Regular same size oscillator
            frequency_right = 0.5
            frequency_left = 1.5

            oscillators = OscillatorClass(num_oscillators=num_oscillators,
                                                  initial_phase=initial_phase, frequency=frequency_right,
                                                  amplitude=amplitude,
                                                  neuron_number=neuron_number, tau_r=tau_r,
                                                  tau_a=tau_a, device=device)

            return oscillators, None

    def oscillators_feedback(self,outputs_prev):
        """
        Function to produce the feedback into the oscillators, between each other
        """
        outputs = torch.roll(outputs_prev, shifts=1, dims=0)
        # Update each oscillator with the output of the previous one
        adaptive_feedback_strength = self.feedback_strength * torch.sigmoid(torch.mean(outputs))
        if self.num_oscillators > 1:
            feedback = adaptive_feedback_strength * torch.sum(outputs, dim=1,
                                                          keepdim=True) - adaptive_feedback_strength * outputs
        else:
            feedback = adaptive_feedback_strength * torch.sum(outputs) - adaptive_feedback_strength * outputs
        return outputs, feedback

    def update_u(self, u_value, feed_y=None, y_prev=None):
        if feed_y is None:
            feed_y = 0
        if self.num_oscillators>1:
            return (torch.tensor(u_value.reshape(self.num_oscillators, self.neuron_number), device=self.device) +
                    feed_y*y_prev)
        else:
            return torch.tensor(u_value, device=self.device) + feed_y*y_prev

    def reset(self):
        self.oscillator.reset_oscillator()

    def get_osc_states(self):
        numpy_output = self.osc_output.cpu().numpy()
        # Flatten if it is not already 1D
        if numpy_output.ndim > 1:
            numpy_output = numpy_output.flatten()
        return numpy_output


class HHMatsuokaOscillator(MatsuokaOscillator):
    def __init__(self, num_oscillators, amplitude=1.0, frequency=1.0, initial_phase=0.0, neuron_number=2, tau_r=16.0,
                 tau_a=48.0, weights=None, u=None, beta=2.5, dt=0.5, isMuscular=False, device='cuda'):
        super(HHMatsuokaOscillator, self).__init__(num_oscillators, amplitude, frequency, initial_phase,
                                                         neuron_number, tau_r, tau_a, weights, u, beta, isMuscular, device)

        self.neurons = HHNeuron()
        # print(self.neurons)
        self.v = self.init_output_tensor(neuron_number, num_oscillators, -67.0, 10, device=self.device)
        # Initialize external input (ones by default)
        if u is None:
            if num_oscillators > 1:
                self.u = torch.ones((num_oscillators, neuron_number), dtype=torch.float32, device=self.device)
                self.u *= self.excitation_signal
            else:
                self.u = torch.ones(neuron_number, dtype=torch.float32, device=self.device) * self.excitation_signal
        else:
            assert len(u) == neuron_number, "Input array u - (fire rate) must match the number of neurons."
            self.u = torch.tensor(u, dtype=torch.float32, device=self.device)
        self.output = self.amplitude * self.y

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


    def step(self, weights=None, weights_origin=None, right_form=False):
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
        else:
            weights_tmp = weights_tmp.reshape(self.num_oscillators, self.neuron_number)

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