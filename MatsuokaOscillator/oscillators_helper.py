import torch
import numpy as np


W_osc = 0.5
W_drl = 1-W_osc


class PIDController:
    def __init__(self, Kp, Ki, Kd, dt, margin=0.1):
        """
        Initialize the PID controller with specified parameters.

        Parameters:
        - Kp, Ki, Kd: PID gains.
        - dt: Time step size.
        - margin: Acceptable phase error margin.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.prev_error = 0.0
        self.integral = 0.0
        self.margin = margin

    def step(self, error):
        """
        Calculate the control signal using PID control logic.

        Parameters:
        - error: The phase error (difference between actual and desired phase).

        Returns:
        - control_signal: The computed control signal to minimize phase error.
        """
        # Ensure error is either a tensor or a scalar
        is_tensor = torch.is_tensor(error)

        # If prev_error and integral aren't tensors but error is, convert them
        if is_tensor:
            self.prev_error = torch.tensor(self.prev_error, dtype=error.dtype, device=error.device)
            self.integral = torch.tensor(self.integral, dtype=error.dtype, device=error.device)
            self.margin = torch.tensor(self.margin, dtype=error.dtype, device=error.device)
            zero = torch.tensor(0.0, dtype=error.dtype, device=error.device)
        else:
            margin = self.margin
            zero = 0.0

        # Check if the error is within the margin
        if (torch.abs(error) if is_tensor else abs(error)) <= self.margin:
            return zero  # No correction needed within margin

        # Proportional term
        P = self.Kp * error

        # Integral term (accumulating error over time)
        self.integral += error * self.dt
        I = self.Ki * self.integral

        # Derivative term (rate of change of the error)
        derivative = (error - self.prev_error) / self.dt
        D = self.Kd * derivative

        # Update the previous error for the next time step
        self.prev_error = error

        # Calculate the control signal
        control_signal = P + I + D

        return control_signal


def env_selection(action_dim, weights, device, output=None):
    """
    Used to determine automatically which environment we are working now
    :param action_dim: action dimension for the different enviroments
    :param weights: weights comming from the DRL algorithm
    :param device: device to run the algorithm on
    :param output: None when weights are received, otherwise the output of the CPG
    :return: either weights if weights are received or the output of the CPG
    """
    if action_dim == 6:
        cpg_values = weight_conversion_walker(weights, device, output=output)
    elif action_dim == 17:
        cpg_values = weight_conversion_humanoid(weights, device, output=output)
    elif action_dim == 70:
        cpg_values = weight_conversion_myoleg(weights, device, output=output)
    elif action_dim == 7:
        cpg_values = weight_conversion_ant(weights, device, output=output)
    else:
        print("Not an implemented environment")
        return None

    return cpg_values


def weight_conversion_ant(weights, device, output=None):
    if output is None:
        weights_tmp = torch.tensor([weights[0], weights[6], weights[4], weights[2]], dtype=torch.float32,
                                   device=device)
        return weights_tmp
    else:
        output_tensor = weights
        output_tensor[0] = output[0, 0]
        output_tensor[2] = output[1, 1]
        output_tensor[4] = output[1, 0]
        output_tensor[6] = output[0, 1]
        return output_tensor


def weight_conversion_walker(weights, device, output=None):
    if output is None:
        weights_tmp = torch.tensor([weights[2], weights[5]], dtype=torch.float32, device=device)
        return weights_tmp
    else:
        # first = output[0,0].item()
        # third = output[0,1].item()
        second = output[0].item()
        fourth = output[1].item()
        output_tensor = np.array(
            [weights[0], weights[1], second,
             weights[3], weights[4], fourth],
            )
        return output_tensor


def weight_conversion_humanoid(weights, device, output=None):
    if output is None:
        weights_tmp = torch.tensor([weights[5], weights[9]], dtype=torch.float32, device=device)
        return weights_tmp
    else:
        output_tensor = weights
        output_tensor[5] = output[0, 0].item()
        output_tensor[9] = output[0, 1].item()
        # output_tensor[6] = output[1, 0].item()
        # output_tensor[10] = output[1, 1].item()
        return output_tensor


# Define muscle groups and their corresponding neurons/oscillators
MUSCLE_GROUP_MYOSIM = {
    'quadriceps_right': [21, 26, 27, 28],  # recfem_r, vasint_r, vaslat_r, vasmed_r
    'hamstrings_right': [6, 7, 23, 24],  # semimem_r, semiten_r
    'hip_flexors_right': [18, 20, 22],  # iliacus_r, psoas_r, sart_r
    'hip_extensors_right': [0, 1, 8, 9, 10],  # addbrev_r, addlong_r, glmax1_r, glmax2_r, glmax3_r
    'ankle_motor_right': [69],  #

    'quadriceps_left': [58, 66, 67, 68],  # recfem_l, vasint_l, vaslat_l, vasmed_l
    'hamstrings_left': [35, 36, 60, 61],  # bflh_l, bfsh_l, semimem_l, semiten_l
    'hip_flexors_left': [53, 57, 59],  # iliacus_l, psoas_l, sart_l
    'hip_extensors_left': [29, 30, 43, 44, 45],  # addbrev_l, addlong_l, glmax1_l, glmax2_l, glmax3_l
    'ankle_dorsiflexors_left': [37, 38, 64],  # edl, ehl, tib_ant (right dorsiflexors)
    'ankle_plantarflexors_left': [39, 40, 41, 42, 54, 55, 62, 65],  # fdl, fhl, gast_lat, gas_med, per_brev, per_long, soleus, tib_post (right plantarflexors)
}

def weight_conversion_myoleg(weights, device, output=None):
    """
    Map the weights or CPG output to specific muscle groups in the MyoLeg model.

    Parameters:
    - weights: The DRL-generated control weights (or external input).
    - device: Device on which the tensors are stored.
    - output: If CPG-generated output is available, map it to muscles instead of using weights.

    Returns:
    - Mapped values for the muscles in the model.
    """

    if output is None:
        weight_motor = weights[MUSCLE_GROUP_MYOSIM['ankle_motor_right']].item()
        # Ankle decision based on absolute value of weights
        dorsiflexor_strength = (weights[MUSCLE_GROUP_MYOSIM['ankle_dorsiflexors_left']]).sum()
        plantarflexor_strength = (weights[MUSCLE_GROUP_MYOSIM['ankle_plantarflexors_left']]).sum()

        if dorsiflexor_strength > plantarflexor_strength:
            weight_ankle = dorsiflexor_strength
        else:
            weight_ankle = -plantarflexor_strength
        weights_tmp = torch.tensor([weight_motor, weight_ankle], dtype=torch.float32, device=device)
        return weights_tmp
    else:

        # We assume weights or output has dimension 71 (for 70 muscles + 1 motor)
        # Map CPG output to muscle groups (for when the CPG outputs the control signals)
        output_tensor = weights  # Assuming weights is preallocated
        hip_flexor_value = max(output[0,0].item(), 0)  # Positive values for dorsiflexion.
        hip_extensor_value = max(-output[0,0].item(), 0)  # Negative values for hip extensor.
        # output_tensor[MUSCLE_GROUP_MYOSIM['quadriceps_right']] = weights_output_helper_myosim(output_tensor[MUSCLE_GROUP_MYOSIM['quadriceps_right']], output[1,0])
        # output_tensor[MUSCLE_GROUP_MYOSIM['hamstrings_right']] = weights_output_helper_myosim(output_tensor[MUSCLE_GROUP_MYOSIM['hamstrings_right']], output[1,0])
        output_tensor[MUSCLE_GROUP_MYOSIM['hip_flexors_right']] = weights_output_helper_myosim(output_tensor[MUSCLE_GROUP_MYOSIM['hip_flexors_right']], hip_flexor_value)
        output_tensor[MUSCLE_GROUP_MYOSIM['hip_extensors_right']] = weights_output_helper_myosim(output_tensor[MUSCLE_GROUP_MYOSIM['hip_extensors_right']], hip_extensor_value)
        output_tensor[MUSCLE_GROUP_MYOSIM['ankle_motor_right']] = weights_output_helper_myosim(output_tensor[MUSCLE_GROUP_MYOSIM['ankle_motor_right']]*2.88, output[1,0].item()*2.88)

        # Left ankle dorsiflexor and plantarflexor muscles
        hip_flexor_value = max(output[0,1].item(), 0)  # Positive values for dorsiflexion.
        hip_extensor_value = max(-output[0,1].item(), 0)  # Negative values for plantarflexion.
        dorsiflexor_value = max(output[1,1].item(), 0)  # Positive values for dorsiflexion.
        plantarflexor_value = max(-output[1,1].item(), 0)  # Negative values for plantarflexion.
        # output_tensor[MUSCLE_GROUP_MYOSIM['quadriceps_left']] = weights_output_helper_myosim(output_tensor[MUSCLE_GROUP_MYOSIM['quadriceps_left']], output[1,1])
        # output_tensor[MUSCLE_GROUP_MYOSIM['hamstrings_left']] = weights_output_helper_myosim(output_tensor[MUSCLE_GROUP_MYOSIM['hamstrings_left']], output[1,1])
        output_tensor[MUSCLE_GROUP_MYOSIM['hip_flexors_left']] = weights_output_helper_myosim(output_tensor[MUSCLE_GROUP_MYOSIM['hip_flexors_left']], hip_flexor_value)
        output_tensor[MUSCLE_GROUP_MYOSIM['hip_extensors_left']] = weights_output_helper_myosim(output_tensor[MUSCLE_GROUP_MYOSIM['hip_extensors_left']], hip_extensor_value)
        output_tensor[MUSCLE_GROUP_MYOSIM['ankle_dorsiflexors_left']] = weights_output_helper_myosim(output_tensor[MUSCLE_GROUP_MYOSIM['ankle_dorsiflexors_left']], dorsiflexor_value)
        output_tensor[MUSCLE_GROUP_MYOSIM['ankle_plantarflexors_left']] = weights_output_helper_myosim(output_tensor[MUSCLE_GROUP_MYOSIM['ankle_plantarflexors_left']], plantarflexor_value)
        return output_tensor


def weights_output_helper_myosim(weights, oscillator):
    return W_drl*weights + W_osc*oscillator

def leg_synchronize(data_left, data_right):
    pass


def scale_osc_weights(weights, old_min=-1.0, old_max=1.0, new_min=1.1, new_max=4.0):
    """
    Scale weights from one range to another.

    Parameters:
    - weights: Tensor or array of weights to scale.
    - old_min: Minimum value of the original range.
    - old_max: Maximum value of the original range.
    - new_min: Minimum value of the target range.
    - new_max: Maximum value of the target range.

    Returns:
    - scaled_weights: Tensor or array of scaled weights.
    """
    return new_min + (weights - old_min) * (new_max - new_min) / (old_max - old_min)
