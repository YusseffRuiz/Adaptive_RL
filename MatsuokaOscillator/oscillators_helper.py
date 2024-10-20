import torch
import numpy as np


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
        # Check if the error is within the margin
        if abs(error) <= self.margin:
            return 0.0  # No correction needed within margin

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
        weights_tmp = torch.tensor([weights[0], weights[1], weights[3], weights[4]], dtype=torch.float32, device=device)
        return weights_tmp
    else:
        output_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        output_tensor[0] = output[0, 0]
        output_tensor[1] = output[0, 1]
        output_tensor[3] = output[1, 0]
        output_tensor[4] = output[1, 1]

        return output_tensor


def weight_conversion_humanoid(weights, device, output=None):
    if output is None:
        weights_tmp = torch.tensor([weights[5], weights[6], weights[9], weights[10]], dtype=torch.float32, device=device)
        return weights_tmp
    else:
        output_tensor = weights
        output_tensor[5] = output[0, 0]
        output_tensor[6] = output[0, 1]
        output_tensor[9] = output[1, 0]
        output_tensor[10] = output[1, 1]
        return output_tensor


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
    # Define muscle groups and their corresponding neurons/oscillators
    muscle_groups = {
        'quadriceps_right': [21, 36, 27, 28],  # recfem_r, vasint_r, vaslat_r, vasmed_r
        'hamstrings_right': [23, 24],  # semimem_r, semiten_r
        'hip_flexors_right': [18, 20, 22],  # iliacus_r, psoas_r, sart_r
        'hip_extensors_right': [0, 1, 8, 9, 10],  # addbrev_r, addlong_r, glmax1_r, glmax2_r, glmax3_r
        'ankle_motor_right': [69],  #

        'quadriceps_left': [58, 66, 67, 68],  # recfem_l, vasint_l, vaslat_l, vasmed_l
        'hamstrings_left': [35, 36, 60, 61],  # bflh_l, bfsh_l, semimem_l, semiten_l
        'hip_flexors_left': [53, 57, 59],  # iliacus_l, psoas_l, sart_l
        'hip_extensors_left': [29, 30, 43, 44, 45],  # addbrev_l, addlong_l, glmax1_l, glmax2_l, glmax3_l
        'ankle_dorsiflexors_left': [37, 38, 64],  # edl_l, ehl_l, fdl_l (right dorsiflexors)
        'ankle_plantarflexors_left': [39, 40, 41, 42, 62, 65],  # fhl_r, soleus_r (right plantarflexors)
    }

    # We assume weights or output has dimension 71 (for 70 muscles + 1 motor)
    if output is None:
        # Map weights to muscle groups
        quadriceps_weights_right = weights[muscle_groups['quadriceps_right']]
        hamstrings_weights_right = weights[muscle_groups['hamstrings_right']]
        hip_flexors_weights_right = weights[muscle_groups['hip_flexors_right']]
        hip_extensors_weights_right = weights[muscle_groups['hip_extensors_right']]
        motor_weight_right = weights[muscle_groups['ankle_motor_right']]

        quadriceps_weights_left = weights[muscle_groups['quadriceps_left']]
        hamstrings_weights_left = weights[muscle_groups['hamstrings_left']]
        hip_flexors_weights_left = weights[muscle_groups['hip_flexors_left']]
        hip_extensors_weights_left = weights[muscle_groups['hip_extensors_left']]
        dorsiflexors_weights_left = weights[muscle_groups['ankle_dorsiflexors_left']]
        plantarflexors_weights_left = weights[muscle_groups['ankle_plantarflexors_left']]

        # Combine all weights into one tensor (or split them for separate control)
        combined_weights = torch.cat([
            quadriceps_weights_right, hamstrings_weights_right, hip_flexors_weights_right,
            hip_extensors_weights_right, motor_weight_right,
            quadriceps_weights_left, hamstrings_weights_left, hip_flexors_weights_left,
            hip_extensors_weights_left, dorsiflexors_weights_left, plantarflexors_weights_left
        ]).to(device)

        return combined_weights

    else:
        # Map CPG output to muscle groups (for when the CPG outputs the control signals)
        output_tensor = weights  # Assuming weights is preallocated

        output_tensor[muscle_groups['quadriceps_right']] = output[muscle_groups['quadriceps_right']]
        output_tensor[muscle_groups['hamstrings_right']] = output[muscle_groups['hamstrings_right']]
        output_tensor[muscle_groups['hip_flexors_right']] = output[muscle_groups['hip_flexors_right']]
        output_tensor[muscle_groups['hip_extensors_right']] = output[muscle_groups['hip_extensors_right']]
        output_tensor[muscle_groups['ankle_motor_right']] = output[muscle_groups['ankle_motor_right']]

        output_tensor[muscle_groups['quadriceps_left']] = output[muscle_groups['quadriceps_left']]
        output_tensor[muscle_groups['hamstrings_left']] = output[muscle_groups['hamstrings_left']]
        output_tensor[muscle_groups['hip_flexors_left']] = output[muscle_groups['hip_flexors_left']]
        output_tensor[muscle_groups['hip_extensors_left']] = output[muscle_groups['hip_extensors_left']]
        output_tensor[muscle_groups['ankle_dorsiflexors_left']] = output[muscle_groups['ankle_dorsiflexors_left']]
        output_tensor[muscle_groups['ankle_plantarflexors_left']] = output[muscle_groups['ankle_plantarflexors_left']]

    return output_tensor


def leg_synchronize(data_left, data_right):
    pass
