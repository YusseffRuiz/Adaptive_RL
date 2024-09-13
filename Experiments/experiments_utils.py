"""
Python wall designed to create the functions used to run the experiments.
Comparison of different DRL methods with and without CPG.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error


def evaluate_experiment(agent, env, alg, episodes_num=5, duration=1500):
    import numpy as np

    total_rewards = []
    total_joint_angles = []
    total_velocity_angles = []
    total_accelerations = []
    total_torques = []
    total_energy = []
    total_velocity = []
    dt = 5  # how many steps per dt
    obs, *_ = env.reset()
    previous_vel = 0
    traveled_distance = 0
    for episode in range(episodes_num):
        obs, *_ = env.reset()
        reward = 0
        ep_energy = []
        position = 0
        for step in range(duration):
            with torch.no_grad():
                if alg == "mpo":
                    action = agent.test_step(obs)
                elif alg == "sac":
                    action, *_ = agent.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()
            obs, rw, done, truncated, info = env.step(action)
            position, velocity, joint_angles, joint_velocity, torques, step_energy = get_data(info)
            if step % dt == 0:
                acceleration = get_acceleration(previous_vel, velocity, dt)
                total_accelerations.append(acceleration)
                previous_vel = velocity
            total_joint_angles.append(joint_angles)
            total_velocity_angles.append(joint_velocity)
            total_velocity.append(velocity)
            total_torques.append(torques)
            reward += rw
            ep_energy.append(step_energy)
        print(f"Episode {episode}: Average speed of: {np.mean(total_velocity)} m/s, with reward: {reward}")
        traveled_distance = position
        total_rewards.append(reward)
        total_energy.append(np.trapz(np.sum(ep_energy, axis=1), dx=1))
    average_reward = np.sum(total_rewards)/episodes_num
    velocity_total = np.mean(total_velocity)
    average_energy = (np.sum(total_energy)/episodes_num) / traveled_distance
    print(f"Average Reward over {episodes_num} episodes: {average_reward:.2f}")
    print(f"Average Speed over {episodes_num} episodes: {velocity_total:.2f} m/s with "
          f"total energy: {average_energy:.2f} Joules per meter")
    right_hip_joints, right_knee_joints, right_ankle_joints, left_hip_joints, left_knee_joints, left_ankle_joints = (
        separate_joints(total_joint_angles))
    right_hip_vels, right_knee_vels, right_ankle_vels, left_hip_vels, left_knee_vels, left_ankle_vels = (
        separate_joints(total_velocity_angles))
    right_hip_torque, right_knee_torque, right_ankle_torque, left_hip_torque, left_knee_torque, left_ankle_torque = (
        separate_joints(total_torques))
    #plot_data(right_hip_joints, y_axis_name="Angle (°/s)", title="Right Hip Joint movement")
    #plot_data(right_hip_vels, y_axis_name="Velocity (m/s)", title="Right Hip Velocity")
    #plot_data(right_hip_torque, y_axis_name="Torque (N/m)", title="Right Hip Torque")
    plot_data(total_velocity, y_axis_name="Velocity(m/s", title="Velocity (m/s)")
    cross_fourier_transform(np.array(right_hip_joints), np.array(left_hip_joints))
    perform_autocorrelation(np.array(right_hip_joints), np.array(left_hip_joints))
    jerk = get_jerk(total_accelerations, dt)
    plot_data(total_accelerations, data2=jerk, y_min_max=[-1, 1], x_axis_name="Time", y_axis_name="Acceleration (m/s^2)", title="jerk and acceleration (m/s^2)")
    plot_data(np.sum(ep_energy, axis=1), x_axis_name="Time", y_axis_name="Energy (Joules)", title="Energy Consumption over time")


def cross_fourier_transform(data1, data2, sampling_rate=1):
    # Perform FFT analysis for right and left hip angles
    freqs_right, magnitudes_right = perform_fourier_transform(data1, sampling_rate)
    freqs_left, magnitudes_left = perform_fourier_transform(data2, sampling_rate)

    # Pearson correlation coefficient  #

    # Extract the indices for the desired frequency range (0 to 0.1 Hz)
    # Frequency bins corresponding to the FFT output
    freq_range = (freqs_right >= 0) & (freqs_right <= 0.06)
    freqs = freqs_left[freq_range]

    # Filter the FFT magnitude spectra for the 0-2 Hz range
    filtered_magnitude_spectrum1 = magnitudes_right[freq_range]
    filtered_magnitude_spectrum2 = magnitudes_left[freq_range]

    correlation_coefficient = np.corrcoef(filtered_magnitude_spectrum1, filtered_magnitude_spectrum2)[0, 1]

    plt.figure(figsize=(10, 6))
    plt.plot(freqs, filtered_magnitude_spectrum1, label='Right Hip')
    plt.plot(freqs, filtered_magnitude_spectrum2, label='Left Hip')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(f'Fourier Transform of Hip Angles with Pearson coeff: {correlation_coefficient:.2f}')
    plt.legend()
    plt.show()


def perform_fourier_transform(data, sampling_rate=1, plot=False):
    """
    Perform a Fast Fourier Transform (FFT) on the given data to analyze the frequency content.

    Parameters:
    - data: numpy array of time-series data (e.g., joint angles or velocities).
    - sampling_rate: Sampling rate of the data collection (default is 1).

    Returns:
    - freqs: Frequencies corresponding to the FFT results.
    - magnitudes: Magnitudes of the FFT results.
    """
    n = len(data)
    fft_result = np.fft.fft(data)
    magnitudes = np.abs(fft_result[:n // 2])  # Only take the positive frequencies
    freqs = np.fft.fftfreq(n, d=sampling_rate)[:n // 2]
    if plot:
        # Plot the FFT results
        plt.figure(figsize=(10, 6))
        plt.plot(freqs, magnitudes)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('Fourier Transform of Joint Angles')
        plt.show()
    return freqs, magnitudes


def perform_autocorrelation(data1, data2):
    """
    Perform cross-correlation between two joint data sequences to analyze synchronization.

    Parameters:
    - joint_data1: numpy array of time-series data for joint 1 (e.g., right leg).
    - joint_data2: numpy array of time-series data for joint 2 (e.g., left leg).

    Returns:
    - lags: Time lags used in the cross-correlation.
    - cross_corr: Cross-correlation values.
    """
    cross_corr = np.correlate(data1 - np.mean(data1), data2 - np.mean(data2), mode='full')
    cross_corr /= np.max(cross_corr)  # Normalize
    lags = np.arange(-len(data1) + 1, len(data1))

    # Assuming lags and cross_corr are computed from the cross-correlation function
    peak_correlation = np.max(cross_corr)
    lag_of_peak = lags[np.argmax(cross_corr)]

    # Calculate RMSE between right and left hip angles
    rmse = np.sqrt(mean_squared_error(data1, data2))

    print(f"Peak Cross-Correlation Value: {peak_correlation:.2f}")
    print(f"Lag of Peak Cross-Correlation: {lag_of_peak}")
    # Plot the autocorrelation results
    plt.figure(figsize=(10, 6))
    plt.plot(lags, cross_corr)
    plt.xlabel('Lag')
    plt.ylabel('Cross-Correlation')
    plt.title(f'Cross-Correlation between Right and Left Hip Angles, RMSE: {rmse}')
    plt.show()


def get_data(info):
    """
    Get data from info dictionary.
    :param info: is separated into what the environment is giving:
    "x_position": position moving to the right in m
    "x_velocity": Velocity of the agent in m/s
    "joint_angles": Angles of the joint
    | 1   | angle of the thigh joint in rad
    | 2   | angle of the leg joint in rad
    | 3   | angle of the foot joint in rad
    | 4   | angle of the left thigh joint in rad
    | 5   | angle of the left leg joint in rad
    | 6   | angle of the left foot joint in rad
    "joint_velocities": vel of the joints in rad/s
    | 1   | thigh joint in rad/s
    | 2   | leg joint in rad/s
    | 3   | foot joint in rad/s
    | 4   | left thigh joint in rad/s
    | 5   | left leg joint in rad/s
    | 6   | left foot joint in rad/s
    "torques": in the range -1 to 1 in N/m
    | 1   | Torque of the thigh joint
    | 2   | Torque of the leg joint
    | 3   | Torque of the foot joint
    | 4   | Torque of the left thigh
    | 5   | Torque of the left leg joint
    | 6   | Torque of the left foot joint
    "total_energy": np.sum(action) in N/m
    :return: specific values
    """
    position = info["x_position"]
    velocity = info["x_velocity"]
    joint_angles = info["joint_angles"] * 180/math.pi
    joint_velocity = info["joint_velocities"] * 180 / math.pi
    torques = info["torques"]
    step_energy = torques*info["joint_velocities"]

    return position, velocity, joint_angles, joint_velocity, torques, step_energy


def separate_joints(joint_list):
    right_hip = [joint[0] for joint in joint_list]
    right_knee = [joint[1] for joint in joint_list]
    right_ankle = [joint[2] for joint in joint_list]
    left_hip = [joint[3] for joint in joint_list]
    left_knee = [joint[4] for joint in joint_list]
    left_ankle = [joint[5] for joint in joint_list]

    return right_hip, right_knee, right_ankle, left_hip, left_knee, left_ankle


def plot_data(data, data2=None, y_min_max=None,  x_data=None, y_axis_name="data", x_axis_name="time", title="data plot over time"):
    plt.figure()
    plt.title(title)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    if y_min_max is not None:
        plt.ylim(y_min_max[0], y_min_max[1])
    if data2 is not None:
        if x_data is not None:
            plt.plot(x_data, data)
            plt.plot(x_data, data2)
        else:
            plt.plot(data)
            plt.plot(data2)
    else:
        if x_data is not None:
            plt.plot(x_data, data)
        else:
            plt.plot(data)
    plt.show()


def get_acceleration(previous_velocity, velocity, dt):
    return (velocity - previous_velocity) / dt


def get_jerk(acceleration, dt):
    return np.diff(acceleration)/dt


def compare_velocity(velocity1, velocity2, dt):
    pass


def compare_jerk(jerk1, jerk2, dt):
    pass


def compare_motion(action_list):
    pass


def compare_energy_consumption(energy_1, energy_2, time):
    # Total energy consumption for each algorithm
    total_energy_rl = np.trapz(energy_1, time)  # Integrate over time
    total_energy_rl_cpg = np.trapz(energy_2, time)

    # Create a bar chart
    algorithms = ['RL', 'RL + CPG']
    total_energy = [total_energy_rl, total_energy_rl_cpg]

    plt.figure(figsize=(8, 5))
    plt.bar(algorithms, total_energy, color=['blue', 'green'])
    plt.xlabel('Algorithm')
    plt.ylabel('Total Energy Consumption (Joules)')
    plt.title('Total Energy Consumption Comparison')
    plt.show()
