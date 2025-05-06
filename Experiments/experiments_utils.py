"""
Python wall designed to create the functions used to run the experiments.
Comparison of different DRL methods with and without CPG.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
import os
from scipy.signal import savgol_filter
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks

def get_name_environment(name, cpg_flag=False, algorithm=None, experiment_number=0, create=False, external_folder=None,
                         hh_neuron=False):
    """
    :param hh_neuron:
    :param external_folder: external folder where data will be saved
    :param algorithm: algorithm being used to create the required folder
    :param name: of the environment
    :param cpg_flag: either we are looking for the cpg env or not
    :param experiment_number: if required, we can create a subfolder
    :param create: If required, create new folder if it doesn't exist
    :return: env_name, save_folder, log_dir
    """
    env_name = name
    cpg = cpg_flag

    if cpg:
        algorithm = algorithm + '-CPG'
        if hh_neuron:
            algorithm = algorithm + '-HH'

    if external_folder:
        if create:
            if os.path.exists(f"{external_folder}"):
                print(f"Using existing folder: {external_folder}")
            if experiment_number > 0:
                save_folder = f"{external_folder}/{experiment_number}"
            else:
                save_folder = f"{external_folder}"
        else:
            if experiment_number > 0:
                save_folder = f"{external_folder}/{experiment_number}"
            else:
                save_folder = f"{external_folder}"
    else:
        if experiment_number > 0:
            save_folder = f"training/{env_name}-{algorithm}/{experiment_number}"
        else:
            save_folder = f"training/{env_name}-{algorithm}"
    log_dir = f"{save_folder}/logs/"
    if create:
        # Create log dir
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        print(f"Folders {save_folder} and {log_dir} created")
    return name, save_folder, log_dir


# Define function to search for trained algorithms in specified folders
def search_trained_algorithms(env_name, algorithms_list, save_folder="training", experiment_number=0):
    """
    The folder to look for the algorithms must be in the form: save_folder/environment_name-algorithm
    logs/checkpoints folder must be inside environment_name-algorithm
    """
    if save_folder is None:
        save_folder = "training"
    # List of algorithms to look for
    algorithms_found = []

    for algo in algorithms_list:
        # Check both the regular and CPG environment names
        if experiment_number > 0:
            possible_folders = [
                os.path.join(f"{save_folder}/{env_name}-{algo}", str(experiment_number)),
            ]
        else:
            possible_folders = [
                os.path.join(f"{save_folder}/{env_name}-{algo}"),
            ]
        for folder in possible_folders:
            if os.path.exists(folder):
                algorithms_found.append((algo, folder))
                print(f"Found folder for {algo} at {folder}")


    return algorithms_found


def evaluate_envs(model=None, model2=None, env=None, env2=None, algorithm="random", num_episodes=5, no_done=False, max_episode_steps=1000):
    total_rewards = []
    range_episodes = num_episodes
    mujoco_env = hasattr(env, "sim")
    muscle_flag = hasattr(env, "muscles_enable")
    for step in range(range_episodes):
        if mujoco_env:
            obs = env.reset()
            if env2:
                obs2 = env2.reset()
        else:
            obs = env.reset()[0]
        if muscle_flag:
            muscle_states = env.muscle_states
        done = False
        episode_reward = 0
        cnt = 0
        phases_1 = []
        phases_2 = []
        phases_3 = []
        phases_4 = []
        while not done:
            cnt += 1
            with torch.no_grad():
                if algorithm != "random":
                    if muscle_flag:
                        action = model.test_step(observations=obs, muscle_states=muscle_states, steps=step)
                    else:
                        action = model.test_step(observations=obs, steps=step)
                else:
                    action = env.action_space.sample()
                    # action = model.test_step(observations=obs, muscle_states=muscle_states, steps=step)
                    if env2:
                        action2 = model2.test_step(observations=obs2, muscle_states=muscle_states, steps=step)
            if len(action.shape) > 1:
                action = action[0, :]
            obs, reward, done, info, _ = env.step(action2)
            if env2:
                obs2, *_ = env2.step(action2)
            if muscle_flag:
                muscle_states = env.muscle_states
            # phase_1, phase_2, phase_3, phase_4 = env.get_osc_output()
            # public_obs = env.unwrapped.public_joints()
            # phases_1.append(phase_1)
            # phases_2.append(phase_2)
            # phases_3.append(phase_3)
            # phases_4.append(phase_4)
            # phases_1.append(public_obs[0,0]) # 0,0 is left Hip
            # phases_2.append(public_obs[0,1]) # 0,1 is right Hip
            # phases_1.append(public_obs[1,0]) # 1,0 is left Ankle
            # phases_2.append(public_obs[1,1]) # 1,1 is right Ankle
            # cntrl.append(control_signal)
            if mujoco_env:
                #Try rendering for MyoSuite
                # extras = env.extras
                env.sim.renderer.render_to_window()
                # env2.sim.renderer.render_to_window()
            episode_reward += reward
            if no_done:
                done = False
            if cnt >= max_episode_steps:
                done = True

        # plot_data(phases_1, phases_2, data1_name="hip_l", data2_name="hip_r", title="Hip Motion")
        # plot_data(phases_3, phases_4, data1_name="ankle_l", data2_name="ankle_r", title="Ankle Motion")
        # get_motion_pattern(phases_1, joint="Right Hip")
        total_rewards.append(episode_reward)
        print(f"Episode {step + 1}/{range_episodes}: Reward = {episode_reward}")
    average_reward = np.mean(total_rewards)
    env.close()
    if env2:
        env2.close()
    print(f"Average Reward over {range_episodes} episodes: {average_reward}")

def evaluate(model=None, env=None, algorithm="random", num_episodes=5, no_done=False, max_episode_steps=1000):
    total_rewards = []
    range_episodes = num_episodes
    mujoco_env = hasattr(env, "sim")
    muscle_flag = hasattr(env, "muscles_enable")
    for step in range(range_episodes):
        if mujoco_env:
            obs = env.reset()
        else:
            obs = env.reset()[0]
        if muscle_flag:
            muscle_states = env.muscle_states
        done = False
        episode_reward = 0
        cnt = 0
        phases_1 = []
        phases_2 = []
        phases_3 = []
        phases_4 = []
        while not done:
            cnt += 1
            with torch.no_grad():
                if algorithm != "random":
                    if muscle_flag:
                        action = model.test_step(observations=obs, muscle_states=muscle_states, steps=step)
                    else:
                        action = model.test_step(observations=obs, steps=step)
                else:
                    action = env.action_space.sample()
                    # action = model.test_step(observations=obs, muscle_states=muscle_states, steps=step)
            if len(action.shape) > 1:
                action = action[0, :]
            obs, reward, done, info, extras = env.step(action)
            position, velocity, joint_angles, joint_velocity, torques, step_energy, distance = get_data(extras, muscles=muscle_flag)
            if muscle_flag:
                muscle_states = env.muscle_states
            # phase_1, phase_2, phase_3, phase_4 = env.get_osc_output()
            # public_obs = env.unwrapped.public_joints()
            # phases_1.append(phase_1)
            # phases_2.append(phase_2)
            # phases_3.append(phase_3)
            # phases_4.append(phase_4)
            # phases_1.append(public_obs[0,0]) # 0,0 is left Hip
            # phases_2.append(public_obs[0,1]) # 0,1 is right Hip
            # phases_1.append(public_obs[1,0]) # 1,0 is left Ankle
            # phases_2.append(public_obs[1,1]) # 1,1 is right Ankle
            # cntrl.append(control_signal)
            if mujoco_env:
                #Try rendering for MyoSuite
                # extras = env.extras
                env.sim.renderer.render_to_window()
            episode_reward += reward
            if no_done:
                done = False
            if cnt >= max_episode_steps:
                done = True

        # plot_data(phases_1, phases_2, data1_name="hip_l", data2_name="hip_r", title="Hip Motion")
        # plot_data(phases_3, phases_4, data1_name="ankle_l", data2_name="ankle_r", title="Ankle Motion")
        # get_motion_pattern(phases_1, joint="Right Hip")
        total_rewards.append(episode_reward)
        print(f"Episode {step + 1}/{range_episodes}: Reward = {episode_reward}, distance = {distance}")
    average_reward = np.mean(total_rewards)
    env.close()
    print(f"Average Reward over {range_episodes} episodes: {average_reward}")


def evaluate_experiment(agent=None, env=None, alg="random", episodes_num=5, duration=1000, env_name=None, deterministic=False, cpg=False):
    save_folder = f"Experiments/{env_name}/images"
    action_dim = len(env.action_space.sample())
    if cpg:
        action_dim = env.da
    successful_episodes = 0
    max_attempts_per_episode = 50  # Maximum retries per episode
    min_reward = 1800
    min_distance = 3
    muscle_flag = hasattr(env, "muscle_states")
    if alg == "PPO" or alg == "PPO-CPG":
        min_reward = 100
    if action_dim == 70:
        min_reward = 10000
    total_rewards = []
    total_joint_angles = []
    total_velocity_angles = []
    total_accelerations = []
    total_torques = []
    total_energy = []
    total_velocity = []
    ending_steps = []
    total_error = []
    total_distance = []
    dt = 5  # Steps per dt
    ep_fall = 0

    while successful_episodes < episodes_num:
        attempts = 0
        success = False
        while not success and attempts < max_attempts_per_episode:
            attempts += 1
            # print(f"Starting episode {successful_episodes + 1}, attempt {attempts}")
            if alg == "random":
                if muscle_flag:
                    obs = env.reset()
                else:
                    obs = env.reset()[0]
            else:
                if muscle_flag:
                    obs = env.reset()
                    muscle_states = env.muscle_states
                else:
                    obs, *_ = env.reset()
            lstm_states = None
            reward = 0
            distance = 0
            ep_energy = []
            ep_velocity = []
            ep_acceleration = []
            ep_torques = []
            ep_joints = []
            ep_joint_velocities = []
            position = 0
            ending_step = 0
            fall = False
            assistance_value = 1  # Normalize when agent falls
            ep_error = []
            previous_vel = 0

            for step in range(duration):
                if alg != "random":
                    if muscle_flag:
                        action = agent.test_step(observations=obs, muscle_states=muscle_states, steps=step)
                    else:
                        action = agent.test_step(obs, steps=step)
                else:
                    action, lstm_states = agent.predict(
                        obs,
                        state=lstm_states,
                        episode_start=np.ones((env.num_envs,), dtype=bool),
                        deterministic=deterministic,
                    )
                if len(action.shape) > 1:
                    action = action[0, :]
                if alg == "random":
                    obs, rw, done, info = env.step(action)
                    position, velocity, joint_angles, joint_velocity, torques, step_energy, distance = get_data(info[0])
                else:
                    obs, rw, done, info, extras = env.step(action)
                    if muscle_flag:
                        muscle_states = env.muscle_states
                    position, velocity, joint_angles, joint_velocity, torques, step_energy, distance = get_data(extras, muscles=muscle_flag)

                if done and not fall:
                    ending_step = step
                    fall = True
                    assistance_value = 0

                if step % dt == 0:
                    acceleration = get_acceleration(previous_vel, velocity, dt)
                    ep_acceleration.append(acceleration * assistance_value)
                    previous_vel = velocity

                ep_joints.append(joint_angles * assistance_value)
                ep_joint_velocities.append(joint_velocity * assistance_value)
                ep_velocity.append(velocity * assistance_value)
                ep_torques.append(torques * assistance_value)
                reward += rw * assistance_value
                ep_energy.append(step_energy * assistance_value)

            # Check success criteria
            if distance >= min_distance:  # Successful episode
                success = True
                successful_episodes += 1
                total_distance.append(distance)
                ending_steps.append(ending_step)
                total_rewards.append(reward)
                total_energy.append(np.sum(ep_energy))
                total_velocity_angles.append(np.array(ep_joint_velocities))
                total_torques.append(np.array(ep_torques))
                total_joint_angles.append(np.array(ep_joints))
                total_velocity.append(np.array(ep_velocity))
                total_accelerations.append(np.array(ep_acceleration))
                total_error.append(np.array(ep_error))
                if fall:
                    ep_fall += 1
                print(f"Episode {successful_episodes} successful with reward {reward:.2f}")

        if not success:
            print(f"Failed to achieve success for episode {successful_episodes + 1} after {attempts} attempts.")
            break  # Avoid infinite loops if the agent cannot succeed

    # Convert all arrays to numpy for analysis
    total_distance = np.array(total_distance)
    total_rewards = np.array(total_rewards)
    total_velocity_angles = np.array(total_velocity_angles)
    total_torques = np.array(total_torques)
    total_joint_angles = np.array(total_joint_angles)
    total_energy = np.array(total_energy)
    total_energy_per_meter = np.array(get_energy_per_meter(total_energy, total_distance))
    total_velocity = np.array(total_velocity)
    total_accelerations = np.array(total_accelerations)
    total_error = np.array(total_error)

    average_reward = np.mean(total_rewards)
    velocity_total = np.mean(total_velocity)
    average_energy = (np.sum(np.trapz(total_energy, dx=1) / np.mean(total_distance)) / episodes_num)
    average_distance = np.mean(total_distance)
    if average_energy < 0: average_energy=0
    print(f"Average Reward over {episodes_num} episodes: {average_reward:.2f}")
    print(f"Average Speed and Distance over {episodes_num} episodes: {velocity_total:.2f} m/s with "
          f"total energy: {np.sum(total_energy_per_meter)/episodes_num:.2f} Joules per meter, travelled {average_distance:.2f} meters")
    joints = np.array(separate_joints(total_joint_angles, action_dim))
    # joints_vel = separate_joints(total_velocity_angles, action_dim)
    # joints_torque = separate_joints(total_torques, action_dim)
    results_dict = {
        'velocity': total_velocity,
        'reward': total_rewards,
        'distance': total_distance,
        'phase_error': total_error,
        'total_energy': total_energy,
        'total_energy_per_meter': total_energy_per_meter,
        'energy': average_energy,
        'joints': joints,
        'falls' : ep_fall,
    }

    return results_dict


def cross_fourier_transform(data1, data2, sampling_rate=1, joint="joint", save_folder=None):
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
    plt.plot(freqs, filtered_magnitude_spectrum1, label=f'Right {joint}')
    plt.plot(freqs, filtered_magnitude_spectrum2, label=f'Left {joint}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(f'Fourier Transform of Hip Angles with Pearson coeff: {correlation_coefficient:.2f}')
    plt.legend()
    if save_folder is not None:
        # Define the path where the image will be saved
        image_path = os.path.join(save_folder, "Fourier.png")
        plt.savefig(image_path)
    plt.show()


def perform_fourier_transform(data, sampling_rate=1, plot=False, save_folder=None):
    """
    Perform a Fast Fourier Transform (FFT) on the given data to analyze the frequency content.

    Parameters:
    - data: numpy array of time-series data (e.g., joint angles or velocities).
    - sampling_rate: Sampling rate of the data collection (default is 1).

    Returns:
    - freqs: Frequencies corresponding to the FFT results.
    - magnitudes: Magnitudes of the FFT results.
    """
    data = np.mean(data, axis=0)
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
        if save_folder is not None:
            # Define the path where the image will be saved
            image_path = os.path.join(save_folder, "Fourier.png")
            plt.savefig(image_path)
        plt.show()
    return freqs, magnitudes


def compute_frequency(signal, dt):
    """
    Compute the dominant frequency of a signal using FFT.

    Parameters:
    - signal: numpy array, the signal to analyze (1D array for a single neuron).
    - dt: float, time step between samples.

    Returns:
    - freq: float, the dominant frequency in Hz.
    """
    n = len(signal)  # Number of samples
    sample_rate = 1 / dt  # Sampling rate in Hz

    # Detrend the signal (remove mean)
    signal = signal - np.mean(signal)

    # Perform FFT
    fft_result = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(n, dt)  # Frequency bins

    # Take the magnitude of the FFT and ignore the negative frequencies
    fft_magnitude = np.abs(fft_result[:n // 2])
    fft_freq = fft_freq[:n // 2]

    # Find the dominant frequency
    dominant_index = np.argmax(fft_magnitude)
    dominant_frequency = fft_freq[dominant_index]

    return dominant_frequency


def analyze_neurons_frequencies(output_signals, dt):
    """
    Analyze the dominant frequencies of multiple neurons.

    Parameters:
    - output_signals: numpy array of shape (steps, neuron_number), signals for all neurons.
    - dt: float, time step between samples.

    Returns:
    - frequencies: list of dominant frequencies for each neuron.
    """
    neuron_number = output_signals.shape[1]
    frequencies = []
    period = []
    for i in range(neuron_number):
        freq = compute_frequency(output_signals[:, i], dt)
        frequencies.append(freq)
        period.append(1/freq)
    return frequencies, period


def perform_autocorrelation(data1, data2, joint="joint", save_folder=None):
    """
    Perform cross-correlation between two joint data sequences to analyze synchronization.

    Parameters:
    - joint_data1: numpy array of time-series data for joint 1 (e.g., right leg).
    - joint_data2: numpy array of time-series data for joint 2 (e.g., left leg).

    Returns:
    - lags: Time lags used in the cross-correlation.
    - cross_corr: Cross-correlation values.
    """
    data1 = np.mean(data1, axis=0)
    data2 = np.mean(data2, axis=0)
    data1 = cut_values_at_zero(data1)
    data2 = cut_values_at_zero(data2)
    assert data1.shape == data2.shape, "Data must be the same shape"

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
    plt.title(f'Cross-Correlation between Right and Left {joint} Angles, RMSE: {rmse:.2f}, Lag of Peak: {lag_of_peak}')
    if save_folder is not None:
        # Define the path where the image will be saved
        image_path = os.path.join(save_folder, "CrossCorrelation.png")
        plt.savefig(image_path)
        plt.show(block=False)
        plt.waitforbuttonpress()
        plt.close()
    else:
        plt.show(block=False)
        plt.waitforbuttonpress()
        plt.close()
    return lags, cross_corr

def get_data(info, muscles=False):
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
    position = info["position"]
    joint_angles = info["joint_angles"] * 180 / math.pi
    joint_velocity = info["joint_velocities"] * 180 / math.pi
    torques = info["torques"]
    distance = info["distance_from_origin"]
    if not muscles:
        velocity = info["x_velocity"]
        step_energy = torques*info["joint_velocities"]
    else:
        velocity = info["velocity"]
        step_energy = info["step_energy"]

    return position, velocity, joint_angles, joint_velocity, torques, step_energy, distance


def separate_joints(joint_list, action_dim):
    # Split the joints for each step across all episodes
    if action_dim == 6:  # Walker2d-v4
        right_hip = joint_list[:, :, 0]  # Extract right hip
        right_knee = joint_list[:, :, 1]  # Extract right knee
        right_ankle = joint_list[:, :, 2]  # Extract right ankle
        left_hip = joint_list[:, :, 3]  # Extract left hip
        left_knee = joint_list[:, :, 4]  # Extract left knee
        left_ankle = joint_list[:, :, 5]  # Extract left ankle

        return (np.array(right_hip), np.array(right_knee), np.array(right_ankle), np.array(left_hip),
                np.array(left_knee), np.array(left_ankle))
    elif action_dim == 17:  #Humanoid-v4
        right_hip = joint_list[:, :, 0]  # Extract right hip
        right_knee = joint_list[:, :, 1]  # Extract right knee
        left_hip = joint_list[:, :, 2]  # Extract left hip
        left_knee = joint_list[:, :, 3]  # Extract left knee
        none_value = np.zeros_like(right_hip)

        return np.array(right_hip), np.array(right_knee), none_value, np.array(left_hip), np.array(left_knee), none_value
    elif action_dim == 70:
        right_hip = joint_list[:, :, 0, 1]  # Extract right hip
        right_ankle = joint_list[:, :, 1, 1]  # Extract right ankle
        left_hip = joint_list[:, :, 0, 0]  # Extract left hip
        left_ankle = joint_list[:, :, 1, 0]  # Extract left ankle
        none_value = np.zeros_like(right_hip)
        # 0 right_hip, 1 none, 2 right_ankle, 3 left hip, 4 none, 5 left ankle
        return np.array(right_hip), none_value, np.array(right_ankle), np.array(left_hip), none_value, np.array(left_ankle)
    else:
        print("Not implemented Action Space")


def plot_data(data, data2=None, data1_name=None, data2_name=None, y_min_max=None,  x_data=None, y_axis_name="data", x_axis_name="time", title="data plot over time", save_folder=None, figure_name="figure"):
    # data = np.mean(data, axis=0)
    plt.figure()
    plt.title(title)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    if y_min_max is not None:
        plt.ylim(y_min_max[0], y_min_max[1])
    if data2 is not None:
        # data2 = np.mean(data2, axis=0)
        if x_data is not None:
            plt.plot(x_data, data, label=data1_name)
            plt.plot(x_data, data2, label=data2_name)
        else:
            plt.plot(data, label=data1_name)
            plt.plot(data2, label=data2_name)
    else:
        if x_data is not None:
            plt.plot(x_data, data, label=data1_name)
        else:
            plt.plot(data, label=data1_name)
    plt.legend()
    if save_folder is not None:
        # Define the path where the image will be saved
        image_path = os.path.join(save_folder, f"{figure_name}.png")
        plt.savefig(image_path)
    plt.show(block=False)
    plt.waitforbuttonpress()
    plt.close()


def statistical_analysis(data, y_axis_name="Value", x_axis_name="Time", title="Data", mean_calc=True, save_folder=None):

    # Calculate moving mean with window size and standard deviation
    if mean_calc:
        mean_data = np.mean(data, axis=0)
        mean_data = savitzky_golay_smoothing(mean_data)
        x = np.arange(data.shape[1])  # Time steps
    else:
        mean_data = data
        x = np.arange(len(data))  # Time steps
    std_dev = np.std(data, axis=0)
    var = np.var(data, axis=0)

    max_std = np.max(std_dev)
    max_var = np.max(var)

    fig, ax = plt.subplots()
    ax.plot(x, mean_data, label=f"Mean {y_axis_name}", color='blue')
    # Plot shaded region for standard deviation (mean ± std_dev)
    # Shade the area for standard deviation (mean ± std_dev)
    ax.fill_between(x, mean_data - std_dev, mean_data + std_dev, color='gray', alpha=0.3, label=f"{max_std:.2f} Std Dev")

    # Plot shaded region for variance (mean ± 2 * std_dev for demonstration purposes)
    ax.fill_between(x, mean_data - var, mean_data + var, color='gray', alpha=0.1, label=f"±{max_var:.2f} Variance")

    # Add labels and legend
    ax.set_xlabel(x_axis_name)
    ax.set_ylabel(y_axis_name)
    ax.set_title(title + ' Mean with Standard Deviation and Variance')
    ax.legend()
    if save_folder is not None:
        # Define the path where the image will be saved
        image_path = os.path.join(save_folder, f"{title}.png")
        plt.savefig(image_path)
    # Show the plot
    plt.show(block=False)
    plt.waitforbuttonpress()
    plt.close()
    return x, mean_data, std_dev, var


def get_acceleration(previous_velocity, velocity, dt):
    return (velocity - previous_velocity) / dt


def get_jerk(acceleration, dt):
    return np.diff(acceleration)/dt


def get_energy_per_meter(total_energy, total_distance, save_folder=None, plot_fig=False, x_range=(0,40), norm=False):
    # Calculate energy per meter for each episode
    energy_per_meter = total_energy / total_distance  # This gives the energy per meter for each episode
    # Normalize values for better visualization
    energy_per_meter_normalized = (energy_per_meter - np.min(energy_per_meter)) / (
            np.max(energy_per_meter) - np.min(energy_per_meter)
    )
    distance_normalized = (total_distance - np.min(total_distance)) / (
            np.max(total_distance) - np.min(total_distance)
    )
    # Create a heatmap-compatible 2D array
    if norm:
        data = np.vstack((distance_normalized, energy_per_meter_normalized)).T
    else:
        data = np.vstack((total_distance, energy_per_meter)).T

    # Now we can plot Joules (y-axis) vs Meters (x-axis)
    if plot_fig:
        plt.figure()
        sns.kdeplot(
            x=data[:, 0],
            y=data[:, 1],
            cmap="inferno",
            fill=True,
            cbar=True,
        )
        # Label the axes and add a title
        plt.xlabel('Distance (Meters)')
        plt.ylabel('Energy (Joules)')
        plt.title(f'Energy Consumption per Meter, Average: {np.mean(energy_per_meter):.2f} J/m')
        plt.xlim(x_range)
        plt.grid(True)
        if save_folder is not None:
            # Define the path where the image will be saved
            image_path = os.path.join(save_folder, "EnergyPerMeter.png")
            plt.savefig(image_path)
        # Show the plot
        plt.show(block=False)
        plt.waitforbuttonpress()
        plt.close()
    return energy_per_meter


def get_statistical_values(data):
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    var = np.var(data, axis=1)
    return mean, std, var


def compare_velocity(velocities, algos, dt=1, save_folder=None, auto_close=False):
    """
        Compare the velocity between two models (RL and RL + CPG) and plot them.
        :param velocities:
        :param algos: algorithm list being plotted
        :param dt: Time step between velocity measurements.
        :param save_folder:
    """
    colors = plt.colormaps.get_cmap("tab10").colors  # Default color cycle
    min_len = 1500
    for i, velocity in enumerate(velocities):
        mean_velocity = np.mean(velocity, axis=0)
        mean_velocity = cut_values_at_zero(mean_velocity)
        zero_index = len(mean_velocity)
        if zero_index < min_len:
            min_len = zero_index

    # Trim all arrays to the minimum length
    for i, velocity in enumerate(velocities):
        # Cut the arrays after the first occurrence of zero
        mean_velocity = np.mean(velocity, axis=0)
        mean_velocity = savitzky_golay_smoothing(mean_velocity)
        mean_velocity = cut_values_at_zero(mean_velocity)
        std_dev = np.std(velocity, axis=0)
        var = np.var(velocity, axis=0)

        #Make shape of arrays the same
        mean_velocity = mean_velocity[:min_len]
        std_dev = std_dev[:min_len]
        var = var[:min_len]

        # Get the max values for the deviations
        max_std = np.max(std_dev)
        max_var = np.max(var)
        max_vel = np.max(mean_velocity)

        # Adjust time array to match the length of the truncated velocities
        time = np.arange(0, len(mean_velocity) * dt, dt)

        # Plot velocities
        label_tmp = algos[i]
        plt.plot(time, mean_velocity, label=label_tmp + f" max vel: {max_vel:.2f} m/s", color=colors[i])

        # Shade the area for standard deviation (mean ± std_dev)
        plt.fill_between(time, mean_velocity - std_dev, mean_velocity + std_dev, color=colors[i], alpha=0.3,
                         label=f"{max_std:.2f} Std Dev ({label_tmp})")

        # Plot shaded region for variance (mean ± 2 * std_dev for demonstration purposes)
        plt.fill_between(time, mean_velocity - var, mean_velocity + var, color=colors[i], alpha=0.1,
                         label=f"±{max_var:.2f} Variance ({label_tmp})")

    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Velocity Comparison Across Environments")
    plt.legend()
    plt.grid(True)
    if save_folder is not None:
        # Define the path where the image will be saved
        image_path = os.path.join(save_folder, f"velocities_comparison.png")
        plt.savefig(image_path)
    plt.show(block=False)
    if not auto_close:
        plt.waitforbuttonpress()
    plt.close()


def plot_phase(data, data2=None, algo=None, save_folder=None, name="Phase"):
    plt.plot(data)
    if data2 is not None:
        plt.plot(data2)
    plt.xlabel('Steps')
    plt.ylabel(name)
    plt.title(f'{name} for joints in {algo} algorithm')
    if save_folder is not None:
        # Define the path where the image will be saved
        image_path = os.path.join(save_folder, f"{name}_{algo}.png")
        plt.savefig(image_path)
    plt.show(block=False)
    plt.waitforbuttonpress()
    plt.close()


def compare_motion(data):
    """
        Perform cross-correlation for multiple algorithms and plot all in the same graph.

        Parameters:
        - data_algos: List of tuples [(data1_algo1, data2_algo1), (data1_algo2, data2_algo2), ...].
                      Each tuple contains two arrays: one for the right leg and one for the left leg.
        - algos: List of algorithm names corresponding to the data_algos.
        - joint: Name of the joint being analyzed (e.g., "hip").
        - save_folder: Folder to save the plot (optional).

        Returns:
        - best_algo: The algorithm name with the lowest RMSE.
        """
    plt.figure(figsize=(10, 6))
    data1 = data[0]
    data2 = data[1]
    # Process data
    data1 = np.mean(data1, axis=0)
    data2 = np.mean(data2, axis=0)
    data1 = cut_values_at_zero(data1)
    data2 = cut_values_at_zero(data2)

    # Calculate cross-correlation
    cross_corr = np.correlate(data1 - np.mean(data1), data2 - np.mean(data2), mode='full')
    cross_corr /= np.max(cross_corr)  # Normalize
    lags = np.arange(-len(data1) + 1, len(data1))

    # Find peak correlation and its lag
    peak_correlation = np.max(cross_corr)
    lag_of_peak = lags[np.argmax(cross_corr)]

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(data1, data2))


    return lags, cross_corr, rmse, lag_of_peak


def compare_motion_pair(results, algos, save_folder=None, auto_close=False, place='knee'):
    """
    Compare autocorrelations between normal and CPG-based algorithms on the same graph,
    and plot separate graphs for other algorithm comparisons.

    Parameters:
    - algos_compare: List of algorithm names including CPG variants (e.g., ['PPO', 'PPO-CPG', ...]).
    - results: Dictionary containing the results for each algorithm.
    - joint: Joint name for the comparison (default: "Hip").
    - save_folder: Folder to save plots (optional).
    - auto_close:
    - place: knee or ankle comparison
    """
    normal_vs_cpg_pairs = []
    others = []
    if place == 'hip':
        indexes = [0,3]
    elif place == 'ankle':
        indexes = [2,5]
    else:
        indexes = [1,4]
        print("not implemented")

    # Separate algorithms into "Normal vs CPG" pairs and others
    for algo in algos:
        if '-CPG' in algo:
            base_algo = algo.replace('-CPG', '')
            if base_algo in algos:
                normal_vs_cpg_pairs.append((base_algo, algo))
        else:
            if not any(f"{algo}-CPG" == a for a in algos):
                others.append(algo)
    # Compare "Normal vs CPG" for each algorithm
    for base_algo, cpg_algo in normal_vs_cpg_pairs:
        if base_algo in results and cpg_algo in results:
            # Get joint data
            base_values = results[base_algo]['joints'][indexes[0]], results[base_algo]['joints'][indexes[1]]
            cpg_values = results[cpg_algo]['joints'][indexes[0]], results[cpg_algo]['joints'][indexes[1]]

            # Calculate cross-correlation for each
            lags_base, cross_corr_base, rmse_base, peak_lag_base = compare_motion(base_values)
            lags_cpg, cross_corr_cpg, rmse_cpg, peak_lag_cpg = compare_motion(cpg_values)

            # Plot autocorrelation comparison
            plt.plot(lags_base, cross_corr_base, label=f"{base_algo} (RMSE: {rmse_base:.2f}, Lag: {peak_lag_base})")
            plt.plot(lags_cpg, cross_corr_cpg, label=f"{cpg_algo} (RMSE: {rmse_cpg:.2f}, Lag: {peak_lag_cpg})")
            plt.xlabel('Lag')
            plt.ylabel('Cross-Correlation')
            plt.title(f"Autocorrelation Comparison at the {place} joint: {base_algo} vs {cpg_algo}")
            plt.legend()
            if save_folder:
                plt.savefig(f"{save_folder}/{place}_{base_algo}_vs_{cpg_algo}_autocorrelation.png")
            plt.show(block=False)
            if not auto_close:
                plt.waitforbuttonpress()
            plt.close()


def compare_vertical(data, algos, data_name="data_comparison", units=" ", save_folder=None, auto_close=False):
    # Total energy consumption for each algorithm

    # Create a bar chart
    total_values = [np.mean(dat) for dat in data]
    std_values = [np.std(dat) for dat in data]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(algos, total_values, yerr=std_values, capsize=5, color=plt.colormaps.get_cmap("tab20").colors, label="Mean ± Std")
    # Add precise value labels on top of each bar
    for bar, val in zip(bars, total_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.2f}", ha='center', va='bottom', fontsize=10)
    plt.xlabel('Algorithm')
    plt.ylabel(f'Total {data_name} ({units})')
    plt.title(f'Total {data_name} Comparison')
    if save_folder is not None:
        # Define the path where the image will be saved
        image_path = os.path.join(save_folder, f"{data_name}.png")
        plt.savefig(image_path)
    plt.show(block=False)
    if not auto_close:
        plt.waitforbuttonpress()
    plt.close()

    if len(data) == 2:
        statistical_test(data[0], data[1], value=data_name)


def compare_horizontal(data, algos, data_name="data_comparison", units=" ", save_folder=None, auto_close=False):
    """
        Compare the distance traveled between multiple models and plot them using horizontal bars.

        :param data: List of total distance arrays from different models. Each element is an array of distances for one model.
        :param data_name: Name of the data to be analyzed
        :param units: units in which the data is measured
        :param algos: List of labels for the models (for the plot legend).
        :param save_folder:
        """
    # Calculate the mean distance for each model
    mean_data = [np.mean(dat) for dat in data]
    std_data = [np.std(dat) for dat in data]
    colors = plt.colormaps.get_cmap("tab20").colors  # Default color cycle

    # Create a horizontal bar chart
    plt.figure(figsize=(10, 6))
    y_positions = np.arange(len(algos))

    plt.barh(y_positions, mean_data, color=colors, xerr=std_data, capsize=5, )
    plt.yticks(y_positions, algos)
    plt.xlabel(f'Total {data_name} ({units})')
    plt.ylabel('Algorithm')
    plt.title(f'Comparison of {data_name} in {units}')
    plt.grid(True)

    for i, v in enumerate(mean_data):
        plt.text(v + 0.05, i, f'{v:.2f}', va='center')  # Adding the exact value next to the bar

    if save_folder is not None:
        # Define the path where the image will be saved
        image_path = os.path.join(save_folder, f"{data_name}.png")
        plt.savefig(image_path)

    plt.show(block=False)
    if not auto_close:
        plt.waitforbuttonpress()
    plt.close()
    if len(data) == 2:
        statistical_test(data[0], data[1], value=data_name)


def statistical_test(cpg_data, base_data, value=''):
    t_stat, p_value = stats.ttest_ind(cpg_data, base_data)
    print("The t-test statistic is", t_stat, "and the p-value is", p_value, " for the ", value, " value")
    if p_value < 0.05:
        print("P-value < 0.05, showing significant difference between groups, rejecting null Hypothesis for the ", value, " value")
    else:
        print("P-value >= 0.05, Fail to reject null Hypothesis for the ", value, " value")
    return t_stat, p_value


def retrieve_cpg(config):
    # Loading CPG configuration
    cpg_oscillators = config.cpg_oscillators
    cpg_neurons = config.cpg_neurons
    cpg_tau_r = config.cpg_tau_r
    cpg_tau_a = config.cpg_tau_a

    return cpg_oscillators, cpg_neurons, cpg_tau_r, cpg_tau_a


def savitzky_golay_smoothing(data, window_length=100, polyorder=3):
    return savgol_filter(data, window_length, polyorder)


def cut_values_at_zero(data):
    # Return the array already cut with only valid values
    def cut_values_single_dim(data):
        # Find where the values are zero
        zero_mask = (data == 0)
        zero_threshold = 10
        # Find where continuous zeros begin
        continuous_zero_start = -1
        for i in range(len(zero_mask) - zero_threshold + 1):
            if all(zero_mask[i:i + zero_threshold]):
                continuous_zero_start = i
                break
        if continuous_zero_start == -1:
            # No continuous zeros found; return the original array
            return data
        else:
            return data[:continuous_zero_start]

    if data.ndim == 1:
        # Single-dimensional input
        return cut_values_single_dim(data)
    elif data.ndim > 1:
        # Multi-dimensional input: Apply truncation to each row
        output = []
        for i in range(len(data)):
            output.append(cut_values_single_dim(data[i]))
        return np.array(output)
    else:
        raise ValueError("Input array must be at least 1-dimensional.")


def get_motion_pattern(data, joint="Joint", plot=False):
    autocorr = np.correlate(data, data, mode='full')
    # autocorr = correlate(data, data, mode='full')  # Compute autocorrelation
    autocorr = autocorr[len(autocorr) // 2:]  # Take positive lags

    # Find peaks in autocorrelation to detect cycles
    peaks, _ = find_peaks(autocorr, height=0.1 * max(autocorr), distance=10)
    if len(peaks) > 1:
        cycle_length = peaks[1] - peaks[0]  # Approximate cycle length
    else:
        cycle_length = len(data) // 5  # Default fallback

    if cycle_length < 5:
        print("No clear pattern detected.")
        return

    pattern_segment = data[:cycle_length]  # Extract first cycle
    if plot:
        plt.figure(figsize=(8, 5))
        label = "Detected " + joint + " Motion Pattern"
        plt.plot(pattern_segment, label=label, color='b')
        plt.xlabel("Time (frames)")
        plt.ylabel(f"{joint} Joint Angle")
        plt.title(label)
        plt.legend()
        plt.show(block=False)
        plt.waitforbuttonpress()
        plt.close()

    return cycle_length, peaks


# Process multiple episodes and extract aligned motion cycles
def process_multiple_episodes(motion_episodes, joint="Joint"):
    """
    Process hip motion data across multiple episodes and extract aligned cycles.

    Returns:
    - all_patterns: Aligned motion cycles from all episodes.
    - mean_pattern: Average motion cycle across all episodes.
    - std_pattern: Standard deviation of motion cycles.
    """
    all_patterns = []

    for episode_idx, hip_motion in enumerate(motion_episodes):
        cycle_length, peaks = get_motion_pattern(hip_motion)

        if cycle_length < 5:
            print(f"Episode {episode_idx}: No clear pattern detected.")
            continue

        # Extract motion cycles based on detected peaks
        extracted_cycles = []
        for i in range(len(peaks) - 1):
            cycle_start, cycle_end = peaks[i], peaks[i] + int(cycle_length)
            if cycle_end <= len(hip_motion):
                extracted_cycles.append(hip_motion[cycle_start:cycle_end])

        if extracted_cycles:
            all_patterns.append(np.mean(extracted_cycles, axis=0))  # Compute mean per episode

    # Convert to array for statistical analysis
    all_patterns = np.array(all_patterns)

    # Compute mean and standard deviation across all episodes
    mean_pattern = np.mean(all_patterns, axis=0)
    std_pattern = np.std(all_patterns, axis=0)

    plt.figure(figsize=(10, 6))

    # Compute 95% confidence interval
    conf_int = stats.norm.interval(0.95, loc=mean_pattern, scale=std_pattern)

    # Plot mean motion pattern
    label = "Mean " + joint + "Motion"
    plt.plot(mean_pattern, label=label, color="blue")

    # Confidence interval shading
    plt.fill_between(range(len(mean_pattern)), conf_int[0], conf_int[1], alpha=0.3, color="blue", label="CI")

    plt.xlabel("Time (frames)")
    plt.ylabel(f"{joint} Angle")
    title = "Mean " + joint + " Motion Cycle Across Episodes"
    plt.title(title)
    plt.legend()
    plt.show()

    return all_patterns, mean_pattern, std_pattern