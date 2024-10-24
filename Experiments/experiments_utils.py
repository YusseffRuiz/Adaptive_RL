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


def get_name_environment(name, cpg_flag=False, algorithm=None, experiment_number=0, create=False, external_folder=None):
    """
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
        env_name = env_name + "-CPG"

    if external_folder:
        if create:
            if os.path.exists(f"training/{external_folder}"):
                print(f"Using existing folder: {external_folder}")
            if experiment_number > 0:
                save_folder = f"training/{external_folder}/{experiment_number}"
            else:
                save_folder = f"training/{external_folder}"
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
def search_trained_algorithms(env_name, algorithms_list, save_folder=None, experiment_number=0):
    # List of algorithms to look for
    algorithms_found = []

    save_folder = "training" or save_folder

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


def evaluate(model=None, env=None, algorithm="random", num_episodes=5, no_done=False, max_episode_steps=1000):
    total_rewards = []
    range_episodes = num_episodes
    mujoco_env = hasattr(env, "sim")
    for i in range(range_episodes):
        obs, *_ = env.reset()
        done = False
        episode_reward = 0
        cnt = 0
        while not done:
            cnt += 1
            with torch.no_grad():
                if algorithm != "random":
                    action = model.test_step(obs)
                else:
                    action = env.action_space.sample()
            obs, reward, done, info, *_ = env.step(action)
            if mujoco_env:
                #Try rendering for MyoSuite
                env.sim.renderer.render_to_window()
            episode_reward += reward
            if no_done:
                done = False
            if cnt >= max_episode_steps:
                done = True

        total_rewards.append(episode_reward)
        print(f"Episode {i + 1}/{range_episodes}: Reward = {episode_reward}")
    average_reward = np.mean(total_rewards)
    print(f"Average Reward over {range_episodes} episodes: {average_reward}")


def evaluate_experiment(agent=None, env=None, alg="random", episodes_num=5, duration=1500, env_name=None, deterministic=False):

    save_folder = f"Experiments/{env_name}/images"
    action_dim = len(env.action_space.sample())
    total_rewards = []
    total_joint_angles = []
    total_velocity_angles = []
    total_accelerations = []
    total_torques = []
    total_energy = []
    total_velocity = []
    dt = 5  # how many steps per dt
    previous_vel = 0
    if alg == "random":
        episode_start = np.ones((env.num_envs,), dtype=bool)
    total_distance = []
    total_steps = []
    for episode in range(episodes_num):
        if alg == "random":
            obs = env.reset()
        else:
            obs, *_ = env.reset()
        lstm_states = None
        reward = 0
        ep_energy = []
        ep_velocity = []
        ep_acceleration = []
        ep_torques = []
        ep_joints = []
        ep_joint_velocities = []
        position = 0
        step_position = 0
        for step in range(duration):
            step_position = step
            if alg != "random":
                action = agent.test_step(obs)
            else:
                action, lstm_states = agent.predict(
                        obs,  # type: ignore[arg-type]
                        state=lstm_states,
                        episode_start=episode_start,
                        deterministic=deterministic,
                )
            if alg == "random":
                obs, rw, done, info = env.step(action)
                position, velocity, joint_angles, joint_velocity, torques, step_energy = get_data(info[0])
            else:
                obs, rw, done, info, extras = env.step(action)
                position, velocity, joint_angles, joint_velocity, torques, step_energy = get_data(extras)
            episode_start = done
            if step % dt == 0:
                acceleration = get_acceleration(previous_vel, velocity, dt)
                ep_acceleration.append(acceleration)
                previous_vel = velocity
            ep_joints.append(joint_angles)
            ep_joint_velocities.append(joint_velocity)
            ep_velocity.append(velocity)
            ep_torques.append(torques)
            reward += rw
            ep_energy.append(step_energy)
        if reward < 0:
            print("Episode failed")
        total_distance.append(position)
        total_steps.append(step_position)
        total_rewards.append(reward)
        total_energy.append(np.sum(ep_energy, axis=1))
        total_velocity_angles.append(np.array(ep_joint_velocities))
        total_torques.append(np.array(ep_torques))
        total_joint_angles.append(np.array(ep_joints))
        total_velocity.append(np.array(ep_velocity))
        total_accelerations.append(np.array(ep_acceleration))

    # Convert all arrays to numpy for analysis
    total_distance = np.array(total_distance)
    total_rewards = np.array(total_rewards)
    total_velocity_angles = np.array(total_velocity_angles)
    total_torques = np.array(total_torques)
    total_joint_angles = np.array(total_joint_angles)
    total_energy = np.array(total_energy)
    total_velocity = np.array(total_velocity)
    total_accelerations = np.array(total_accelerations)

    average_reward = np.mean(total_rewards)
    velocity_total = np.mean(total_velocity)
    average_energy = (np.sum(np.trapz(total_energy, dx=1)/np.mean(total_steps))/episodes_num)
    average_distance = np.mean(total_distance)
    print(f"Average Reward over {episodes_num} episodes: {average_reward:.2f}")
    print(f"Average Speed and Distance over {episodes_num} episodes: {velocity_total:.2f} m/s with "
          f"total energy: {average_energy:.2f} Joules per second, travelled {average_distance:.2f} meters")
    joints = separate_joints(total_joint_angles, action_dim)
    joints_vel = separate_joints(total_velocity_angles, action_dim)
    joints_torque = separate_joints(total_torques, action_dim)

    os.makedirs(save_folder, exist_ok=True)
    # plot_data(data=joints[0], data2=joints[2], data1_name="right hip", data2_name="left hip", y_axis_name="Angle (°/s)", title="Hip Joint movement")
    # statistical_analysis(total_velocity, y_axis_name="Velocity(m/s", title=f"Velocity (m/s) {velocity_total:.2f} m/s", save_folder=save_folder, figure_name="Total_Velocity")
    # cross_fourier_transform(joints[0], joints[2], joint="Hip", save_folder=save_folder)
    perform_autocorrelation(joints[0], joints[2], joint="Hip", save_folder=save_folder)
    jerk = get_jerk(total_accelerations, dt)
    # statistical_analysis(total_accelerations, x_axis_name="Time", y_axis_name="Acceleration (m/s^2)", title="Acceleration (m/s^2)", save_folder=save_folder, figure_name="Acceleration")
    # statistical_analysis(jerk, x_axis_name="Time", y_axis_name="Jerk (m/s^3)", title="jerk (m/s^3)", save_folder=save_folder, figure_name="Jerk")
    energy_consumption = get_energy_per_meter(total_energy, np.mean(total_steps), average_energy, save_folder=save_folder)

    results_dict = {
        'velocity': total_velocity,
        'energy': energy_consumption,
        'reward': total_rewards,
        'distance': total_distance,
        'steps': total_steps,
        'jerk': jerk,
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
        plt.close()
    else:
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

    position = info["distance_from_origin"]
    velocity = info["x_velocity"]
    joint_angles = info["joint_angles"] * 180/math.pi
    joint_velocity = info["joint_velocities"] * 180 / math.pi
    torques = info["torques"]
    step_energy = torques*info["joint_velocities"]

    return position, velocity, joint_angles, joint_velocity, torques, step_energy


def separate_joints(joint_list, action_dim):
    # Split the joints for each step across all episodes
    if action_dim == 6:  # Walker2d-v4
        right_hip = joint_list[:, :, 0]  # Extract right hip
        right_knee = joint_list[:, :, 1]  # Extract right knee
        right_ankle = joint_list[:, :, 2]  # Extract right ankle
        left_hip = joint_list[:, :, 3]  # Extract left hip
        left_knee = joint_list[:, :, 4]  # Extract left knee
        left_ankle = joint_list[:, :, 5]  # Extract left ankle

        return np.array(right_hip), np.array(right_knee), np.array(left_hip), np.array(left_knee)
    elif action_dim == 17:  #Humanoid-v4
        right_hip = joint_list[:, :, 2]  # Extract right hip
        right_knee = joint_list[:, :, 3]  # Extract right knee
        left_hip = joint_list[:, :, 6]  # Extract left hip
        left_knee = joint_list[:, :, 7]  # Extract left knee

        return np.array(right_hip), np.array(right_knee), np.array(left_hip), np.array(left_knee)
    elif action_dim == 70:
        pass
    else:
        right_hip = joint_list[:, :, 0]  # Extract right hip
        right_knee = joint_list[:, :, 1]  # Extract right knee
        left_hip = joint_list[:, :, 2]  # Extract left hip
        left_knee = joint_list[:, :, 3]  # Extract left knee
        return np.array(right_hip), np.array(right_knee), np.array(left_hip), np.array(left_knee)


def plot_data(data, data2=None, data1_name=None, data2_name=None, y_min_max=None,  x_data=None, y_axis_name="data", x_axis_name="time", title="data plot over time", save_folder=None, figure_name="figure"):
    data = np.mean(data, axis=0)
    plt.figure()
    plt.title(title)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    if y_min_max is not None:
        plt.ylim(y_min_max[0], y_min_max[1])
    if data2 is not None:
        data2 = np.mean(data2, axis=0)
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
    plt.show()


def statistical_analysis(data, y_axis_name="Value", x_axis_name="Time", title="Data", mean_calc=True, save_folder=None, figure_name="figure"):

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
        image_path = os.path.join(save_folder, f"{figure_name}.png")
        plt.savefig(image_path)
    # Show the plot
    plt.show()


def get_acceleration(previous_velocity, velocity, dt):
    return (velocity - previous_velocity) / dt


def get_jerk(acceleration, dt):
    return np.diff(acceleration)/dt


def get_energy_per_meter(total_energy, total_distance, average, save_folder=None, plot_fig=False):
    energy_per_episode = np.sum(total_energy, axis=1)  # Sum energy across steps for each episode

    # Calculate energy per meter for each episode
    energy_per_meter = energy_per_episode / total_distance  # This gives the energy per meter for each episode

    # Now we can plot Joules (y-axis) vs Meters (x-axis)
    if plot_fig:
        plt.figure()
        sns.kdeplot(x=total_distance, y=energy_per_meter, cmap="Reds", fill=True, thresh=0, levels=100)

        # Label the axes and add a title
        plt.xlabel('Distance (Meters)')
        plt.ylabel('Energy (Joules)')
        plt.title(f'Energy Consumption per Meter, Average: {average:.2f} J/m')
        plt.grid(True)
        if save_folder is not None:
            # Define the path where the image will be saved
            image_path = os.path.join(save_folder, "EnergyPerMeter.png")
            plt.savefig(image_path)
        # Show the plot
        plt.show()
    return energy_per_meter


def compare_velocity(velocities, algos, dt=1, save_folder=None):
    """
        Compare the velocity between two models (RL and RL + CPG) and plot them.
        :param velocities:
        :param algos: algorithm list being plotted
        :param dt: Time step between velocity measurements.
        :param save_folder:
    """
    time = np.arange(0, len(velocities[0][0]) * dt, dt)
    print(len(velocities))
    colors = plt.colormaps.get_cmap("tab10").colors  # Default color cycle

    for i, velocity in enumerate(velocities):
        mean_velocity = np.mean(velocity, axis=0)
        mean_velocity = savitzky_golay_smoothing(mean_velocity)
        std_dev = np.std(mean_velocity, axis=0)
        var = np.var(mean_velocity, axis=0)

        max_std = np.max(std_dev)
        max_var = np.max(var)

        # Plot velocities
        label_tmp = algos[i]
        plt.plot(time, mean_velocity, label=label_tmp, color=colors[i])

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
    plt.waitforbuttonpress()
    plt.close()


def compare_jerk(jerk1, jerk2, dt):
    pass


def compare_motion(action_list):
    pass


def compare_vertical(data, algos, data_name="data_comparison", units=" ", save_folder=None):
    # Total energy consumption for each algorithm

    # Create a bar chart
    total_values= [np.mean(dat) for dat in data]

    plt.figure(figsize=(8, 5))
    plt.bar(algos, total_values, color=plt.colormaps.get_cmap("tab10").colors)
    plt.xlabel('Algorithm')
    plt.ylabel(f'Total {data_name} ({units})')
    plt.title(f'Total {data_name} Comparison')
    if save_folder is not None:
        # Define the path where the image will be saved
        image_path = os.path.join(save_folder, f"{data_name}.png")
        plt.savefig(image_path)
    plt.show(block=False)
    plt.waitforbuttonpress()
    plt.close()


def compare_horizontal(data, algos, data_name="data_comparison", units=" ", save_folder=None):
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
    colors = plt.colormaps.get_cmap("tab10").colors  # Default color cycle

    # Create a horizontal bar chart
    plt.figure(figsize=(10, 6))
    y_positions = np.arange(len(algos))

    plt.barh(y_positions, mean_data, color=colors)
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
    plt.waitforbuttonpress()
    plt.close()


def savitzky_golay_smoothing(data, window_length=100, polyorder=3):
    return savgol_filter(data, window_length, polyorder)
