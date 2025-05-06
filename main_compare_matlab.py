import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import Experiments.experiments_utils as utils
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error


def min_max_normalize(data):
    """
    Normalize a 1D array to the range [0, 1].

    Parameters:
        data (np.ndarray): Input array.

    Returns:
        np.ndarray: Normalized array with values between 0 and 1.
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)  # avoid division by zero


def extract_mean_cycle(signal, num_points=100, separation=4):
    # Detect peaks (we assume they mark gait cycles, adjust height/distance as needed)
    peaks, _ = find_peaks(signal, distance=len(signal)//separation)  # tune `distance` if needed

    cycles = []
    for i in range(len(peaks) - 1):
        start, end = peaks[i], peaks[i+1]
        cycle = signal[start:end]
        if len(cycle) > 5:  # avoid very short segments
            # Interpolate each cycle to a common length
            x_old = np.linspace(0, 1, len(cycle))
            f = interp1d(x_old, cycle, kind='linear')
            x_new = np.linspace(0, 1, num_points)
            cycles.append(f(x_new))

    if not cycles:
        raise ValueError("No full cycles could be extracted. Check the peak detection.")

    mean_cycle = np.mean(cycles, axis=0)
    std_cycle = np.std(cycles, axis=0)
    return mean_cycle, std_cycle


def plot_motion(x_value, mean_data_1, mean_data_2, std_data_1, std_data_2, rmse, title="Joint"):
    plt.figure(figsize=(12, 6))
    plt.plot(x_value, mean_data_1, label=f'{title} Motion MC source')
    plt.plot(x_value, mean_data_2, label=f'{title} Motion RL source')
    plt.fill_between(x_value, mean_data_1 - std_data_1, mean_data_1 + std_data_1, alpha=0.3, label=f"Std Dev MC", color='blue')
    plt.fill_between(x_value, mean_data_2 - std_data_2, mean_data_2 + std_data_2, alpha=0.2, label=f"Std Dev RL",color='orange')
    plt.title(f'Motion Capture vs RL: {title}. RMSE: {rmse:.3f})')
    plt.xlabel('Gait Cycle (%)')
    plt.ylabel('Normalized Joint Angle')
    plt.legend()
    plt.grid(True)
    plt.show()


def normalize_and_plot_crosscorr(lags1, corr1, lags2, corr2, label1='Mocap', label2='RL+CPG', title="Joint"):
    # Normalize lag axis to -1 to 1
    norm_lag1 = np.linspace(-1, 1, len(corr1))
    norm_lag2 = np.linspace(-1, 1, len(corr2))

    # Interpolate to common frame size
    num_points = 200
    interp1 = interp1d(norm_lag1, corr1, kind='linear', fill_value="extrapolate")
    interp2 = interp1d(norm_lag2, corr2, kind='linear', fill_value="extrapolate")
    common_lags = np.linspace(-1, 1, num_points)
    norm_corr1 = interp1(common_lags)
    norm_corr2 = interp2(common_lags)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(common_lags, norm_corr1, label=label1)
    plt.plot(common_lags, norm_corr2, label=label2)
    plt.title(f"Normalized Cross-Correlation Comparison from {title}")
    plt.xlabel("Normalized Lag")
    plt.ylabel("Cross-Correlation")
    plt.legend()
    plt.grid(True)
    plt.show()

hip_data = sio.loadmat('utilities_repository/matlab_files/TrajectoryHip.mat')

ankle_data = sio.loadmat('utilities_repository/matlab_files/TrajectoryAnkle.mat')

crossAnkle = sio.loadmat('utilities_repository/matlab_files/crosscorrDataAnkle.mat')

crossHip = sio.loadmat('utilities_repository/matlab_files/crosscorrDataHip.mat')


# Get MC data from matlab
hip_motion_l = hip_data['HipLAngleDataNormalized']
mean_hml, std_hml, var_hml = utils.get_statistical_values(hip_motion_l)
mean_hml = mean_hml[1000:2000]

ankle_motion_l = ankle_data['ankleLAngleDataNormalized']
mean_aml, std_aml, var_aml = utils.get_statistical_values(ankle_motion_l)
mean_aml = mean_aml[1000:2000]

hip_motion_r = hip_data['HipRAngleDataNormalized']
mean_hmr, std_hmr, var_hmr = utils.get_statistical_values(hip_motion_r)
mean_hmr = mean_hmr[1000:2000]

ankle_motion_r = ankle_data['AnkleRAngleDataNormalized']
mean_amr, std_amr, var_amr = utils.get_statistical_values(ankle_motion_r)
mean_amr = mean_amr[1000:2000]

crossHip_mc = crossHip['ccorHip'].squeeze()
lagHip_mc = crossHip['lagsHip'].squeeze()

crossAnkle_mc = crossAnkle['ccorAnkle'].squeeze()
lagAnkle_mc = crossAnkle['lagsAnkle'].squeeze()

# Load numpy data
hip_motion_np_l = np.load('utilities_repository/rl-files/right_hip_movement.npy') # 300 - 600 in timesteps
hip_motion_np_r = np.load('utilities_repository/rl-files/left_hip_movement.npy')

ankle_motion_np_l = np.load('utilities_repository/rl-files/left_ankle_movement.npy')
ankle_motion_np_r = np.load('utilities_repository/rl-files/right_ankle_movement.npy')

crossHip_np = np.load('utilities_repository/rl-files/crossCorrelation_hip.npy')
lagsHip_np = np.load('utilities_repository/rl-files/lags_hip.npy')
crossAnkle_np = np.load('utilities_repository/rl-files/crossCorrelation_ankle.npy')
lagsAnkle_np = np.load('utilities_repository/rl-files/lags_ankle.npy')

# Normalize data


mean_hml = min_max_normalize(mean_hml)
hip_motion_np_l = min_max_normalize(hip_motion_np_l[400:620])
mean_hmr = min_max_normalize(mean_hmr)
hip_motion_np_r = min_max_normalize(hip_motion_np_r[400:620])

mean_aml = min_max_normalize(mean_aml)
ankle_motion_np_l = min_max_normalize(ankle_motion_np_l[400:620])
mean_amr = min_max_normalize(mean_amr)
ankle_motion_np_r = min_max_normalize(ankle_motion_np_r[400:620])

## We got 3-4 cycles of steps. With normalized data, either in steps and in scale.
# Next: Find the cycle pattern and compare.

meanhml, stdhml = extract_mean_cycle(mean_hml, separation=6)
meanhrl, stdhrl = extract_mean_cycle(hip_motion_np_l, separation=5)

meanhmr, stdhmr = extract_mean_cycle(mean_hmr, separation=6)
meanhrr, stdhrr = extract_mean_cycle(hip_motion_np_r, separation=5)

meanaml, stdaml = extract_mean_cycle(mean_aml, separation=8)
meanarl, stdarl = extract_mean_cycle(ankle_motion_np_l, separation=4)

meanamr, stdamr = extract_mean_cycle(mean_amr, separation=8)
meanarr, stdarr = extract_mean_cycle(ankle_motion_np_r, separation=4)

x_value = np.linspace(0, 100, len(meanhml))
errorhl = np.abs(meanhml - meanhrl)
rmsehl = np.sqrt(mean_squared_error(meanhml, meanhrl))*100
rmsehr = np.sqrt(mean_squared_error(meanhmr, meanhrr))*100

rmseal = np.sqrt(mean_squared_error(meanaml, meanarl))*100
rmsear = np.sqrt(mean_squared_error(meanamr, meanarr))*100

#
plt.figure(figsize=(12, 6))
plt.plot(lagAnkle_mc, crossAnkle_mc, label='cross Hip Motion MC source')
plt.show()

plot_motion(x_value, meanhml, meanhrl, stdhml, stdhrl, rmsehl, title="Hip Left")

plot_motion(x_value, meanhmr, meanhrr, stdhmr, stdhrr, rmsehr, title="Hip Right")

plot_motion(x_value, meanaml, meanarl, stdaml, stdarl, rmseal, title="Ankle Left")

plot_motion(x_value, meanamr, meanarr, stdamr, stdarr, rmsear, title="Ankle Right")


normalize_and_plot_crosscorr(lagHip_mc, crossHip_mc, lagsHip_np, crossHip_np, title="Hip Joint")

normalize_and_plot_crosscorr(lagAnkle_mc, crossAnkle_mc, lagsAnkle_np, crossAnkle_np, title="Ankle Joint")
