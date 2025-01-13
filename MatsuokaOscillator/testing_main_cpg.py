import matplotlib.pyplot as plt
import matplotlib as mpl
from fontTools.unicodedata import block

mpl.use('TkAgg')  # Comment if we want regular images
import torch
import numpy as np
from random import random

from MatsuokaOscillator import MatsuokaNetworkWithNN, MatsuokaNetwork, MatsuokaOscillator
from MatsuokaOscillator.matsuokaOscillator import HHMatsuokaNetwork, HHMatsuokaOscillator
from Experiments.experiments_utils import analyze_neurons_frequencies


# Basic Matsuoka Oscillator Implementation
def matsuoka_main():
    # Parameters
    neural_net = False
    hh_neuron = False  # Running with hudgkin huxley neurons if True
    num_oscillators = 2
    neuron_number = 2
    tau_r = 4.0
    tau_a = 24.0
    t_min = 1
    t_max = 8
    amplitude = 1.0
    w12_ = 2.5
    min_w12 = 1.0
    max_w12 = 4.0
    beta = 2.5
    dt = 1
    steps = 1000
    step = 0.1
    # Generate the sequence for w12
    t_values = np.arange(t_min, t_max+1, 1)
    t_values = np.concatenate([t_values, t_values[::-1]])  # Include reverse direction

    if len(t_values) > steps:
        t_values = t_values[:steps]
    elif len(t_values) < steps:
        repetitions = (steps // len(t_values)) + 1
        t_values = np.tile(t_values, repetitions)[:steps]

    w12_values = np.arange(min_w12, max_w12 + step, step)
    w12_values = np.concatenate([w12_values, w12_values[::-1]])  # Include reverse direction
    # Ensure w12_values matches the total steps
    if len(w12_values) > steps:
        w12_values = w12_values[:steps]
    elif len(w12_values) < steps:
        repetitions = (steps // len(w12_values)) + 1
        w12_values = np.tile(w12_values, repetitions)[:steps]

    # weights = torch.tensor([[w12, -w12]], device="cuda")  # Example weights
    time = np.linspace(0, steps * 1, steps)

    if neural_net is True:
        # Neural Network Implementation
        input_size = num_oscillators  # Example input size
        hidden_size = 10  # Hidden layer size
        output_size = 3  # tau_r, weights, and beta for each oscillator

        matsuoka_network = MatsuokaNetworkWithNN(num_oscillators=num_oscillators,
                                                 neuron_number=neuron_number,
                                                 tau_r=tau_r, tau_a=tau_a, hh=hh_neuron)
        # Create a sample sensory input sequence
        sensory_input_seq = torch.rand(steps, num_oscillators, input_size, dtype=torch.float32, device="cuda")

        # Run the coupled system with NN control
        outputs = matsuoka_network.run(steps=steps, sensory_input_seq=sensory_input_seq)
        outputs = outputs.cpu().numpy()

        for i in range(num_oscillators):
            plt.plot(time, outputs[:, i, 0], label=f'Oscillator {i + 1} Neuron 1')
            plt.plot(time, outputs[:, i, 1], label=f'Oscillator {i + 1} Neuron 2')

        plt.xlabel('Time step')
        plt.ylabel('Output')
        plt.title('Outputs of Coupled Matsuoka Oscillators Controlled by NN')
        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        # Construct weights_seq
        weights_seq = []
        times_seq = []
        cnt = 0
        for w12 in w12_values:
            if num_oscillators > 1:
                weights = torch.full((num_oscillators, neuron_number), w12)+torch.rand(num_oscillators, neuron_number)
            else:
                weights = np.full((neuron_number,), w12) * (1+random())
            times_seq.append([int(tau_a*t_values[cnt]/6), tau_a*t_values[cnt]])
            weights_seq.append(weights)
            cnt += 1
        # print(weights_seq)
        # Run of the events
        if num_oscillators == 1:
            # Create Matsuoka Oscillator with N neurons
            if hh_neuron is False:
                oscillator = MatsuokaOscillator(neuron_number=neuron_number, tau_r=tau_r, tau_a=tau_a,
                                            beta=beta, dt=dt,
                                            num_oscillators=num_oscillators, amplitude=amplitude)
            else:
                oscillator = HHMatsuokaOscillator(neuron_number=neuron_number, tau_r=tau_r, tau_a=tau_a,
                                            beta=beta, dt=dt,
                                            num_oscillators=num_oscillators, amplitude=amplitude)
            oscillator.w = 0
            y_output = oscillator.run(steps=steps, weights_seq=weights_seq)

            y_output = y_output.cpu().numpy()
            for i in range(y_output.shape[1]):
                plt.plot(time, y_output[:, i], label=f'y{i + 1} (Neuron {i + 1})')
            plt.xlabel('Time')
            plt.ylabel('Output')
            plt.title('Matsuoka Oscillator Output')
            plt.legend()
            plt.grid(True)
            plt.show(block=False)
            frequencies, period = analyze_neurons_frequencies(y_output, dt / 2)
            print("Freq: ", frequencies, "Period: ", period)
            plt.waitforbuttonpress()
            plt.close()
        else:
            # Coupled System
            if hh_neuron is False:
                coupled_system = MatsuokaNetwork(num_oscillators=num_oscillators, neuron_number=neuron_number, tau_r=tau_r,
                                             tau_a=tau_a, beta=beta, dt=dt, amplitude=amplitude)
            else:
                coupled_system = HHMatsuokaNetwork(num_oscillators=num_oscillators, neuron_number=neuron_number, tau_r=tau_r,
                                             tau_a=tau_a, beta=beta, dt=dt, amplitude=amplitude)
            coupled_system.oscillators.w = 0
            y_output = coupled_system.run(steps=steps)

            # Coupled Oscillators
            for i in range(num_oscillators):
                for j in range(neuron_number):
                    plt.plot(time, y_output[i][:, j], label=f'Oscillator {i + 1} Neuron {j + 1}')
            plt.xlabel('Time step')
            plt.ylabel('Output')
            plt.title('Outputs of Coupled Matsuoka Oscillators')
            plt.legend()
            plt.grid(True)
            plt.show(block=False)
            plt.waitforbuttonpress()
            plt.close()


if __name__ == "__main__":
    matsuoka_main()
