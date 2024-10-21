import matplotlib.pyplot as plt
import torch
import numpy as np
from MatsuokaOscillator import MatsuokaNetworkWithNN, MatsuokaNetwork, MatsuokaOscillator
from MatsuokaOscillator.matsuokaOscillator import HHMatsuokaNetwork, HHMatsuokaOscillator


# Basic Matsuoka Oscillator Implementation
def matsuoka_main():
    # Parameters
    neural_net = False
    hh_neuron = True  # Running with hudgkin huxley neurons if True
    num_oscillators = 1
    neuron_number = 2
    tau_r = 1
    tau_a = 6
    amplitude = 1.5
    w12 = 2.5
    beta = 2.5
    dt = 0.0001
    steps = 50
    weights = np.full(neuron_number, w12)
    time = np.linspace(0, steps * dt, steps)

    if neural_net is True:
        # Neural Network Implementation
        input_size = num_oscillators  # Example input size
        hidden_size = 10  # Hidden layer size
        output_size = 3  # tau_r, weights, and beta for each oscillator

        matsuoka_network = MatsuokaNetworkWithNN(num_oscillators=num_oscillators,
                                                 neuron_number=neuron_number,
                                                 tau_r=tau_r, tau_a=tau_a, amplitude=amplitude, hh=hh_neuron)
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
        # Run of the events
        if num_oscillators == 1:
            # Create Matsuoka Oscillator with N neurons
            if hh_neuron is False:
                oscillator = MatsuokaOscillator(neuron_number=neuron_number, tau_r=tau_r, tau_a=tau_a,
                                            beta=beta, dt=dt,
                                            num_oscillators=num_oscillators)
            else:
                oscillator = HHMatsuokaOscillator(neuron_number=neuron_number, tau_r=tau_r, tau_a=tau_a,
                                            beta=beta, dt=dt,
                                            num_oscillators=num_oscillators)
            weights_seq = [weights]*steps
            y_output = oscillator.run(steps=steps, weights_seq=weights_seq)

            y_output = y_output.cpu().numpy()
            for i in range(y_output.shape[1]):
                plt.plot(time, y_output[:, i], label=f'y{i + 1} (Neuron {i + 1})')
            plt.xlabel('Time')
            plt.ylabel('Output')
            plt.title('Matsuoka Oscillator Output')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            # Coupled System
            if hh_neuron is False:
                coupled_system = MatsuokaNetwork(num_oscillators=num_oscillators, neuron_number=neuron_number, tau_r=tau_r,
                                             tau_a=tau_a, beta=beta, dt=dt)
            else:
                coupled_system = HHMatsuokaNetwork(num_oscillators=num_oscillators, neuron_number=neuron_number, tau_r=tau_r,
                                             tau_a=tau_a, beta=beta, dt=dt)
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
            plt.show()

if __name__ == "__main__":
    matsuoka_main()