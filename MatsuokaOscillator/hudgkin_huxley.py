import math
import torch



class HHNeuron(torch.nn.Module):
    def __init__(self, C_m=1.0, g_Na=120.0, g_K=36.0, g_L=0.3, E_Na=50.0, E_K=-77.0, E_L=-54.387, dt=0.5):
        super(HHNeuron, self).__init__()
        self.C_m = C_m  # membrane capacitance [uF/cm2]
        self.g_Na = g_Na  # sodium conductance [mS/cm2]
        self.g_K = g_K  # potassium conductance [mS/cm2]
        self.g_L = g_L  # leak conductance [mS/cm2]
        self.E_Na = E_Na  # sodium equilibrium potential [mV]
        self.E_K = E_K  # potassium equilibrium potential [mV]
        self.E_L = E_L  # leak equilibrium potential [mV]
        self.V_max = 65
        self.V_min = -80
        self.dt = dt

    # """
    def alpha_m(self, V):
        V = torch.clamp(V, self.V_min, self.V_max)  # Avoids overflow
        return (0.182 * (V + 35)) / (1 - torch.exp(-(V + 35) / 9))

    def beta_m(self, V):
        V = torch.clamp(V, self.V_min, self.V_max)  # Avoids overflow
        return (-0.124 * (V + 35)) / (1 - torch.exp((V + 35) / 9))

    def alpha_h(self, V):
        V = torch.clamp(V, self.V_min, self.V_max)  # Avoids overflow
        return 0.25 * torch.exp(-(V + 90) / 12)

    def beta_h(self, V):
        V = torch.clamp(V, self.V_min, self.V_max)  # Avoids overflow
        return 0.25 * torch.exp((V + 62) / 12)

    def alpha_n(self, V):
        V = torch.clamp(V, self.V_min, self.V_max)  # Avoids overflow
        return (0.02 * (V - 25)) / (1 - torch.exp(-(V -25) / 9))

    def beta_n(self, V):
        V = torch.clamp(V, self.V_min, self.V_max)  # Avoids overflow
        return (-0.002 * (V - 25)) / (1 - torch.exp((V -25) / 9))

    def compute_currents(self, V, m, h, n, I_ext):
        # Calculate the ionic currents
        I_Na = self.g_Na * (m ** 3) * h * (V - self.E_Na)
        I_K = self.g_K * (n ** 4) * (V - self.E_K)
        I_L = self.g_L * (V - self.E_L)

        I_sum = I_Na + I_K + I_L
        # print("I sum", I_sum)

        # Total current derivative (dV/dt)
        dVdt = (I_ext - (I_sum)) * (1/self.C_m)
        # print("dV Tensor: ", dVdt)

        # Gating variable derivatives
        dmdt = self.alpha_m(V) * (1 - m) - self.beta_m(V) * m
        dhdt = self.alpha_h(V) * (1 - h) - self.beta_h(V) * h
        dndt = self.alpha_n(V) * (1 - n) - self.beta_n(V) * n

        return dVdt, dmdt, dhdt, dndt

    def forward(self, V, m, h, n, I_ext):
        # Compute the RK4 steps
        dt = self.dt
        k1_V, k1_m, k1_h, k1_n = self.compute_currents(V, m, h, n, I_ext)

        k2_V, k2_m, k2_h, k2_n = self.compute_currents(V + k1_V * dt / 2,
                                                       m + k1_m * dt / 2,
                                                       h + k1_h * dt / 2,
                                                       n + k1_n * dt / 2,
                                                       I_ext/2)

        k3_V, k3_m, k3_h, k3_n = self.compute_currents(V + k2_V * dt / 2,
                                                       m + k2_m * dt / 2,
                                                       h + k2_h * dt / 2,
                                                       n + k2_n * dt / 2,
                                                       I_ext/2)

        k4_V, k4_m, k4_h, k4_n = self.compute_currents(V + k3_V * dt,
                                                       m + k3_m * dt,
                                                       h + k3_h * dt,
                                                       n + k3_n * dt,
                                                       I_ext)

        # Combine the RK4 results for each variable

        dV_new = (dt / 6.0) * (k1_V + 2 * k2_V + 2 * k3_V + k4_V)
        # print("dV_new: ", dV_new)
        dm_new = (dt / 6.0) * (k1_m + 2 * k2_m + 2 * k3_m + k4_m)
        dh_new = (dt / 6.0) * (k1_h + 2 * k2_h + 2 * k3_h + k4_h)
        dn_new = (dt / 6.0) * (k1_n + 2 * k2_n + 2 * k3_n + k4_n)

        return dV_new, dm_new, dh_new, dn_new


# Example of simulation step
def simulate_step(neuron, V, m, h, n, I_ext, dt):
    dVdt, dmdt, dhdt, dndt = neuron(V, m, h, n, I_ext)

    # Update values with Euler's method
    V_new = V + dVdt * dt
    m_new = m + dmdt * dt
    h_new = h + dhdt * dt
    n_new = n + dndt * dt

    return V_new, m_new, h_new, n_new


def run():
    # Initialize the neuron
    hh_neuron = HHNeuron()

    # Initial conditions
    V = torch.tensor(-65.0)  # Membrane potential
    m = torch.tensor(0.05)
    h = torch.tensor(0.6)
    n = torch.tensor(0.32)
    I_ext = torch.tensor(10.0)  # External current
    dt = 0.01  # Time step

    # Simulate one step
    V, m, h, n = simulate_step(hh_neuron, V, m, h, n, I_ext, dt)
    print(f"V: {V}, m: {m}, h: {h}, n: {n}")