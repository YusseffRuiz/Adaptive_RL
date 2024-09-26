# Adaptive_RL

Adaptive_RL is an open-source framework for implementing state-of-the-art Reinforcement Learning (RL) algorithms, 
designed to be modular and easily adaptable for different research and real-world applications. This repository has 
emphasis in Continuous space problems

Currently, the repository supports DDPG (Deep Deterministic Policy Gradient), SAC (Soft Actor-Critic)  and MPO (Maximum 
a Posteriori Optimization) with a flexible architecture that allows for the integration of new algorithms.

**Features:**

Replay Buffer and experience sampling to stabilize training.


Customizable neural network architectures for the actor and critic networks.


Modular framework allowing for easy extension to additional algorithms.
Comprehensive logging for training metrics and performance evaluation.

**Implemented Algorithms**
- DDPG: Suitable for environments requiring deterministic, high-precision control. Works well in tasks such as robotic
control or simulated physical systems.
- SAC: Best used in tasks requiring robust exploration, where a stochastic policy can adapt better to uncertain 
environments.
- MPO: Ideal for complex environments requiring a careful balance between policy stability and adaptability.


**CPG**

Development of a DRL framework in the MatsuokaOscillator Folder.

MatsuokaOscillator Directory created with multiple neurons and multiple oscillators implementation for oscillated
pattern movements. To be controlled by manual parameters or a DRL algorithm.


**Quickstart**

Clone the repository and install dependencies:
```
git clone https://github.com/YusseffRuiz/Adaptive_RL
cd Adaptive_RL
pip install -r requirements.txt
```

**How to use**

To train a new environment, everything can be run in main.py class.

Create the agent, specifying learning rates, however, they can be left for the default values:

**Example Usage**

To Run the SAC agent on the Mujoco Walker2d-v4 environment

```
from agents import SAC
from environments import GymEnvironment

env = GymEnvironment('Walker2d-v4')
agent = SAC()
agent.train(env, episodes=1000)
```

**Sequential:** Run environments in a for loop.

**Parallel:** Run different processes over either CPU or GPU, using multiprocess class.


**To Do:**

ARS, PPO implementation.

MyoSuite environment builder and algorithm adaptation for muscles and DoF.
At the moment, it can only interact with Gymnasium environments.

Automatic CPG adding into the environment. At the moment, CPG must be added into the env class.
Implementation must be directly into the RL class.

Development and Implementation of Hodgkin-Huxley neurons CPG. 


**Credit**

This repository is based on [Tonic RL](https://github.com/fabiopardo/tonic).
Based on Tonic RL Library and modified to match current Gymnasium implementation and for MyoSuite development.

Changes includes:
- Direct control over learning rates and neuron size.
- Usage of Sigmoid Linear Unit instead of Relu
- Simplification of classes.