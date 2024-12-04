# Adaptive_RL
<div align="center">
<img src="utilities_repository/Adaptive_RL_CPG_Logo.jpg" alt="Adaptive_RL" width=40% height=40%>
</div>
Adaptive_RL is an open-source framework for implementing state-of-the-art Reinforcement Learning (RL) algorithms, 
designed to be modular and easily adaptable for different research and real-world applications. This repository has 
emphasis in Continuous space problems.

Current implementation for Gymnasium and MyoSuite deployment, using torch.

Currently, the repository supports DDPG (Deep Deterministic Policy Gradient), SAC (Soft Actor-Critic), PPO (Proximal Policy Optimization),
and MPO (Maximum a Posteriori Optimization) with a flexible architecture that allows for the integration of new algorithms.

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
- PPO: Strikes a balance between performance and simplicity. It prevents large policy updates, stabilizing the 
learning process by clipping updates to the policy to keep changes within a fixed range, 
ensuring more stable and reliable training performance.

**CPG**

Development of a DRL framework in the MatsuokaOscillator Folder.

MatsuokaOscillator Directory created with multiple neurons and multiple oscillators implementation for oscillated
pattern movements. To be controlled by manual parameters or a DRL algorithm.

Hudgkin and Huxley Neurons: Simulate ionic currents such as sodium and potassium, allowing the system to generate more
biologically accurate oscillations. This addition provides the flexibility to model more complex neuronal behaviors and 
enriches the simulation of Central Pattern Generators (CPGs) in robotic control, prosthetics, and other neural control systems.

**Quickstart**

Clone the repository and install dependencies:
```
git clone https://github.com/YusseffRuiz/Adaptive_RL
cd Adaptive_RL
pip install -r requirements.txt
```

**How to use**

Running the Training Script
You can use the training script to train an agent on a specified environment using different algorithms like PPO, SAC, MPO, or DDPG. The training script supports several command-line arguments to customize the training process.

Here is how to run the training script:

To Run the SAC agent on the Mujoco Humanoid-v4 environment

```
python train.py --algorithm SAC --env Humanoid-v4 --steps 1000000
```


**To Plot:**

Plot every algorithm found inside a folder (you can specify only a specific folder):

```
python plot.py --path training/ --x_axis "train/seconds" --x_label "Time (s)"  --title "Training Progress" --name "DDPG-Training"
```

To plot multiple folders:
```
python plot.py --path training/Walker2d/Walker2d-v4-DDPG training/Walker2d/Walker2d-v4-DDPG-CPG/ --x_axis "train/seconds" --y_axis "test/episode_score/mean" --title "Training Progress DDPG" --name "DDPG-Training"
```

Test an Agent:

```
python simulate.py --algorithm SAC --env Humanoid-v4
```



**Parallelization training via Multiprocess class.**

## Command-Line Arguments

Below are the key arguments that can be passed to the script for customizing the training:

| Argument               | Type         | Default         | Description                                                                 |
|------------------------|--------------|-----------------|-----------------------------------------------------------------------------|
| `--algorithm`           | `str`        | `'PPO'`         | The RL algorithm to use for training. Options: `'PPO'`, `'SAC'`, `'MPO'`, `'DDPG'`. |
| `--env`                | `str`        | `'Humanoid-v4'` | The environment name to train on (e.g., `'Humanoid-v4'`, `'Walker2d-v4'`).   |
| `--cpg`                | `flag`       | `False`         | Enable Central Pattern Generator (CPG) for training.                        |
| `--f`                  | `str`        | `None`          | Folder to save logs, models, and results. If not specified, it will create one based on the environment and algorithm. |
| `--experiment_number`   | `int`        | `0`             | Specify the experiment number for logging purposes.                         |
| `--steps`              | `int`        | `1e7`           | Maximum steps for training.                                                 |
| `--seq`                | `int`        | `1`             | Number of sequential environments.                                          |
| `--parallel`           | `int`        | `1`             | Number of parallel environments to run.                                     |

### Hyperparameters

| Argument               | Type         | Default         | Description                                                                 |
|------------------------|--------------|-----------------|-----------------------------------------------------------------------------|
| `--learning_rate`      | `float`      | `3.57e-05`      | Learning rate for the actor network.                                        |
| `--lr_critic`          | `float`      | `3e-4`          | Learning rate for the critic network.                                       |
| `--ent_coeff`          | `float`      | `0.00238306`    | Entropy coefficient used by PPO or SAC algorithms.                          |
| `--clip_range`         | `float`      | `0.3`           | Clip range for PPO.                                                         |
| `--lr_dual`            | `float`      | `0.000356987`   | Learning rate for the dual variables (MPO).                                 |
| `--gamma`              | `float`      | `0.95`          | Discount factor.                                                            |
| `--neuron_number`      | `int/list`   | `256`           | Number of neurons in hidden layers. Can be a single integer or a list for specifying different layer sizes. |
| `--layers_number`      | `int`        | `2`             | Number of hidden layers in the neural network.                              |
| `--batch_size`         | `int`        | `256`           | Batch size used during training.                                            |
| `--replay_buffer_size` | `int`        | `10e5`          | Size of the replay buffer used by off-policy algorithms like SAC and DDPG.   |
| `--epsilon`            | `float`      | `0.1`           | Exploration rate for epsilon-greedy algorithms.                             |
| `--learning_starts`    | `int`        | `10000`         | Number of steps before learning starts.                                     |
| `--noise`              | `float`      | `0.01`          | Noise added to future rewards to promote exploration (DDPG).                |

**Example Commands**

Train PPO on Humanoid-v4 for 1 million steps:
```
python train.py --algorithm PPO --env Humanoid-v4 --steps 1000000
```
Train SAC on Walker2d-v4 with a custom learning rate:

```
python train.py --algorithm SAC --env Walker2d-v4 --learning_rate 0.0003
```
Enable CPG training with DDPG:
```
python train.py --algorithm DDPG --env Humanoid-v4 --cpg
```
To Run the SAC agent on the Mujoco Humanoid-v4 environment with CPG

```
python train.py --algorithm SAC --env Humanoid-v4 --steps 1000000 --cpg
```

Train MPO on Humanoid-v4 with parallel environments and multiple sequenced environments:
```
python train.py --algorithm MPO --env Humanoid-v4 --parallel 4 --seq 4
```

Train SAC with optimized HyperParameters based on rl3-zoo library:
```
python train.py --algorithm SAC --env Humanoid-v4 --params utilities_repository/hyperparameters.yaml
```


**To Do:**

ARS, TRPO implementation, analysing other state-of-the-art algorithms.
Tensorflow Implementation.

**Upcoming Features**
- Integration of Trust Region Policy Optimization (TRPO) and Augmented Random Search (ARS).
- Enhanced support for discrete action spaces.

**Credit**

This repository is based on [Tonic RL](https://github.com/fabiopardo/tonic) and inspired by SB3
Based on Tonic RL Library and modified to match current Gymnasium implementation and for MyoSuite development.

Changes includes:
- Direct control over learning rates and neuron size.
- Simplification of classes.
- Updated Libraries.
- Parallelization of algorithm and buffer by using torch tensors.