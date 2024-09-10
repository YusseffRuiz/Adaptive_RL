**CPG + MPO Implementation Repository**

Development of a DRL library for the use in Gymnasium and Myosuite.

Based on Tonic RL Library and modified to match current Gymnasium implementation and for MyoSuite development.

MatsuokaOscillator Directory created with multiple neurons and multiple oscillators implementation for oscillated
pattern movements. To be controlled by manual parameters or a DRL algorithm.

**How to use**

To train a new environment, everything can be run in main.py class.

Create the agent, specifying learning rates, however, they can be left for the default values:

```
agent = MPO_Algorithm.agents.MPO(lr_actor=3.53e-5, lr_critic=6.081e-5, lr_dual=0.00213)
```  

main training function is to be used with the following command:

```
train_mpo(agent=agent,
                environment=env_name,
                sequential=2, parallel=3,
                trainer=MPO_Algorithm.Trainer(steps=max_steps, epoch_steps=epochs, save_steps=save_steps),
                log_dir=log_dir)
```
**Sequential:** Run environments in a for loop.

**Parallel:** Run different processes over either CPU or GPU, using multiprocess class.


**To Do:**

ARS, DDPG, SAC implementation.

MyoSuite environment builder and algorithm adaptation for muscles and DoF.
At the moment, it can only interact with Gymnasium environments.

Automatic CPG adding into the environment. At the moment, CPG must be added into the env class.
Implementation must be directly into the RL class.

Development and Implementation of Hodgkin-Huxley neurons CPG. 


**Tested Environments**


| Model name                                                         | Algorithm | Movement                                              |
|--------------------------------------------------------------------|-----------|-------------------------------------------------------|
| **Walker-2d-v4** <br> - 8 Action Space <br> - 17 Observation State | SAC       | ![](https://github.com/YusseffRuiz/MPO_CPG/blob/main/Experiments/Videos/mpo-agent-Walker2d-v4-CPG.mp4)     |
|                                                                    | SAC + CPG | ![](Experiments/Videos/sac-agent-Walker2d-v4-CPG.mp4) |
|                                                                    | MPO       | ![](Experiments/Videos/mpo-agent-Walker2d-v4.mp4)     |
|                                                                    | MPO + CPG | ![](Experiments/Videos/mpo-agent-Walker2d-v4-CPG.mp4) |


**Credit**

This repository is based on [Tonic RL](https://github.com/fabiopardo/tonic).