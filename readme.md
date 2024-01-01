# Sim2Real Manipulation on Unknown Objects with Tactile-based Reinforcement Learning (ICRA2024 under review)



## Installation

```shell
conda env create -f env.yaml
```


## Quick Start: 

**stable_baselines3**: Customized for the tactile RL training(plus DAgger)

**assets**: Customized Objects for the manipulation tasks

**conf**: configuration files for the tactile simulation

**estimator**: baselines methods for the angle estimation from tactile images.

**eval**: Evaluation tool for the tasks(pivoting, insertion, stable placing)

**simulation**: 

    **env**: Simulation environment for a batch of tactile based manipulation tasks on Xarm and allegro hand
    
    **tactile_render**: Simulation environment for the tactile images.
    
**main**: Training for the tactile based，point-cloud, DAgger policy for the manipulation tasks
