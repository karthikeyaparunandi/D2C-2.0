copyright @ Karthikeya S Parunandi

# Decoupled Data-Based Approach for Learning to Control Nonlinear Systems using Model free DDP - D2C-2.0
This repository is an implementation of model free DDP using MuJoCo simulator.

Working systems:

- Inverted Pendulum
- Cartpole
- Acrobot
- 3-link swimmer
- 6-link swimmer
- Fish


Libraries needed:
- numpy-1.14.5
- mujoco_py
- json

Simulator:
- Mujoco
- Some models from Deepmind's control suite


Main files:
- model_free_DDP.py (model free DDP/ILQR)
- ltv_sys_id.py (for system identification)
