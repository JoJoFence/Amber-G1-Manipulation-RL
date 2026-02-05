# Amber-G1-Manipulation-RL

Reinforcement learning for bimanual manipulation on the Unitree G1 humanoid robot using [Isaac Lab](https://isaac-sim.github.io/IsaacLab/).

![G1 Reaching Demo](assets/reach_demo_video.mp4)


## Overview

This project trains the G1 humanoid's upper body (14 DOF dual arms) to perform precise bimanual reaching tasks. The policy learns to:

- Move each arm independently to its respective target
- Reach targets with ~3-6cm precision
- Hold position at target without drift
- Handle domain randomization for sim-to-real transfer

### Key Features

- **Position-only Differential IK**: 3D position control per arm (6D action space total)
- **Error-vector observations**: Direct target-relative encoding eliminates arm-target confusion
- **Multi-level reward shaping**: Coarse → medium → fine tracking with anti-drift penalties
- **Domain randomization**: Joint noise, base pose variation for robustness

## Results

| Metric | Value |
|--------|-------|
| Position Error (avg) | 6.0 - 6.5 cm |
| Success Rate (< 3cm) | ~35% of episode |
| Training Time | ~2.5 hours (5000 iterations) |
| Environment | 4096 parallel envs on RTX 4090 |

<!-- Add your results images/videos here -->
<!-- ![Training Curves](assets/training_curves.png) -->
<!-- ![Reaching Demo](assets/reaching_demo.mp4) -->

## Project Structure

```
Amber-G1-Manipulation-RL/
├── g1_description/                 # Robot description files
│   ├── g1_dual_arm.urdf           # G1 upper body URDF
│   └── config.yaml                # Asset configuration
│
├── g1_upper_body_tasks/           # Isaac Lab task implementation
│   ├── g1_tasks/
│   │   ├── g1_upper_body_cfg.py   # Robot articulation config
│   │   └── reach_g1_upper_body/   # Reaching task
│   │       ├── reach_env_cfg.py   # Environment configuration
│   │       ├── mdp/
│   │       │   ├── rewards.py     # Custom reward functions
│   │       │   └── observations.py # Custom observations
│   │       └── agents/
│   │           └── rsl_rl_ppo_cfg.py  # PPO hyperparameters
│   │
│   ├── scripts/
│   │   ├── train_g1_upper.py      # Training script
│   │   └── play_g1_upper.py       # Visualization script
│   │
│   └── setup.py
│
└── README.md
```

## Installation

### Prerequisites

- Ubuntu 22.04 / 24.04
- NVIDIA GPU with RTX 30xx or newer (tested on RTX 4090)
- [Isaac Sim 4.5+](https://developer.nvidia.com/isaac-sim)
- [Isaac Lab 1.0+](https://isaac-sim.github.io/IsaacLab/)

### Setup

1. **Install Isaac Lab** following the [official instructions](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

2. **Clone this repository**
   ```bash
   git clone https://github.com/JoJoFence/Amber-G1-Manipulation-RL.git
   cd Amber-G1-Manipulation-RL
   ```

3. **Install the task package**
   ```bash
   cd g1_upper_body_tasks
   pip install -e .
   ```

4. **Update the URDF path** in `g1_tasks/g1_upper_body_cfg.py`:
   ```python
   # Change this line to point to your local path:
   usd_path="/path/to/Amber-G1-Manipulation-RL/g1_description/configuration/g1_dual_arm_physics.usd"
   ```

   If you only have the URDF, convert it to USD using Isaac Sim's asset converter.

## Usage

### Training

```bash
cd g1_upper_body_tasks

# Train with default settings (4096 envs, headless)
python scripts/train_g1_upper.py --task Isaac-Reach-G1-Upper-v0 --num_envs 4096 --headless

# Train with visualization (fewer envs for performance)
python scripts/train_g1_upper.py --task Isaac-Reach-G1-Upper-v0 --num_envs 64

# Resume training from checkpoint
python scripts/train_g1_upper.py --task Isaac-Reach-G1-Upper-v0 --num_envs 4096 --headless \
    --resume --load_run <run_name>
```

Training logs are saved to `logs/rsl_rl/g1_upper_body_reach/`.

### Visualization

```bash
# Visualize the latest trained policy
python scripts/play_g1_upper.py --task Isaac-Reach-G1-Upper-v0 --num_envs 32

# Visualize a specific checkpoint
python scripts/play_g1_upper.py --task Isaac-Reach-G1-Upper-v0 --num_envs 32 \
    --load_run <run_name> --checkpoint <model_file.pt>
```

### TensorBoard

```bash
tensorboard --logdir logs/rsl_rl/g1_upper_body_reach --bind_all
```

Then open `http://localhost:6006` in your browser.

## Architecture

### Control Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Policy Network                        │
│                     [256, 128, 64] MLP                      │
├─────────────────────────────────────────────────────────────┤
│  Observations (46D)           │  Actions (6D)               │
│  ├─ Joint positions (14D)     │  ├─ Left arm Δposition (3D) │
│  ├─ Joint velocities (14D)    │  └─ Right arm Δposition (3D)│
│  ├─ Left EE error (3D)        │                             │
│  ├─ Right EE error (3D)       │                             │
│  ├─ Left EE velocity (3D)     │                             │
│  ├─ Right EE velocity (3D)    │                             │
│  └─ Last action (6D)          │                             │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│              Differential IK Controller                      │
│  ├─ Left arm: 4 DOF (shoulder + elbow)                      │
│  └─ Right arm: 4 DOF (shoulder + elbow)                     │
│  Wrists held at neutral by PD controllers                   │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    G1 Upper Body                             │
│            14 DOF: 7 per arm (3 shoulder + 1 elbow + 3 wrist)│
└─────────────────────────────────────────────────────────────┘
```

### Reward Structure

| Reward Term | Weight | Description |
|-------------|--------|-------------|
| **Tracking (per arm)** | | |
| Coarse (tanh, σ=0.5) | 2.0 | Gradient at any distance |
| Medium (tanh, σ=0.25) | 6.0 | Main tracking driver |
| Fine (tanh, σ=0.1) | 15.0 | Precision within 10cm |
| **Target Holding** | | |
| Holding reward | 10.0 | Low velocity when < 4cm |
| Success bonus | 15.0 | Binary reward when < 3cm |
| EE velocity penalty | -8.0 | Penalize drift when < 6cm |
| **Regularization** | | |
| Action rate | -1.0 | Smooth actions |
| Joint velocity | -0.005 | Damping |
| Joint acceleration | -0.002 | Smooth trajectories |

### Why Error Vectors?

Previous approaches used absolute EE positions and target positions as separate observations. This caused **arm-target confusion** where both arms would sometimes go to the same target.

By using **error vectors** (target - current position), each arm's observation directly encodes the direction and distance to *its own* target, making the arm-target association trivial for the policy to learn.

## Configuration

Key parameters in `reach_env_cfg.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `episode_length_s` | 8.0 | Episode duration |
| `decimation` | 2 | Control at 50Hz (sim at 100Hz) |
| `action_scale` | 0.03 | Max 3cm position delta per step |
| `resampling_time` | 4-6s | Target randomization interval |

Workspace bounds (body frame, relative to robot base):
- **X (forward)**: 0.20 - 0.45 m
- **Y (lateral)**: ±0.10 - 0.30 m
- **Z (vertical)**: -0.15 - 0.15 m

## Roadmap

- [x] Bimanual reaching with position control
- [x] Anti-drift holding behavior
- [ ] Wrist orientation control
- [ ] Dexterous hand integration
- [ ] Grasp task training
- [ ] Sim-to-real transfer

## Acknowledgments

- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) by NVIDIA
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl) by Robotic Systems Lab, ETH Zurich
- [Unitree Robotics](https://www.unitree.com/) for the G1 humanoid platform

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{amber_g1_manipulation_rl,
  author = {Hansen, Jonas},
  title = {Amber-G1-Manipulation-RL: Bimanual Manipulation for Unitree G1},
  year = {2025},
  url = {https://github.com/JoJoFence/Amber-G1-Manipulation-RL}
}
```
