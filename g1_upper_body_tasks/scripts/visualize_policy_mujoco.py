"""
Visualize trained joint-space policy in MuJoCo.

Loads the G1 dual arm model, runs the trained policy, and renders in MuJoCo viewer.
Uses proper PD control dynamics to match Isaac Lab's ImplicitActuator behavior,
so the policy sees realistic joint positions and velocities.

Usage:
    python visualize_policy_mujoco.py
    python visualize_policy_mujoco.py --checkpoint path/to/model.pt
"""

import argparse
import numpy as np
import torch
import mujoco
import mujoco.viewer
import time
from pathlib import Path


# Joint names in policy order (must match training config and URDF tree order)
POLICY_JOINT_NAMES = [
    # Left arm (7 joints)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # Right arm (7 joints)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Default joint positions (matching g1_upper_body_cfg.py InitialStateCfg)
DEFAULT_POSITIONS = np.array([
    # Left arm
    0.4, 0.3, 0.0, 0.8, 0.0, 0.0, 0.0,
    # Right arm
    0.4, -0.3, 0.0, 0.8, 0.0, 0.0, 0.0,
])

# EE body names - wrist_yaw_link is the parent body of the rubber_hand geom
LEFT_EE_BODY = "left_wrist_yaw_link"
RIGHT_EE_BODY = "right_wrist_yaw_link"

# Rubber hand offset from wrist_yaw_link (from MJCF geom pos attributes)
# Must match to compute same EE position as Isaac Lab's left_rubber_hand body
LEFT_RUBBER_HAND_OFFSET = np.array([0.0415, 0.003, 0.0])
RIGHT_RUBBER_HAND_OFFSET = np.array([0.0415, -0.003, 0.0])

# PD gains matching Isaac Lab's ImplicitActuatorCfg
# Shoulders and elbows: kp=300, kd=100, effort_limit=25
# Wrists: kp=200, kd=80, effort_limit=5
PD_KP = np.array([300, 300, 300, 300, 200, 200, 200,
                   300, 300, 300, 300, 200, 200, 200], dtype=np.float64)
PD_KD = np.array([100, 100, 100, 100, 80, 80, 80,
                   100, 100, 100, 100, 80, 80, 80], dtype=np.float64)
EFFORT_LIMIT = np.array([25, 25, 25, 25, 5, 5, 5,
                          25, 25, 25, 25, 5, 5, 5], dtype=np.float64)


class MuJoCoPolicyVisualizer:
    def __init__(self, checkpoint_path: str, mjcf_path: str):
        # Load MuJoCo model
        print(f"Loading MuJoCo model from: {mjcf_path}")
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)

        # Set simulation timestep to match Isaac Lab (0.01s)
        self.model.opt.timestep = 0.01

        # Map joint names to MuJoCo joint IDs and addresses
        self.joint_ids = []
        self.qpos_addrs = []
        self.qvel_addrs = []
        self.actuator_ids = []
        for name in POLICY_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                raise ValueError(f"Joint '{name}' not found in model")
            self.joint_ids.append(jid)
            self.qpos_addrs.append(self.model.jnt_qposadr[jid])
            self.qvel_addrs.append(self.model.jnt_dofadr[jid])
            # Motor actuator has same name as joint
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.actuator_ids.append(aid)
        print(f"Mapped {len(self.joint_ids)} joints")

        # Get EE body IDs
        self.left_ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, LEFT_EE_BODY)
        self.right_ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_EE_BODY)
        print(f"Left EE body ID: {self.left_ee_id}, Right EE body ID: {self.right_ee_id}")

        # Load policy
        self.load_policy(checkpoint_path)

        # State
        self.last_action = np.zeros(14)
        self.joint_targets = DEFAULT_POSITIONS.copy()

        # Targets for reaching
        self.left_target = np.array([0.35, 0.20, 0.0])
        self.right_target = np.array([0.35, -0.20, 0.0])

        # Target resampling
        self.target_resample_time = 5.0
        self.last_resample = 0.0

        # Control settings matching training
        self.decimation = 2  # Policy runs every 2 physics steps
        self.action_scale = 0.05

    def load_policy(self, checkpoint_path: str):
        """Load trained policy from checkpoint."""
        print(f"Loading policy from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        if 'actor.0.weight' in state_dict:
            input_dim = state_dict['actor.0.weight'].shape[1]
            h1 = state_dict['actor.0.weight'].shape[0]
            h2 = state_dict['actor.2.weight'].shape[0]
            h3 = state_dict['actor.4.weight'].shape[0]
            output_dim = state_dict['actor.6.weight'].shape[0]
            print(f"Network: {input_dim} -> {h1} -> {h2} -> {h3} -> {output_dim}")

            self.policy = torch.nn.Sequential(
                torch.nn.Linear(input_dim, h1),
                torch.nn.ELU(),
                torch.nn.Linear(h1, h2),
                torch.nn.ELU(),
                torch.nn.Linear(h2, h3),
                torch.nn.ELU(),
                torch.nn.Linear(h3, output_dim),
            )

            new_state_dict = {}
            for layer_idx in [0, 2, 4, 6]:
                new_state_dict[f'{layer_idx}.weight'] = state_dict[f'actor.{layer_idx}.weight']
                new_state_dict[f'{layer_idx}.bias'] = state_dict[f'actor.{layer_idx}.bias']
            self.policy.load_state_dict(new_state_dict)
        else:
            raise RuntimeError("Could not find actor weights in checkpoint")

        if 'actor_obs_normalizer._mean' in state_dict:
            self.obs_mean = state_dict['actor_obs_normalizer._mean'].numpy().flatten()
            self.obs_std = state_dict['actor_obs_normalizer._std'].numpy().flatten()
            print(f"Loaded obs normalization (dim: {len(self.obs_mean)})")
        else:
            obs_dim = input_dim
            self.obs_mean = np.zeros(obs_dim)
            self.obs_std = np.ones(obs_dim)
            print("WARNING: No observation normalization found")

        self.policy.eval()
        print("Policy loaded successfully")

    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions using proper qpos addresses."""
        return np.array([self.data.qpos[addr] for addr in self.qpos_addrs])

    def get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities using proper dof addresses."""
        return np.array([self.data.qvel[addr] for addr in self.qvel_addrs])

    def get_ee_position(self, body_id: int, offset: np.ndarray) -> np.ndarray:
        """Get EE position including rubber hand offset (matching Isaac Lab)."""
        body_pos = self.data.xpos[body_id].copy()
        body_rot = self.data.xmat[body_id].reshape(3, 3)
        return body_pos + body_rot @ offset

    def get_ee_velocity(self, body_id: int) -> np.ndarray:
        """Get EE linear velocity in world frame."""
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data,
                                  mujoco.mjtObj.mjOBJ_BODY, body_id, vel, 0)
        # MuJoCo returns [angular(3), linear(3)] in local frame when flg=0
        # We want world frame linear velocity
        body_rot = self.data.xmat[body_id].reshape(3, 3)
        return body_rot @ vel[3:6]  # rotate local linear velocity to world frame

    def build_observation(self) -> np.ndarray:
        """Build observation vector matching training format (54D).

        Order matches reach_env_cfg.py ObservationsCfg.PolicyCfg:
        1. joint_pos_rel (14D) - joint positions relative to default
        2. joint_vel (14D) - joint velocities
        3. left_ee_error (3D) - target - ee_pos (world frame)
        4. right_ee_error (3D) - target - ee_pos (world frame)
        5. left_ee_vel (3D) - EE linear velocity (world frame)
        6. right_ee_vel (3D) - EE linear velocity (world frame)
        7. last_action (14D) - previous action
        """
        joint_pos = self.get_joint_positions()
        joint_pos_rel = joint_pos - DEFAULT_POSITIONS
        joint_vel = self.get_joint_velocities()

        # EE positions with rubber hand offset
        left_ee_pos = self.get_ee_position(self.left_ee_id, LEFT_RUBBER_HAND_OFFSET)
        right_ee_pos = self.get_ee_position(self.right_ee_id, RIGHT_RUBBER_HAND_OFFSET)

        # Error vectors (target - current) in world frame
        left_ee_error = self.left_target - left_ee_pos
        right_ee_error = self.right_target - right_ee_pos

        # EE velocities in world frame
        left_ee_vel = self.get_ee_velocity(self.left_ee_id)
        right_ee_vel = self.get_ee_velocity(self.right_ee_id)

        obs = np.concatenate([
            joint_pos_rel,    # 14D
            joint_vel,        # 14D
            left_ee_error,    # 3D
            right_ee_error,   # 3D
            left_ee_vel,      # 3D
            right_ee_vel,     # 3D
            self.last_action, # 14D
        ])

        return obs

    def run_policy(self, obs: np.ndarray) -> np.ndarray:
        """Run policy inference."""
        obs_normalized = (obs - self.obs_mean) / (self.obs_std + 1e-8)

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs_normalized).unsqueeze(0)
            action = self.policy(obs_tensor).squeeze(0).numpy()

        return np.clip(action, -100.0, 100.0)

    def apply_pd_control(self, targets: np.ndarray):
        """Apply PD control torques matching Isaac Lab's ImplicitActuator.

        Computes: torque = kp * (target - qpos) - kd * qvel
        Clips to effort limits matching the training config.
        """
        joint_pos = self.get_joint_positions()
        joint_vel = self.get_joint_velocities()

        torques = PD_KP * (targets - joint_pos) - PD_KD * joint_vel
        torques = np.clip(torques, -EFFORT_LIMIT, EFFORT_LIMIT)

        for i, aid in enumerate(self.actuator_ids):
            self.data.ctrl[aid] = torques[i]

    def randomize_targets(self):
        """Generate random reachable targets matching training command ranges."""
        self.left_target = np.array([
            np.random.uniform(0.20, 0.45),
            np.random.uniform(0.10, 0.30),
            np.random.uniform(-0.15, 0.15),
        ])
        self.right_target = np.array([
            np.random.uniform(0.20, 0.45),
            np.random.uniform(-0.30, -0.10),
            np.random.uniform(-0.15, 0.15),
        ])

    def run(self):
        """Main visualization loop with PD-controlled dynamics."""
        print("\n" + "=" * 60)
        print("MuJoCo Policy Visualization (PD Dynamics)")
        print("=" * 60)
        print("Targets will resample every 5 seconds")
        print("Close the viewer window to exit")
        print()

        # Initialize to default pose
        for i, addr in enumerate(self.qpos_addrs):
            self.data.qpos[addr] = DEFAULT_POSITIONS[i]
        for addr in self.qvel_addrs:
            self.data.qvel[addr] = 0.0
        # Start with PD holding the default pose
        self.joint_targets = DEFAULT_POSITIONS.copy()
        self.apply_pd_control(self.joint_targets)
        mujoco.mj_forward(self.model, self.data)

        # Let the PD controller settle for a moment
        for _ in range(100):
            self.apply_pd_control(self.joint_targets)
            mujoco.mj_step(self.model, self.data)

        # Print initial EE positions
        left_ee = self.get_ee_position(self.left_ee_id, LEFT_RUBBER_HAND_OFFSET)
        right_ee = self.get_ee_position(self.right_ee_id, RIGHT_RUBBER_HAND_OFFSET)
        print(f"Initial left EE:  {left_ee.round(4)}")
        print(f"Initial right EE: {right_ee.round(4)}")

        # Set initial targets near default EE positions
        self.left_target = left_ee + np.array([0.05, 0.0, 0.05])
        self.right_target = right_ee + np.array([0.05, 0.0, 0.05])
        print(f"Initial left target:  {self.left_target.round(4)}")
        print(f"Initial right target: {self.right_target.round(4)}")

        sim_time = 0.0
        log_counter = 0
        policy_step = 0

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                step_start = time.time()

                # Resample targets periodically
                if sim_time - self.last_resample > self.target_resample_time:
                    self.randomize_targets()
                    self.last_resample = sim_time
                    print(f"\n[{sim_time:.1f}s] New targets: "
                          f"L={self.left_target.round(3)} R={self.right_target.round(3)}")

                # Run policy (observations are read from current physics state)
                obs = self.build_observation()
                action = self.run_policy(obs)
                self.last_action = action

                # Compute joint position targets
                self.joint_targets = DEFAULT_POSITIONS + action * self.action_scale

                # Run physics with PD control for `decimation` substeps
                for _ in range(self.decimation):
                    self.apply_pd_control(self.joint_targets)
                    mujoco.mj_step(self.model, self.data)

                sim_time += self.decimation * self.model.opt.timestep
                policy_step += 1

                # Logging
                log_counter += 1
                if log_counter % 50 == 0:
                    left_ee = self.get_ee_position(self.left_ee_id, LEFT_RUBBER_HAND_OFFSET)
                    right_ee = self.get_ee_position(self.right_ee_id, RIGHT_RUBBER_HAND_OFFSET)
                    left_err = np.linalg.norm(self.left_target - left_ee)
                    right_err = np.linalg.norm(self.right_target - right_ee)
                    print(f"[{sim_time:.1f}s] L_err: {left_err:.3f}m  R_err: {right_err:.3f}m  "
                          f"Action max: {np.abs(action).max():.2f}")

                viewer.sync()

                # Real-time sync
                control_dt = self.decimation * self.model.opt.timestep
                elapsed = time.time() - step_start
                sleep_time = control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)


def main():
    parser = argparse.ArgumentParser(description="Visualize policy in MuJoCo")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint")
    args = parser.parse_args()

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        log_dir = Path(__file__).parent.parent / "logs" / "rsl_rl" / "g1_upper_body_reach_joint_space"
        if log_dir.exists():
            runs = sorted(log_dir.iterdir())
            if runs:
                latest_run = runs[-1]
                models = sorted(latest_run.glob("model_*.pt"),
                               key=lambda p: int(p.stem.split('_')[1]))
                if models:
                    checkpoint_path = str(models[-1])
                    print(f"Auto-found checkpoint: {checkpoint_path}")
                else:
                    raise FileNotFoundError(f"No model files in {latest_run}")
            else:
                raise FileNotFoundError(f"No runs in {log_dir}")
        else:
            raise FileNotFoundError(f"Log directory not found: {log_dir}")

    # Find MJCF
    mjcf_candidates = [
        Path(__file__).parent.parent.parent / "g1_description" / "g1_dual_arm.xml",
        Path("/home/jonas/g1_env/unitree_ros/robots/g1_description/g1_dual_arm.xml"),
    ]
    mjcf_path = None
    for p in mjcf_candidates:
        if p.exists():
            mjcf_path = str(p)
            break
    if mjcf_path is None:
        raise FileNotFoundError("Could not find g1_dual_arm.xml")

    viz = MuJoCoPolicyVisualizer(checkpoint_path, mjcf_path)
    viz.run()


if __name__ == "__main__":
    main()
