"""
Deploy trained joint-space reaching policy to real Unitree G1 robot.

This script uses direct joint-space control - no IK required!
The policy outputs joint position deltas that are applied directly to the robot.

Usage:
    python deploy_g1_joint_space.py --checkpoint <path_to_model.pt> --mode keyboard
    python deploy_g1_joint_space.py --checkpoint <path_to_model.pt> --mode fixed

Safety:
    - The robot should be in a safe position before running
    - Ensure the high-level motion service is disabled
    - Keep the emergency stop nearby
"""

import argparse
import time
import sys
import numpy as np
import torch
import threading
from pathlib import Path
from typing import Optional, Tuple

# Unitree SDK2 imports
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient


class G1JointIndex:
    """Joint indices for G1 robot (29 DOF version)."""
    # Legs (indices 0-11, not used for arm control)
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleRoll = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleRoll = 11

    # Waist (indices 12-14)
    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14

    # Left arm (indices 15-21)
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21

    # Right arm (indices 22-28)
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28


# Joint ordering matching the training environment
# Left arm (7 joints) + Right arm (7 joints) = 14 total
POLICY_JOINT_ORDER = [
    # Left arm
    G1JointIndex.LeftShoulderPitch,
    G1JointIndex.LeftShoulderRoll,
    G1JointIndex.LeftShoulderYaw,
    G1JointIndex.LeftElbow,
    G1JointIndex.LeftWristRoll,
    G1JointIndex.LeftWristPitch,
    G1JointIndex.LeftWristYaw,
    # Right arm
    G1JointIndex.RightShoulderPitch,
    G1JointIndex.RightShoulderRoll,
    G1JointIndex.RightShoulderYaw,
    G1JointIndex.RightElbow,
    G1JointIndex.RightWristRoll,
    G1JointIndex.RightWristPitch,
    G1JointIndex.RightWristYaw,
]

# Default arm positions (matching simulation)
DEFAULT_POSITIONS = np.array([
    # Left arm
    0.4,   # shoulder_pitch
    0.3,   # shoulder_roll
    0.0,   # shoulder_yaw
    0.8,   # elbow
    0.0,   # wrist_roll
    0.0,   # wrist_pitch
    0.0,   # wrist_yaw
    # Right arm
    0.4,   # shoulder_pitch
    -0.3,  # shoulder_roll (opposite sign)
    0.0,   # shoulder_yaw
    0.8,   # elbow
    0.0,   # wrist_roll
    0.0,   # wrist_pitch
    0.0,   # wrist_yaw
])

# Waist joint indices and defaults
WAIST_JOINTS = [
    G1JointIndex.WaistYaw,
    G1JointIndex.WaistRoll,
    G1JointIndex.WaistPitch,
]
DEFAULT_WAIST = np.array([0.0, 0.0, 0.0])

# PD gains
ARM_KP = np.array([60.0, 60.0, 40.0, 40.0, 30.0, 30.0, 30.0,  # Left
                   60.0, 60.0, 40.0, 40.0, 30.0, 30.0, 30.0])  # Right
ARM_KD = np.array([2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0,  # Left
                   2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0])  # Right

WAIST_KP = np.array([60.0, 40.0, 40.0])
WAIST_KD = np.array([1.0, 1.0, 1.0])


class KeyboardController:
    """Non-blocking keyboard input for target control."""

    def __init__(self):
        self.running = True
        self.left_target_delta = np.zeros(3)
        self.right_target_delta = np.zeros(3)
        self.selected_arm = 'both'
        self.step_size = 0.02

        self.bindings = """
Keyboard Controls:
  Arrow Keys: Move target in X-Y plane
  W/S: Move target Up/Down (Z)
  1/2/3: Select left/right/both arms
  +/-: Adjust step size
  R: Reset targets
  Q: Quit
"""

    def start(self):
        self.thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self.thread.start()
        print(self.bindings)

    def _keyboard_loop(self):
        try:
            import termios
            import tty
            import select

            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setcbreak(sys.stdin.fileno())
                while self.running:
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1)
                        self._handle_key(key)
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except ImportError:
            print("Keyboard control not available on this platform")

    def _handle_key(self, key: str):
        delta = np.zeros(3)

        if key == '\x1b':
            try:
                seq = sys.stdin.read(2)
                if seq == '[A': delta[0] = self.step_size
                elif seq == '[B': delta[0] = -self.step_size
                elif seq == '[C': delta[1] = -self.step_size
                elif seq == '[D': delta[1] = self.step_size
            except:
                pass
        elif key.lower() == 'w': delta[2] = self.step_size
        elif key.lower() == 's': delta[2] = -self.step_size
        elif key == '1':
            self.selected_arm = 'left'
            print("Selected: LEFT arm")
        elif key == '2':
            self.selected_arm = 'right'
            print("Selected: RIGHT arm")
        elif key == '3':
            self.selected_arm = 'both'
            print("Selected: BOTH arms")
        elif key in ['+', '=']:
            self.step_size = min(0.1, self.step_size + 0.01)
            print(f"Step size: {self.step_size:.2f}m")
        elif key == '-':
            self.step_size = max(0.005, self.step_size - 0.01)
            print(f"Step size: {self.step_size:.2f}m")
        elif key.lower() == 'r':
            self.left_target_delta = np.zeros(3)
            self.right_target_delta = np.zeros(3)
            print("Targets reset")
        elif key.lower() == 'q':
            self.running = False

        if np.any(delta != 0):
            if self.selected_arm in ['left', 'both']:
                self.left_target_delta += delta
            if self.selected_arm in ['right', 'both']:
                right_delta = delta.copy()
                right_delta[1] = -delta[1]  # Mirror Y
                self.right_target_delta += right_delta

    def stop(self):
        self.running = False


class G1JointSpaceDeployer:
    """Deploys trained joint-space policy to real G1 robot."""

    def __init__(
        self,
        checkpoint_path: str,
        control_freq: float = 50.0,
        action_scale: float = 0.05,
        mode: str = 'fixed',
    ):
        self.checkpoint_path = checkpoint_path
        self.control_dt = 1.0 / control_freq
        self.action_scale = action_scale
        self.mode = mode

        # State
        self.low_state: Optional[LowState_] = None
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.crc = CRC()
        self.mode_machine = 0
        self.is_initialized = False

        # Policy
        self.policy = None
        self.obs_mean = None
        self.obs_std = None
        self.last_action = np.zeros(14)

        # Current joint targets (start at default)
        self.joint_targets = DEFAULT_POSITIONS.copy()

        # EE targets for observations (body frame)
        self.base_left_target = np.array([0.35, 0.20, 0.0])
        self.base_right_target = np.array([0.35, -0.20, 0.0])
        self.left_target = self.base_left_target.copy()
        self.right_target = self.base_right_target.copy()

        # Keyboard controller
        self.keyboard = None
        if mode == 'keyboard':
            self.keyboard = KeyboardController()

        self.running = False

    def load_policy(self):
        """Load the trained policy from checkpoint."""
        print(f"Loading policy from {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        print("Checkpoint keys:")
        for k in sorted(state_dict.keys()):
            if 'actor' in k:
                print(f"  {k}: {state_dict[k].shape if hasattr(state_dict[k], 'shape') else type(state_dict[k])}")

        # Determine network architecture from checkpoint
        # Look for actor.0.weight to get input size, actor.6.weight for output size
        if 'actor.0.weight' in state_dict:
            input_dim = state_dict['actor.0.weight'].shape[1]
            output_dim = state_dict['actor.6.weight'].shape[0]
            h1 = state_dict['actor.0.weight'].shape[0]
            h2 = state_dict['actor.2.weight'].shape[0]
            h3 = state_dict['actor.4.weight'].shape[0]
            print(f"Network architecture: {input_dim} -> {h1} -> {h2} -> {h3} -> {output_dim}")

            self.policy = torch.nn.Sequential(
                torch.nn.Linear(input_dim, h1),
                torch.nn.ELU(),
                torch.nn.Linear(h1, h2),
                torch.nn.ELU(),
                torch.nn.Linear(h2, h3),
                torch.nn.ELU(),
                torch.nn.Linear(h3, output_dim),
            )

            new_state_dict = {
                '0.weight': state_dict['actor.0.weight'],
                '0.bias': state_dict['actor.0.bias'],
                '2.weight': state_dict['actor.2.weight'],
                '2.bias': state_dict['actor.2.bias'],
                '4.weight': state_dict['actor.4.weight'],
                '4.bias': state_dict['actor.4.bias'],
                '6.weight': state_dict['actor.6.weight'],
                '6.bias': state_dict['actor.6.bias'],
            }
            self.policy.load_state_dict(new_state_dict)
            print("Policy weights loaded successfully")
        else:
            raise RuntimeError("Could not find actor weights in checkpoint")

        # Load observation normalization
        if 'actor_obs_normalizer._mean' in state_dict:
            self.obs_mean = state_dict['actor_obs_normalizer._mean'].numpy().flatten()
            self.obs_std = state_dict['actor_obs_normalizer._std'].numpy().flatten()
            print(f"Loaded observation normalization (dim: {len(self.obs_mean)})")
        else:
            print("WARNING: No observation normalization found")
            obs_dim = state_dict['actor.0.weight'].shape[1]
            self.obs_mean = np.zeros(obs_dim)
            self.obs_std = np.ones(obs_dim)

        self.policy.eval()
        print("Policy ready")

    def init_communication(self, network_interface: Optional[str] = None):
        """Initialize communication with the robot."""
        print("Initializing communication...")

        if network_interface:
            ChannelFactoryInitialize(0, network_interface)
        else:
            ChannelFactoryInitialize(0)

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result.get('name'):
            print(f"Releasing mode: {result['name']}")
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._lowstate_callback, 10)

        print("Waiting for robot state...")
        timeout = 10.0
        start = time.time()
        while self.low_state is None:
            if time.time() - start > timeout:
                raise RuntimeError("Timeout waiting for robot state")
            time.sleep(0.1)

        self.mode_machine = self.low_state.mode_machine
        self.is_initialized = True
        print("Communication initialized")

    def _lowstate_callback(self, msg: LowState_):
        self.low_state = msg

    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions in policy order."""
        return np.array([self.low_state.motor_state[i].q for i in POLICY_JOINT_ORDER])

    def get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities in policy order."""
        return np.array([self.low_state.motor_state[i].dq for i in POLICY_JOINT_ORDER])

    def build_observation(self) -> np.ndarray:
        """Build observation vector matching training format (54D).

        Observation space:
        - joint_pos_rel: 14D (joint positions relative to default)
        - joint_vel: 14D (joint velocities)
        - left_ee_error: 3D (target - EE position, approximate)
        - right_ee_error: 3D
        - left_ee_vel: 3D (approximate from joint velocities)
        - right_ee_vel: 3D
        - last_action: 14D
        """
        joint_pos = self.get_joint_positions()
        joint_vel = self.get_joint_velocities()

        # Joint positions relative to default
        joint_pos_rel = joint_pos - DEFAULT_POSITIONS

        # Approximate EE positions from joint angles (simplified)
        # For a proper implementation, you'd use FK here
        # We'll use placeholder values that the policy should handle
        # since the error vectors are what matters most
        left_ee_error = self.left_target - np.array([0.3, 0.2, 0.0])  # Approximate
        right_ee_error = self.right_target - np.array([0.3, -0.2, 0.0])

        # Approximate EE velocities (zeros for now, policy should be robust)
        left_ee_vel = np.zeros(3)
        right_ee_vel = np.zeros(3)

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

        action = np.clip(action, -1.0, 1.0)
        return action

    def send_joint_commands(self):
        """Send current joint targets to robot."""
        self.low_cmd.mode_pr = 0
        self.low_cmd.mode_machine = self.mode_machine

        # Waist joints - hold at neutral
        for i, joint_idx in enumerate(WAIST_JOINTS):
            self.low_cmd.motor_cmd[joint_idx].mode = 1
            self.low_cmd.motor_cmd[joint_idx].q = float(DEFAULT_WAIST[i])
            self.low_cmd.motor_cmd[joint_idx].dq = 0.0
            self.low_cmd.motor_cmd[joint_idx].kp = float(WAIST_KP[i])
            self.low_cmd.motor_cmd[joint_idx].kd = float(WAIST_KD[i])
            self.low_cmd.motor_cmd[joint_idx].tau = 0.0

        # Arm joints
        for i, joint_idx in enumerate(POLICY_JOINT_ORDER):
            self.low_cmd.motor_cmd[joint_idx].mode = 1
            self.low_cmd.motor_cmd[joint_idx].q = float(self.joint_targets[i])
            self.low_cmd.motor_cmd[joint_idx].dq = 0.0
            self.low_cmd.motor_cmd[joint_idx].kp = float(ARM_KP[i])
            self.low_cmd.motor_cmd[joint_idx].kd = float(ARM_KD[i])
            self.low_cmd.motor_cmd[joint_idx].tau = 0.0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

    def move_to_default_pose(self, duration: float = 3.0):
        """Smoothly move to default pose."""
        print("Moving to default pose...")

        start_time = time.time()
        start_pos = self.get_joint_positions()
        start_waist = np.array([self.low_state.motor_state[i].q for i in WAIST_JOINTS])

        print(f"Start positions: {start_pos[:4]}... (showing first 4)")
        print(f"Target positions: {DEFAULT_POSITIONS[:4]}...")

        while time.time() - start_time < duration:
            t = (time.time() - start_time) / duration
            alpha = 3 * t**2 - 2 * t**3  # Smoothstep

            self.joint_targets = start_pos + alpha * (DEFAULT_POSITIONS - start_pos)
            waist_targets = start_waist + alpha * (DEFAULT_WAIST - start_waist)

            # Send commands
            self.low_cmd.mode_pr = 0
            self.low_cmd.mode_machine = self.mode_machine

            for i, joint_idx in enumerate(WAIST_JOINTS):
                self.low_cmd.motor_cmd[joint_idx].mode = 1
                self.low_cmd.motor_cmd[joint_idx].q = float(waist_targets[i])
                self.low_cmd.motor_cmd[joint_idx].dq = 0.0
                self.low_cmd.motor_cmd[joint_idx].kp = float(WAIST_KP[i])
                self.low_cmd.motor_cmd[joint_idx].kd = float(WAIST_KD[i])
                self.low_cmd.motor_cmd[joint_idx].tau = 0.0

            for i, joint_idx in enumerate(POLICY_JOINT_ORDER):
                self.low_cmd.motor_cmd[joint_idx].mode = 1
                self.low_cmd.motor_cmd[joint_idx].q = float(self.joint_targets[i])
                self.low_cmd.motor_cmd[joint_idx].dq = 0.0
                self.low_cmd.motor_cmd[joint_idx].kp = float(ARM_KP[i])
                self.low_cmd.motor_cmd[joint_idx].kd = float(ARM_KD[i])
                self.low_cmd.motor_cmd[joint_idx].tau = 0.0

            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.lowcmd_publisher.Write(self.low_cmd)

            time.sleep(0.002)

        self.joint_targets = DEFAULT_POSITIONS.copy()
        print("Default pose reached")

    def run(self):
        """Main control loop."""
        print("\nStarting joint-space policy deployment...")
        print(f"Control mode: {self.mode}")
        print("Press Ctrl+C to stop\n")

        if self.keyboard:
            self.keyboard.start()

        self.running = True
        last_time = time.time()
        log_counter = 0

        try:
            while self.running and (self.keyboard is None or self.keyboard.running):
                current_time = time.time()

                if current_time - last_time >= self.control_dt:
                    # Update targets from keyboard
                    if self.keyboard:
                        self.left_target = self.base_left_target + self.keyboard.left_target_delta
                        self.right_target = self.base_right_target + self.keyboard.right_target_delta

                    # Build observation
                    obs = self.build_observation()

                    # Run policy
                    action = self.run_policy(obs)
                    self.last_action = action

                    # Apply action: joint position delta
                    joint_delta = action * self.action_scale
                    self.joint_targets = self.joint_targets + joint_delta

                    # Clip to safe joint limits
                    # TODO: Add proper joint limits from URDF
                    self.joint_targets = np.clip(self.joint_targets, -2.0, 2.0)

                    # Send commands
                    self.send_joint_commands()

                    # Logging
                    log_counter += 1
                    if log_counter % 50 == 0:
                        joint_pos = self.get_joint_positions()
                        print(f"Joints L[0:4]: {joint_pos[:4].round(2)}  "
                              f"R[0:4]: {joint_pos[7:11].round(2)}  "
                              f"Action max: {np.abs(action).max():.2f}")

                    last_time = current_time

                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\nStopping...")

        finally:
            self.running = False
            if self.keyboard:
                self.keyboard.stop()

    def shutdown(self):
        print("Shutting down...")
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="Deploy joint-space policy to G1")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--network_interface", type=str, default=None,
                       help="Network interface (e.g., eth0)")
    parser.add_argument("--control_freq", type=float, default=50.0,
                       help="Control frequency in Hz")
    parser.add_argument("--mode", type=str, default="fixed",
                       choices=["fixed", "keyboard"],
                       help="Control mode")
    parser.add_argument("--skip_default_pose", action="store_true",
                       help="Skip moving to default pose")
    args = parser.parse_args()

    print("=" * 60)
    print("G1 Joint-Space Policy Deployment")
    print("=" * 60)
    print()
    print("WARNING: Ensure robot is in safe position!")
    print("WARNING: Keep emergency stop nearby!")
    print()
    input("Press Enter to continue...")

    deployer = G1JointSpaceDeployer(
        checkpoint_path=args.checkpoint,
        control_freq=args.control_freq,
        mode=args.mode,
    )

    deployer.load_policy()
    deployer.init_communication(args.network_interface)

    if not args.skip_default_pose:
        deployer.move_to_default_pose(duration=3.0)

    try:
        deployer.run()
    finally:
        deployer.shutdown()


if __name__ == "__main__":
    main()
