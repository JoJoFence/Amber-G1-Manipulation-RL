"""
Deploy trained reaching policy to real Unitree G1 robot.

This script:
1. Loads the trained policy network
2. Reads robot state via unitree_sdk2_python
3. Computes observations and runs policy inference
4. Converts actions to joint commands via differential IK
5. Sends joint position commands to the robot

Control modes:
- Fixed targets: Arms reach to predefined positions
- Keyboard: Use arrow keys to move targets in real-time

Usage:
    python deploy_g1_real.py --checkpoint <path_to_model.pt> --mode keyboard
    python deploy_g1_real.py --checkpoint <path_to_model.pt> --mode fixed

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
from typing import Optional, Dict, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Import kinematics
from g1_kinematics import G1ArmKinematics, DifferentialIK

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
    WaistRoll = 13   # NOTE: May be locked on g1 23dof/29dof with waist locked
    WaistPitch = 14  # NOTE: May be locked on g1 23dof/29dof with waist locked

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


# Waist joint indices
WAIST_JOINTS = [
    G1JointIndex.WaistYaw,
    G1JointIndex.WaistRoll,
    G1JointIndex.WaistPitch,
]

# Default waist position (neutral/zero)
DEFAULT_WAIST_POSITIONS = {
    G1JointIndex.WaistYaw: 0.0,
    G1JointIndex.WaistRoll: 0.0,
    G1JointIndex.WaistPitch: 0.0,
}

# PD gains for waist (from Unitree example)
WAIST_KP = {
    G1JointIndex.WaistYaw: 60.0,
    G1JointIndex.WaistRoll: 40.0,
    G1JointIndex.WaistPitch: 40.0,
}

WAIST_KD = {
    G1JointIndex.WaistYaw: 1.0,
    G1JointIndex.WaistRoll: 1.0,
    G1JointIndex.WaistPitch: 1.0,
}

# Arm joint indices (matching simulation order)
LEFT_ARM_JOINTS = [
    G1JointIndex.LeftShoulderPitch,
    G1JointIndex.LeftShoulderRoll,
    G1JointIndex.LeftShoulderYaw,
    G1JointIndex.LeftElbow,
    G1JointIndex.LeftWristRoll,
    G1JointIndex.LeftWristPitch,
    G1JointIndex.LeftWristYaw,
]

RIGHT_ARM_JOINTS = [
    G1JointIndex.RightShoulderPitch,
    G1JointIndex.RightShoulderRoll,
    G1JointIndex.RightShoulderYaw,
    G1JointIndex.RightElbow,
    G1JointIndex.RightWristRoll,
    G1JointIndex.RightWristPitch,
    G1JointIndex.RightWristYaw,
]

# IK-controlled joints (4 DOF per arm - shoulder + elbow only)
LEFT_ARM_IK_JOINTS = LEFT_ARM_JOINTS[:4]
RIGHT_ARM_IK_JOINTS = RIGHT_ARM_JOINTS[:4]

# Default arm positions (matching simulation)
DEFAULT_ARM_POSITIONS = {
    G1JointIndex.LeftShoulderPitch: 0.4,
    G1JointIndex.LeftShoulderRoll: 0.3,
    G1JointIndex.LeftShoulderYaw: 0.0,
    G1JointIndex.LeftElbow: 0.8,
    G1JointIndex.LeftWristRoll: 0.0,
    G1JointIndex.LeftWristPitch: 0.0,
    G1JointIndex.LeftWristYaw: 0.0,
    G1JointIndex.RightShoulderPitch: 0.4,
    G1JointIndex.RightShoulderRoll: -0.3,
    G1JointIndex.RightShoulderYaw: 0.0,
    G1JointIndex.RightElbow: 0.8,
    G1JointIndex.RightWristRoll: 0.0,
    G1JointIndex.RightWristPitch: 0.0,
    G1JointIndex.RightWristYaw: 0.0,
}

# PD gains for arms
ARM_KP = {i: 60.0 if i in [G1JointIndex.LeftShoulderPitch, G1JointIndex.LeftShoulderRoll,
                           G1JointIndex.RightShoulderPitch, G1JointIndex.RightShoulderRoll]
          else 40.0 for i in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS}

ARM_KD = {i: 2.0 if i in [G1JointIndex.LeftShoulderPitch, G1JointIndex.LeftShoulderRoll,
                          G1JointIndex.RightShoulderPitch, G1JointIndex.RightShoulderRoll]
          else 1.5 for i in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS}


class KeyboardController:
    """Non-blocking keyboard input for target control."""

    def __init__(self):
        self.running = True
        self.left_delta = np.zeros(3)
        self.right_delta = np.zeros(3)
        self.selected_arm = 'left'  # 'left', 'right', or 'both'
        self.step_size = 0.02  # 2cm per keypress

        # Key bindings
        self.bindings = """
Keyboard Controls:
  Arrow Keys: Move in X-Y plane
    Up/Down:    Forward/Backward (X)
    Left/Right: Left/Right (Y)

  W/S: Move Up/Down (Z)

  1: Control left arm only
  2: Control right arm only
  3: Control both arms

  +/-: Increase/Decrease step size
  R: Reset targets to default
  Q: Quit
"""

    def start(self):
        """Start keyboard listener thread."""
        self.thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self.thread.start()
        print(self.bindings)

    def _keyboard_loop(self):
        """Main keyboard reading loop."""
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
            # Windows fallback
            import msvcrt
            while self.running:
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8', errors='ignore')
                    self._handle_key(key)
                time.sleep(0.05)

    def _handle_key(self, key: str):
        """Handle a keypress."""
        delta = np.zeros(3)

        # Movement keys
        if key == '\x1b':  # Arrow key prefix
            # Read the rest of the escape sequence
            try:
                seq = sys.stdin.read(2)
                if seq == '[A':  # Up
                    delta[0] = self.step_size
                elif seq == '[B':  # Down
                    delta[0] = -self.step_size
                elif seq == '[C':  # Right
                    delta[1] = -self.step_size
                elif seq == '[D':  # Left
                    delta[1] = self.step_size
            except:
                pass
        elif key.lower() == 'w':
            delta[2] = self.step_size
        elif key.lower() == 's':
            delta[2] = -self.step_size
        elif key == '1':
            self.selected_arm = 'left'
            print("Selected: LEFT arm")
        elif key == '2':
            self.selected_arm = 'right'
            print("Selected: RIGHT arm")
        elif key == '3':
            self.selected_arm = 'both'
            print("Selected: BOTH arms")
        elif key == '+' or key == '=':
            self.step_size = min(0.1, self.step_size + 0.01)
            print(f"Step size: {self.step_size:.2f}m")
        elif key == '-':
            self.step_size = max(0.005, self.step_size - 0.01)
            print(f"Step size: {self.step_size:.2f}m")
        elif key.lower() == 'r':
            self.left_delta = np.zeros(3)
            self.right_delta = np.zeros(3)
            print("Targets reset")
        elif key.lower() == 'q':
            self.running = False
            print("Quitting...")

        # Apply delta to selected arm(s)
        if np.any(delta != 0):
            if self.selected_arm in ['left', 'both']:
                self.left_delta += delta
            if self.selected_arm in ['right', 'both']:
                # Mirror Y for right arm
                right_delta = delta.copy()
                right_delta[1] = -delta[1]
                self.right_delta += right_delta

    def get_targets(self, base_left: np.ndarray, base_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get current target positions."""
        return base_left + self.left_delta, base_right + self.right_delta

    def stop(self):
        self.running = False


class G1PolicyDeployer:
    """Deploys trained policy to real G1 robot."""

    def __init__(
        self,
        checkpoint_path: str,
        control_freq: float = 50.0,
        action_scale: float = 0.03,
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
        self.obs_var = None
        self.last_action = np.zeros(6)

        # Kinematics
        self.left_kin = G1ArmKinematics('left')
        self.right_kin = G1ArmKinematics('right')
        self.left_ik = DifferentialIK(self.left_kin, lambda_val=0.1)
        self.right_ik = DifferentialIK(self.right_kin, lambda_val=0.1)

        # Base target positions (body frame, matching simulation)
        self.base_left_target = np.array([0.35, 0.20, 0.0])
        self.base_right_target = np.array([0.35, -0.20, 0.0])
        self.left_target = self.base_left_target.copy()
        self.right_target = self.base_right_target.copy()

        # Keyboard controller
        self.keyboard = None
        if mode == 'keyboard':
            self.keyboard = KeyboardController()

        # Running state
        self.running = False

    def load_policy(self):
        """Load the trained policy from checkpoint."""
        print(f"Loading policy from {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)

        # RSL-RL checkpoint structure
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Print keys to understand structure
        actor_keys = [k for k in state_dict.keys() if 'actor' in k.lower()]
        print(f"Found actor keys: {actor_keys[:10]}...")

        # Build MLP matching training config [256, 128, 64]
        # Input: 46D observations, Output: 6D actions
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(46, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 6),
        )

        # Try to load weights - adjust based on RSL-RL naming
        try:
            # RSL-RL typically uses 'actor.0.weight', 'actor.0.bias', etc.
            new_state_dict = {}
            layer_map = {
                'actor.0': '0',  # First linear
                'actor.2': '2',  # Second linear
                'actor.4': '4',  # Third linear
                'actor.6': '6',  # Output linear
            }
            for old_key, new_prefix in layer_map.items():
                if f'{old_key}.weight' in state_dict:
                    new_state_dict[f'{new_prefix}.weight'] = state_dict[f'{old_key}.weight']
                    new_state_dict[f'{new_prefix}.bias'] = state_dict[f'{old_key}.bias']

            if new_state_dict:
                self.policy.load_state_dict(new_state_dict)
                print("Policy weights loaded successfully")
            else:
                print("Warning: Could not map policy weights, using random initialization")
        except Exception as e:
            print(f"Warning: Error loading weights: {e}")
            print("Using random initialization - policy may not work correctly")

        # Load observation normalization
        if 'obs_rms' in checkpoint:
            self.obs_mean = checkpoint['obs_rms']['mean'].numpy()
            self.obs_var = checkpoint['obs_rms']['var'].numpy()
            print("Loaded observation normalization")
        else:
            self.obs_mean = np.zeros(46)
            self.obs_var = np.ones(46)
            print("Using default observation normalization")

        self.policy.eval()

    def init_communication(self, network_interface: Optional[str] = None):
        """Initialize communication with the robot."""
        print("Initializing communication with G1...")

        if network_interface:
            ChannelFactoryInitialize(0, network_interface)
        else:
            ChannelFactoryInitialize(0)

        # Disable high-level motion service
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result.get('name'):
            print(f"Releasing mode: {result['name']}")
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

        # Create publisher and subscriber
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._lowstate_callback, 10)

        # Wait for first state
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
        """Callback for robot state updates."""
        self.low_state = msg

    def get_arm_joint_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current arm joint positions."""
        left_pos = np.array([self.low_state.motor_state[i].q for i in LEFT_ARM_JOINTS])
        right_pos = np.array([self.low_state.motor_state[i].q for i in RIGHT_ARM_JOINTS])
        return left_pos, right_pos

    def get_arm_joint_velocities(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current arm joint velocities."""
        left_vel = np.array([self.low_state.motor_state[i].dq for i in LEFT_ARM_JOINTS])
        right_vel = np.array([self.low_state.motor_state[i].dq for i in RIGHT_ARM_JOINTS])
        return left_vel, right_vel

    def build_observation(self) -> np.ndarray:
        """Build observation vector matching simulation format (46D)."""
        left_pos, right_pos = self.get_arm_joint_positions()
        left_vel, right_vel = self.get_arm_joint_velocities()

        # Default positions
        left_default = np.array([DEFAULT_ARM_POSITIONS[i] for i in LEFT_ARM_JOINTS])
        right_default = np.array([DEFAULT_ARM_POSITIONS[i] for i in RIGHT_ARM_JOINTS])

        # Joint positions relative to default
        joint_pos_rel = np.concatenate([left_pos - left_default, right_pos - right_default])

        # Joint velocities
        joint_vel = np.concatenate([left_vel, right_vel])

        # Compute EE positions using FK
        left_ee_pos, _ = self.left_kin.forward_kinematics(left_pos)
        right_ee_pos, _ = self.right_kin.forward_kinematics(right_pos)

        # Error vectors (target - current)
        left_ee_error = self.left_target - left_ee_pos
        right_ee_error = self.right_target - right_ee_pos

        # EE velocities (approximate from Jacobian)
        J_left = self.left_kin.jacobian(left_pos[:4], n_joints=4)
        J_right = self.right_kin.jacobian(right_pos[:4], n_joints=4)
        left_ee_vel = J_left @ left_vel[:4]
        right_ee_vel = J_right @ right_vel[:4]

        # Build observation
        obs = np.concatenate([
            joint_pos_rel,      # 14D
            joint_vel,          # 14D
            left_ee_error,      # 3D
            right_ee_error,     # 3D
            left_ee_vel,        # 3D
            right_ee_vel,       # 3D
            self.last_action,   # 6D
        ])

        return obs

    def run_policy(self, obs: np.ndarray) -> np.ndarray:
        """Run policy inference to get actions."""
        # Normalize observation
        obs_normalized = (obs - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs_normalized).unsqueeze(0)
            action = self.policy(obs_tensor).squeeze(0).numpy()

        # Clip actions
        action = np.clip(action, -1.0, 1.0)
        return action

    def action_to_joint_deltas(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert policy actions to joint position deltas via IK."""
        # Scale actions to position deltas
        left_delta_pos = action[:3] * self.action_scale
        right_delta_pos = action[3:] * self.action_scale

        # Get current joint positions
        left_pos, right_pos = self.get_arm_joint_positions()

        # Solve IK for joint deltas
        left_delta_q = self.left_ik.solve(left_pos, left_delta_pos, n_joints=4)
        right_delta_q = self.right_ik.solve(right_pos, right_delta_pos, n_joints=4)

        return left_delta_q, right_delta_q

    def send_joint_commands(self, left_target: np.ndarray, right_target: np.ndarray):
        """Send joint position commands to the robot."""
        self.low_cmd.mode_pr = 0
        self.low_cmd.mode_machine = self.mode_machine

        # Waist joints - hold at neutral position
        for joint_idx in WAIST_JOINTS:
            self.low_cmd.motor_cmd[joint_idx].mode = 1
            self.low_cmd.motor_cmd[joint_idx].q = DEFAULT_WAIST_POSITIONS[joint_idx]
            self.low_cmd.motor_cmd[joint_idx].dq = 0.0
            self.low_cmd.motor_cmd[joint_idx].kp = WAIST_KP[joint_idx]
            self.low_cmd.motor_cmd[joint_idx].kd = WAIST_KD[joint_idx]
            self.low_cmd.motor_cmd[joint_idx].tau = 0.0

        # Left arm
        for i, joint_idx in enumerate(LEFT_ARM_IK_JOINTS):
            self.low_cmd.motor_cmd[joint_idx].mode = 1
            self.low_cmd.motor_cmd[joint_idx].q = float(left_target[i])
            self.low_cmd.motor_cmd[joint_idx].dq = 0.0
            self.low_cmd.motor_cmd[joint_idx].kp = ARM_KP[joint_idx]
            self.low_cmd.motor_cmd[joint_idx].kd = ARM_KD[joint_idx]
            self.low_cmd.motor_cmd[joint_idx].tau = 0.0

        # Right arm
        for i, joint_idx in enumerate(RIGHT_ARM_IK_JOINTS):
            self.low_cmd.motor_cmd[joint_idx].mode = 1
            self.low_cmd.motor_cmd[joint_idx].q = float(right_target[i])
            self.low_cmd.motor_cmd[joint_idx].dq = 0.0
            self.low_cmd.motor_cmd[joint_idx].kp = ARM_KP[joint_idx]
            self.low_cmd.motor_cmd[joint_idx].kd = ARM_KD[joint_idx]
            self.low_cmd.motor_cmd[joint_idx].tau = 0.0

        # Hold wrist joints at neutral
        for joint_idx in LEFT_ARM_JOINTS[4:] + RIGHT_ARM_JOINTS[4:]:
            self.low_cmd.motor_cmd[joint_idx].mode = 1
            self.low_cmd.motor_cmd[joint_idx].q = 0.0
            self.low_cmd.motor_cmd[joint_idx].dq = 0.0
            self.low_cmd.motor_cmd[joint_idx].kp = 30.0
            self.low_cmd.motor_cmd[joint_idx].kd = 1.0
            self.low_cmd.motor_cmd[joint_idx].tau = 0.0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

    def get_waist_joint_positions(self) -> np.ndarray:
        """Get current waist joint positions."""
        return np.array([self.low_state.motor_state[i].q for i in WAIST_JOINTS])

    def move_to_default_pose(self, duration: float = 3.0):
        """Smoothly move arms and waist to default pose."""
        print("Moving to default pose...")

        start_time = time.time()
        left_start, right_start = self.get_arm_joint_positions()
        waist_start = self.get_waist_joint_positions()

        left_default = np.array([DEFAULT_ARM_POSITIONS[i] for i in LEFT_ARM_JOINTS])
        right_default = np.array([DEFAULT_ARM_POSITIONS[i] for i in RIGHT_ARM_JOINTS])
        waist_default = np.array([DEFAULT_WAIST_POSITIONS[i] for i in WAIST_JOINTS])

        while time.time() - start_time < duration:
            t = (time.time() - start_time) / duration
            alpha = 3 * t**2 - 2 * t**3  # Smoothstep

            left_target = left_start + alpha * (left_default - left_start)
            right_target = right_start + alpha * (right_default - right_start)
            waist_target = waist_start + alpha * (waist_default - waist_start)

            # Send commands
            self.low_cmd.mode_pr = 0
            self.low_cmd.mode_machine = self.mode_machine

            # Waist joints
            for i, joint_idx in enumerate(WAIST_JOINTS):
                self.low_cmd.motor_cmd[joint_idx].mode = 1
                self.low_cmd.motor_cmd[joint_idx].q = float(waist_target[i])
                self.low_cmd.motor_cmd[joint_idx].dq = 0.0
                self.low_cmd.motor_cmd[joint_idx].kp = WAIST_KP[joint_idx]
                self.low_cmd.motor_cmd[joint_idx].kd = WAIST_KD[joint_idx]
                self.low_cmd.motor_cmd[joint_idx].tau = 0.0

            # Left arm joints
            for i, joint_idx in enumerate(LEFT_ARM_JOINTS):
                self.low_cmd.motor_cmd[joint_idx].mode = 1
                self.low_cmd.motor_cmd[joint_idx].q = float(left_target[i])
                self.low_cmd.motor_cmd[joint_idx].dq = 0.0
                self.low_cmd.motor_cmd[joint_idx].kp = ARM_KP[joint_idx]
                self.low_cmd.motor_cmd[joint_idx].kd = ARM_KD[joint_idx]
                self.low_cmd.motor_cmd[joint_idx].tau = 0.0

            # Right arm joints
            for i, joint_idx in enumerate(RIGHT_ARM_JOINTS):
                self.low_cmd.motor_cmd[joint_idx].mode = 1
                self.low_cmd.motor_cmd[joint_idx].q = float(right_target[i])
                self.low_cmd.motor_cmd[joint_idx].dq = 0.0
                self.low_cmd.motor_cmd[joint_idx].kp = ARM_KP[joint_idx]
                self.low_cmd.motor_cmd[joint_idx].kd = ARM_KD[joint_idx]
                self.low_cmd.motor_cmd[joint_idx].tau = 0.0

            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.lowcmd_publisher.Write(self.low_cmd)

            time.sleep(0.002)

        print("Default pose reached")

    def run(self):
        """Main control loop."""
        print("\nStarting policy deployment...")
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
                    # Update targets from keyboard if applicable
                    if self.keyboard:
                        self.left_target, self.right_target = self.keyboard.get_targets(
                            self.base_left_target, self.base_right_target
                        )

                    # Build observation
                    obs = self.build_observation()

                    # Run policy
                    action = self.run_policy(obs)
                    self.last_action = action

                    # Convert to joint deltas
                    left_delta_q, right_delta_q = self.action_to_joint_deltas(action)

                    # Get current positions and compute targets
                    left_pos, right_pos = self.get_arm_joint_positions()
                    left_joint_target = left_pos[:4] + left_delta_q
                    right_joint_target = right_pos[:4] + right_delta_q

                    # Send commands
                    self.send_joint_commands(left_joint_target, right_joint_target)

                    # Logging
                    log_counter += 1
                    if log_counter % 50 == 0:  # Log every second
                        left_ee, _ = self.left_kin.forward_kinematics(left_pos)
                        right_ee, _ = self.right_kin.forward_kinematics(right_pos)
                        left_err = np.linalg.norm(self.left_target - left_ee)
                        right_err = np.linalg.norm(self.right_target - right_ee)
                        print(f"L_err: {left_err:.3f}m  R_err: {right_err:.3f}m  "
                              f"L_target: {self.left_target}  R_target: {self.right_target}")

                    last_time = current_time

                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\nStopping...")

        finally:
            self.running = False
            if self.keyboard:
                self.keyboard.stop()

    def shutdown(self):
        """Cleanup and shutdown."""
        print("Shutting down...")
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="Deploy reaching policy to G1 robot")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint (.pt file)")
    parser.add_argument("--network_interface", type=str, default=None,
                       help="Network interface connected to robot (e.g., eth0)")
    parser.add_argument("--control_freq", type=float, default=50.0,
                       help="Control frequency in Hz (default: 50)")
    parser.add_argument("--mode", type=str, default="fixed", choices=["fixed", "keyboard"],
                       help="Control mode: 'fixed' for preset targets, 'keyboard' for manual control")
    parser.add_argument("--skip_default_pose", action="store_true",
                       help="Skip moving to default pose on startup")
    args = parser.parse_args()

    print("=" * 60)
    print("G1 Reaching Policy Deployment")
    print("=" * 60)
    print()
    print("WARNING: Ensure the robot is in a safe position!")
    print("WARNING: Keep emergency stop nearby!")
    print()
    input("Press Enter to continue...")

    deployer = G1PolicyDeployer(
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
