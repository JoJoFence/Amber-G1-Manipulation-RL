"""
G1 Arm Kinematics Module

Forward kinematics and Jacobian computation for G1 humanoid arms,
extracted from URDF specifications.
"""

import numpy as np
from typing import Tuple


def rotation_x(angle: float) -> np.ndarray:
    """Rotation matrix around X axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def rotation_y(angle: float) -> np.ndarray:
    """Rotation matrix around Y axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rotation_z(angle: float) -> np.ndarray:
    """Rotation matrix around Z axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def rpy_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert roll-pitch-yaw to rotation matrix."""
    return rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)


def transform_matrix(translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Create 4x4 homogeneous transformation matrix."""
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


class G1ArmKinematics:
    """Forward kinematics and Jacobian for G1 arm.

    Kinematic chain from torso_link to rubber_hand (end-effector).
    Uses URDF-derived transforms.
    """

    def __init__(self, arm: str = 'left'):
        """Initialize kinematics for specified arm.

        Args:
            arm: 'left' or 'right'
        """
        self.arm = arm
        self.n_joints = 7  # Full arm (4 for IK, 7 total)

        # Joint axes (in local frame)
        # Shoulder pitch: Y, Shoulder roll: X, Shoulder yaw: Z
        # Elbow: Y, Wrist roll: X, Wrist pitch: Y, Wrist yaw: Z
        self.joint_axes = [
            np.array([0, 1, 0]),  # shoulder_pitch (Y)
            np.array([1, 0, 0]),  # shoulder_roll (X)
            np.array([0, 0, 1]),  # shoulder_yaw (Z)
            np.array([0, 1, 0]),  # elbow (Y)
            np.array([1, 0, 0]),  # wrist_roll (X)
            np.array([0, 1, 0]),  # wrist_pitch (Y)
            np.array([0, 0, 1]),  # wrist_yaw (Z)
        ]

        # Fixed transforms between joints (from URDF)
        if arm == 'left':
            self._init_left_arm()
        else:
            self._init_right_arm()

    def _init_left_arm(self):
        """Initialize left arm transforms from URDF."""
        # torso_link to left_shoulder_pitch_link
        # origin xyz="0.0039563 0.10022 0.23778" rpy="0.27931 5.4949E-05 -0.00019159"
        self.T_base_to_shoulder = transform_matrix(
            np.array([0.0039563, 0.10022, 0.23778]),
            rpy_to_rotation(0.27931, 5.4949e-05, -0.00019159)
        )

        # Transforms between consecutive joints
        self.joint_transforms = [
            # shoulder_pitch to shoulder_roll
            # origin xyz="0 0.038 -0.013831" rpy="-0.27925 0 0"
            (np.array([0, 0.038, -0.013831]), rpy_to_rotation(-0.27925, 0, 0)),

            # shoulder_roll to shoulder_yaw
            # origin xyz="0 0.00624 -0.1032" rpy="0 0 0"
            (np.array([0, 0.00624, -0.1032]), np.eye(3)),

            # shoulder_yaw to elbow
            # origin xyz="0.015783 0 -0.080518" rpy="0 0 0"
            (np.array([0.015783, 0, -0.080518]), np.eye(3)),

            # elbow to wrist_roll
            # origin xyz="0.100 0.00188791 -0.010" rpy="0 0 0"
            (np.array([0.100, 0.00188791, -0.010]), np.eye(3)),

            # wrist_roll to wrist_pitch
            # origin xyz="0.038 0 0" rpy="0 0 0"
            (np.array([0.038, 0, 0]), np.eye(3)),

            # wrist_pitch to wrist_yaw
            # origin xyz="0.046 0 0" rpy="0 0 0"
            (np.array([0.046, 0, 0]), np.eye(3)),
        ]

        # wrist_yaw to end-effector (rubber_hand)
        # origin xyz="0.0415 0.003 0" rpy="0 0 0"
        self.T_wrist_to_ee = transform_matrix(
            np.array([0.0415, 0.003, 0]),
            np.eye(3)
        )

    def _init_right_arm(self):
        """Initialize right arm transforms from URDF."""
        # torso_link to right_shoulder_pitch_link
        # origin xyz="0.0039563 -0.10021 0.23778" rpy="-0.27931 5.4949E-05 0.00019159"
        self.T_base_to_shoulder = transform_matrix(
            np.array([0.0039563, -0.10021, 0.23778]),
            rpy_to_rotation(-0.27931, 5.4949e-05, 0.00019159)
        )

        # Transforms between consecutive joints
        self.joint_transforms = [
            # shoulder_pitch to shoulder_roll
            # origin xyz="0 -0.038 -0.013831" rpy="0.27925 0 0"
            (np.array([0, -0.038, -0.013831]), rpy_to_rotation(0.27925, 0, 0)),

            # shoulder_roll to shoulder_yaw
            # origin xyz="0 -0.00624 -0.1032" rpy="0 0 0"
            (np.array([0, -0.00624, -0.1032]), np.eye(3)),

            # shoulder_yaw to elbow
            # origin xyz="0.015783 0 -0.080518" rpy="0 0 0"
            (np.array([0.015783, 0, -0.080518]), np.eye(3)),

            # elbow to wrist_roll
            # origin xyz="0.100 -0.00188791 -0.010" rpy="0 0 0"
            (np.array([0.100, -0.00188791, -0.010]), np.eye(3)),

            # wrist_roll to wrist_pitch
            # origin xyz="0.038 0 0" rpy="0 0 0"
            (np.array([0.038, 0, 0]), np.eye(3)),

            # wrist_pitch to wrist_yaw
            # origin xyz="0.046 0 0" rpy="0 0 0"
            (np.array([0.046, 0, 0]), np.eye(3)),
        ]

        # wrist_yaw to end-effector (rubber_hand)
        # origin xyz="0.0415 -0.003 0" rpy="0 0 0"
        self.T_wrist_to_ee = transform_matrix(
            np.array([0.0415, -0.003, 0]),
            np.eye(3)
        )

    def _joint_rotation(self, joint_idx: int, angle: float) -> np.ndarray:
        """Get rotation matrix for a joint at given angle."""
        axis = self.joint_axes[joint_idx]
        if np.allclose(axis, [1, 0, 0]):
            return rotation_x(angle)
        elif np.allclose(axis, [0, 1, 0]):
            return rotation_y(angle)
        elif np.allclose(axis, [0, 0, 1]):
            return rotation_z(angle)
        else:
            # General axis rotation (Rodrigues' formula)
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K

    def forward_kinematics(self, joint_angles: np.ndarray,
                           n_joints: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute end-effector position and orientation.

        Args:
            joint_angles: Array of joint angles (up to 7 for full arm)
            n_joints: Number of joints to use (default: len(joint_angles))

        Returns:
            position: 3D position of end-effector in base frame
            rotation: 3x3 rotation matrix of end-effector
        """
        if n_joints is None:
            n_joints = len(joint_angles)

        # Start with base to first joint transform
        T = self.T_base_to_shoulder.copy()

        # Apply first joint rotation
        R_joint = self._joint_rotation(0, joint_angles[0])
        T[:3, :3] = T[:3, :3] @ R_joint

        # Chain through remaining joints
        for i in range(1, n_joints):
            # Fixed transform to next joint
            trans, rot = self.joint_transforms[i - 1]
            T_fixed = transform_matrix(trans, rot)
            T = T @ T_fixed

            # Joint rotation
            R_joint = self._joint_rotation(i, joint_angles[i])
            T[:3, :3] = T[:3, :3] @ R_joint

        # Apply remaining fixed transforms if not using all joints
        for i in range(n_joints - 1, len(self.joint_transforms)):
            trans, rot = self.joint_transforms[i]
            T_fixed = transform_matrix(trans, rot)
            T = T @ T_fixed

        # Transform to end-effector
        T = T @ self.T_wrist_to_ee

        return T[:3, 3], T[:3, :3]

    def jacobian(self, joint_angles: np.ndarray, n_joints: int = 4) -> np.ndarray:
        """Compute the geometric Jacobian for position control.

        Uses numerical differentiation for robustness.

        Args:
            joint_angles: Current joint angles
            n_joints: Number of joints to include (default 4 for IK)

        Returns:
            J: 3 x n_joints Jacobian matrix (position only)
        """
        eps = 1e-6
        J = np.zeros((3, n_joints))

        pos_0, _ = self.forward_kinematics(joint_angles, n_joints)

        for i in range(n_joints):
            q_plus = joint_angles.copy()
            q_plus[i] += eps
            pos_plus, _ = self.forward_kinematics(q_plus, n_joints)
            J[:, i] = (pos_plus - pos_0) / eps

        return J

    def jacobian_analytical(self, joint_angles: np.ndarray,
                           n_joints: int = 4) -> np.ndarray:
        """Compute analytical Jacobian (more efficient).

        J_i = z_i x (p_ee - p_i) for revolute joints
        where z_i is the joint axis in world frame
        and p_i is the joint origin in world frame
        """
        # Get all joint positions and axes in world frame
        positions = []
        axes = []

        T = self.T_base_to_shoulder.copy()

        for i in range(n_joints):
            # Store joint origin and axis
            positions.append(T[:3, 3].copy())
            axes.append(T[:3, :3] @ self.joint_axes[i])

            # Apply joint rotation
            R_joint = self._joint_rotation(i, joint_angles[i])
            T[:3, :3] = T[:3, :3] @ R_joint

            # Apply fixed transform to next joint (if not last)
            if i < len(self.joint_transforms):
                trans, rot = self.joint_transforms[i]
                T_fixed = transform_matrix(trans, rot)
                T = T @ T_fixed

        # Get end-effector position
        # Apply remaining transforms
        for i in range(n_joints - 1, len(self.joint_transforms)):
            trans, rot = self.joint_transforms[i]
            T_fixed = transform_matrix(trans, rot)
            T = T @ T_fixed
        T = T @ self.T_wrist_to_ee
        p_ee = T[:3, 3]

        # Compute Jacobian columns
        J = np.zeros((3, n_joints))
        for i in range(n_joints):
            J[:, i] = np.cross(axes[i], p_ee - positions[i])

        return J


class DifferentialIK:
    """Differential inverse kinematics solver."""

    def __init__(self, kinematics: G1ArmKinematics, lambda_val: float = 0.1):
        """
        Args:
            kinematics: Kinematics model for the arm
            lambda_val: Damping factor for DLS
        """
        self.kin = kinematics
        self.lambda_val = lambda_val

    def solve(self, joint_angles: np.ndarray,
              delta_pos: np.ndarray,
              n_joints: int = 4) -> np.ndarray:
        """Solve IK for position delta.

        Args:
            joint_angles: Current joint angles
            delta_pos: Desired position change (3D)
            n_joints: Number of joints to use

        Returns:
            delta_q: Joint angle changes
        """
        J = self.kin.jacobian(joint_angles[:n_joints], n_joints)

        # Damped least squares: J^T (J J^T + lambda^2 I)^-1 delta_pos
        JJT = J @ J.T
        damping = self.lambda_val ** 2 * np.eye(3)
        delta_q = J.T @ np.linalg.solve(JJT + damping, delta_pos)

        return delta_q


# Test functions
def test_kinematics():
    """Test forward kinematics with default joint positions."""
    print("Testing G1 Arm Kinematics")
    print("=" * 50)

    # Default joint positions from simulation
    default_left = np.array([0.4, 0.3, 0.0, 0.8, 0.0, 0.0, 0.0])
    default_right = np.array([0.4, -0.3, 0.0, 0.8, 0.0, 0.0, 0.0])

    left_kin = G1ArmKinematics('left')
    right_kin = G1ArmKinematics('right')

    # Test FK
    left_pos, left_rot = left_kin.forward_kinematics(default_left)
    right_pos, right_rot = right_kin.forward_kinematics(default_right)

    print(f"Left arm EE position:  {left_pos}")
    print(f"Right arm EE position: {right_pos}")

    # Test Jacobian
    J_left = left_kin.jacobian(default_left[:4], n_joints=4)
    print(f"\nLeft arm Jacobian (4 DOF):\n{J_left}")

    # Test IK
    ik_left = DifferentialIK(left_kin, lambda_val=0.1)
    delta_pos = np.array([0.01, 0.0, 0.0])  # 1cm forward
    delta_q = ik_left.solve(default_left, delta_pos, n_joints=4)
    print(f"\nIK solution for 1cm forward: {delta_q}")

    # Verify IK
    new_joints = default_left.copy()
    new_joints[:4] += delta_q
    new_pos, _ = left_kin.forward_kinematics(new_joints)
    print(f"Position after IK: {new_pos}")
    print(f"Actual delta: {new_pos - left_pos}")


if __name__ == "__main__":
    test_kinematics()
