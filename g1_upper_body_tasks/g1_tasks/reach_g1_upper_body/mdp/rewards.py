# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def flat_orientation_l2(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize non-flat base orientation to prevent robot from falling.
    
    The function computes the L2 penalty on the roll and pitch of the base.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get base orientation quaternion (w, x, y, z)
    quat_w = asset.data.root_quat_w
    
    # Convert to roll and pitch
    # roll (x-axis rotation)
    roll = torch.atan2(2 * (quat_w[:, 0] * quat_w[:, 1] + quat_w[:, 2] * quat_w[:, 3]),
                       1 - 2 * (quat_w[:, 1]**2 + quat_w[:, 2]**2))
    
    # pitch (y-axis rotation)  
    pitch = torch.asin(2 * (quat_w[:, 0] * quat_w[:, 2] - quat_w[:, 3] * quat_w[:, 1]))
    
    # Penalize deviation from upright
    penalty = torch.square(roll) + torch.square(pitch)
    
    # CRITICAL: Replace any NaN/inf with zero
    penalty = torch.nan_to_num(penalty, nan=0.0, posinf=0.0, neginf=0.0)
    
    return penalty

def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def position_command_success(
    env: ManagerBasedRLEnv, threshold: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Sparse reward for reaching the target position within a threshold.

    Returns 1.0 when the end-effector is within the threshold distance of the target,
    and 0.0 otherwise. This provides a clear success signal to the policy.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return (distance < threshold).float()


def position_command_error_exp(
    env: ManagerBasedRLEnv, sigma: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Exponential reward for position tracking - very strong gradient near target.

    reward = exp(-(distance/sigma)^2)

    This provides much stronger gradient near the target compared to tanh.
    At distance=0: reward=1.0
    At distance=sigma: reward=0.37
    At distance=2*sigma: reward=0.02
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return torch.exp(-torch.square(distance / sigma))


def position_holding_reward(
    env: ManagerBasedRLEnv, threshold: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward for holding position steady when near target.

    When within threshold distance of target, reward low end-effector velocity.
    This reduces drift and encourages stable positioning.
    Returns 0 when far from target (no holding needed).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Get distance to target
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)

    # Get end-effector velocity
    ee_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids[0]]  # type: ignore
    vel_magnitude = torch.norm(ee_vel, dim=1)

    # Only apply holding reward when close to target
    near_target = (distance < threshold).float()

    # Reward low velocity when near target: 1.0 for zero velocity, decreasing with velocity
    holding_reward = torch.exp(-vel_magnitude * 10.0)  # Exponential decay with velocity

    return near_target * holding_reward


def joint_deviation_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize joint positions deviating from default positions.

    This encourages the robot to maintain natural arm configurations
    and prevents drift into awkward poses.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get current joint positions and default positions
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    # Compute squared deviation from default
    deviation = torch.sum(torch.square(joint_pos - default_pos), dim=1)

    return deviation


def wrist_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Strongly penalize wrist joints deviating from neutral (zero) position.

    Wrists should stay neutral during reaching to prevent contortion.
    This specifically targets wrist roll, pitch, and yaw joints.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get wrist joint positions (asset_cfg should specify wrist joint indices)
    wrist_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

    # Penalize any deviation from zero (neutral wrist)
    penalty = torch.sum(torch.square(wrist_pos), dim=1)

    return penalty


def wrist_velocity_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize wrist joint velocities to prevent rapid wrist movements.

    This helps keep wrists stable and prevents oscillation.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get wrist joint velocities
    wrist_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    # Penalize velocity magnitude
    penalty = torch.sum(torch.square(wrist_vel), dim=1)

    return penalty


def action_near_target_penalty(
    env: ManagerBasedRLEnv, threshold: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize large actions when near the target position.

    When close to target, the robot should make small corrective actions,
    not large movements. This reduces wobble and overshoot at the end of reach.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Get distance to target
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)

    # Get action magnitude (from last action)
    action_magnitude = torch.sum(torch.square(env.action_manager.action), dim=1)

    # Only penalize large actions when near target
    near_target = (distance < threshold).float()

    return near_target * action_magnitude


def joint_velocity_near_target_penalty(
    env: ManagerBasedRLEnv, threshold: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize joint velocities when near target - encourages stillness.

    When close to target, all joints should slow down to reduce wobble.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Get distance to target
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)

    # Get joint velocities for specified joints
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    vel_magnitude = torch.sum(torch.square(joint_vel), dim=1)

    # Only penalize when near target
    near_target = (distance < threshold).float()

    return near_target * vel_magnitude


def ee_velocity_penalty_near_target(
    env: ManagerBasedRLEnv, threshold: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize end-effector velocity when near target to prevent drift.

    This directly penalizes EE movement when close to target, providing
    stronger anti-drift signal than joint-level penalties. Uses smooth
    activation that increases as distance decreases.

    Returns squared EE velocity magnitude, scaled by proximity to target.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Get distance to target
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)

    # Get EE velocity
    ee_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids[0]]
    vel_sq = torch.sum(torch.square(ee_vel), dim=1)

    # Smooth activation: 1.0 at target, 0.0 at threshold, smooth transition
    # Using (1 - d/threshold)^2 for smooth falloff
    proximity = torch.clamp(1.0 - distance / threshold, min=0.0)
    activation = proximity * proximity  # Quadratic for smooth gradient

    return activation * vel_sq
