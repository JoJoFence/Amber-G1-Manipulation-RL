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
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, quat_box_minus

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


# === Helper for state-dependent rewards ===

def _compute_ee_distance(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Compute distance from EE to commanded target position. Returns (N,) tensor."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


# === State-Dependent Rewards for Curriculum Phase 2 ===


def adaptive_action_rate_penalty(
    env: ManagerBasedRLEnv, sigma: float, command_name: str, asset_cfg: SceneEntityCfg,
    action_start: int = 0, action_end: int = 7,
) -> torch.Tensor:
    """Penalize action rate (change between timesteps) scaled by Gaussian proximity to target.

    Near the target, action jitter is heavily penalized to prevent wobble.
    Far from the target, the penalty is negligible so the policy can make big moves.

    Only penalizes the arm's own action dimensions (action_start:action_end) to avoid
    cross-arm interference where one arm's proximity suppresses the other arm's actions.
    """
    distance = _compute_ee_distance(env, command_name, asset_cfg)
    proximity = torch.exp(-torch.square(distance / sigma))

    # Action rate for THIS arm only
    action = env.action_manager.action[:, action_start:action_end]
    prev_action = env.action_manager.prev_action[:, action_start:action_end]
    action_rate_sq = torch.sum(torch.square(action - prev_action), dim=1)

    return proximity * action_rate_sq


def terminal_damping_reward(
    env: ManagerBasedRLEnv, pos_sigma: float, command_name: str, asset_cfg: SceneEntityCfg,
    action_start: int = 0, action_end: int = 7,
) -> torch.Tensor:
    """Positive reward for being still at the target — the 'stop and hold' signal.

    Combines two factors multiplicatively:
    - Position closeness: exp(-(distance/pos_sigma)^2)
    - Low EE velocity: exp(-||ee_vel|| * 10)

    Note: Does NOT penalize action magnitude, since wrist actions are needed
    for orientation control even when the position is on target.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    distance = _compute_ee_distance(env, command_name, asset_cfg)

    # Position closeness (Gaussian)
    pos_reward = torch.exp(-torch.square(distance / pos_sigma))

    # Low EE velocity
    ee_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids[0]]
    vel_magnitude = torch.norm(ee_vel, dim=1)
    vel_damping = torch.exp(-vel_magnitude * 10.0)

    return pos_reward * vel_damping


def proximity_action_magnitude_penalty(
    env: ManagerBasedRLEnv, sigma: float, command_name: str, asset_cfg: SceneEntityCfg,
    action_start: int = 0, action_end: int = 7,
) -> torch.Tensor:
    """Penalize action magnitude near target — the 'become deterministic' signal.

    Near the target, the optimal policy should output near-zero actions (effectively
    deterministic). Far from the target, large actions are acceptable for fast reaching.

    Only penalizes THIS arm's action dimensions to avoid cross-arm interference.
    """
    distance = _compute_ee_distance(env, command_name, asset_cfg)
    proximity = torch.exp(-torch.square(distance / sigma))

    arm_action = env.action_manager.action[:, action_start:action_end]
    action_sq = torch.sum(torch.square(arm_action), dim=1)

    return proximity * action_sq


def bimanual_position_balance(
    env: ManagerBasedRLEnv, std: float,
    left_command: str, right_command: str,
    left_asset_cfg: SceneEntityCfg, right_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward the WORSE arm's position tracking — forces balanced bimanual convergence.

    Returns min(left_tracking, right_tracking) where tracking = 1 - tanh(distance/std).
    The policy can only increase this reward by improving whichever arm is lagging.
    This eliminates gradient imbalance from one arm converging first.
    """
    left_dist = _compute_ee_distance(env, left_command, left_asset_cfg)
    right_dist = _compute_ee_distance(env, right_command, right_asset_cfg)
    left_reward = 1 - torch.tanh(left_dist / std)
    right_reward = 1 - torch.tanh(right_dist / std)
    return torch.min(left_reward, right_reward)


def bimanual_orient_balance(
    env: ManagerBasedRLEnv, std: float,
    left_command: str, right_command: str,
    left_asset_cfg: SceneEntityCfg, right_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward the WORSE arm's orientation tracking — forces balanced bimanual convergence."""
    asset_left: Articulation = env.scene[left_asset_cfg.name]
    asset_right: Articulation = env.scene[right_asset_cfg.name]

    left_cmd = env.command_manager.get_command(left_command)
    right_cmd = env.command_manager.get_command(right_command)

    # Left orientation error
    left_des_quat_w = quat_mul(asset_left.data.root_quat_w, left_cmd[:, 3:7])
    left_curr_quat_w = asset_left.data.body_quat_w[:, left_asset_cfg.body_ids[0]]
    left_err = quat_error_magnitude(left_curr_quat_w, left_des_quat_w)

    # Right orientation error
    right_des_quat_w = quat_mul(asset_right.data.root_quat_w, right_cmd[:, 3:7])
    right_curr_quat_w = asset_right.data.body_quat_w[:, right_asset_cfg.body_ids[0]]
    right_err = quat_error_magnitude(right_curr_quat_w, right_des_quat_w)

    left_reward = 1 - torch.tanh(left_err / std)
    right_reward = 1 - torch.tanh(right_err / std)
    return torch.min(left_reward, right_reward)


def orientation_tracking_reward(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward orientation tracking using tanh kernel — always active, not proximity-scaled.

    reward = 1 - tanh(orient_error / std)

    Provides smooth gradient signal for orientation at ALL distances from target,
    so the policy learns position and orientation simultaneously.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    orient_error = quat_error_magnitude(curr_quat_w, des_quat_w)
    return 1 - torch.tanh(orient_error / std)


def adaptive_orientation_penalty(
    env: ManagerBasedRLEnv, sigma: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize orientation error scaled by position proximity to target.

    Only cares about wrist alignment when the hand is close to the target position.
    Far from target, the policy focuses purely on reaching; once close, it must also
    align the wrist orientation.

    proximity = exp(-(distance/sigma)^2)
    penalty = proximity * quat_error_magnitude(current, desired)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    distance = _compute_ee_distance(env, command_name, asset_cfg)
    proximity = torch.exp(-torch.square(distance / sigma))

    # Orientation error
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    orient_error = quat_error_magnitude(curr_quat_w, des_quat_w)

    return proximity * orient_error
