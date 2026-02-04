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

"""Custom observation functions for G1 upper body reach task.

Provides error-vector and velocity observations that directly encode
the information each arm needs to reach its target.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_position_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """3D position error vector from current EE position to target position.

    Computes (target_world - ee_world), giving a vector that points from the
    current end-effector position toward the target. This directly encodes
    both direction and distance for the policy, making it trivial to associate
    each arm with its own target.

    Returns:
        Tensor of shape (N, 3) with the position error in world frame.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Target position: body frame -> world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )

    # Current EE position in world frame
    ee_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]

    # Error vector: points from current toward target
    return des_pos_w - ee_pos_w


def ee_velocity(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """End-effector linear velocity in world frame.

    Provides velocity information so the policy can learn to decelerate
    when approaching targets, preventing overshoot.

    Returns:
        Tensor of shape (N, 3) with EE linear velocity.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_lin_vel_w[:, asset_cfg.body_ids[0]]
