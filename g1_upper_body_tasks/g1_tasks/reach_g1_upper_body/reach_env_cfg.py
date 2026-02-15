"""
G1 Upper Body Reach Task Environment Configuration for Isaac Lab 5.1.0

Bimanual reaching task with:
- Direct joint-space control (no IK) for sim-to-real transfer
- 7 DOF per arm (shoulder + elbow + wrist)
- Error-vector observations for clear arm-target association
- EE velocity observations for learned deceleration
- Multi-level tanh rewards for smooth approach without overshoot
"""

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ActionTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils.configclass import configclass
from isaaclab.envs.mdp.actions import JointPositionActionCfg

# Import MDP functions from Isaac Lab
from . import mdp

from ..g1_upper_body_cfg import G1_UPPER_BODY_CFG, LEFT_EE_FRAME, RIGHT_EE_FRAME, DECIMATION, BASE_FRAME

##
# Scene definition
##

@configclass
class G1UpperBodyReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the G1 upper body reach scene."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    )

    # Robot
    robot: ArticulationCfg = G1_UPPER_BODY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # End-effector frames for tracking
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + BASE_FRAME,
        debug_vis=True,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + LEFT_EE_FRAME,
                name="left_ee",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.0),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + RIGHT_EE_FRAME,
                name="right_ee",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.0),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            ),
        ],
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # Target poses in BODY FRAME (relative to robot base at z=1.0)
    # Workspace tightened to stay within arm reach (~0.35m from shoulder).
    # Shoulder is at (0.004, ±0.100, 0.238) in body frame; max arm reach ~0.41m.
    left_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=LEFT_EE_FRAME,
        resampling_time_range=(4.0, 6.0),  # Longer hold for precision learning
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.30),   # Forward reach (tightened from 0.20-0.45)
            pos_y=(0.05, 0.22),   # Left side workspace (tightened from 0.10-0.30)
            pos_z=(0.05, 0.25),   # Biased UP toward shoulder height (was -0.15 to 0.15)
            roll=(-0.5, 0.5),     # ±29° (wrist roll limit is ±113°)
            pitch=(-0.3, 0.3),    # ±17° (wrist pitch limit is ±92°)
            yaw=(-0.5, 0.5),      # ±29° (wrist yaw limit is ±92°)
        ),
    )

    right_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=RIGHT_EE_FRAME,
        resampling_time_range=(4.0, 6.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.30),   # Same forward range
            pos_y=(-0.22, -0.05), # Right side workspace (symmetric)
            pos_z=(0.05, 0.25),   # Same vertical range
            roll=(-0.5, 0.5),     # ±29° (same as left)
            pitch=(-0.3, 0.3),    # ±17°
            yaw=(-0.5, 0.5),      # ±29°
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP.

    Direct joint-space control for all arm joints (7 DOF per arm).
    Policy outputs joint position deltas, applied directly to joint targets.
    Total action space: 14D (7 joints per arm × 2 arms).

    This eliminates IK dependency for sim-to-real transfer - policy outputs
    can be applied directly to the robot without any kinematic conversion.
    """

    arm_actions = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
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
        ],
        scale=0.03,  # Reduced from 0.05 to limit jitter amplitude
        use_default_offset=True,  # Actions are relative to default joint positions
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP.

    Key design: error vectors directly tell each arm where its target is,
    eliminating the ambiguity that caused both arms to go to the same target.
    EE velocity enables learned deceleration to prevent overshoot.

    Observation space (60D):
    - joint_pos_rel: 14D (all joint positions relative to default)
    - joint_vel_rel: 14D (all joint velocities)
    - left_ee_error: 3D (vector from left EE to left target)
    - right_ee_error: 3D (vector from right EE to right target)
    - left_ee_orient_error: 3D (axis-angle orientation error for left EE)
    - right_ee_orient_error: 3D (axis-angle orientation error for right EE)
    - left_ee_vel: 3D (left EE linear velocity)
    - right_ee_vel: 3D (right EE linear velocity)
    - last_action: 14D (previous action for temporal consistency)
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy."""

        # Proprioception: joint states for all 14 joints
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        # Error vectors: directly encode target-relative position for each arm
        # This makes arm-target association trivial for the policy
        left_ee_error = ObsTerm(
            func=mdp.ee_position_error,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[LEFT_EE_FRAME]),
                "command_name": "left_ee_pose",
            },
        )
        right_ee_error = ObsTerm(
            func=mdp.ee_position_error,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_FRAME]),
                "command_name": "right_ee_pose",
            },
        )

        # Orientation error vectors: axis-angle representation (3D each)
        left_ee_orient_error = ObsTerm(
            func=mdp.ee_orientation_error,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[LEFT_EE_FRAME]),
                "command_name": "left_ee_pose",
            },
        )
        right_ee_orient_error = ObsTerm(
            func=mdp.ee_orientation_error,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_FRAME]),
                "command_name": "right_ee_pose",
            },
        )

        # EE velocities: enables learned deceleration near target
        left_ee_vel = ObsTerm(
            func=mdp.ee_velocity,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[LEFT_EE_FRAME])},
        )
        right_ee_vel = ObsTerm(
            func=mdp.ee_velocity,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_FRAME])},
        )

        # Action history for temporal consistency
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # Observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP — Single-phase with all rewards active.

    Position tracking + state-dependent anti-jitter + orientation rewards
    all active from the start with calibrated weights.
    """

    # === BIMANUAL BALANCE (forces both arms to converge together) ===
    # Returns min(left, right) so only improving the WORST arm earns reward.
    # Prevents gradient imbalance from one arm converging first.
    bimanual_pos_coarse = RewTerm(
        func=mdp.bimanual_position_balance, weight=15.0,
        params={
            "left_asset_cfg": SceneEntityCfg("robot", body_names=[LEFT_EE_FRAME]),
            "right_asset_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_FRAME]),
            "left_command": "left_ee_pose", "right_command": "right_ee_pose", "std": 0.5,
        },
    )
    bimanual_pos_fine = RewTerm(
        func=mdp.bimanual_position_balance, weight=20.0,
        params={
            "left_asset_cfg": SceneEntityCfg("robot", body_names=[LEFT_EE_FRAME]),
            "right_asset_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_FRAME]),
            "left_command": "left_ee_pose", "right_command": "right_ee_pose", "std": 0.1,
        },
    )
    bimanual_orient = RewTerm(
        func=mdp.bimanual_orient_balance, weight=25.0,
        params={
            "left_asset_cfg": SceneEntityCfg("robot", body_names=[LEFT_EE_FRAME]),
            "right_asset_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_FRAME]),
            "left_command": "left_ee_pose", "right_command": "right_ee_pose", "std": 1.5,
        },
    )

    # === POSITION TRACKING ===
    left_ee_coarse = RewTerm(
        func=mdp.position_command_error_tanh, weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[LEFT_EE_FRAME]), "command_name": "left_ee_pose", "std": 0.5},
    )
    right_ee_coarse = RewTerm(
        func=mdp.position_command_error_tanh, weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_FRAME]), "command_name": "right_ee_pose", "std": 0.5},
    )
    left_ee_medium = RewTerm(
        func=mdp.position_command_error_tanh, weight=6.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[LEFT_EE_FRAME]), "command_name": "left_ee_pose", "std": 0.25},
    )
    right_ee_medium = RewTerm(
        func=mdp.position_command_error_tanh, weight=6.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_FRAME]), "command_name": "right_ee_pose", "std": 0.25},
    )
    left_ee_fine = RewTerm(
        func=mdp.position_command_error_tanh, weight=8.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[LEFT_EE_FRAME]), "command_name": "left_ee_pose", "std": 0.1},
    )
    right_ee_fine = RewTerm(
        func=mdp.position_command_error_tanh, weight=8.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_FRAME]), "command_name": "right_ee_pose", "std": 0.1},
    )

    # === HOLDING + SUCCESS ===
    left_ee_holding = RewTerm(
        func=mdp.position_holding_reward, weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[LEFT_EE_FRAME]), "command_name": "left_ee_pose", "threshold": 0.04},
    )
    right_ee_holding = RewTerm(
        func=mdp.position_holding_reward, weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_FRAME]), "command_name": "right_ee_pose", "threshold": 0.04},
    )
    left_ee_success = RewTerm(
        func=mdp.position_command_success, weight=8.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[LEFT_EE_FRAME]), "command_name": "left_ee_pose", "threshold": 0.03},
    )
    right_ee_success = RewTerm(
        func=mdp.position_command_success, weight=8.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_FRAME]), "command_name": "right_ee_pose", "threshold": 0.03},
    )

    # === EE VELOCITY NEAR TARGET ===
    left_ee_vel_near_target = RewTerm(
        func=mdp.ee_velocity_penalty_near_target, weight=-8.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[LEFT_EE_FRAME]), "command_name": "left_ee_pose", "threshold": 0.06},
    )
    right_ee_vel_near_target = RewTerm(
        func=mdp.ee_velocity_penalty_near_target, weight=-8.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_FRAME]), "command_name": "right_ee_pose", "threshold": 0.06},
    )

    # === REGULARIZATION ===
    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.2)
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # wrist_position_penalty REMOVED — conflicts with orientation control.
    # With randomized target orientations, wrists MUST deviate from zero.
    # wrist_velocity_penalty REMOVED — same reason; wrists need to move for orientation.

    # === STATE-DEPENDENT REWARDS (active from start, gentle weights) ===
    # action_start/action_end isolate each arm's 7 joints to prevent cross-arm interference.
    # Action layout: left arm [0:7], right arm [7:14].

    # Adaptive action rate: Gaussian proximity-scaled action jitter penalty
    left_adaptive_action_rate = RewTerm(
        func=mdp.adaptive_action_rate_penalty, weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[LEFT_EE_FRAME]), "command_name": "left_ee_pose", "sigma": 0.08, "action_start": 0, "action_end": 7},
    )
    right_adaptive_action_rate = RewTerm(
        func=mdp.adaptive_action_rate_penalty, weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_FRAME]), "command_name": "right_ee_pose", "sigma": 0.08, "action_start": 7, "action_end": 14},
    )

    # Terminal damping: positive reward for being still at target
    left_terminal_damping = RewTerm(
        func=mdp.terminal_damping_reward, weight=10.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[LEFT_EE_FRAME]), "command_name": "left_ee_pose", "pos_sigma": 0.05, "action_start": 0, "action_end": 7},
    )
    right_terminal_damping = RewTerm(
        func=mdp.terminal_damping_reward, weight=10.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_FRAME]), "command_name": "right_ee_pose", "pos_sigma": 0.05, "action_start": 7, "action_end": 14},
    )

    # proximity_action_magnitude REMOVED — incompatible with orientation control.
    # Penalizes ALL action magnitude near target, including wrist actions needed for orientation.
    # At 1cm distance it produced -7.38 episode cost, overwhelming tracking and creating
    # cross-arm interference through shared network gradients.

    # === ORIENTATION TRACKING (always active — multi-level like position) ===
    # Coarse: gradient for large errors (1-2 rad). std=1.5 so tanh doesn't saturate.
    left_orient_coarse = RewTerm(
        func=mdp.orientation_tracking_reward, weight=12.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[LEFT_EE_FRAME]), "command_name": "left_ee_pose", "std": 1.5},
    )
    right_orient_coarse = RewTerm(
        func=mdp.orientation_tracking_reward, weight=12.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_FRAME]), "command_name": "right_ee_pose", "std": 1.5},
    )
    # Fine: strong gradient for sub-0.5 rad precision
    left_orient_fine = RewTerm(
        func=mdp.orientation_tracking_reward, weight=15.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[LEFT_EE_FRAME]), "command_name": "left_ee_pose", "std": 0.3},
    )
    right_orient_fine = RewTerm(
        func=mdp.orientation_tracking_reward, weight=15.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_FRAME]), "command_name": "right_ee_pose", "std": 0.3},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Time limit
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class EventCfg:
    """Configuration for events - includes domain randomization."""

    # Reset robot joints with randomization for robustness
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.15, 0.15),
            "velocity_range": (-0.05, 0.05),
        },
    )

    # Randomize robot root state slightly (for robustness)
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "z": (0.0, 0.0),
                "roll": (-0.02, 0.02),
                "pitch": (-0.02, 0.02),
                "yaw": (-0.05, 0.05),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


##
# Environment configuration
##

@configclass
class G1UpperBodyReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the G1 upper body reach environment."""

    # Scene settings
    scene: G1UpperBodyReachSceneCfg = G1UpperBodyReachSceneCfg(num_envs=4096, env_spacing=2.5)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = DECIMATION  # 2 for 50Hz control
        self.episode_length_s = 8.0  # Longer episodes for reaching + holding

        # Simulation settings
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )

##
# Register agent configuration
##

import g1_tasks.reach_g1_upper_body.agents as agents

G1UpperBodyReachEnvCfg.agent = agents.G1UpperReachPPORunnerCfg()
