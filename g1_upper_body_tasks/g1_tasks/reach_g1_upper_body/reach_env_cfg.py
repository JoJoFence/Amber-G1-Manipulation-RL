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
    # Workspace is conservative to ensure all targets are reachable
    left_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=LEFT_EE_FRAME,
        resampling_time_range=(4.0, 6.0),  # Longer hold for precision learning
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.20, 0.45),   # Forward reach (conservative max for reachability)
            pos_y=(0.10, 0.30),   # Left side workspace (+Y is left)
            pos_z=(-0.15, 0.15),  # Vertical range relative to base
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )

    right_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=RIGHT_EE_FRAME,
        resampling_time_range=(4.0, 6.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.20, 0.45),   # Same forward range
            pos_y=(-0.30, -0.10), # Right side workspace (symmetric, -Y is right)
            pos_z=(-0.15, 0.15),  # Same vertical range
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
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
        scale=0.05,  # Joint position delta scale (radians per action unit)
        use_default_offset=True,  # Actions are relative to default joint positions
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP.

    Key design: error vectors directly tell each arm where its target is,
    eliminating the ambiguity that caused both arms to go to the same target.
    EE velocity enables learned deceleration to prevent overshoot.

    Observation space (54D):
    - joint_pos_rel: 14D (all joint positions relative to default)
    - joint_vel_rel: 14D (all joint velocities)
    - left_ee_error: 3D (vector from left EE to left target)
    - right_ee_error: 3D (vector from right EE to right target)
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
    """Reward terms for the MDP.

    Simplified structure with multi-level tanh rewards (natural deceleration)
    instead of exponential rewards (which caused overshoot due to increasing
    gradient near target). No wrist penalties needed since wrists are excluded
    from IK chain.
    """

    # === LEVEL 1: COARSE TRACKING (gradient at any distance) ===
    left_ee_coarse = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[LEFT_EE_FRAME]),
            "command_name": "left_ee_pose",
            "std": 0.5,
        },
    )

    right_ee_coarse = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_FRAME]),
            "command_name": "right_ee_pose",
            "std": 0.5,
        },
    )

    # === LEVEL 2: MEDIUM TRACKING (main driver, gradient ~5-50cm) ===
    left_ee_medium = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=6.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[LEFT_EE_FRAME]),
            "command_name": "left_ee_pose",
            "std": 0.25,
        },
    )

    right_ee_medium = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=6.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_FRAME]),
            "command_name": "right_ee_pose",
            "std": 0.25,
        },
    )

    # === LEVEL 3: FINE PRECISION (strong gradient within ~10cm) ===
    left_ee_fine = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=15.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[LEFT_EE_FRAME]),
            "command_name": "left_ee_pose",
            "std": 0.1,
        },
    )

    right_ee_fine = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=15.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_FRAME]),
            "command_name": "right_ee_pose",
            "std": 0.1,
        },
    )

    # === HOLDING REWARD (reduce drift when close) ===
    left_ee_holding = RewTerm(
        func=mdp.position_holding_reward,
        weight=10.0,  # Increased from 5.0
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[LEFT_EE_FRAME]),
            "command_name": "left_ee_pose",
            "threshold": 0.04,  # Tightened from 5cm to 4cm
        },
    )

    right_ee_holding = RewTerm(
        func=mdp.position_holding_reward,
        weight=10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_FRAME]),
            "command_name": "right_ee_pose",
            "threshold": 0.04,
        },
    )

    # === SUCCESS BONUS (sparse reward at target) ===
    left_ee_success = RewTerm(
        func=mdp.position_command_success,
        weight=15.0,  # Increased from 10.0
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[LEFT_EE_FRAME]),
            "command_name": "left_ee_pose",
            "threshold": 0.03,  # 3cm success threshold
        },
    )

    right_ee_success = RewTerm(
        func=mdp.position_command_success,
        weight=15.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_FRAME]),
            "command_name": "right_ee_pose",
            "threshold": 0.03,
        },
    )

    # === ANTI-DRIFT: EE velocity penalty near target ===
    left_ee_vel_near_target = RewTerm(
        func=mdp.ee_velocity_penalty_near_target,
        weight=-8.0,  # Strong penalty for moving when should be holding
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[LEFT_EE_FRAME]),
            "command_name": "left_ee_pose",
            "threshold": 0.06,  # 6cm activation zone
        },
    )

    right_ee_vel_near_target = RewTerm(
        func=mdp.ee_velocity_penalty_near_target,
        weight=-8.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_FRAME]),
            "command_name": "right_ee_pose",
            "threshold": 0.06,
        },
    )

    # === REGULARIZATION ===

    # Prevent base tilt
    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Smooth actions - key for preventing overshoot
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.5)

    # Light global joint velocity damping
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Light joint acceleration penalty
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-0.002,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Wrist regularization - keep wrists near neutral during reaching
    # Wrists should stay relatively neutral for reaching tasks
    wrist_position_penalty = RewTerm(
        func=mdp.wrist_position_penalty,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wrist_.*"])},
    )

    # Wrist velocity penalty - discourage excessive wrist movement
    wrist_velocity_penalty = RewTerm(
        func=mdp.wrist_velocity_penalty,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wrist_.*"])},
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
