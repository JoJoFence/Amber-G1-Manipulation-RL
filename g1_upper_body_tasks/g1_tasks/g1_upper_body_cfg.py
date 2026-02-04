"""
G1 Upper Body Robot Configuration for Isaac Lab 5.1.0

Defines the Unitree G1 dual-arm robot configuration with:
- Fixed waist (0 DOF) - waist joints are fixed in URDF
- 7 DOF left arm
- 7 DOF right arm
Total: 14 DOF upper body
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##

G1_UPPER_BODY_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/jonas/g1_env/g1_description/g1_dual_arm.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            fix_root_link=True,  # CRITICAL: Fix base in place since no legs
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            # Left arm - FORWARD REACHING POSE
            # Arms extended forward, hands in front of torso for grasping
            "left_shoulder_pitch_joint": 0.4,     # Shoulder pitched forward (~23°)
            "left_shoulder_roll_joint": 0.3,      # Slight outward roll
            "left_shoulder_yaw_joint": 0.0,       # No yaw
            "left_elbow_joint": 0.8,              # Moderate elbow bend (~46°)
            "left_wrist_roll_joint": 0.0,         # Neutral wrist
            "left_wrist_pitch_joint": 0.0,        # Neutral wrist
            "left_wrist_yaw_joint": 0.0,          # Neutral wrist

            # Right arm - symmetric forward reaching pose
            "right_shoulder_pitch_joint": 0.4,    # Same as left
            "right_shoulder_roll_joint": -0.3,    # Opposite roll (symmetric)
            "right_shoulder_yaw_joint": 0.0,      # No yaw
            "right_elbow_joint": 0.8,             # Same elbow bend
            "right_wrist_roll_joint": 0.0,        # Neutral wrist
            "right_wrist_pitch_joint": 0.0,       # Neutral wrist
            "right_wrist_yaw_joint": 0.0,         # Neutral wrist
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "shoulders": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*"],
            stiffness=300.0,  # INCREASED back for stability
            damping=100.0,     # INCREASED back
            effort_limit=25.0,
            velocity_limit=20.0,
        ),
        "elbows": ImplicitActuatorCfg(
            joint_names_expr=[".*_elbow_joint"],
            stiffness=300.0,  # INCREASED back
            damping=100.0,     # INCREASED back
            effort_limit=25.0,
            velocity_limit=20.0,
        ),
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_.*"],
            stiffness=200.0,  # INCREASED for wrist stability
            damping=80.0,     # INCREASED
            effort_limit=5.0,
            velocity_limit=15.0,
        ),
    },
)

# End-effector frame names (use existing hand links from URDF)
LEFT_EE_FRAME = "left_rubber_hand"
RIGHT_EE_FRAME = "right_rubber_hand"

# Body frame for base
BASE_FRAME = "waist_yaw_link"

# Controller settings
DECIMATION = 2  # Increased control frequency (50Hz) for precise reaching
CONTROL_FREQ = 50.0
