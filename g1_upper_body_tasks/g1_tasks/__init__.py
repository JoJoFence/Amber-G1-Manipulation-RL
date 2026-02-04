"""
G1 Upper Body Manipulation Tasks for Isaac Lab 5.1.0

This package contains environments and configurations for training
the Unitree G1 humanoid robot's upper body (torso + dual arms) using
inverse kinematics control.
"""

# Import to register environments
from .reach_g1_upper_body import *

__all__ = [
    # Reach tasks
    "G1UpperBodyReachEnvCfg",
]
