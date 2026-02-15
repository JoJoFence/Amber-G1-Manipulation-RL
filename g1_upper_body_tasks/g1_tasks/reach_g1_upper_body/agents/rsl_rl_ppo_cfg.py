"""RSL-RL PPO configuration for G1 upper body reach task.

Joint-space control with 14D action space (7 joints × 2 arms) and 54D observation space.
No IK required - policy outputs joint position deltas directly for sim-to-real transfer.
"""

from dataclasses import MISSING
from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class G1UpperReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for RSL-RL PPO runner for G1 upper body reach."""

    num_steps_per_env = 64  # Longer rollouts for better value estimation
    max_iterations = 10000
    save_interval = 200
    experiment_name = "g1_upper_body_reach_joint_space"
    empirical_normalization = True

    clip_obs = 100.0
    clip_actions = 100.0

    # Logging
    logger = "tensorboard"

    # Observation groups
    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy"],
    }

    # Policy configuration - larger network for 14D joint-space control
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,  # Higher for larger 14D action space
        actor_hidden_dims=[512, 256, 256],  # Larger for 6D pose task with orientation
        critic_hidden_dims=[512, 256, 256],
        activation="elu",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )

    # Algorithm configuration
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,  # Slightly higher - cleaner obs enables faster learning
        schedule="adaptive",
        gamma=0.99,
        lam=0.97,  # Higher lambda for better long-horizon value estimation
        desired_kl=0.01,
        max_grad_norm=0.5,
    )
