"""RSL-RL PPO configuration for G1 upper body reach task.

Tuned for position-only IK with 6D action space and 46D observation space.
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
    max_iterations = 5000
    save_interval = 200
    experiment_name = "g1_upper_body_reach"
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

    # Policy configuration
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,  # Lower for smaller 6D action space
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
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
