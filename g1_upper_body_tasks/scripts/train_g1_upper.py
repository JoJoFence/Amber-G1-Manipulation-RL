#!/usr/bin/env python3
"""Train G1 upper body manipulation with RSL-RL."""

import argparse
from isaaclab.app import AppLauncher
#!/usr/bin/env python3
"""Train G1 upper body with RSL-RL using Isaac Lab's official pattern."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train G1 upper body.")
parser.add_argument("--task", type=str, default="Isaac-Reach-G1-Upper-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
from datetime import datetime
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry

import g1_tasks

def main():
    # Parse environment config
    env_cfg = parse_env_cfg(
        args_cli.task,
        device="cuda:0",
        num_envs=args_cli.num_envs,
        use_fabric=True
    )
    
    # Load agent config
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    
    # Override from CLI
    if args_cli.seed:
        agent_cfg.seed = args_cli.seed
    if args_cli.max_iterations:
        agent_cfg.max_iterations = args_cli.max_iterations
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Wrap
    env = RslRlVecEnvWrapper(env)
    
    # Setup logging
    log_dir = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name,
                           datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # Train
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
