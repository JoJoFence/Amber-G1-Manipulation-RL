#!/usr/bin/env python3
"""
Play script for visualizing trained G1 upper body policies (Isaac Lab 5.1.0)

Usage:
    python scripts/play_g1_upper.py --task Isaac-Reach-G1-Upper-v0 --num_envs 32
    python scripts/play_g1_upper.py --task Isaac-Reach-G1-Upper-v0 --checkpoint 500
"""

import argparse
import os
import re

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Play trained G1 upper body manipulation policies.")
parser.add_argument("--task", type=str, default="Isaac-Reach-G1-Upper-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to load.")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Always run with GUI for play
args_cli.headless = False

# Launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after launching the app
import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry

# Import your tasks
import g1_tasks



def main():
    # Parse configs
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    
    # Create runner with agent config as dict
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    
    # Load checkpoint
    if args_cli.checkpoint:
        print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
        runner.load(args_cli.checkpoint)
    else:
        # Find latest checkpoint
        log_dir = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        if os.path.exists(log_dir):
            runs = sorted([d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))])
            if runs:
                latest_run = runs[-1]
                checkpoint_dir = os.path.join(log_dir, latest_run)
                # Find all model files
                model_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("model_") and f.endswith(".pt")]
                
                if model_files:
                    # Use a key to sort by the integer value within the filename
                    # re.findall(r'\d+', f)[0] extracts the first number found in the string
                    model_files.sort(key=lambda f: int(re.findall(r'\d+', f)[0]))
                    
                    checkpoint_path = os.path.join(checkpoint_dir, model_files[-1])
                    print(f"[INFO] Loading latest checkpoint: {checkpoint_path}")
                    runner.load(checkpoint_path)
                else:
                    print("[WARN] No checkpoints found")
            else:
                print("[WARN] No training runs found")
        else:
            print(f"[WARN] Log directory not found: {log_dir}")
    
    # Play
    obs, _ = env.reset()
    print("[INFO] Playing trained policy. Close window to exit.")
    
    with torch.no_grad():
        while simulation_app.is_running():
            actions = runner.alg.act(obs)
            obs, rewards, dones, info = env.step(actions)
    
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
