#!/usr/bin/env python3
"""
Debug script for G1 upper body environment (Isaac Lab 5.1.0)

Usage:
    python scripts/debug_g1_env.py --task Isaac-Reach-G1-Upper-v0
"""

import argparse

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Debug G1 environment setup.")
parser.add_argument("--task", type=str, default="Isaac-Reach-G1-Upper-v0", help="Task to debug.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments.")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force GUI for debugging
args_cli.headless = False

# Launch simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after launching
import gymnasium as gym
import torch
import numpy as np

# Import your tasks
import g1_tasks


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def check_environment(task_name: str, num_envs: int):
    """Run comprehensive environment checks."""
    
    print_section("CREATING ENVIRONMENT")
    
    # Create environment - pass num_envs directly
    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg(task_name, num_envs=num_envs)
    env = gym.make(task_name, cfg=env_cfg)
    print(f"✓ Environment created: {task_name}")
    print(f"  Number of environments: {num_envs}")
    print(f"  Device: {env.unwrapped.device}")
    
    # Check spaces
    print_section("CHECKING SPACES")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Reset environment
    print_section("RESETTING ENVIRONMENT")
    obs, info = env.reset()
    print(f"✓ Environment reset successful")
    
    # Check observations
    print_section("CHECKING OBSERVATIONS")
    if isinstance(obs, dict):
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                print(f"    min={value.min().item():.3f}, max={value.max().item():.3f}, mean={value.mean().item():.3f}")
            elif isinstance(value, dict):
                print(f"  {key}: (nested dict)")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, torch.Tensor):
                        print(f"    {subkey}: shape={subvalue.shape}, dtype={subvalue.dtype}")
    else:
        print(f"  Observation: shape={obs.shape}, dtype={obs.dtype}")
    
    # Check robot state
    print_section("CHECKING ROBOT STATE")
    robot = env.unwrapped.scene["robot"]
    print(f"Robot prim path: {robot.cfg.prim_path}")

    print(f"\nBody names:")
    for i, name in enumerate(robot.body_names):
    	print(f"  {i}: {name}")

    print(f"Number of DOFs: {robot.num_joints}")
    print(f"\nJoint names:")
    joint_pos = robot.data.joint_pos[0].cpu().numpy()
    for i, (name, pos) in enumerate(zip(robot.joint_names, joint_pos)):
        print(f"  {i:2d}. {name:30s}: {pos:7.3f} rad ({np.rad2deg(pos):7.2f}°)")
    
    # Check end-effector frames
    print_section("CHECKING END-EFFECTOR FRAMES")
    if "ee_frame" in env.unwrapped.scene.sensors:
        ee_frame = env.unwrapped.scene.sensors["ee_frame"]
        print(f"✓ Frame transformer found")
        print(f"  Target frames: {[f.name for f in ee_frame.cfg.target_frames]}")
    else:
        print("⚠ No ee_frame sensor found")
    
    # Test actions
    print_section("TESTING ACTIONS")
    
    actions = torch.from_numpy(env.action_space.sample()).to(env.unwrapped.device)
    print(f"Action shape: {actions.shape}")
    print(f"Action range: [{actions.min():.3f}, {actions.max():.3f}]")
    
    print("\nTaking 10 steps...")
    for step in range(10):
        obs, reward, terminated, truncated, info = env.step(actions)
        
        if step % 5 == 0:
            print(f"  Step {step}:")
            print(f"    Reward: {reward[0].item():.3f}")
            print(f"    Terminated: {terminated[0].item()}")
    
    print("✓ Actions executed successfully")
    
    # Check rewards
    print_section("CHECKING REWARD MANAGER")
    if hasattr(env.unwrapped, 'reward_manager'):
        reward_manager = env.unwrapped.reward_manager
        print(f"Number of reward terms: {len(reward_manager.active_terms)}")
        for term_name in reward_manager.active_terms:
            print(f"  - {term_name}")
    
    # Check commands
    print_section("CHECKING COMMAND MANAGER")
    if hasattr(env.unwrapped, 'command_manager'):
        command_manager = env.unwrapped.command_manager
        print(f"Number of command terms: {len(command_manager.active_terms)}")
        for term_name in command_manager.active_terms:
            print(f"  - {term_name}")
    
    # Summary
    print_section("SUMMARY")
    print("✓ All checks passed!")
    print(f"\nEnvironment is ready for training.")
    print(f"\nTo start training, run:")
    print(f"  python scripts/train_g1_upper.py --task {task_name} --headless")
    
    # Keep running for visual inspection
    print_section("VISUAL INSPECTION")
    print("Simulation is running. Close the window or press Ctrl+C to exit.")
    print("Observe:")
    print("  - Robot is standing (waist at correct height)")
    print("  - Arms are in reasonable starting pose")
    print("  - No collision errors or warnings")
    
    try:
        while simulation_app.is_running():
            actions = torch.from_numpy(env.action_space.sample()).to(env.unwrapped.device) * 0.1
            env.step(actions)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    
    env.close()


if __name__ == "__main__":
    try:
        check_environment(args_cli.task, args_cli.num_envs)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
