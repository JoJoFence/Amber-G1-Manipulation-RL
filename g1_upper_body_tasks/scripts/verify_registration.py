#!/usr/bin/env python3
"""
Verify G1 task registration in Isaac Lab 5.1.0

This script helps debug why environments might not be showing up.
"""

import sys

print("=" * 80)
print("G1 Task Registration Verification")
print("=" * 80)

# Step 1: Check Isaac Lab installation
print("\n[1/5] Checking Isaac Lab installation...")
try:
    import isaaclab
    print(f"✓ Isaac Lab found: {isaaclab.__file__}")
except ImportError as e:
    print(f"✗ Isaac Lab not found: {e}")
    print("  Solution: Install Isaac Lab with ./isaaclab.sh --install")
    sys.exit(1)

# Step 2: Check gymnasium
print("\n[2/5] Checking gymnasium...")
try:
    import gymnasium as gym
    print(f"✓ Gymnasium found: {gym.__version__}")
except ImportError as e:
    print(f"✗ Gymnasium not found: {e}")
    sys.exit(1)

# Step 3: Check g1_tasks package
print("\n[3/5] Checking g1_tasks package...")
try:
    import g1_tasks
    print(f"✓ g1_tasks package found: {g1_tasks.__file__}")
except ImportError as e:
    print(f"✗ g1_tasks package not found: {e}")
    print("  Solution: Run 'pip install -e .' from your g1_upper_body_tasks directory")
    sys.exit(1)

# Step 4: List all registered environments
print("\n[4/5] Checking registered environments...")
all_envs = list(gym.envs.registry.keys())
print(f"Total environments registered: {len(all_envs)}")

# Look for G1 environments
g1_envs = [e for e in all_envs if 'G1' in e or 'g1' in e.lower()]
if g1_envs:
    print(f"✓ Found {len(g1_envs)} G1 environment(s):")
    for env in g1_envs:
        print(f"  - {env}")
else:
    print("✗ No G1 environments found")
    print("\nAll Isaac environments:")
    isaac_envs = [e for e in all_envs if 'Isaac' in e]
    for env in sorted(isaac_envs)[:10]:
        print(f"  - {env}")
    if len(isaac_envs) > 10:
        print(f"  ... and {len(isaac_envs) - 10} more")

# Step 5: Try to create the environment
print("\n[5/5] Attempting to create Isaac-Reach-G1-Upper-v0...")
try:
    env = gym.make("Isaac-Reach-G1-Upper-v0", num_envs=1, render_mode=None)
    print("✓ Environment created successfully!")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    env.close()
except Exception as e:
    print(f"✗ Failed to create environment: {e}")
    print("\nDebugging information:")
    print(f"  Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()

# Final summary
print("\n" + "=" * 80)
print("Verification Summary")
print("=" * 80)

if g1_envs:
    print("✓ SUCCESS: G1 environments are properly registered!")
    print("\nYou can now run:")
    print("  python scripts/train_g1_upper.py --task Isaac-Reach-G1-Upper-v0")
    print("  python scripts/debug_g1_env.py --task Isaac-Reach-G1-Upper-v0")
else:
    print("✗ FAILURE: G1 environments are not registered")
    print("\nTroubleshooting steps:")
    print("1. Make sure you're in the correct conda environment:")
    print("   conda activate isaaclab")
    print("\n2. Install the g1_tasks package:")
    print("   cd ~/workspace/g1_upper_body_tasks")
    print("   pip install -e .")
    print("\n3. Verify package installation:")
    print("   pip list | grep openarm")
    print("\n4. Check for import errors:")
    print("   python -c 'import g1_tasks'")
    print("\n5. Re-run this script")

print("=" * 80)
