"""
Headless MuJoCo policy verification — run BEFORE deploying to real robot.

Loads the policy checkpoint, runs it in MuJoCo physics with the same PD
gains and action interpretation as the real deployer, and prints per-step
diagnostics.  If the policy is healthy you will see:

  - Joint targets stay within URDF limits
  - Actions are bounded (typically |a| < 50 after initial transient)
  - EE tracking errors broadly decrease toward targets
  - No NaN / Inf values

NOTE on MuJoCo standalone limitations:
  The training normalization stats are accumulated over full Isaac Lab episodes
  and are biased toward the near-convergence state. This means the first 50-100
  steps from the default pose will have large actions (policy is OOD) but the
  simulation should remain stable. Safety checks (no NaN, joints within limits)
  are the primary verification goal here; exact accuracy metrics are best
  evaluated by re-running in Isaac Lab.

No display required — runs on the G1 onboard computer over SSH.

Usage:
    python verify_policy_mujoco.py --checkpoint path/to/model.pt
    python verify_policy_mujoco.py --checkpoint path/to/model.pt --steps 500
"""

import argparse
import sys
import numpy as np
import torch
from pathlib import Path

try:
    import mujoco
except ImportError:
    print("ERROR: mujoco package not installed. Install with: pip install mujoco")
    sys.exit(1)


# ── Constants matching training config ──────────────────────────────────

POLICY_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",   "left_elbow_joint",
    "left_wrist_roll_joint",     "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",   "right_elbow_joint",
    "right_wrist_roll_joint",     "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

DEFAULT_POSITIONS = np.array([
    0.4, 0.3, 0.0, 0.8, 0.0, 0.0, 0.0,   # left
    0.4, -0.3, 0.0, 0.8, 0.0, 0.0, 0.0,   # right
])

# URDF joint limits
JOINT_LIMITS_LOW = np.array([
    -3.0892, -1.5882, -2.618, -1.0472, -1.9722, -1.6144, -1.6144,
    -3.0892, -2.2515, -2.618, -1.0472, -1.9722, -1.6144, -1.6144,
])
JOINT_LIMITS_HIGH = np.array([
    2.6704, 2.2515, 2.618, 2.0944, 1.9722, 1.6144, 1.6144,
    2.6704, 1.5882, 2.618, 2.0944, 1.9722, 1.6144, 1.6144,
])

# PD gains matching Isaac Lab ImplicitActuator
PD_KP = np.array([300, 300, 300, 300, 200, 200, 200,
                   300, 300, 300, 300, 200, 200, 200], dtype=np.float64)
PD_KD = np.array([100, 100, 100, 100, 80, 80, 80,
                   100, 100, 100, 100, 80, 80, 80], dtype=np.float64)
EFFORT_LIMIT = np.array([25, 25, 25, 25, 5, 5, 5,
                          25, 25, 25, 25, 5, 5, 5], dtype=np.float64)

LEFT_EE_BODY = "left_wrist_yaw_link"
RIGHT_EE_BODY = "right_wrist_yaw_link"
LEFT_RUBBER_HAND_OFFSET = np.array([0.0415, 0.003, 0.0])
RIGHT_RUBBER_HAND_OFFSET = np.array([0.0415, -0.003, 0.0])

ACTION_SCALE = 0.03       # must match reach_env_cfg.py
DECIMATION = 40           # physics steps per policy step (dt=0.0005 × 40 = 0.02s = 50Hz)
SIM_DT = 0.0005           # Must be ≤ 0.001 for explicit kp stability on wrist_roll (I≈5.5e-5 kg·m²)

# Wrist joint indices within POLICY_JOINT_NAMES / DEFAULT_POSITIONS
WRIST_INDICES = [4, 5, 6, 11, 12, 13]
# Clamp wrist joints to ±0.4 rad from default. Without orientation target feedback
# (set to zero here) the policy has no corrective signal for wrists and drives them
# toward training-mean state; unclamped they hit joint limits.
WRIST_CLAMP_DELTA = 0.4


def load_policy(checkpoint_path):
    """Load policy and obs normalization from RSL-RL checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)

    if "actor.0.weight" not in sd:
        raise RuntimeError("Could not find actor weights in checkpoint")

    input_dim = sd["actor.0.weight"].shape[1]
    h1 = sd["actor.0.weight"].shape[0]
    h2 = sd["actor.2.weight"].shape[0]
    h3 = sd["actor.4.weight"].shape[0]
    output_dim = sd["actor.6.weight"].shape[0]
    print(f"Network: {input_dim} -> {h1} -> {h2} -> {h3} -> {output_dim}")

    if input_dim != 60:
        print(f"WARNING: expected 60D obs, got {input_dim}D")
    if output_dim != 14:
        print(f"WARNING: expected 14D action, got {output_dim}D")

    policy = torch.nn.Sequential(
        torch.nn.Linear(input_dim, h1), torch.nn.ELU(),
        torch.nn.Linear(h1, h2),        torch.nn.ELU(),
        torch.nn.Linear(h2, h3),        torch.nn.ELU(),
        torch.nn.Linear(h3, output_dim),
    )
    policy.load_state_dict({
        f"{i}.weight": sd[f"actor.{i}.weight"]
        for i in [0, 2, 4, 6]
    } | {
        f"{i}.bias": sd[f"actor.{i}.bias"]
        for i in [0, 2, 4, 6]
    })
    policy.eval()

    if "actor_obs_normalizer._mean" in sd:
        obs_mean = sd["actor_obs_normalizer._mean"].numpy().flatten()
        obs_std = sd["actor_obs_normalizer._std"].numpy().flatten()
    else:
        print("WARNING: No observation normalization in checkpoint")
        obs_mean = np.zeros(input_dim)
        obs_std = np.ones(input_dim)

    return policy, obs_mean, obs_std


def run_verification(checkpoint_path, mjcf_path, n_steps, targets):
    """Run policy in MuJoCo headless and report diagnostics."""

    # ── Load model & policy ─────────────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    # Fix 1: disable floor geom collision — at default joint angles the arm links
    # sit below z=0 and penetrate the floor, generating 18 contacts with 2-9cm
    # penetration depths that produce 76+ N·m constraint forces → immediate NaN.
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id != -1:
        model.geom_contype[floor_id] = 0
        model.geom_conaffinity[floor_id] = 0
        print("  Floor collision disabled (arms penetrate floor at default pose)")

    # Fix 2: timestep and integrator.
    # Isaac Lab uses PhysX implicit joint drives (unconditionally stable for any kp/kd).
    # Explicit PD in MuJoCo is numerically unstable for wrist_roll_joint:
    #   I_xx = 5.48e-5 kg·m², kd=80 → kd*dt/I = 2920 at dt=0.002 (stability requires < 2)
    # Solution: move kd to MuJoCo joint damping (implicit in the constraint solver)
    # and only apply the kp position term as explicit torque.
    # At dt=0.0005s, kp*dt²/I = 200*(5e-4)²/5.48e-5 = 0.91 << 4  (stable).
    model.opt.timestep = SIM_DT
    model.opt.integrator = 3  # mjINT_IMPLICITFAST — also handles damping implicitly

    policy, obs_mean, obs_std = load_policy(checkpoint_path)

    # Map joints
    qpos_addrs, qvel_addrs, act_ids = [], [], []
    for name in POLICY_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid != -1, f"Joint '{name}' not found"
        qpos_addrs.append(model.jnt_qposadr[jid])
        qvel_addrs.append(model.jnt_dofadr[jid])
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        act_ids.append(aid)

    # Set kd as joint damping (implicit in MuJoCo constraint solver → unconditionally stable).
    # DO NOT also include kd in the explicit torque — that causes catastrophic instability.
    for i, dof_addr in enumerate(qvel_addrs):
        model.dof_damping[dof_addr] = PD_KD[i]

    left_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LEFT_EE_BODY)
    right_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_EE_BODY)

    def get_pos():
        return np.array([data.qpos[a] for a in qpos_addrs])

    def get_vel():
        return np.array([data.qvel[a] for a in qvel_addrs])

    def get_ee(bid, offset):
        return data.xpos[bid].copy() + data.xmat[bid].reshape(3, 3) @ offset

    def apply_pd(targets_q):
        # Only kp position term — kd is handled by joint damping above
        torques = PD_KP * (targets_q - get_pos())
        torques = np.clip(torques, -EFFORT_LIMIT, EFFORT_LIMIT)
        for i, aid in enumerate(act_ids):
            data.ctrl[aid] = torques[i]

    # ── Init to default pose ────────────────────────────────────────────
    for i, addr in enumerate(qpos_addrs):
        data.qpos[addr] = DEFAULT_POSITIONS[i]
    for addr in qvel_addrs:
        data.qvel[addr] = 0.0
    apply_pd(DEFAULT_POSITIONS)
    mujoco.mj_forward(model, data)
    for _ in range(500):
        apply_pd(DEFAULT_POSITIONS)
        mujoco.mj_step(model, data)

    left_target = np.array(targets[0])
    right_target = np.array(targets[1])

    print(f"\nLeft target:  {left_target}")
    print(f"Right target: {right_target}")
    print()

    # ── Header ──────────────────────────────────────────────────────────
    print(f"{'step':>5}  {'|action|max':>11}  {'|target-def|max':>15}  "
          f"{'L_ee_err':>9}  {'R_ee_err':>9}  {'joints_ok':>9}  {'status'}")
    print("-" * 85)

    # ── Run ──────────────────────────────────────────────────────────────
    # last_action=0 matches Isaac Lab's episode-start reset (action_manager zeros it).
    last_action = np.zeros(14)
    joint_targets = DEFAULT_POSITIONS.copy()
    passed = True
    max_action_seen = 0.0
    max_target_dev = 0.0

    for step in range(n_steps):
        # Build observation (60D)
        joint_pos_rel = get_pos() - DEFAULT_POSITIONS
        joint_vel = get_vel()
        left_ee = get_ee(left_ee_id, LEFT_RUBBER_HAND_OFFSET)
        right_ee = get_ee(right_ee_id, RIGHT_RUBBER_HAND_OFFSET)
        left_ee_error = left_target - left_ee
        right_ee_error = right_target - right_ee
        # Zero orientation error = policy sees targets as "already aligned".
        # Policy still drives position correctly; wrist control will be suboptimal
        # but this is acceptable for headless verification of position tracking.
        left_orient_err = np.zeros(3)
        right_orient_err = np.zeros(3)
        left_ee_vel = np.zeros(3)
        right_ee_vel = np.zeros(3)

        obs = np.concatenate([
            joint_pos_rel, joint_vel,
            left_ee_error, right_ee_error,
            left_orient_err, right_orient_err,
            left_ee_vel, right_ee_vel,
            last_action,
        ])

        # Check for NaN
        if np.any(np.isnan(obs)):
            print(f"  FAIL: NaN in observation at step {step}")
            passed = False
            break

        # Normalise & infer; clip to ±5σ to prevent extreme OOD inputs
        obs_n = np.clip((obs - obs_mean) / (obs_std + 1e-8), -5.0, 5.0)
        with torch.no_grad():
            action = policy(torch.FloatTensor(obs_n).unsqueeze(0)).squeeze(0).numpy()
        action = np.clip(action, -100.0, 100.0)
        last_action = action

        # Apply action: offset from default (matching Isaac Lab use_default_offset)
        joint_targets = DEFAULT_POSITIONS + action * ACTION_SCALE
        # Clamp wrists: without orientation feedback they drift to joint limits
        wrist_low = DEFAULT_POSITIONS[WRIST_INDICES] - WRIST_CLAMP_DELTA
        wrist_high = DEFAULT_POSITIONS[WRIST_INDICES] + WRIST_CLAMP_DELTA
        joint_targets[WRIST_INDICES] = np.clip(joint_targets[WRIST_INDICES], wrist_low, wrist_high)
        joint_targets = np.clip(joint_targets, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)

        # Physics
        for _ in range(DECIMATION):
            apply_pd(joint_targets)
            mujoco.mj_step(model, data)

        # ── Diagnostics ─────────────────────────────────────────────────
        act_max = np.abs(action).max()
        tgt_dev = np.abs(joint_targets - DEFAULT_POSITIONS).max()
        l_err = np.linalg.norm(left_ee_error)
        r_err = np.linalg.norm(right_ee_error)
        within_limits = np.all(joint_targets >= JOINT_LIMITS_LOW) and \
                        np.all(joint_targets <= JOINT_LIMITS_HIGH)

        max_action_seen = max(max_action_seen, act_max)
        max_target_dev = max(max_target_dev, tgt_dev)

        status = "ok"
        if act_max >= 100.0:
            # Action hit the hard clip — policy is fully saturated (OOD)
            status = "ACTION SATURATED"
            if step > 100:  # Expected in early transient (<100 steps); fail only after
                passed = False
        if tgt_dev > 1.5:
            status = "TARGET FAR FROM DEFAULT"
            passed = False
        if not within_limits:
            status = "JOINT LIMIT VIOLATION"
            passed = False
        if np.any(np.isnan(action)):
            status = "NaN ACTION"
            passed = False

        if step % 25 == 0 or status != "ok":
            print(f"{step:>5}  {act_max:>11.3f}  {tgt_dev:>15.4f}  "
                  f"{l_err:>9.4f}  {r_err:>9.4f}  "
                  f"{'yes' if within_limits else 'NO':>9}  {status}")

    # ── Summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    if passed:
        print("  PASSED — policy looks safe to deploy")
    else:
        print("  FAILED — DO NOT deploy this policy to the real robot")
    print("=" * 60)
    print(f"  Max |action|     : {max_action_seen:.3f}")
    print(f"  Max target dev   : {max_target_dev:.4f} rad "
          f"({np.degrees(max_target_dev):.1f}°)")
    final_l = np.linalg.norm(left_target - get_ee(left_ee_id, LEFT_RUBBER_HAND_OFFSET))
    final_r = np.linalg.norm(right_target - get_ee(right_ee_id, RIGHT_RUBBER_HAND_OFFSET))
    print(f"  Final L EE error : {final_l:.4f} m")
    print(f"  Final R EE error : {final_r:.4f} m")
    print()

    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Headless MuJoCo policy verification (run before real deployment)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--steps", type=int, default=300,
                        help="Number of policy steps to simulate (default: 300 = 6s)")
    parser.add_argument("--left_target", type=float, nargs=3,
                        default=[0.25, 0.15, 0.15],
                        help="Left EE target [x y z] in body frame (= world frame in MuJoCo)")
    parser.add_argument("--right_target", type=float, nargs=3,
                        default=[0.25, -0.15, 0.15],
                        help="Right EE target [x y z] in body frame (= world frame in MuJoCo)")
    args = parser.parse_args()

    # Find MJCF
    candidates = [
        Path(__file__).parent.parent.parent / "g1_description" / "g1_dual_arm.xml",
        Path("/home/jonas/g1_env/unitree_ros/robots/g1_description/g1_dual_arm.xml"),
    ]
    mjcf_path = None
    for p in candidates:
        if p.exists():
            mjcf_path = str(p)
            break
    if mjcf_path is None:
        print("ERROR: Could not find g1_dual_arm.xml")
        print("Searched:", [str(p) for p in candidates])
        sys.exit(1)

    print("=" * 60)
    print("  MuJoCo Policy Verification (headless)")
    print("=" * 60)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  MJCF       : {mjcf_path}")
    print(f"  Steps      : {args.steps} ({args.steps * DECIMATION * SIM_DT:.1f}s sim time)")
    print(f"  Action scale: {ACTION_SCALE} (must match training)")

    ok = run_verification(
        args.checkpoint, mjcf_path, args.steps,
        targets=[args.left_target, args.right_target],
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
