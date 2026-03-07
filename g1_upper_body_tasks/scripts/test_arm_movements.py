"""
Multi-joint arm movement test for G1 robot.

Runs a sequence of predefined arm poses to verify all 14 arm joints
respond correctly before deploying a trained policy.  E-stop is active
throughout — press SPACEBAR at any time to halt.

Poses exercised:
  1. Default (neutral) pose
  2. Arms forward (reach)
  3. Arms wide (shoulders abducted)
  4. Elbows bent (flex/extend)
  5. Wrists rotate
  6. Asymmetric pose (left up, right down)
  7. Return to default

Usage:
    python test_arm_movements.py
    python test_arm_movements.py --network_interface eth0
    python test_arm_movements.py --speed slow     # 5 s per pose
    python test_arm_movements.py --speed fast     # 2 s per pose
"""

import argparse
import time
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from estop import EStopMonitor, make_dampen_callback

from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)

# ── Joint indices (29-DOF G1) ───────────────────────────────────────────
class G1JointIndex:
    WaistYaw = 12;  WaistRoll = 13;  WaistPitch = 14
    LeftShoulderPitch = 15;  LeftShoulderRoll = 16;  LeftShoulderYaw = 17
    LeftElbow = 18;  LeftWristRoll = 19;  LeftWristPitch = 20;  LeftWristYaw = 21
    RightShoulderPitch = 22; RightShoulderRoll = 23; RightShoulderYaw = 24
    RightElbow = 25; RightWristRoll = 26; RightWristPitch = 27; RightWristYaw = 28

POLICY_JOINT_ORDER = [
    G1JointIndex.LeftShoulderPitch, G1JointIndex.LeftShoulderRoll,
    G1JointIndex.LeftShoulderYaw,   G1JointIndex.LeftElbow,
    G1JointIndex.LeftWristRoll,     G1JointIndex.LeftWristPitch,
    G1JointIndex.LeftWristYaw,
    G1JointIndex.RightShoulderPitch, G1JointIndex.RightShoulderRoll,
    G1JointIndex.RightShoulderYaw,   G1JointIndex.RightElbow,
    G1JointIndex.RightWristRoll,     G1JointIndex.RightWristPitch,
    G1JointIndex.RightWristYaw,
]

# Joint labels for status printing
JOINT_NAMES = [
    "L_ShPitch", "L_ShRoll", "L_ShYaw", "L_Elbow",
    "L_WrRoll",  "L_WrPitch", "L_WrYaw",
    "R_ShPitch", "R_ShRoll", "R_ShYaw", "R_Elbow",
    "R_WrRoll",  "R_WrPitch", "R_WrYaw",
]

DEFAULT_POSITIONS = np.array([
    # Left arm:  pitch  roll   yaw   elbow  wr_roll wr_pitch wr_yaw
                 0.4,   0.3,   0.0,  0.8,   0.0,    0.0,     0.0,
    # Right arm: pitch  roll   yaw   elbow  wr_roll wr_pitch wr_yaw
                 0.4,  -0.3,   0.0,  0.8,   0.0,    0.0,     0.0,
])

WAIST_JOINTS = [G1JointIndex.WaistYaw, G1JointIndex.WaistRoll, G1JointIndex.WaistPitch]
DEFAULT_WAIST = np.array([0.0, 0.0, 0.0])

ARM_KP = np.array([60.0, 60.0, 40.0, 40.0, 30.0, 30.0, 30.0,
                   60.0, 60.0, 40.0, 40.0, 30.0, 30.0, 30.0])
ARM_KD = np.array([2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0,
                   2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0])
WAIST_KP = np.array([60.0, 40.0, 40.0])
WAIST_KD = np.array([1.0, 1.0, 1.0])


# ── Test poses ──────────────────────────────────────────────────────────
# Each pose is (name, 14-element array in POLICY_JOINT_ORDER)
# Keep all values conservative — well within joint limits.

TEST_POSES = [
    ("Default (neutral)", DEFAULT_POSITIONS.copy()),

    ("Arms forward (reach)", np.array([
        #  L: pitch  roll   yaw   elbow  wr_r  wr_p  wr_y
           0.15,  0.15,  0.0,  0.3,   0.0,  0.0,  0.0,
        #  R: pitch  roll   yaw   elbow  wr_r  wr_p  wr_y
           0.15, -0.15,  0.0,  0.3,   0.0,  0.0,  0.0,
    ])),

    ("Arms wide (abducted)", np.array([
           0.4,   0.6,   0.0,  0.8,   0.0,  0.0,  0.0,
           0.4,  -0.6,   0.0,  0.8,   0.0,  0.0,  0.0,
    ])),

    ("Elbows bent (flex)", np.array([
           0.4,   0.3,   0.0,  1.4,   0.0,  0.0,  0.0,
           0.4,  -0.3,   0.0,  1.4,   0.0,  0.0,  0.0,
    ])),

    ("Elbows straight (extend)", np.array([
           0.4,   0.3,   0.0,  0.2,   0.0,  0.0,  0.0,
           0.4,  -0.3,   0.0,  0.2,   0.0,  0.0,  0.0,
    ])),

    ("Wrist rotation", np.array([
           0.4,   0.3,   0.0,  0.8,   0.4,  0.3,  0.0,
           0.4,  -0.3,   0.0,  0.8,  -0.4, -0.3,  0.0,
    ])),

    ("Asymmetric (left high, right low)", np.array([
        #  Left arm reaches up & forward
           0.1,   0.4,   0.0,  0.3,   0.0,  0.0,  0.0,
        #  Right arm stays tucked
           0.6,  -0.2,   0.0,  1.2,   0.0,  0.0,  0.0,
    ])),

    ("Return to default", DEFAULT_POSITIONS.copy()),
]


def main():
    parser = argparse.ArgumentParser(description="Multi-joint arm movement test on G1")
    parser.add_argument("--network_interface", type=str, default=None)
    parser.add_argument("--speed", choices=["slow", "normal", "fast"], default="normal",
                        help="Transition speed between poses")
    parser.add_argument("--hold", type=float, default=1.5,
                        help="Seconds to hold each pose before moving to the next")
    args = parser.parse_args()

    transition_time = {"slow": 5.0, "normal": 3.0, "fast": 2.0}[args.speed]

    print("=" * 60)
    print("  G1 Multi-Joint Arm Movement Test")
    print("=" * 60)
    print()
    print(f"  Transition speed : {args.speed} ({transition_time:.0f}s per move)")
    print(f"  Hold time        : {args.hold:.1f}s per pose")
    print(f"  Total poses      : {len(TEST_POSES)}")
    print()
    print("  Pose sequence:")
    for i, (name, _) in enumerate(TEST_POSES):
        print(f"    {i+1}. {name}")
    print()
    print("  >>> Press SPACEBAR at any time to E-STOP <<<")
    print()
    input("Press Enter to start...")

    # ── Init comms ───────────────────────────────────────────────────────
    if args.network_interface:
        ChannelFactoryInitialize(0, args.network_interface)
    else:
        ChannelFactoryInitialize(0)

    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()

    status, result = msc.CheckMode()
    while result.get("name"):
        print(f"Releasing mode: {result['name']}")
        msc.ReleaseMode()
        status, result = msc.CheckMode()
        time.sleep(1)

    low_cmd = unitree_hg_msg_dds__LowCmd_()
    crc = CRC()
    low_state = [None]

    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(lambda msg: low_state.__setitem__(0, msg), 10)

    print("Waiting for robot state...")
    t0 = time.time()
    while low_state[0] is None:
        if time.time() - t0 > 10.0:
            raise RuntimeError("Timeout waiting for robot state")
        time.sleep(0.1)

    mode_machine = low_state[0].mode_machine
    print("Communication OK\n")

    # ── helpers ──────────────────────────────────────────────────────────
    def send(joint_targets, waist_targets=DEFAULT_WAIST):
        low_cmd.mode_pr = 0
        low_cmd.mode_machine = mode_machine
        for i, idx in enumerate(WAIST_JOINTS):
            low_cmd.motor_cmd[idx].mode = 1
            low_cmd.motor_cmd[idx].q    = float(waist_targets[i])
            low_cmd.motor_cmd[idx].dq   = 0.0
            low_cmd.motor_cmd[idx].kp   = float(WAIST_KP[i])
            low_cmd.motor_cmd[idx].kd   = float(WAIST_KD[i])
            low_cmd.motor_cmd[idx].tau  = 0.0
        for i, idx in enumerate(POLICY_JOINT_ORDER):
            low_cmd.motor_cmd[idx].mode = 1
            low_cmd.motor_cmd[idx].q    = float(joint_targets[i])
            low_cmd.motor_cmd[idx].dq   = 0.0
            low_cmd.motor_cmd[idx].kp   = float(ARM_KP[i])
            low_cmd.motor_cmd[idx].kd   = float(ARM_KD[i])
            low_cmd.motor_cmd[idx].tau  = 0.0
        low_cmd.crc = crc.Crc(low_cmd)
        pub.Write(low_cmd)

    def get_joint_positions():
        return np.array([low_state[0].motor_state[i].q for i in POLICY_JOINT_ORDER])

    def move_to(target, duration):
        """Smoothstep interpolation from current position to target."""
        start = get_joint_positions()
        start_waist = np.array([low_state[0].motor_state[i].q for i in WAIST_JOINTS])
        t0 = time.time()
        while time.time() - t0 < duration:
            if estop.triggered:
                return False
            t = (time.time() - t0) / duration
            a = 3 * t**2 - 2 * t**3   # smoothstep
            send(start + a * (target - start),
                 start_waist + a * (DEFAULT_WAIST - start_waist))
            time.sleep(0.002)
        send(target)
        return True

    def hold(duration):
        """Hold current command for a duration, checking e-stop."""
        t0 = time.time()
        while time.time() - t0 < duration:
            if estop.triggered:
                return False
            time.sleep(0.01)
        return True

    # ── Init e-stop ─────────────────────────────────────────────────────
    kd_lookup = {POLICY_JOINT_ORDER[i]: float(ARM_KD[i]) for i in range(14)}
    waist_kd_lookup = {WAIST_JOINTS[i]: float(WAIST_KD[i]) for i in range(3)}

    dampen_cb = make_dampen_callback(
        low_cmd=low_cmd,
        crc=crc,
        lowcmd_publisher=pub,
        low_state_getter=lambda: low_state[0],
        joint_indices=POLICY_JOINT_ORDER,
        kd_values=kd_lookup,
        waist_joints=WAIST_JOINTS,
        waist_kd=waist_kd_lookup,
        mode_machine_getter=lambda: mode_machine,
    )

    estop = EStopMonitor(
        trigger_key=" ",
        dampen_callback=dampen_cb,
        hold_duration=3.0,
        exit_after=True,
    )
    estop.start()

    # ── First move to default from wherever the robot is ────────────────
    print("Moving to default pose from current position...")
    if not move_to(DEFAULT_POSITIONS, transition_time):
        return
    print()

    # ── Run through test poses ──────────────────────────────────────────
    try:
        for i, (name, target) in enumerate(TEST_POSES):
            print(f"[{i+1}/{len(TEST_POSES)}] {name}")

            # Show which joints differ from default
            diff = target - DEFAULT_POSITIONS
            changed = np.nonzero(np.abs(diff) > 0.01)[0]
            if len(changed) > 0:
                parts = [f"{JOINT_NAMES[j]}={target[j]:+.2f} (Δ{diff[j]:+.2f})" for j in changed]
                print(f"        Changes: {', '.join(parts)}")
            else:
                print(f"        (no change from default)")

            if not move_to(target, transition_time):
                break

            # Print actual joint positions after reaching target
            actual = get_joint_positions()
            error = np.abs(actual - target)
            max_err_idx = np.argmax(error)
            print(f"        Reached. Max tracking error: "
                  f"{JOINT_NAMES[max_err_idx]} = {np.degrees(error[max_err_idx]):.1f}°")

            if not hold(args.hold):
                break

            print()

        if not estop.triggered:
            print("=" * 60)
            print("  All poses completed successfully!")
            print("=" * 60)
            print("\nHolding default pose. Press Ctrl-C to exit.\n")

            # Hold default until user exits
            while not estop.triggered:
                send(DEFAULT_POSITIONS)
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nCtrl-C — returning to default pose...")
        move_to(DEFAULT_POSITIONS, 2.0)
        t1 = time.time()
        while time.time() - t1 < 1.0:
            send(DEFAULT_POSITIONS)
            time.sleep(0.002)
        estop.stop()
        print("Done.")


if __name__ == "__main__":
    main()
