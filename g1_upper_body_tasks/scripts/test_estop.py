"""
E-Stop & movement test for G1 robot.

Connects to the robot, moves arms to the default pose with a small
test wiggle, and keeps the e-stop active so you can verify SPACEBAR
stops all motion.  No trained policy is needed.

Usage:
    python test_estop.py
    python test_estop.py --network_interface eth0
"""

import argparse
import math
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

DEFAULT_POSITIONS = np.array([
    0.4, 0.3, 0.0, 0.8, 0.0, 0.0, 0.0,   # left arm
    0.4, -0.3, 0.0, 0.8, 0.0, 0.0, 0.0,   # right arm
])

WAIST_JOINTS = [G1JointIndex.WaistYaw, G1JointIndex.WaistRoll, G1JointIndex.WaistPitch]
DEFAULT_WAIST = np.array([0.0, 0.0, 0.0])

ARM_KP = np.array([60.0, 60.0, 40.0, 40.0, 30.0, 30.0, 30.0,
                   60.0, 60.0, 40.0, 40.0, 30.0, 30.0, 30.0])
ARM_KD = np.array([2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0,
                   2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0])
WAIST_KP = np.array([60.0, 40.0, 40.0])
WAIST_KD = np.array([1.0, 1.0, 1.0])

# ── Wiggle parameters ───────────────────────────────────────────────────
WIGGLE_AMPLITUDE = 0.08   # radians (~4.5 degrees) — small & safe
WIGGLE_PERIOD    = 3.0    # seconds per cycle
WIGGLE_JOINTS    = [0, 7] # indices into POLICY_JOINT_ORDER (both shoulder pitches)


def main():
    parser = argparse.ArgumentParser(description="Test E-Stop & basic movement on G1")
    parser.add_argument("--network_interface", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  G1 E-Stop & Movement Test")
    print("=" * 60)
    print()
    print("This script will:")
    print("  1. Connect to the robot")
    print("  2. Smoothly move arms to the default pose (3 s)")
    print("  3. Gently wiggle both shoulder pitches back & forth")
    print("  4. Keep wiggling until you press SPACEBAR (e-stop)")
    print("     or Q / Ctrl-C to quit gracefully")
    print()
    print("  >>> Press SPACEBAR at any time to test the E-Stop <<<")
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

    low_cmd  = unitree_hg_msg_dds__LowCmd_()
    crc      = CRC()
    low_state = [None]           # mutable container for callback

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
    print("Communication OK")

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

    # ── Step 1: move to default pose (3 s smoothstep) ───────────────────
    print("\nMoving to default pose...")
    start_pos = np.array([low_state[0].motor_state[i].q for i in POLICY_JOINT_ORDER])
    start_waist = np.array([low_state[0].motor_state[i].q for i in WAIST_JOINTS])
    duration = 3.0
    t0 = time.time()
    while time.time() - t0 < duration:
        t = (time.time() - t0) / duration
        a = 3 * t**2 - 2 * t**3
        targets = start_pos + a * (DEFAULT_POSITIONS - start_pos)
        w_targets = start_waist + a * (DEFAULT_WAIST - start_waist)
        send(targets, w_targets)
        time.sleep(0.002)
    send(DEFAULT_POSITIONS)
    print("Default pose reached.")

    # ── Step 2: init e-stop ──────────────────────────────────────────────
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

    # ── Step 3: wiggle loop ─────────────────────────────────────────────
    print(f"\nWiggling shoulder pitches  (±{np.degrees(WIGGLE_AMPLITUDE):.1f}°, "
          f"period {WIGGLE_PERIOD:.1f}s)")
    print("Press SPACEBAR to test e-stop, or Ctrl-C to quit.\n")

    t0 = time.time()
    try:
        while not estop.triggered:
            elapsed = time.time() - t0
            offset = WIGGLE_AMPLITUDE * math.sin(2 * math.pi * elapsed / WIGGLE_PERIOD)

            targets = DEFAULT_POSITIONS.copy()
            for j in WIGGLE_JOINTS:
                targets[j] += offset

            send(targets)
            time.sleep(0.002)
    except KeyboardInterrupt:
        print("\nCtrl-C received — holding default pose for 2 s then exiting.")
        t1 = time.time()
        while time.time() - t1 < 2.0:
            send(DEFAULT_POSITIONS)
            time.sleep(0.002)
        estop.stop()
        print("Done.")


if __name__ == "__main__":
    main()
