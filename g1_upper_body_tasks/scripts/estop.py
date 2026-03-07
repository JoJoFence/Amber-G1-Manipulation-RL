"""
Software Emergency Stop (E-Stop) for G1 Robot Deployment.

Monitors for an emergency stop trigger (SPACEBAR by default) in a background
thread. When triggered, it:
  1. Signals the main control loop to stop sending policy actions
  2. Commands all controlled joints to hold their current positions with
     high damping (Kd) and zero velocity — this smoothly arrests all motion
  3. Maintains the damping commands for a configurable hold duration
  4. Then exits the process

Usage:
    from estop import EStopMonitor

    estop = EStopMonitor()
    estop.start()

    while not estop.triggered:
        # ... normal control loop ...

    # E-stop was triggered — estop.dampen() is called automatically,
    # or you can call it manually in your finally block.

The E-stop can also be triggered programmatically:
    estop.trigger()
"""

import sys
import time
import signal
import threading
from typing import Optional, Callable


class EStopMonitor:
    """Monitors keyboard for emergency stop trigger.

    When SPACEBAR (or the configured trigger key) is pressed, sets
    `self.triggered = True` and calls the registered dampen callback.

    Args:
        trigger_key: The character that triggers the E-stop. Default is ' ' (spacebar).
        dampen_callback: Function called when E-stop fires. Receives no arguments.
            Typically this sends damping commands to the robot.
        hold_duration: How long (seconds) to maintain damping commands after trigger.
        exit_after: Whether to call sys.exit() after the hold duration.
    """

    def __init__(
        self,
        trigger_key: str = ' ',
        dampen_callback: Optional[Callable] = None,
        hold_duration: float = 3.0,
        exit_after: bool = True,
    ):
        self.trigger_key = trigger_key
        self.dampen_callback = dampen_callback
        self.hold_duration = hold_duration
        self.exit_after = exit_after

        self.triggered = False
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._old_terminal_settings = None

        # Key name for display
        self._key_name = "SPACEBAR" if trigger_key == ' ' else repr(trigger_key)

    def start(self):
        """Start the E-stop monitor in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

        print(f"\n{'='*60}")
        print(f"  E-STOP ACTIVE: Press {self._key_name} to emergency stop")
        print(f"  Robot will dampen to current position and halt.")
        print(f"{'='*60}\n")

    def stop(self):
        """Stop the monitor thread (does NOT trigger E-stop)."""
        self._running = False
        self._restore_terminal()

    def trigger(self):
        """Programmatically trigger the E-stop."""
        if self.triggered:
            return  # Already triggered, avoid re-entry
        self.triggered = True
        self._running = False

        print(f"\n{'!'*60}")
        print(f"  *** E-STOP TRIGGERED ***")
        print(f"  Dampening all joints to current positions...")
        print(f"{'!'*60}\n")

        self._restore_terminal()
        self._execute_dampen()

    def _monitor_loop(self):
        """Background thread: reads single keystrokes looking for the trigger."""
        try:
            import termios
            import tty
            import select

            fd = sys.stdin.fileno()
            self._old_terminal_settings = termios.tcgetattr(fd)

            try:
                tty.setcbreak(fd)

                while self._running and not self.triggered:
                    # Poll stdin with a short timeout so we can check _running
                    if select.select([sys.stdin], [], [], 0.05)[0]:
                        ch = sys.stdin.read(1)
                        if ch == self.trigger_key:
                            self.trigger()
                            return
            finally:
                self._restore_terminal()

        except (ImportError, OSError):
            # Fallback: just watch for the flag (programmatic trigger only)
            while self._running and not self.triggered:
                time.sleep(0.05)

    def _restore_terminal(self):
        """Restore the terminal to its original settings."""
        if self._old_terminal_settings is not None:
            try:
                import termios
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN,
                                  self._old_terminal_settings)
            except Exception:
                pass
            self._old_terminal_settings = None

    def _execute_dampen(self):
        """Run the dampen callback for the hold duration, then optionally exit."""
        if self.dampen_callback is None:
            print("[E-STOP] No dampen callback registered — stopping immediately.")
            if self.exit_after:
                sys.exit(1)
            return

        start = time.time()
        # Send damping commands repeatedly for the hold duration to ensure
        # the robot receives them even if a few packets are dropped.
        while time.time() - start < self.hold_duration:
            try:
                self.dampen_callback()
            except Exception as e:
                print(f"[E-STOP] Warning: dampen callback error: {e}")
            time.sleep(0.002)  # ~500 Hz command rate

        elapsed = time.time() - start
        print(f"[E-STOP] Damping held for {elapsed:.1f}s. Robot should be stationary.")

        if self.exit_after:
            print("[E-STOP] Exiting process.")
            # Use os._exit to bypass any lingering threads
            import os
            os._exit(0)


def make_dampen_callback(low_cmd, crc, lowcmd_publisher, low_state_getter,
                         joint_indices, kd_values, waist_joints=None,
                         waist_kd=None, mode_machine_getter=None):
    """Factory that creates a dampen callback for the G1 robot.

    This returns a function that, when called, sends a single command frame
    with all controlled joints set to:
      - target position = current position (hold in place)
      - target velocity = 0
      - Kp = 0 (no position tracking — let damping do the work)
      - Kd = high value (strong velocity damping to arrest motion)
      - tau = 0

    Args:
        low_cmd: The LowCmd_ message object (reused across calls).
        crc: The CRC calculator.
        lowcmd_publisher: The channel publisher for rt/lowcmd.
        low_state_getter: Callable that returns the current LowState_ message.
        joint_indices: List of motor indices to dampen (arm joints).
        kd_values: Dict or array mapping joint index to damping gain.
            During E-stop, Kd is doubled from normal operating values.
        waist_joints: Optional list of waist joint indices to also dampen.
        waist_kd: Optional dict/array of waist Kd values.
        mode_machine_getter: Callable returning current mode_machine value.
    """

    def _dampen():
        state = low_state_getter()
        if state is None:
            return

        low_cmd.mode_pr = 0
        if mode_machine_getter:
            low_cmd.mode_machine = mode_machine_getter()

        # Dampen arm joints
        for idx in joint_indices:
            motor = state.motor_state[idx]
            low_cmd.motor_cmd[idx].mode = 1
            low_cmd.motor_cmd[idx].q = motor.q       # Hold current position
            low_cmd.motor_cmd[idx].dq = 0.0           # Zero target velocity
            low_cmd.motor_cmd[idx].kp = 0.0            # No position spring
            low_cmd.motor_cmd[idx].kd = kd_values[idx] * 2.0  # Double damping
            low_cmd.motor_cmd[idx].tau = 0.0           # No feedforward torque

        # Dampen waist joints if provided
        if waist_joints and waist_kd:
            for idx in waist_joints:
                motor = state.motor_state[idx]
                low_cmd.motor_cmd[idx].mode = 1
                low_cmd.motor_cmd[idx].q = motor.q
                low_cmd.motor_cmd[idx].dq = 0.0
                low_cmd.motor_cmd[idx].kp = 0.0
                low_cmd.motor_cmd[idx].kd = waist_kd[idx] * 2.0
                low_cmd.motor_cmd[idx].tau = 0.0

        low_cmd.crc = crc.Crc(low_cmd)
        lowcmd_publisher.Write(low_cmd)

    return _dampen
