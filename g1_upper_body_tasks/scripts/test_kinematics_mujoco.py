"""
Test G1 arm kinematics against MuJoCo ground truth.

This script:
1. Loads the G1 dual arm model in MuJoCo
2. Sets various joint configurations
3. Compares our FK computation vs MuJoCo's actual body positions
4. Visualizes the results

Run with: python test_kinematics_mujoco.py
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
from pathlib import Path

# Import our kinematics
from g1_kinematics import G1ArmKinematics, DifferentialIK


def quaternion_to_rotation_matrix(quat):
    """Convert quaternion [w, x, y, z] to rotation matrix."""
    w, x, y, z = quat
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


class KinematicsTest:
    def __init__(self):
        # Find the MJCF file
        script_dir = Path(__file__).parent
        mjcf_path = script_dir.parent.parent / "g1_description" / "g1_dual_arm.xml"

        if not mjcf_path.exists():
            # Try alternate location
            mjcf_path = Path("/home/jonas/g1_env/unitree_ros/robots/g1_description/g1_dual_arm.xml")

        print(f"Loading model from: {mjcf_path}")
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)

        # Initialize our kinematics
        self.left_kin = G1ArmKinematics('left')
        self.right_kin = G1ArmKinematics('right')

        # Get joint and body IDs
        self.joint_names = {
            'left': [
                'left_shoulder_pitch_joint',
                'left_shoulder_roll_joint',
                'left_shoulder_yaw_joint',
                'left_elbow_joint',
                'left_wrist_roll_joint',
                'left_wrist_pitch_joint',
                'left_wrist_yaw_joint',
            ],
            'right': [
                'right_shoulder_pitch_joint',
                'right_shoulder_roll_joint',
                'right_shoulder_yaw_joint',
                'right_elbow_joint',
                'right_wrist_roll_joint',
                'right_wrist_pitch_joint',
                'right_wrist_yaw_joint',
            ]
        }

        # Get joint IDs
        self.joint_ids = {}
        for arm in ['left', 'right']:
            self.joint_ids[arm] = []
            for name in self.joint_names[arm]:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                self.joint_ids[arm].append(jid)
                print(f"  Joint {name}: id={jid}")

        # Get end-effector body IDs (wrist_yaw_link is the last actuated link)
        # The rubber_hand is attached but we use wrist_yaw as reference
        self.ee_body_ids = {
            'left': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'left_wrist_yaw_link'),
            'right': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'right_wrist_yaw_link'),
        }
        print(f"Left EE body id: {self.ee_body_ids['left']}")
        print(f"Right EE body id: {self.ee_body_ids['right']}")

        # Print all body names for debugging
        print("\nAll bodies in model:")
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            print(f"  {i}: {name}")

    def set_joint_angles(self, left_angles, right_angles):
        """Set joint angles in MuJoCo simulation."""
        for i, jid in enumerate(self.joint_ids['left']):
            if i < len(left_angles):
                self.data.qpos[jid] = left_angles[i]

        for i, jid in enumerate(self.joint_ids['right']):
            if i < len(right_angles):
                self.data.qpos[jid] = right_angles[i]

        mujoco.mj_forward(self.model, self.data)

    def get_mujoco_ee_pose(self, arm):
        """Get end-effector position from MuJoCo."""
        body_id = self.ee_body_ids[arm]
        pos = self.data.xpos[body_id].copy()
        quat = self.data.xquat[body_id].copy()  # [w, x, y, z]
        rot = quaternion_to_rotation_matrix(quat)
        return pos, rot

    def compare_fk(self, left_angles, right_angles, verbose=True):
        """Compare our FK with MuJoCo ground truth."""
        # Set angles in MuJoCo
        self.set_joint_angles(left_angles, right_angles)

        # Get MuJoCo positions
        mj_left_pos, mj_left_rot = self.get_mujoco_ee_pose('left')
        mj_right_pos, mj_right_rot = self.get_mujoco_ee_pose('right')

        # Compute our FK
        our_left_pos, our_left_rot = self.left_kin.forward_kinematics(left_angles)
        our_right_pos, our_right_rot = self.right_kin.forward_kinematics(right_angles)

        # Compute errors
        left_pos_err = np.linalg.norm(mj_left_pos - our_left_pos)
        right_pos_err = np.linalg.norm(mj_right_pos - our_right_pos)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Joint angles:")
            print(f"  Left:  {left_angles}")
            print(f"  Right: {right_angles}")
            print(f"\nLeft arm:")
            print(f"  MuJoCo EE pos:  {mj_left_pos}")
            print(f"  Our FK pos:     {our_left_pos}")
            print(f"  Position error: {left_pos_err*1000:.1f} mm")
            print(f"\nRight arm:")
            print(f"  MuJoCo EE pos:  {mj_right_pos}")
            print(f"  Our FK pos:     {our_right_pos}")
            print(f"  Position error: {right_pos_err*1000:.1f} mm")

        return left_pos_err, right_pos_err, mj_left_pos, mj_right_pos

    def test_ik(self, arm='left'):
        """Test IK by moving to a target and checking if we get there."""
        kin = self.left_kin if arm == 'left' else self.right_kin
        ik_solver = DifferentialIK(kin, lambda_val=0.1)

        # Start at default position
        if arm == 'left':
            q = np.array([0.4, 0.3, 0.0, 0.8, 0.0, 0.0, 0.0])
            other_q = np.array([0.4, -0.3, 0.0, 0.8, 0.0, 0.0, 0.0])
        else:
            q = np.array([0.4, -0.3, 0.0, 0.8, 0.0, 0.0, 0.0])
            other_q = np.array([0.4, 0.3, 0.0, 0.8, 0.0, 0.0, 0.0])

        # Get initial EE position
        pos_init, _ = kin.forward_kinematics(q)
        print(f"\n{arm.upper()} ARM IK TEST")
        print(f"Initial EE position: {pos_init}")

        # Target: move 5cm forward in X
        target_delta = np.array([0.05, 0.0, 0.0])
        target_pos = pos_init + target_delta
        print(f"Target position: {target_pos}")

        # Iteratively solve IK
        max_iters = 50
        for i in range(max_iters):
            pos_curr, _ = kin.forward_kinematics(q)
            error = target_pos - pos_curr
            err_norm = np.linalg.norm(error)

            if err_norm < 0.001:  # 1mm tolerance
                print(f"Converged in {i} iterations!")
                break

            # IK step
            delta_q = ik_solver.solve(q, error * 0.5, n_joints=4)  # Scaled error for stability
            q[:4] += delta_q

        pos_final, _ = kin.forward_kinematics(q)
        final_error = np.linalg.norm(target_pos - pos_final)
        print(f"Final EE position: {pos_final}")
        print(f"Final error: {final_error*1000:.1f} mm")
        print(f"Final joint angles: {q[:4]}")

        return q, final_error

    def run_visual_test(self):
        """Run visual test with MuJoCo viewer."""
        print("\nStarting visual test...")
        print("Controls:")
        print("  1-4: Test different joint configurations")
        print("  I: Test IK")
        print("  ESC: Exit")

        # Test configurations
        configs = [
            # Default pose
            (np.array([0.4, 0.3, 0.0, 0.8, 0.0, 0.0, 0.0]),
             np.array([0.4, -0.3, 0.0, 0.8, 0.0, 0.0, 0.0])),
            # Arms forward
            (np.array([0.0, 0.3, 0.0, 0.5, 0.0, 0.0, 0.0]),
             np.array([0.0, -0.3, 0.0, 0.5, 0.0, 0.0, 0.0])),
            # Arms down
            (np.array([1.0, 0.3, 0.0, 0.3, 0.0, 0.0, 0.0]),
             np.array([1.0, -0.3, 0.0, 0.3, 0.0, 0.0, 0.0])),
            # Zero pose
            (np.zeros(7), np.zeros(7)),
        ]

        config_idx = 0
        self.set_joint_angles(*configs[config_idx])
        self.compare_fk(*configs[config_idx])

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.01)


def main():
    print("G1 Kinematics Test")
    print("=" * 60)

    test = KinematicsTest()

    # Test various configurations
    print("\n" + "=" * 60)
    print("TESTING FK ACCURACY")
    print("=" * 60)

    test_configs = [
        # (left_angles, right_angles, description)
        (np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
         np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
         "Zero pose"),
        (np.array([0.4, 0.3, 0.0, 0.8, 0.0, 0.0, 0.0]),
         np.array([0.4, -0.3, 0.0, 0.8, 0.0, 0.0, 0.0]),
         "Default pose"),
        (np.array([0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0]),
         np.array([0.0, -0.5, 0.0, 1.0, 0.0, 0.0, 0.0]),
         "Arms out"),
        (np.array([0.8, 0.2, 0.3, 0.6, 0.0, 0.0, 0.0]),
         np.array([0.8, -0.2, -0.3, 0.6, 0.0, 0.0, 0.0]),
         "Random pose 1"),
    ]

    total_left_err = 0
    total_right_err = 0

    for left_q, right_q, desc in test_configs:
        print(f"\n--- {desc} ---")
        l_err, r_err, _, _ = test.compare_fk(left_q, right_q)
        total_left_err += l_err
        total_right_err += r_err

    n = len(test_configs)
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"Average left arm error:  {total_left_err/n*1000:.1f} mm")
    print(f"Average right arm error: {total_right_err/n*1000:.1f} mm")

    # Test IK
    print("\n" + "=" * 60)
    print("TESTING IK")
    print("=" * 60)
    test.test_ik('left')
    test.test_ik('right')

    # Visual test
    print("\n" + "=" * 60)
    print("VISUAL TEST")
    print("=" * 60)
    test.run_visual_test()


if __name__ == "__main__":
    main()
