import json
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
from capstone_vla.franka import FrankaActionContract, FrankaCartesianAdapter

ROOT = Path(__file__).resolve().parents[1]


class FrankaActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = json.loads((ROOT/'configs/franka_action_interface.json').read_text())
        cls.adapter = FrankaCartesianAdapter(cfg['urdf'])
        cls.q = np.array(cfg['initial_arm_joints_rad'])

    def test_gripper_and_invalid_action(self):
        c = FrankaActionContract()
        self.assertEqual(c.decode([0,0,0,0,0,0,1])[2], .04)
        self.assertEqual(c.decode([0,0,0,0,0,0,-1])[2], 0)
        for action in ([0]*7, [0]*6, [np.nan]*7):
            with self.assertRaises(ValueError):c.decode(action)

    def test_frame_identity_has_no_offset_bias(self):
        a = self.adapter
        action = np.r_[a.state(self.q), 1.0]
        target, _ = a.joint_target(self.q, action)
        np.testing.assert_allclose(target, self.q, atol=2e-7, rtol=0)

    def test_jacobian_against_finite_difference(self):
        a = self.adapter
        p, r, jac = a.pose_jacobian(self.q)
        numerical = np.empty_like(jac)
        for i in range(7):
            q = self.q.copy();q[i] += 1e-6
            pp, rr, _ = a.pose_jacobian(q)
            numerical[:,i] = np.r_[(pp-p)/1e-6, (rr*r.inv()).as_rotvec()/1e-6]
        np.testing.assert_allclose(jac, numerical, atol=1e-6, rtol=0)

    def test_euler_periodicity_does_not_move_robot(self):
        a = self.adapter
        action = np.r_[a.state(self.q), -1.]
        action[3] += 2*np.pi;action[5] -= 2*np.pi
        target, _ = a.joint_target(self.q, action)
        np.testing.assert_allclose(target, self.q, atol=2e-7, rtol=0)

    def test_small_goal_is_kinematically_reachable(self):
        a = self.adapter
        p, r, _ = a.pose_jacobian(self.q)
        goal = a.contract.encode(p+[.02,0,0],r,1)
        q = self.q.copy()
        for _ in range(15):q, _ = a.joint_target(q,goal)
        pp, rr, _ = a.pose_jacobian(q)
        self.assertLess(np.linalg.norm(pp-goal[:3]), 1e-5)
        self.assertLess((rr*r.inv()).magnitude(), 1e-5)


if __name__ == '__main__':unittest.main()
