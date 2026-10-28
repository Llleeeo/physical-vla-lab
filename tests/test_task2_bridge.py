"""Control-bridge tests without loading Isaac or simulator ground truth."""
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from capstone_vla.task2_bridge import SceneController, patch_worker, bounded_dls, ConstraintRecovery
ROOT=Path(__file__).resolve().parents[1]

class Robot:
    dof_names=[f'panda_joint{i}' for i in range(1,8)]+['panda_finger_joint1','panda_finger_joint2']
    def __init__(self,q):self.q=np.asarray(q).copy()
    def get_joint_positions(self):return self.q.copy()

class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.cfg=ROOT/'configs/franka_action_interface.json'
        self.env=patch.dict(os.environ,{'CAPSTONE_FRANKA_CONFIG':str(self.cfg),'CAPSTONE_VELOCITY_REFERENCE':'zero','CAPSTONE_IK_SOLVER':'clipped','CAPSTONE_BOUNDED_FROM_COMMAND':'0'})
        self.env.start();self.addCleanup(self.env.stop)
        cfg=json.loads(self.cfg.read_text());self.q=np.r_[cfg['initial_arm_joints_rad'],.04,.04]

    def test_current_worker_patch_and_fail_closed(self):
        text=(ROOT/'data/derived/hf_pilot_render_repo/Scripts/expert_data/collect_worker.py').read_text()
        self.assertIn('pilot_velocity_target = pilot_ik_controller.velocity_target()',patch_worker(text))
        self.assertIn("sim.add_physics_callback('capstone_task2_servo', pilot_ik_controller.physics_step)",patch_worker(text))
        with self.assertRaises(RuntimeError):patch_worker(text.replace('pilot_trace.attrs["controller_ee_offset_m"] = 0.107','changed'))

    def test_held_cartesian_action_reacts_to_new_robot_measurement(self):
        robot=Robot(self.q);controller=SceneController(robot,np.arange(7),'cartesian')
        np.testing.assert_allclose(controller.set_source(self.q),self.q,atol=2e-7)
        held=controller.action.copy();robot.q[0]+=.02
        target=controller.target()
        np.testing.assert_array_equal(controller.action,held)
        self.assertGreater(np.linalg.norm(target[:7]-robot.q[:7]),.005)

    def test_causal_velocity_uses_only_present_and_previous_commands(self):
        with patch.dict(os.environ,{'CAPSTONE_VELOCITY_REFERENCE':'causal'}):
            c=SceneController(Robot(self.q),np.arange(7),'joint')
            c.set_source(self.q)
            np.testing.assert_array_equal(c.velocity_target(),np.zeros(9))
            new=self.q.copy();new[0]+=.01
            c.set_source(new)
            self.assertAlmostEqual(float(c.velocity_target()[0]),.6,places=5)
            np.testing.assert_array_equal(c.velocity_target()[7:],[0,0])
            new[0]+=1;c.set_source(new)
            self.assertAlmostEqual(float(c.velocity_target()[0]),2.175,places=5)

    def test_recovery_requires_error_and_constraint_and_has_hysteresis(self):
        gate=ConstraintRecovery()
        self.assertFalse(gate.update(.03,0,False))
        self.assertFalse(gate.update(.005,0,True))
        self.assertTrue(gate.update(.03,0,True))
        self.assertTrue(gate.update(.02,0,False))
        self.assertFalse(gate.update(.005,0,False))
        self.assertTrue(gate.update(0,np.deg2rad(6),True))
        with self.assertRaises(ValueError):gate.update(np.nan,0,True)

    def test_adaptive_decision_and_target_do_not_depend_on_command_index(self):
        with patch.dict(os.environ,{'CAPSTONE_IK_SOLVER':'adaptive'}):
            controllers=[]
            for index in (1,116,10000):
                c=SceneController(Robot(self.q),np.arange(7),'cartesian')
                c.set_source(self.q);c.command_index=index
                c.action[:3]+=[.1,0,0]
                controllers.append(c)
            targets=[c.target() for c in controllers]
            for c,target in zip(controllers,targets):
                self.assertTrue(c.recovery.active)
                np.testing.assert_array_equal(target,targets[0])
            with patch.dict(os.environ,{'CAPSTONE_BOUNDED_FROM_COMMAND':'116'}):
                with self.assertRaises(ValueError):SceneController(Robot(self.q),np.arange(7),'cartesian')

    def test_bounded_solver_preserves_objective_better_than_component_clip(self):
        # Redundant coupled task: clipping an unconstrained solution loses
        # achievable task motion that another unconstrained joint can supply.
        jac=np.array([[1.,2.]])
        goal=np.array([3.]);lo=np.array([-.1,-2]);hi=np.array([.1,2])
        raw=jac.T@np.linalg.solve(jac@jac.T+.01**2*np.eye(1),goal)
        clipped=np.clip(raw,lo,hi)
        solved=bounded_dls(jac,goal,.01,lo,hi)
        self.assertTrue(np.all((solved>=lo)&(solved<=hi)))
        self.assertLess(np.linalg.norm(jac@solved-goal),np.linalg.norm(jac@clipped-goal)/10)

    def test_recorded_velocity_is_explicit_diagnostic_only(self):
        with patch.dict(os.environ,{'CAPSTONE_VELOCITY_REFERENCE':'recorded-diagnostic'}):
            c=SceneController(Robot(self.q),np.arange(7),'joint')
            with self.assertRaises(ValueError):c.set_source(self.q)
            velocity=np.r_[np.full(7,.2),1,1]
            c.set_source(self.q,velocity)
            np.testing.assert_allclose(c.velocity_target()[:7],.2)
            np.testing.assert_array_equal(c.velocity_target()[7:],[0,0])
            c=SceneController(Robot(self.q),np.arange(7),'cartesian')
            with self.assertRaises(ValueError):c.set_source(self.q,velocity)

    def test_cartesian_velocity_maps_same_frame_twist(self):
        with patch.dict(os.environ,{'CAPSTONE_VELOCITY_REFERENCE':'causal'}):
            c=SceneController(Robot(self.q),np.arange(7),'cartesian')
            c.set_source(self.q)
            new=self.q.copy();new[1]+=.001;c.set_source(new)
            _,_,jac=c.adapter.pose_jacobian(self.q[:7])
            np.testing.assert_allclose(jac@c.velocity_target()[:7],c.reference_twist,atol=1e-4)

    def test_physics_callback_validates_each_tick(self):
        controller=SceneController(Robot(self.q),np.arange(7),'joint')
        with patch.object(controller,'servo') as servo:
            for _ in range(3):controller.physics_step(1/180)
            self.assertEqual(servo.call_count,3)
            self.assertEqual(len(controller.physics_dt_samples),3)
            with self.assertRaises(RuntimeError):controller.physics_step(1/60)
            self.assertIsNotNone(controller.callback_error)
            self.assertEqual(servo.call_count,3)

    def test_joint_baseline_and_binary_gripper(self):
        c=SceneController(Robot(self.q),np.arange(7),'joint')
        np.testing.assert_array_equal(c.set_source(self.q),self.q.astype(np.float32))
        bad=self.q.copy();bad[-1]=.02
        with self.assertRaises(ValueError):c.set_source(bad)

if __name__=='__main__':unittest.main()
