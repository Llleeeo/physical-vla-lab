import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'data/derived/hgwind_current_baseline_20261027/repo/Scripts')]
from expert_data.skills import catch
from capstone_vla.hgwind_control import (ContactDwell,SmoothLiftConfig,install_profile,
                                        smooth_reference,_new_stats,profile_summary,profile_configuration)


class ControlTests(unittest.TestCase):
    def test_dwell_requires_consecutive_eligible_ticks(self):
        d=ContactDwell()
        self.assertAlmostEqual(d.observe(0,True,.02),.02)
        self.assertAlmostEqual(d.observe(0,True,.02),.02)
        self.assertEqual(d.observe(1,False,.02),0)
        self.assertAlmostEqual(d.observe(2,True,.02),.02)
        self.assertAlmostEqual(d.observe(4,True,.02),.02)
        self.assertAlmostEqual(d.observe(5,True,.02),.04)

    def test_invalid_dwell_dt(self):
        for dt in (0,-.1,float('nan')):
            with self.assertRaises(ValueError):ContactDwell().observe(0,True,dt)

    def skill(self):
        return SimpleNamespace(_capstone_lift_config=SmoothLiftConfig(),
            _capstone_lift_stats=_new_stats(),
            _capstone_reference_step=catch._jerk_limited_cartesian_target_step,
            _carry_position=np.zeros(3),_carry_velocity=np.array([.14,0.,0.]),
            _carry_acceleration=np.zeros(3),param=lambda k,d=None:d,
            _environment_safe_translation=lambda p,q,v,**kw:v)

    def test_reference_stays_bounded_under_velocity_reversals(self):
        hand=self.skill();dt=1/60;position=np.zeros(3)
        for tick in range(360):
            desired=np.array([.23 if tick<120 else -.23,0.,.05 if tick<240 else 0.])
            old_v=hand._carry_velocity.copy();old_a=hand._carry_acceleration.copy()
            goal=hand._carry_position+desired*dt
            position,velocity=smooth_reference(hand,goal,desired,position,[1,0,0,0],dt)
            self.assertLessEqual(np.linalg.norm(velocity-old_v)/dt,.5+1e-10)
            self.assertLessEqual(np.linalg.norm(hand._carry_acceleration-old_a)/dt,5+1e-10)
        self.assertLess(np.linalg.norm(hand._carry_velocity-np.array([-.23,0,0])),.01)

    def test_collision_projection_remains_authoritative(self):
        hand=self.skill();hand._environment_safe_translation=lambda *a,**kw:np.zeros(3)
        _,velocity=smooth_reference(hand,np.array([.01,0,0]),np.array([.2,0,0]),
                                    np.zeros(3),[1,0,0,0],1/60)
        np.testing.assert_array_equal(velocity,np.zeros(3))
        self.assertEqual(hand._capstone_lift_stats['collision_projection_overrides'],1)

    def test_reference_position_lead_is_capped(self):
        hand=self.skill();hand._carry_position=np.array([1.,0,0])
        position,_=smooth_reference(hand,np.array([2.,0,0]),np.array([.2,0,0]),
                                    np.zeros(3),[1,0,0,0],1/60)
        self.assertLessEqual(np.linalg.norm(position),.08+1e-12)

    def make_installed_probe(self,config=None):
        class Probe:
            _drive_carry=catch.Catch._drive_carry
            def __init__(self):
                self.params={'lift_speed':.12,'r_hold':.02};self.reset()
            def reset(self):self.lifts=0
            def param(self,name,default=None):return self.params.get(name,default)
            def _begin_lift_from_physical_grasp(self,**kw):self.lifts+=1
        install_profile(SimpleNamespace(Catch=Probe,
            _jerk_limited_cartesian_target_step=catch._jerk_limited_cartesian_target_step),config)
        h=Probe();h._uses_runtime_attachment=lambda:False
        h._physical_grasp_confirmed=h._jaws_closed=h._contact_pair_now=h._finger_segment_solid_ready=True
        h._capture_relative_speed_mps=.02;h._capture_speed_limit_mps=lambda:.087
        h._grasp_center_world=np.zeros(3);h._pinch_band=lambda:.02
        h._dt=1/60;return h

    def advance(self,h,tick):
        h.step_count=tick
        h._begin_lift_from_physical_grasp(p_A=np.zeros(3),p_ee=np.zeros(3),
                                         q_ee=[1,0,0,0],v_ee=np.zeros(3))

    def test_real_transition_waits_and_preserves_success_tolerance(self):
        h=self.make_installed_probe()
        for tick in range(4):self.advance(h,tick);self.assertEqual(h.lifts,0)
        self.advance(h,4);self.assertEqual(h.lifts,1)
        self.assertEqual(h.params['r_hold'],.02)
        self.assertAlmostEqual(h.params['lift_speed'],.06)

    def test_lost_contact_and_skipped_ticks_restart_settle(self):
        h=self.make_installed_probe()
        for tick in range(3):self.advance(h,tick)
        h._contact_pair_now=False;self.advance(h,3)
        h._contact_pair_now=True
        self.advance(h,4);self.advance(h,7)
        self.assertEqual(h.lifts,0)
        self.assertAlmostEqual(h._capstone_contact_dwell.elapsed_s,1/60)

    def test_high_relative_speed_blocks_lift(self):
        h=self.make_installed_probe();h._capture_relative_speed_mps=.4
        for tick in range(20):self.advance(h,tick)
        self.assertEqual(h.lifts,0)

    def test_off_center_capture_is_not_admitted(self):
        h=self.make_installed_probe();h._grasp_center_world=np.array([.03,0,0])
        for tick in range(20):self.advance(h,tick)
        self.assertEqual(h.lifts,0)

    def test_actual_carry_method_passes_filtered_velocity_to_controller(self):
        h=self.make_installed_probe();h.phase=catch.Phase.LIFT
        h._logical_fix_joint_attached=False;h._exact_segment_close_ready=lambda *a:True
        h._up=np.array([0.,0.,1.]);h._z=lambda p:float(p[2]);h._z_ref=0.
        h._norm=lambda value:float(np.linalg.norm(value))
        h._carry_position=np.zeros(3);h._carry_velocity=np.array([.14,0.,0.])
        h._carry_acceleration=np.zeros(3);h._lift_target=np.zeros(3)
        h._carry_quaternion=np.array([1.,0.,0.,0.])
        h._environment_safe_translation=lambda p,q,v,**kw:v
        h._gripper_axis_world=lambda *a:np.array([0.,0.,1.])
        h._closed_gripper_action=lambda:'grip';h.merge_actions=lambda a,b:(a,b)
        calls=[]
        def action(p,q,**kw):calls.append(kw);return 'arm'
        h.expert=SimpleNamespace(operational_space_pinch_action=action)
        result=h._drive_carry(None,1/60,np.zeros(3),[1,0,0,0],np.array([.23,0,0]),
                              np.zeros(3),np.zeros(3),[1,0,0,0],np.array([.14,0,0]))
        self.assertEqual(result,('arm','grip'))
        velocity=calls[0]['target_linear_velocity']
        self.assertLess(abs(velocity[2]),.06)
        self.assertEqual(h._capstone_lift_stats['reference_updates'],1)
        self.assertFalse(calls[0]['continuous_fusion'])

    def test_smooth_only_preserves_original_transition_without_new_dwell(self):
        h=self.make_installed_probe(profile_configuration('smooth-only'))
        self.advance(h,0)
        self.assertEqual(h.lifts,1)
        self.assertEqual(h._capstone_lift_stats['settle_deferred_calls'],0)
        self.assertAlmostEqual(h.params['lift_speed'],.06)
        self.assertEqual(h.params['r_hold'],.02)

    def test_smooth_only_keeps_upstream_physical_capture_admission(self):
        physical = dict(jaws_closed=True, bilateral_contact=True,
                        capture_speed_safe=True, finger_drives_blocked=True,
                        runtime_attachment=False)
        for key in ('jaws_closed', 'bilateral_contact', 'finger_drives_blocked'):
            altered = dict(physical, **{key: False})
            self.assertFalse(catch._capture_contact_transition_ready(**altered))
        self.assertTrue(catch._capture_contact_transition_ready(**physical))
        transition = dict(physical_grasp_confirmed=False, runtime_attachment=False,
                          logical_attachment_active=False, handoff_after_capture=False,
                          require_lift=True)
        self.assertEqual(catch._capture_phase_transition(**transition), 'none')
        self.assertEqual(catch._capture_phase_transition(**dict(transition, physical_grasp_confirmed=True)), 'lift')

    def test_only_added_dwell_differs_between_candidate_and_old_profile(self):
        from dataclasses import asdict
        old = asdict(profile_configuration('smooth-lift'))
        new = asdict(profile_configuration('smooth-only'))
        self.assertEqual({key for key in old if old[key] != new[key]}, {'name', 'settle_s'})

    def test_smooth_only_metadata_is_distinct(self):
        h=self.make_installed_probe(profile_configuration('smooth-only'))
        report=profile_summary(SimpleNamespace(active_primitives=lambda:[h]))
        self.assertEqual(report['profile'],'smooth_only_v1')
        self.assertEqual(report['primitives'][0]['configuration']['settle_s'],0.)

    def test_unknown_profile_rejected(self):
        with self.assertRaises(ValueError):profile_configuration('unknown')

    def test_runtime_metadata_contains_applied_configuration(self):
        h=self.make_installed_probe()
        report=profile_summary(SimpleNamespace(active_primitives=lambda:[h]))
        self.assertEqual(report['primitives'][0]['applied_lift_speed_mps'],.06)
        self.assertFalse(report['success_thresholds_changed'])


if __name__=='__main__':unittest.main()
