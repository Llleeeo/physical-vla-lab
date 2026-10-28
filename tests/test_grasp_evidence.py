import unittest
import numpy as np
from capstone_vla.grasp_evidence import evaluate_grasp, GraspEvidenceCriteria

class GraspEvidenceTests(unittest.TestCase):
    def sample(self):
        n=120;t=np.arange(n)/60
        obj=np.zeros((n,1,3));obj[:,0,0]=.3-np.minimum(np.arange(n),30)*.001
        obj[:,0,2]=.02+np.clip((np.arange(n)-30)/60,0,1)*.2
        both=np.arange(n)>=30
        vel=np.zeros_like(obj);vel[:30,0,0]=-.06
        quat=np.zeros((n,4));quat[:,0]=1
        return dict(timestamp_s=t,object_pos=obj,object_vel=vel,ee_pos=obj[:,0]-[0,0,.05],ee_quat=quat,
                    contact_L=both.copy(),contact_R=both.copy(),impulse_L=both*.1,impulse_R=both*.1,
                    grasp_assistance_audit=np.zeros((n,3)))
    def test_contact_lift_retention_and_motion_pass(self):
        self.assertEqual(evaluate_grasp(self.sample())['status'],'pass')
    def test_rotating_stationary_target_uses_angular_motion_and_relative_arrest(self):
        data=self.sample();data['object_vel'][:]=0
        self.assertFalse(evaluate_grasp(data)['checks']['object_moving_before_first_contact'])
        data['object_ang_vel']=np.zeros((120,1,3));data['object_ang_vel'][:30,0,2]=5
        data['object_quat']=np.zeros((120,1,4));data['object_quat'][:,0,0]=1
        criteria=GraspEvidenceCriteria(motion_requirement='angular',require_lift=False)
        self.assertEqual(evaluate_grasp(data,criteria)['status'],'pass')
        data['object_ang_vel'][-30:,0,2]=2
        self.assertFalse(evaluate_grasp(data,criteria)['checks']['final_rotation_arrest_relative_to_hand'])
        del data['object_ang_vel']
        with self.assertRaises(KeyError):evaluate_grasp(data,criteria)

    def test_missing_assistance_evidence_is_undetermined(self):
        data=self.sample();del data['grasp_assistance_audit']
        self.assertEqual(evaluate_grasp(data)['status'],'undetermined')
    def test_contact_only_without_lift_fails(self):
        data=self.sample();data['object_pos'][:,0,2]=.02
        result=evaluate_grasp(data);self.assertFalse(result['checks']['sustained_lift'])
    def test_relative_slip_and_assistance_fail(self):
        data=self.sample();data['ee_pos'][-15:,0]+=.02
        self.assertFalse(evaluate_grasp(data)['checks']['relative_position_retention'])
        data=self.sample();data['grasp_assistance_audit'][60,1]=1
        self.assertFalse(evaluate_grasp(data)['checks']['no_grasp_assistance'])
    def test_lost_final_contact_and_bad_timing_rejected(self):
        data=self.sample();data['contact_R'][-1]=False
        self.assertFalse(evaluate_grasp(data)['checks']['final_bilateral_hold'])
        data=self.sample();data['timestamp_s'][50]=data['timestamp_s'][49]
        with self.assertRaises(ValueError):evaluate_grasp(data)

if __name__=='__main__':unittest.main()
