import unittest
import numpy as np
from capstone_vla.prediction_metrics import absolute_from_delta,action_errors

class PredictionMetricTests(unittest.TestCase):
    def test_delta_is_added_once_and_gripper_remains_absolute(self):
        delta=np.zeros((1,2,7));delta[...,0]=.01;delta[...,6]=-1
        current=np.array([[.4,.1,.2,.3,.4,.5]])
        absolute=absolute_from_delta(delta,current)
        np.testing.assert_allclose(absolute[...,0],.41)
        np.testing.assert_allclose(absolute[...,6],-1)
        np.testing.assert_allclose(delta[...,0],.01)

    def test_wrapped_rotations_and_padded_steps_are_handled_physically(self):
        target=np.zeros((1,2,7));pred=target.copy()
        pred[0,0,0]=.01;pred[0,0,3]=2*np.pi
        pred[0,1,:3]=100
        metrics,_,angle,_=action_errors(pred,target,np.array([[False,True]]))
        self.assertAlmostEqual(metrics['all_valid_chunk_targets']['position_mean_mm'],10.)
        self.assertLess(angle[0,0],1e-10)
        self.assertEqual(metrics['all_valid_chunk_targets']['valid_targets'],1)

    def test_gripper_sign_errors_do_not_hide_in_pose_error(self):
        target=np.zeros((1,1,7));target[...,6]=1
        pred=target.copy();pred[...,6]=-.001
        metrics,_,_,_=action_errors(pred,target,np.zeros((1,1),dtype=bool))
        self.assertEqual(metrics['first_action']['position_mean_mm'],0.)
        self.assertEqual(metrics['first_action']['gripper_sign_accuracy'],0.)

    def test_nonfinite_prediction_rejected(self):
        with self.assertRaises(ValueError):absolute_from_delta(np.full((1,20,7),np.nan),np.zeros((1,6)))

if __name__=='__main__':unittest.main()
