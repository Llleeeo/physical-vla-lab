import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('capstone_sampling',ROOT/'scripts/diagnose_task2_sampling.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class SamplingTests(unittest.TestCase):
    def test_empirical_position_bias_and_spread_decompose_squared_error(self):
        pred=np.zeros((2,7));pred[:,0]=[.01,.03];pred[:,6]=[-1,1]
        target=np.zeros(7);target[6]=-1
        stats=m.noise_summary(pred,target)
        self.assertAlmostEqual(stats['position_mse_mm2'],500.)
        self.assertAlmostEqual(stats['position_mean_bias_mm'],20.)
        self.assertAlmostEqual(stats['position_sample_spread_rms_mm'],10.)
        self.assertEqual(stats['gripper_open_fraction'],.5)
        self.assertEqual(stats['gripper_sign_accuracy'],.5)

    def test_residual_changes_do_not_count_expert_motion_as_prediction_noise(self):
        target=np.zeros((4,7));target[:,0]=[0,.1,.2,.3]
        predicted=target.copy();predicted[:,0]+=.05
        self.assertAlmostEqual(m.residual_changes(predicted,target)['residual_step_mean_mm'],0.,places=10)

if __name__=='__main__':unittest.main()
