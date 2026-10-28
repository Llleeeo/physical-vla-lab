import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from diagnose_task2_train_teacher import queue_stream,prediction_difference

class TeacherDiagnosticTests(unittest.TestCase):
    def test_queue_uses_prediction_anchor_once_and_truncates_end(self):
        chunks=np.zeros((2,20,7));chunks[0,:,0]=np.arange(20);chunks[1,:,0]=100+np.arange(20)
        chunks[:,:,6]=.5;chunks[1,0,6]=-.2
        out=queue_stream(chunks,[0,4],6)
        np.testing.assert_array_equal(out[:,0],[0,1,2,3,100,101])
        np.testing.assert_array_equal(out[:,6],[1,1,1,1,-1,1])
    def test_missing_or_shifted_anchors_rejected(self):
        with self.assertRaises(ValueError):queue_stream(np.zeros((2,20,7)),[0,5],6)
        with self.assertRaises(ValueError):queue_stream(np.zeros((1,20,7)),[0],6)
    def test_difference_reports_units_and_periodic_orientation(self):
        a=np.zeros((20,7));b=a.copy();b[:,0]=.001;b[:,3]=2*np.pi;b[:,6]=1
        result=prediction_difference(a,b)
        self.assertAlmostEqual(result['max_position_difference_mm'],1)
        self.assertLess(result['max_orientation_difference_deg'],1e-10)
        self.assertEqual(result['gripper_sign_disagreements'],20)

if __name__=='__main__':unittest.main()
