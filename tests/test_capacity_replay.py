import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_task2_capacity_replay import validate_final_stream

class CapacityReplayTests(unittest.TestCase):
    def fixture(self):
        chunks=np.zeros((2,20,7));chunks[0,:,0]=np.arange(20);chunks[1,:,0]=100+np.arange(20);chunks[:,:,6]=.1;chunks[1,0,6]=-.1
        stream=np.zeros((6,7));stream[:,0]=[0,1,2,3,100,101];stream[:,6]=[1,1,1,1,-1,1]
        return chunks,stream,np.array([0,4])
    def test_final_queue_exact_and_tail_truncated(self):
        c,s,a=self.fixture();result=validate_final_stream(c,s,a,6)
        np.testing.assert_array_equal(result,s);result[0,0]=123;self.assertEqual(s[0,0],0)
    def test_event_shift_cannot_pass_as_final_prediction(self):
        c,s,a=self.fixture();s[4,6]=1
        with self.assertRaises(AssertionError):validate_final_stream(c,s,a,6)
    def test_bad_anchors_or_nonfinite_values_rejected(self):
        c,s,a=self.fixture()
        with self.assertRaises(ValueError):validate_final_stream(c,s,[0,5],6)
        c[0,0,0]=np.nan
        with self.assertRaises(ValueError):validate_final_stream(c,s,a,6)

if __name__=='__main__':unittest.main()
