import unittest
import numpy as np
from capstone_vla.chunk_execution import execute_prefix
from capstone_vla.prediction_metrics import absolute_from_delta

class ChunkExecutionTests(unittest.TestCase):
 def test_prefix_resets_at_episode_boundary_and_keeps_tail(self):
  x=np.zeros((7,4,7));x[:,:,0]=np.arange(7)[:,None]*10+np.arange(4)
  stream,anchor,horizon=execute_prefix(x,[3,4],2)
  np.testing.assert_array_equal(stream[:,0],[0,1,20,30,31,50,51])
  np.testing.assert_array_equal(anchor,[0,0,2,3,3,5,5]);np.testing.assert_array_equal(horizon,[0,1,0,0,1,0,1])
 def test_chunk_uses_prediction_anchor_not_future_recorded_state(self):
  delta=np.zeros((3,3,7));delta[:,:,0]=[.01,.02,.03];delta[:,:,6]=-1
  state=np.zeros((3,6));state[:,0]=[.4,.8,1.2]
  absolute=absolute_from_delta(delta,state);stream,_,_=execute_prefix(absolute,[3],3)
  np.testing.assert_allclose(stream[:,0],[.41,.42,.43]);np.testing.assert_array_equal(stream[:,6],[-1,-1,-1])
 def test_invalid_prefix_rejected(self):
  with self.assertRaises(ValueError):execute_prefix(np.zeros((2,3,7)),[2],4)
