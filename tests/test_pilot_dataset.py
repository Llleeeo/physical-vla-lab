import unittest
import numpy as np
from capstone_vla.pilot_dataset import action_chunk,temporal_indices

class PilotDatasetTests(unittest.TestCase):
    def test_history_is_causal_and_first_frame_padding_explicit(self):
        obs,pad,act,apad=temporal_indices(0,5,chunk_size=3)
        np.testing.assert_array_equal(obs,[0,0])
        np.testing.assert_array_equal(pad,[True,False])
        np.testing.assert_array_equal(act,[0,1,2])
        self.assertFalse(apad.any())
        for i in range(5):self.assertTrue(np.all(temporal_indices(i,5)[0]<=i))

    def test_last_action_is_padded_and_masked(self):
        _,_,act,pad=temporal_indices(4,5,chunk_size=3)
        np.testing.assert_array_equal(act,[4,4,4])
        np.testing.assert_array_equal(pad,[False,True,True])

    def test_delta_uses_current_state_for_every_target(self):
        states=np.arange(30,dtype=np.float32).reshape(5,6)
        actions=np.column_stack([states+1,np.array([1,-1,1,-1,-1])])
        actual,_=action_chunk(actions,states,1,3)
        np.testing.assert_array_equal(actual[:,:6],actions[1:4,:6]-states[1])
        np.testing.assert_array_equal(actual[:,6],[-1,1,-1])

    def test_same_index_command_is_not_shifted(self):
        actions=np.arange(35,dtype=np.float32).reshape(5,7)
        states=np.zeros((5,6),dtype=np.float32)
        chunk,_=action_chunk(actions,states,2,2,delta=False)
        np.testing.assert_array_equal(chunk,actions[2:4])

    def test_future_observations_and_invalid_indices_rejected(self):
        for frame in [-1,5]:
            with self.assertRaises(IndexError):temporal_indices(frame,5)
        with self.assertRaises(ValueError):temporal_indices(0,5,history=(0,1))
        with self.assertRaises(ValueError):temporal_indices(0,5,chunk_size=0)

if __name__=='__main__':unittest.main()
