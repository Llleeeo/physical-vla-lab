import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from capstone_vla.cohort_training import wrap_delta_euler,train_normalization,CyclingFrameSampler

class CohortTrainingTests(unittest.TestCase):
    def test_periodic_labels_keep_physical_rotation_position_and_gripper(self):
        state=np.array([[.1,.2,.3,np.deg2rad(359),.2,.3]],np.float32)
        goal=np.array([[.15,.3,.4,np.deg2rad(1),.25,.35,-1]],np.float32)
        delta=goal.copy();delta[:,:6]-=state
        wrapped=wrap_delta_euler(delta)
        np.testing.assert_array_equal(wrapped[:,:3],delta[:,:3]);np.testing.assert_array_equal(wrapped[:,6],goal[:,6])
        target=state[:,3:6]+wrapped[:,3:6]
        angle=(Rotation.from_euler('xyz',target).inv()*Rotation.from_euler('xyz',goal[:,3:6])).magnitude()
        self.assertLess(angle.max(),1e-6)
        self.assertLess(abs(wrapped[0,3]),.04)

    def test_stats_exclude_padded_targets_and_refuse_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'capstone_cohort.json').write_text(json.dumps({'split':'train','episode_order':['fixture']}))
            (root/'meta').mkdir();(root/'data').mkdir()
            (root/'meta/info.json').write_text(json.dumps({'total_episodes':1,'chunks_size':1000,'data_path':'data/episode_{episode_index:06d}.parquet'}))
            states=np.zeros((2,6),np.float32);states[1,0]=1
            actions=np.zeros((2,7),np.float32);actions[:,0]=[1,2];actions[:,6]=[1,-1]
            pq.write_table(pa.table({'observation.state':states.tolist(),'action':actions.tolist()}),root/'data/episode_000000.parquet')
            stats,audit=train_normalization(root)
            self.assertEqual(audit['valid_chunk_targets'],3)
            self.assertAlmostEqual(float(stats['action']['mean'][0]),4/3,places=6)
            self.assertAlmostEqual(float(stats['observation.state']['mean'][0]),.5)
            (root/'capstone_cohort.json').write_text(json.dumps({'split':'validation'}))
            with self.assertRaises(ValueError):train_normalization(root)

    def test_sampler_covers_each_frame_before_starting_next_pass(self):
        sampler=CyclingFrameSampler(5,0)
        values=sampler.draw(3)+sampler.draw(3)
        self.assertEqual(sorted(values[:5]),list(range(5)))
        same=CyclingFrameSampler(5,0)
        self.assertEqual(values,same.draw(6))

if __name__=='__main__':unittest.main()
