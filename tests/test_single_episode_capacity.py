import json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from fit_task2_single_episode_capacity import episode_window,restricted_indices,closure_runs,validate_plan

class CapacityTests(unittest.TestCase):
    def test_only_declared_train_episode_can_be_selected(self):
        rows=[{'episode':'traj_00009','action':np.zeros((5,7))},{'episode':'traj_00001','action':np.zeros((3,7))}]
        record={'train_episodes':['traj_00009','traj_00001'],'validation_episodes':['traj_00042']}
        offset,row=episode_window(rows,record,'traj_00001');self.assertEqual(offset,5)
        self.assertEqual(restricted_indices([0,2],3,offset),[5,7])
        with self.assertRaises(ValueError):episode_window(rows,record,'traj_00042')
        with self.assertRaises(ValueError):episode_window(rows+rows[-1:],record,'traj_00001')
        record['validation_episodes'].append('traj_00001')
        with self.assertRaises(ValueError):episode_window(rows,record,'traj_00001')

    def test_sampler_cannot_cross_episode_boundaries(self):
        for values in [[-1],[3],[0.5],[]]:
            with self.assertRaises(ValueError):restricted_indices(values,3,5)

    def test_gripper_transient_and_sustained_closure_are_both_reported(self):
        a=np.zeros((8,7));a[:,6]=[1,-1,1,1,-1,-1,-1,-1]
        result=closure_runs(a)
        self.assertEqual([(r['start_frame'],r['end_exclusive']) for r in result],[(1,2),(4,8)])
        self.assertAlmostEqual(result[0]['duration_s'],1/60)

    def test_plan_rejects_changed_scope_or_normalization(self):
        plan=json.loads((ROOT/'configs/task2_single_episode_capacity.json').read_text());validate_plan(plan,232)
        for key,value in [('episode','traj_00042'),('validation_evaluation',True),('test_evaluation',True),('simulation_rollout',True),('normalization','refit'),('steps',0),('probe_frames',[232])]:
            with self.assertRaises(ValueError):validate_plan({**plan,key:value},232)

if __name__=='__main__':unittest.main()
