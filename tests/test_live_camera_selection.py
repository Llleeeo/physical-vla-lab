import json,sys,unittest
from pathlib import Path
import numpy as np,h5py
from pxr import Usd,UsdGeom
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'data/derived/hf_pilot_render_repo/Scripts'))
from capstone_vla.closed_loop import select_policy_camera_records

class LiveCameraSelectionTests(unittest.TestCase):
 def test_manual_environment_matches_actual_training_camera_and_wrist_stays_canonical(self):
  stage=Usd.Stage.Open(str(ROOT/'data/derived/hf_pilot_render_repo/Tasks/1s1p/grasp_sliding_motion_on_flat_surface_1s1p_g9wvkk1.usd'),load=Usd.Stage.LoadNone)
  records={r['name']:r for r in select_policy_camera_records(stage)}
  self.assertIn('/ManualEnvironmentCameras/',records['opst_cam']['prim_path'])
  self.assertIn('/RobotCameras/',records['wrist_cam']['prim_path'])
  source=next(v for v in json.loads((ROOT/'data/derived/task2_expansion_inventory_20261025/render_sources.json').read_text())['sources'] if v['episode']=='traj_00042')
  cache=UsdGeom.XformCache()
  with h5py.File(source['expected_camera_archive']) as f:
   names=json.loads(f.attrs['camera_names_json'])
   for name,record in records.items():
    position=np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(record['prim_path'])).ExtractTranslation())
    np.testing.assert_allclose(position,f['rgb_camera_position_world'][0,names.index(name)],atol=2e-5,rtol=0)
 def test_missing_manual_calibration_is_rejected(self):
  stage=Usd.Stage.CreateInMemory();stage.DefinePrim('/Root')
  with self.assertRaises(ValueError):select_policy_camera_records(stage)
