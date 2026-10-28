import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from capstone_vla.live_policy_protocol import atomic_npz,binary_action,history_pair,validate_observation
from capstone_vla.closed_loop import PolicySceneController
from capstone_vla.closed_loop_patch import patch_policy_worker
ROOT=Path(__file__).resolve().parents[1]

class Robot:
 dof_names=[f'panda_joint{i}' for i in range(1,8)]+['panda_finger_joint1','panda_finger_joint2']
 def __init__(self):self.q=np.array([0,-.6,0,-2.,0,1.5,.7,.04,.04])
 def get_joint_positions(self):return self.q.copy()
class Camera:
 records=[]
 last_poses=[[0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.]]
 def __init__(self,*args):pass
 def capture(self):return {name:np.full((480,640,3),127,dtype=np.uint8) for name in ['wrist_cam','opst_cam']}
class Sim:current_time=0.

class LivePolicyTests(unittest.TestCase):
 def test_history_and_action_validation(self):
  self.assertEqual(history_pair([0]),(0,0));self.assertEqual(history_pair([0,1]),(0,1));self.assertEqual(history_pair([0,1,2,3]),(1,3))
  a=np.array([.4,0,.3,0,0,0,-.01]);np.testing.assert_array_equal(binary_action(a)[:-1],a[:-1]);self.assertEqual(binary_action(a)[-1],-1.)
  with self.assertRaises(ValueError):binary_action(np.full(7,np.nan))
  with self.assertRaises(ValueError):validate_observation(np.zeros((2,6)),{})
 def test_live_feedback_history_and_prefix_queue_without_expert_actions(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);ipc=root/'ipc';ipc.mkdir()
   env={'CAPSTONE_FRANKA_CONFIG':str(ROOT/'configs/franka_action_interface.json'),'CAPSTONE_POLICY_RUN':tmp,'CAPSTONE_POLICY_PREFIX':'4','CAPSTONE_IK_SOLVER':'adaptive','CAPSTONE_VELOCITY_REFERENCE':'causal','CAPSTONE_BOUNDED_FROM_COMMAND':'0'}
   with patch.dict(os.environ,env),patch('capstone_vla.closed_loop.LiveRGB',Camera):
    robot=Robot();controller=PolicySceneController(robot,np.arange(7),'cartesian',None,Sim())
    initial=controller.adapter.state(robot.q[:7]);chunk=np.tile(np.r_[initial,-.7],(20,1));chunk[:,0]+=np.arange(20)*.001
    atomic_npz(ipc/'response_000000.npz',frame=np.array(0),absolute=chunk,inference_s=np.array(.1))
    atomic_npz(ipc/'response_000004.npz',frame=np.array(4),absolute=chunk,inference_s=np.array(.1))
    recorded=[]
    try:
     for k in range(5):
      robot.q[0]=k*.001;recorded.append(controller.adapter.state(robot.q[:7]));controller.set_policy_frame(k)
     self.assertEqual([r['anchor_frame'] for r in controller.records],[0,0,0,0,4])
     self.assertEqual(len(list(ipc.glob('request_*.npz'))),2)
     with np.load(ipc/'request_000004.npz') as f:
      self.assertEqual(set(f.files),{'frame','states','wrist_cam','opst_cam'});np.testing.assert_allclose(f['states'],[recorded[2],recorded[4]])
     self.assertEqual(controller.action[6],-1.)
    finally:controller.finish()
 def test_live_publisher_requests_native_usd_writeback_without_a_simulation_step(self):
  from types import SimpleNamespace
  from unittest.mock import Mock
  from capstone_vla.closed_loop import publish_gui_physics_state
  import sys
  interface=SimpleNamespace(update_transformations=Mock())
  with patch.dict(sys.modules,{'omni.physx':SimpleNamespace(get_physx_interface=lambda:interface)}):publish_gui_physics_state()
  interface.update_transformations.assert_called_once_with(False,True,False,False)
 def test_real_worker_patch_is_guarded_and_compiles(self):
  source=(ROOT/'data/derived/hf_pilot_render_repo/Scripts/expert_data/collect_worker.py').read_text();patched=patch_policy_worker(source)
  self.assertIn('tgt = pilot_ik_controller.set_policy_frame(k)',patched)
  self.assertNotIn('tgt = np.array(replay_actions[k], dtype=np.float32, copy=True)',patched)
  with self.assertRaises(RuntimeError):patch_policy_worker(source.replace('tgt = np.array(replay_actions[k], dtype=np.float32, copy=True)','changed_incompatible_worker'))
