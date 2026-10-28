import os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np,h5py
from test_live_policy import ROOT,Robot,Camera,Sim
from capstone_vla.franka import FrankaCartesianAdapter
from capstone_vla.closed_loop import PolicySceneController
from capstone_vla.fixed_policy_replay import RecordedPolicySceneController
from capstone_vla.live_policy_protocol import atomic_npz
import json

class FixedPolicyTests(unittest.TestCase):
 def test_recorded_commands_have_identical_live_control_semantics_and_no_requests(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);live=root/'live';fixed=root/'fixed';live.mkdir();fixed.mkdir();(live/'ipc').mkdir()
   cfg=ROOT/'configs/franka_action_interface.json';adapter=FrankaCartesianAdapter(json.loads(cfg.read_text())['urdf'])
   pose=adapter.state(Robot().q[:7]);commands=np.tile(np.r_[pose,1.],(8,1));commands[:,0]+=np.arange(8)*.001;commands[3:,6]=-1
   path=root/'commands.hdf5'
   with h5py.File(path,'w') as f:f.create_dataset('action',data=commands)
   env={'CAPSTONE_FRANKA_CONFIG':str(cfg),'CAPSTONE_IK_SOLVER':'adaptive','CAPSTONE_BOUNDED_FROM_COMMAND':'0','CAPSTONE_VELOCITY_REFERENCE':'causal','CAPSTONE_POLICY_PREFIX':'4','CAPSTONE_POLICY_COMMANDS':str(path)}
   with patch.dict(os.environ,{**env,'CAPSTONE_POLICY_RUN':str(live)}),patch('capstone_vla.closed_loop.LiveRGB',Camera):
    a=PolicySceneController(Robot(),np.arange(7),'cartesian',None,Sim())
   with patch.dict(os.environ,{**env,'CAPSTONE_POLICY_RUN':str(fixed)}),patch('capstone_vla.closed_loop.LiveRGB',side_effect=AssertionError('camera must not be created')):
    b=RecordedPolicySceneController(Robot(),np.arange(7),'cartesian',None,Sim())
   for frame in [0,4]:
    chunk=np.tile(commands[-1],(20,1));chunk[:4]=commands[frame:frame+4];atomic_npz(live/'ipc'/f'response_{frame:06d}.npz',frame=np.array(frame),absolute=chunk,inference_s=np.array(0.))
   try:
    for k in range(8):
     np.testing.assert_allclose(a.set_policy_frame(k),b.set_policy_frame(k),atol=0,rtol=0)
     np.testing.assert_array_equal(a.action,b.action);np.testing.assert_allclose(a.reference_twist,b.reference_twist,atol=0,rtol=0);np.testing.assert_allclose(a.velocity_target(),b.velocity_target(),atol=0,rtol=0)
    b.velocity_reference='zero';np.testing.assert_array_equal(b.velocity_target(),np.zeros(9))
    self.assertFalse((fixed/'ipc').exists())
    np.testing.assert_array_equal(b.archive['action'][:],commands)
    with self.assertRaises(ValueError):b.set_policy_frame(9)
   finally:a.finish();b.finish()
 def test_nonbinary_gripper_is_rejected_before_execution(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'commands.hdf5'
   with h5py.File(p,'w') as f:f.create_dataset('action',data=np.zeros((3,7)))
   with patch.dict(os.environ,{'CAPSTONE_FRANKA_CONFIG':str(ROOT/'configs/franka_action_interface.json'),'CAPSTONE_POLICY_RUN':tmp,'CAPSTONE_POLICY_COMMANDS':str(p),'CAPSTONE_IK_SOLVER':'clipped','CAPSTONE_VELOCITY_REFERENCE':'zero','CAPSTONE_BOUNDED_FROM_COMMAND':'0'}):
    with self.assertRaisesRegex(ValueError,'invalid fixed'):RecordedPolicySceneController(Robot(),np.arange(7),'cartesian',None,Sim())
