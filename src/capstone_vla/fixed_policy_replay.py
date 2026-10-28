"""Replay recorded VLA commands to isolate the low-level controller; no new inference."""
from pathlib import Path
import json,os
import h5py,numpy as np
from scipy.spatial.transform import Rotation
from .task2_bridge import SceneController
from .live_policy_protocol import binary_action

class RecordedPolicySceneController(SceneController):
 def __init__(self,robot,arm_indices,mode,stage,sim,publish=None):
  if mode!='cartesian':raise ValueError('fixed policy replay is Cartesian only')
  super().__init__(robot,arm_indices,mode)
  self.sim=sim;self.root=Path(os.environ['CAPSTONE_POLICY_RUN'])
  with h5py.File(os.environ['CAPSTONE_POLICY_COMMANDS'],'r') as f:self.commands=f['action'][:]
  if self.commands.ndim!=2 or self.commands.shape[1]!=7 or not np.isfinite(self.commands).all() or not np.all(np.abs(self.commands[:,6])==1):raise ValueError('invalid fixed absolute VLA commands')
  self.archive=h5py.File(self.root/'live_observations.hdf5','x');self.archive.attrs['mode']='fixed_policy_commands_controller_diagnostic';self.archive.attrs['new_model_inference']=False
  for name,width in [('state',6),('action',7)]:self.archive.create_dataset(name,shape=(0,width),maxshape=(None,width),dtype='f8')
  self.records=[]
 def set_policy_frame(self,k):
  if k!=self.command_index+1 or k>=len(self.commands):raise ValueError('fixed command index mismatch')
  raw=self.commands[k];action=binary_action(raw);state=self.adapter.state(np.asarray(self.robot.get_joint_positions())[self.arm])
  self.command_index+=1;self.source=np.asarray(self.robot.get_joint_positions(),dtype=np.float32).copy()
  p=action[:3];rot=Rotation.from_euler('xyz',action[3:6]);self.reference_twist=np.zeros(6)
  if self.previous_pose is not None:
   previous_p,previous_r=self.previous_pose;self.reference_twist=np.r_[p-previous_p,(rot*previous_r.inv()).as_rotvec()]*self.command_hz
  self.previous_pose=(p.copy(),rot);self.reference_joint_velocity=np.zeros(7);self.action=action;self.adapter.contract.decode(action)
  for name,value in [('state',state),('action',action)]:
   ds=self.archive[name];ds.resize(k+1,axis=0);ds[k]=value
  self.archive.flush();self.records.append({'frame':int(k),'simulation_time_s':float(self.sim.current_time),'mode':'fixed_policy_commands_no_new_inference'})
  return self.target()
 def finish(self):
  self.archive.close();(self.root/'policy_timing.json').write_text(json.dumps(self.records,indent=2,allow_nan=False)+'\n')
