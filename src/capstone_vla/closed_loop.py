"""Live simulator RGB/proprioception -> VLA -> existing Cartesian controller."""
from collections import deque
from pathlib import Path
import json,os,time
import h5py,numpy as np
from scipy.spatial.transform import Rotation
from .task2_bridge import SceneController
from .live_policy_protocol import CAMERAS,atomic_npz,validate_observation,history_pair,binary_action,wait_file,freeze_render_script_dispatch


def publish_gui_physics_state():
 """Publish existing live PhysX poses through the rollout's USD backend; no simulation step."""
 from omni.physx import get_physx_interface
 get_physx_interface().update_transformations(False,True,False,False)


def select_policy_camera_records(stage):
 """Use the same saved manual environment selection as formal training renders."""
 from camera.isaaclab_adapter import discover_authored_cameras
 from camera.manual_camera_render import require_manual_calibration,select_manual_environment
 require_manual_calibration(stage)
 canonical={record['name']:record for record in discover_authored_cameras(stage.GetPrimAtPath('/Root'))}
 selected=select_manual_environment(stage,canonical,required=True)
 missing=set(CAMERAS)-set(selected)
 if missing:raise RuntimeError('missing policy cameras: '+str(sorted(missing)))
 return [selected[name] for name in CAMERAS]

class LiveRGB:
 def __init__(self,stage,sim,robot,publish):
  import omni.replicator.core as rep
  import carb
  from camera.render_settings import apply_render_quality
  self.render_quality=apply_render_quality(carb.settings.get_settings(),'balanced')
  records=select_policy_camera_records(stage);self.annotations={};self.products=[];self.records=[];self.sim=sim
  for name in CAMERAS:
   matching=[v for v in records if v['name']==name]
   if len(matching)!=1:raise RuntimeError('expected exactly one authored camera '+name)
   record=matching[0];paths=record.get('stream_prim_paths') or {};path=paths.get(record.get('policy_stream','')) or record['prim_path']
   product=rep.create.render_product(path,(640,480));ann=rep.AnnotatorRegistry.get_annotator('rgb');ann.attach([product])
   self.products.append(product);self.annotations[name]=ann;self.records.append(record)
  if not callable(publish):raise RuntimeError("live sensor requires native PhysX transform publisher")
  self.publish=publish
  self.rep=rep;self.stage=stage;self.robot=robot
 def capture(self):
  from expert_data.replay_runtime import sync_replay_camera_bindings
  physics_view=self.robot._articulation_view._physics_view
  if physics_view is None:raise RuntimeError('live camera mount requires measured PhysX links')
  sync_replay_camera_bindings(self.stage,None,0,physics_view=physics_view)
  from omni.kit.scripting.scripts.script_manager import ScriptManager
  before=float(self.sim.current_time)
  with freeze_render_script_dispatch(ScriptManager.get_instance()):
   self.publish()
   boundary=getattr(self,'render_boundary',None)
   if boundary is not None:
    boundary(self.rep,self.render_quality)
   else:
    self.sim.render()
    self.rep.orchestrator.step(delta_time=0.0,pause_timeline=False,rt_subframes=max(1,int(self.render_quality.get('temporal_subframes') or 1)))
  after=float(self.sim.current_time)
  if abs(after-before)>1e-9:raise RuntimeError('camera rendering advanced physics time')
  from pxr import UsdGeom
  cache=UsdGeom.XformCache();mpu=float(UsdGeom.GetStageMetersPerUnit(self.stage));self.last_poses=[]
  for record in self.records:
   matrix=cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(record['prim_path']));q=matrix.ExtractRotationQuat()
   self.last_poses.append([*[float(x)*mpu for x in matrix.ExtractTranslation()],float(q.GetReal()),*[float(x) for x in q.GetImaginary()]])
  images={name:np.asarray(ann.get_data()).copy() for name,ann in self.annotations.items()}
  for name,arr in images.items():
   if arr.shape not in [(480,640,4),(480,640,3)] or arr.dtype!=np.uint8:raise RuntimeError('RGB not ready: '+name+' '+str(arr.shape))
   images[name]=arr[:,:,:3].copy()
   if images[name].std()<1.:raise RuntimeError('blank RGB: '+name)
  return images

class PolicySceneController(SceneController):
 def __init__(self,robot,arm_indices,mode,stage,sim,publish=None):
  if mode!='cartesian':raise ValueError('policy execution is Cartesian only')
  super().__init__(robot,arm_indices,mode)
  self.sim=sim;self.root=Path(os.environ['CAPSTONE_POLICY_RUN']);self.ipc=self.root/'ipc';self.prefix=int(os.environ['CAPSTONE_POLICY_PREFIX']);self.history=[];self.queue=deque();self.anchor=None
  self.camera=LiveRGB(stage,sim,robot,publish);self.records=[]
  self.archive=h5py.File(self.root/'live_observations.hdf5','x')
  self.archive.attrs['observation_timing']='pre-action, live simulation';self.archive.attrs['camera_records_json']=json.dumps(self.camera.records)
  self.archive.attrs['feedback_mode']='synchronous simulation pause during inference, not real-time'
  for name in CAMERAS:self.archive.create_dataset(name,shape=(0,480,640,3),maxshape=(None,480,640,3),dtype='u1',compression='gzip',compression_opts=1,chunks=(1,480,640,3))
  self.archive.create_dataset('camera_pose_world',shape=(0,2,7),maxshape=(None,2,7),dtype='f8')
  self.archive.attrs['camera_pose_order_json']=json.dumps(list(CAMERAS))
  self.archive.attrs['camera_selection']='formal-render manual environment and canonical robot cameras'
  for name,width in [('state',6),('action',7)]:self.archive.create_dataset(name,shape=(0,width),maxshape=(None,width),dtype='f8')
 def set_policy_frame(self,k):
  images=self.camera.capture();state=self.adapter.state(np.asarray(self.robot.get_joint_positions())[self.arm]);self.history.append({'state':state.copy(),'images':images})
  # Only a three-frame window is needed for training history offsets [-2,0].
  self.history=self.history[-3:];wait_s=0.;server_s=0.
  if not self.queue:
   past,current=history_pair(self.history);states=np.stack([past['state'],current['state']]);rgb={name:np.stack([past['images'][name],current['images'][name]]) for name in CAMERAS};validate_observation(states,rgb)
   request=self.ipc/f'request_{k:06d}.npz';response=self.ipc/f'response_{k:06d}.npz'
   before=float(self.sim.current_time);start=time.perf_counter();atomic_npz(request,frame=np.asarray(k,dtype=np.int64),states=states,**rgb);wait_file(response,self.ipc)
   wait_s=time.perf_counter()-start
   if abs(float(self.sim.current_time)-before)>1e-9:raise RuntimeError('simulation advanced while waiting for inference')
   with np.load(response,allow_pickle=False) as f:
    chunk=f['absolute'];server_s=float(f['inference_s']);returned=int(f['frame'])
   if returned!=k or chunk.shape!=(20,7) or not np.isfinite(chunk).all():raise RuntimeError('invalid policy chunk response')
   self.queue.extend(chunk[:self.prefix]);self.anchor=k
   print('LIVE_POLICY frame='+str(k)+' prefix='+str(self.prefix)+' wait_s='+format(wait_s,'.3f'),flush=True)
  raw=np.asarray(self.queue.popleft()).copy();action=binary_action(raw)
  self.command_index+=1;self.source=np.asarray(self.robot.get_joint_positions(),dtype=np.float32).copy()
  p=action[:3];rot=Rotation.from_euler('xyz',action[3:6])
  self.reference_twist=np.zeros(6)
  if self.previous_pose is not None:
   previous_p,previous_r=self.previous_pose;self.reference_twist=np.r_[p-previous_p,(rot*previous_r.inv()).as_rotvec()]*self.command_hz
  self.previous_pose=(p.copy(),rot);self.reference_joint_velocity=np.zeros(7);self.action=action
  self.adapter.contract.decode(action)
  for name in CAMERAS:
   ds=self.archive[name];ds.resize(k+1,axis=0);ds[k]=images[name]
  for name,value in [('state',state),('action',action)]:
   ds=self.archive[name];ds.resize(k+1,axis=0);ds[k]=value
  poses=np.asarray(self.camera.last_poses,dtype=float)
  if poses.shape!=(2,7) or not np.isfinite(poses).all():raise RuntimeError('invalid measured camera pose')
  self.archive['camera_pose_world'].resize(k+1,axis=0);self.archive['camera_pose_world'][k]=poses
  self.archive.flush();self.records.append({'frame':int(k),'anchor_frame':int(self.anchor),'raw_gripper':float(raw[6]),'wait_s':wait_s,'server_inference_s':server_s,'simulation_time_s':float(self.sim.current_time)})
  return self.target()
 def finish(self):
  self.archive.close();(self.root/'policy_timing.json').write_text(json.dumps(self.records,indent=2,allow_nan=False)+'\n')
