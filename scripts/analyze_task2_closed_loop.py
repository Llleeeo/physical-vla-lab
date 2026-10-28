#!/usr/bin/env python3
"""CPU evidence/visualization audit of a completed live-feedback development run."""
import argparse,json,sys
from pathlib import Path
import h5py,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import imageio.v2 as imageio
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from capstone_vla.franka import FrankaCartesianAdapter

def main():
 p=argparse.ArgumentParser();p.add_argument('run_dir',type=Path);a=p.parse_args();base=a.run_dir
 config=json.loads((base/'run_config.json').read_text());cfg=json.loads((ROOT/'configs/franka_action_interface.json').read_text());adapter=FrankaCartesianAdapter(cfg['urdf'])
 sources=json.loads((ROOT/'data/derived/task2_expansion_inventory_20261025/render_sources.json').read_text())['sources'];episode=Path(config['episode']);row=next(v for v in sources if v['episode']==episode.name)
 out=base/'analysis';out.mkdir(exist_ok=False)
 with h5py.File(base/'physics_trace.hdf5') as f,h5py.File(base/'live_observations.hdf5') as live,h5py.File(episode/'replay.hdf5') as source,h5py.File(row['expected_camera_archive']) as render:
  n=len(live['state']);q=source['obs/joint_pos'][:n,:7];expert_state=np.stack([adapter.state(v) for v in q]);state=live['state'][:];action=live['action'][:];expert_grip=source['actions'][:n,7]>.02
  left=f['contact_L'][:];right=f['contact_R'][:];bilateral=left&right;object_pos=f['object_pos'][:,0,:]
  t=np.arange(n)/60;initial_diff=float(np.linalg.norm(state[0,:3]-expert_state[0,:3])*1000);angle=float(np.rad2deg((Rotation.from_euler('xyz',state[0,3:]).inv()*Rotation.from_euler('xyz',expert_state[0,3:])).magnitude()))
  first_closed=lambda x:int(np.flatnonzero(~x)[0]) if (~x).any() else None
  result={'status':'CLOSED_LOOP_ROLLOUT_AUDIT_COMPLETED','frames':n,'servo_calls':int(f.attrs['servo_calls']),
   'bilateral_contact_frames':int(bilateral.sum()),'left_contact_frames':int(left.sum()),'right_contact_frames':int(right.sum()),
   'left_impulse_max':float(f['impulse_L'][:].max()),'right_impulse_max':float(f['impulse_R'][:].max()),
   'initial_state_position_difference_mm':initial_diff,'initial_state_orientation_difference_deg':angle,
   'expert_first_close_frame':first_closed(expert_grip),'policy_first_close_frame':first_closed(action[:,6]>0),
   'policy_switch_frames':(np.flatnonzero(np.diff(action[:,6]>0))+1).tolist(),
   'object_start_position_world_m':object_pos[0].tolist(),'object_end_position_world_m':object_pos[-1].tolist(),
   'source_initial_object_difference_mm':float(np.linalg.norm(object_pos[0]-source['obs/object_pos'][0,0])*1000),
   'controller_goal_error_mean_mm':float(np.mean(f['servo_diagnostics/goal_position_error_m'][:])*1000),
   'controller_goal_error_p95_mm':float(np.percentile(f['servo_diagnostics/goal_position_error_m'][:],95))*1000,
   'grasp_assistance_active':bool(np.asarray(f['grasp_assistance_audit']).any()),
   'interpretation':'Control integration completed, no measured finger-target contact and no evidence of grasp/lift in this rollout; closure timing differs from the source demonstration, its cause is not established.',
   'official_success_verified':False,'formal_failure_verdict':None,'real_time_evaluated':False,'reserved_test_used':False}
  fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
  axes[0].step(t,expert_grip.astype(int),where='post',label='expert');axes[0].step(t,(action[:,6]>0).astype(int),where='post',label='live policy',alpha=.8);axes[0].set(yticks=[0,1],yticklabels=['closed','open'],ylabel='gripper');axes[0].legend()
  axes[1].plot(t,np.linalg.norm(action[:,:3]-state[:,:3],axis=1)*1000,label='live target vs measured');axes[1].plot(t,np.linalg.norm(expert_state[:,:3]-state[:,:3],axis=1)*1000,label='deviation from expert state');axes[1].set_ylabel('mm');axes[1].legend()
  axes[2].plot(t,object_pos[:,2],label='live object z');axes[2].plot(t,source['obs/object_pos'][:n,0,2],label='expert object z');axes[2].set(xlabel='simulation time (s)',ylabel='world z (m)');axes[2].legend();fig.tight_layout();fig.savefig(out/'rollout_diagnostics.png',dpi=140);plt.close(fig)
  def tile(camera,index):
   a=np.asarray(render[camera+'_rgb'][index]);b=np.asarray(live[camera][index]);return np.concatenate([a,b],axis=1)
  def composite(index):
   canvas=Image.fromarray(np.concatenate([tile('opst_cam',index),tile('wrist_cam',index)],axis=0));draw=ImageDraw.Draw(canvas)
   for pos,text in [((8,8),'EXPERT render'),((648,8),'LIVE VLA rollout'),((8,488),'EXPERT wrist'),((648,488),'LIVE wrist'),((8,940),f'frame {index}  simulation {index/60:.3f}s')]:
    draw.rectangle([pos[0]-2,pos[1]-2,pos[0]+310,pos[1]+16],fill='black');draw.text(pos,text,fill='white')
   return np.asarray(canvas)
  with imageio.get_writer(out/'expert_vs_live_policy.mp4',fps=60,codec='libx264',pixelformat='yuv420p',quality=7,macro_block_size=1) as writer:
   for k in range(n):writer.append_data(composite(k))
  for k in [0,19,116,n-1]:
   if k<n:Image.fromarray(composite(k)).save(out/f'comparison_frame_{k:04d}.jpg',quality=90)
  result['video']=str(out/'expert_vs_live_policy.mp4');result['figure']=str(out/'rollout_diagnostics.png')
 (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
