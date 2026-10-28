#!/usr/bin/env python3
"""Separate learned-target discontinuity from motor tracking diagnostics, CPU only."""
import argparse,json,sys
from pathlib import Path
import numpy as np,h5py
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from capstone_vla.franka import FrankaCartesianAdapter

def main():
 p=argparse.ArgumentParser();p.add_argument('run_dir',type=Path);a=p.parse_args();base=a.run_dir;cfg=json.loads((ROOT/'configs/franka_action_interface.json').read_text());config=json.loads((base/'run_config.json').read_text());adapter=FrankaCartesianAdapter(cfg['urdf'])
 out=base/'command_jitter_analysis';out.mkdir(exist_ok=False);result={'scope':'command discontinuity diagnostics, not proof of controller instability','prefix':config['prefix']}
 with h5py.File(base/'live_observations.hdf5') as obs,h5py.File(base/'physics_trace.hdf5') as f,h5py.File(Path(config['episode'])/'replay.hdf5') as source:
  action=obs['action'][:];n=len(action);expert=np.stack([adapter.state(v[:7]) for v in source['actions'][:n]]);series={}
  for name,pose in [('policy',action[:,:6]),('expert',expert)]:
   position=np.linalg.norm(np.diff(pose[:,:3],axis=0),axis=1)*1000;r=Rotation.from_euler('xyz',pose[:,3:]);angle=np.rad2deg((r[:-1].inv()*r[1:]).magnitude());boundary=np.arange(1,n)%config['prefix']==0
   result[name]={'position_step_mean_mm':float(position.mean()),'position_step_p95_mm':float(np.percentile(position,95)),'position_step_max_mm':float(position.max()),'orientation_step_p95_deg':float(np.percentile(angle,95)),
    'boundary_position_step_mean_mm':float(position[boundary].mean()) if boundary.any() else None,'within_chunk_position_step_mean_mm':float(position[~boundary].mean()) if (~boundary).any() else None,
    'position_finite_difference_p95_m_s':float(np.percentile(position,95)*.06)}
   series[name]=(position,angle)
  d=f['servo_diagnostics'];result['velocity_clipped_joint_tick_fraction']=float(d['velocity_clipped'][:].mean());result['ik_delta_clipped_joint_tick_fraction']=float(d['ik_delta_clipped'][:].mean())
  result['commanded_joint_velocity_max_abs_rad_s']=np.abs(d['velocity_target'][:]).max(axis=0).tolist();result['measured_joint_velocity_max_abs_rad_s']=np.abs(d['qd'][:]).max(axis=0).tolist()
  result['interpretation']=['Policy chunk boundaries contain larger target changes than within chunks in this run.','Causal velocity feedforward differentiates target changes at60Hz; noisy target changes can produce alternating/saturated velocity references.','These measurements do not isolate camera mismatch, sampling noise, feedforward and actuator response causally. Compare camera alignment first with controller unchanged.']
  fig,axes=plt.subplots(3,1,figsize=(11,9));t=np.arange(1,n)/60
  for name,(position,angle) in series.items():axes[0].plot(t,position,label=name);axes[1].plot(t,angle,label=name)
  axes[0].set(ylabel='target step (mm)');axes[1].set(ylabel='rotation step (deg)')
  ts=np.arange(len(d['qd']))/180
  for joint in [3,4,5,6]:axes[2].plot(ts,d['qd'][:,joint],label='joint'+str(joint+1),alpha=.7)
  axes[2].set(xlabel='simulation time (s)',ylabel='measured joint speed (rad/s)')
  for ax in axes:ax.legend();ax.grid(alpha=.2)
  fig.tight_layout();fig.savefig(out/'command_steps_and_joint_velocity.png',dpi=140);plt.close(fig)
 (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('COMMAND_JITTER_AUDIT_COMPLETED '+str(out/'report.json'))
if __name__=='__main__':main()
