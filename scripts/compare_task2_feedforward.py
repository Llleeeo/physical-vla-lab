#!/usr/bin/env python3
"""Matched fixed VLA commands with causal versus zero velocity feedforward."""
import argparse,json,subprocess,sys,hashlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import h5py,numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/derived/task2_multiepisode_20261026/training_runs/20261026_185108_323227'
DEFAULT=BASE/'closed_loop_runs/20261026_233318_548559'

def trace_metrics(folder):
 with h5py.File(folder/'physics_trace.hdf5') as f:
  d=f['servo_diagnostics'];v=d['qd'][:];dv=np.diff(v,axis=0);valid=(np.abs(v[:-1])>.05)&(np.abs(v[1:])>.05)
  return {'frames':len(f['actions']),'servo_calls':int(f.attrs['servo_calls']),
   'tracking_position_mean_mm':float(d['goal_position_error_m'][:].mean()*1000),'tracking_position_p95_mm':float(np.percentile(d['goal_position_error_m'][:],95)*1000),
   'joint_speed_rms_rad_s':float(np.sqrt(np.mean(v*v))),
   'joint_speed_step_rms_rad_s':float(np.sqrt(np.mean(dv*dv))),
   'joint_speed_sign_reversals_above_005_rad_s':int(np.count_nonzero(valid&(v[:-1]*v[1:]<0))),
   'velocity_reference_clip_joint_tick_fraction':float(d['velocity_clipped'][:].mean()),
   'bilateral_contact_frames':int(np.count_nonzero(f['contact_L'][:]&f['contact_R'][:]))}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--source-run',type=Path,default=DEFAULT);p.add_argument('--frames',type=int,default=0);a=p.parse_args()
 if a.frames<0:p.error('frames must be nonnegative')
 source=a.source_run.resolve();out=BASE/'feedforward_comparisons'/datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
 if a.run:out.mkdir(parents=True,exist_ok=False)
 for mode in ['causal','zero']:
  cmd=[sys.executable,str(ROOT/'scripts/run_task2_closed_loop.py'),'--fixed-policy-run',str(source),'--velocity-reference',mode,'--frames',str(a.frames),'--output',str(out/mode)]
  if a.run:cmd.append('--run')
  print('FEEDFORWARD_CASE '+mode,flush=True);code=subprocess.run(cmd,cwd=ROOT).returncode
  if code:raise RuntimeError('case failed: '+mode+'; preserved outputs at '+str(out))
 if not a.run:print('CPU_COMPARISON_PREFLIGHT_PASS GPU_NOT_STARTED; add --run');return
 with h5py.File(source/'live_observations.hdf5') as f:expected=f['action'][:]
 for mode in ['causal','zero']:
  with h5py.File(out/mode/'live_observations.hdf5') as f:np.testing.assert_array_equal(f['action'][:],expected[:len(f['action'])])
 result={'status':'FEEDFORWARD_COMPARISON_COMPLETED','source_policy_run':str(source),'training':False,'new_model_inference':False,'source_commands_sha256':hashlib.sha256(expected.tobytes()).hexdigest(),
  'cases':{mode:trace_metrics(out/mode) for mode in ['causal','zero']},
  'interpretation':'Controller-only comparison on the same absolute commands and scene, not new closed-loop VLA success. Lower speed alone is not better if tracking lag increases. Sign-reversal threshold0.05rad/s is diagnostic, not a task criterion.'}
 with h5py.File(source/'physics_trace.hdf5') as original,h5py.File(out/'causal/physics_trace.hdf5') as replay:
  n=len(replay['joint_pos']);difference=replay['joint_pos'][:,:7]-original['joint_pos'][:n,:7]
  result['causal_reproduction_joint_position_rms_rad']=float(np.sqrt(np.mean(difference*difference)))
  result['causal_reproduction_joint_position_max_abs_rad']=float(np.abs(difference).max())
 (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2));print('FEEDFORWARD_COMPARISON_COMPLETED report='+str(out/'report.json'),flush=True)
if __name__=='__main__':main()
