#!/usr/bin/env python3
"""Replay model predictions on recorded expert inputs: diagnostic, not a live policy."""
import argparse,hashlib,json,subprocess,sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import h5py,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from capstone_vla.grasp_evidence import evaluate_grasp,GraspEvidenceCriteria
BASE=ROOT/'data/derived/task2_multiepisode_20261026/training_runs/20261026_185108_323227'
DEFAULT=BASE/'train_teacher_diagnostics/20261027_103424_626252'
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--diagnostic',type=Path,default=DEFAULT)
 group=p.add_mutually_exclusive_group();group.add_argument('--run',action='store_true');group.add_argument('--prepare-only',action='store_true')
 a=p.parse_args();d=a.diagnostic.resolve();report=json.loads((d/'report.json').read_text());prior=json.loads((Path(report['live_run'])/'run_config.json').read_text())
 if prior.get('episode_role')!='train-debug' or Path(prior['episode']).name!='traj_00001':raise ValueError('TRAIN001 only')
 if report['checkpoint_sha256']!=hashlib.sha256((Path(prior['checkpoint'])/'model.safetensors').read_bytes()).hexdigest():raise ValueError('weights changed')
 with h5py.File(d/'predictions.hdf5') as f:commands=f['original_render_live_preprocessing/queued_actions'][:]
 if commands.shape!=(232,7) or not np.isfinite(commands).all() or not np.all(np.abs(commands[:,6])==1):raise ValueError('invalid teacher commands')
 for rel,expected in prior['sha256'].items():
  if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=expected:raise ValueError('source/environment/controller changed: '+rel)
 print('CPU_PREFLIGHT_OK frames=232 origin=recorded_expert_observation_predictions physical=true model_process=false source_actions_used=false',flush=True)
 if not (a.run or a.prepare_only):print('GPU_NOT_STARTED; add --run or --prepare-only');return
 out=BASE/'teacher_command_replays'/datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');bundle=out/'command_bundle';bundle.mkdir(parents=True,exist_ok=False)
 config={**prior,'schema':'capstone_teacher_command_bundle_v1','command_origin':'recorded_expert_observation_predictions','purpose':'offline_model_command_execution_diagnostic','training_eligible':False,'simulation_already_executed':False,'source_predictions':str(d/'predictions.hdf5'),'source_predictions_sha256':hashlib.sha256((d/'predictions.hdf5').read_bytes()).hexdigest(),'source_diagnostic_report_sha256':hashlib.sha256((d/'report.json').read_bytes()).hexdigest(),'generalization_claim':False}
 (bundle/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
 with h5py.File(bundle/'commands.hdf5','x') as f:
  f.create_dataset('action',data=commands);f.create_dataset('timestamp_s',data=np.arange(232)/60)
  f.attrs['source']='model predictions on recorded expert RGB/proprioception, not online feedback';f.attrs['training_eligible']=False
 cmd=[sys.executable,str(ROOT/'scripts/run_task2_closed_loop.py'),'--episode-role','train-debug','--fixed-policy-run',str(bundle),'--frames','0','--output',str(out/'simulation')]
 if a.run:cmd.append('--run')
 print('OUTPUT',out,flush=True);subprocess.run(cmd,cwd=ROOT,check=True)
 if not a.run:return
 trace=out/'simulation/physics_trace.hdf5'
 with h5py.File(trace) as f:
  keys=['timestamp_s','ee_pos','ee_quat','object_pos','object_vel','contact_L','contact_R','impulse_L','impulse_R','grasp_assistance_audit','object_quat','object_ang_vel']
  evidence=evaluate_grasp({k:f[k][:] for k in keys if k in f},GraspEvidenceCriteria())
 evidence['scope']='engineering check of fixed teacher-input model commands; not learned-policy closed-loop success'
 evidence['source_trace_sha256']=hashlib.sha256(trace.read_bytes()).hexdigest()
 (out/'simulation/independent_grasp_evidence.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
 print('TEACHER_COMMAND_REPLAY_COMPLETED engineering_grasp='+evidence['status']+' output='+str(out),flush=True)
if __name__=='__main__':main()
