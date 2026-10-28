#!/usr/bin/env python3
"""CPU audit of saved validation predictions; no policy rollout or test access."""
import argparse
from pathlib import Path
import sys,json
import h5py,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from capstone_vla.cohort_training import read_split_arrays
from capstone_vla.prediction_metrics import action_errors

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,default=ROOT/'data/derived/task2_multiepisode_20261026/training_runs/20261026_185108_323227');a=p.parse_args()
 report=json.loads((a.run_dir/'report.json').read_text())
 rows=read_split_arrays(ROOT/'data/derived/task2_multiepisode_20261026/validation')
 target=np.concatenate([r['action'] for r in rows]);state=np.concatenate([r['state'] for r in rows]);n=len(target)
 with h5py.File(a.run_dir/'validation_actions.hdf5','r') as f:
  initial=f['initial/predicted_first_absolute'][:];final=f['final/predicted_first_absolute'][:]
  for phase in ['initial','final']:np.testing.assert_array_equal(f[phase+'/target_absolute'][:],target)
 mask=np.zeros((n,1),bool)
 def metrics(x):return action_errors(x[:,None,:],target[:,None,:],mask)
 before=metrics(initial);after=metrics(final)
 for phase,value in [('initial',before),('final',after)]:
  saved=report['validation_actions_'+phase]['all_validation_frames_first_action']
  for key,v in value[0]['first_action'].items():np.testing.assert_allclose(v,saved[key],rtol=1e-6,atol=1e-7)
 # Gripper is excluded: this baseline predicts pose only, never an oracle gripper.
 hold=target.copy();hold[:,:6]=state
 baseline=metrics(hold)
 pose_baseline={k:v for k,v in baseline[0]['first_action'].items() if k!='gripper_sign_accuracy'}
 out=a.run_dir/'offline_analysis';out.mkdir(exist_ok=False)
 detail=[];curves=[];offset=0
 for row in rows:
  size=len(row['action']);sl=slice(offset,offset+size);time=np.arange(size)/60
  true=target[sl,6]>0;pred=final[sl,6]>0
  curves.append({'episode':row['episode'],'time':time.tolist(),'position':after[1][sl,0].tolist(),'hold_position':baseline[1][sl,0].tolist(),'expert_gripper':true.astype(int).tolist(),'model_gripper':pred.astype(int).tolist()})
  tp=int((~true&~pred).sum());fn=int((~true&pred).sum());fp=int((true&~pred).sum());tn=int((true&pred).sum())
  events=(np.flatnonzero(np.diff(true.astype(int)))+1).tolist()
  detail.append({'episode':row['episode'],'frames':size,'closed_recall':tp/(tp+fn),'open_recall':tn/(tn+fp),
    'gripper_balanced_accuracy':.5*(tp/(tp+fn)+tn/(tn+fp)),
    'expert_switch_frames':events,'model_switch_count':int(np.count_nonzero(np.diff(pred.astype(int)))),
    'expert_switch_frame_sign_correct':int(sum(pred[i]==true[i] for i in events)),
    'expert_switch_frames_total':len(events),'closed_true_positive':tp,'closed_false_negative':fn,'open_false_positive':fp,
    'position_mean_mm':float(after[1][sl,0].mean()),'hold_pose_position_mean_mm':float(baseline[1][sl,0].mean())})
  offset+=size
 (out/'curves.json').write_text(json.dumps({'episodes':curves,'flow_history':report['validation_flow_history']})+'\n')
 result={'status':'OFFLINE_AUDIT_PASS','scope':'same-task validation, recorded observations, fixed per-frame random noise; no rollout',
 'target_alignment':'exact match to validation parquet actions','saved_metrics_reproduced':True,
 'initial':before[0]['first_action'],'final':after[0]['first_action'],'hold_measured_pose_baseline':pose_baseline,
 'per_episode':detail,'reserved_test_opened':False,
 'interpretation':['Pose errors improved relative to unadapted initialization, but first-action errors exceed a hold-current-pose baseline.',
 'Hold baseline has access only to measured pose; it cannot solve the moving-object task by holding still.',
 'Gripper predictions vary strongly across episodes; one per-frame noise draw does not isolate sampling noise from observation sensitivity.',
 'Exact expert switch timing is an imitation diagnostic, not a task-success criterion.',
 'No causal attribution to dataset size alone: budget, normalizers and periodic Euler labels also changed.']}
 (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
 print('OFFLINE_AUDIT_PASS report='+str(out/'report.json'))
 print('FINAL',json.dumps(result['final']));print('HOLD_POSE',json.dumps(pose_baseline))

if __name__=='__main__':main()
