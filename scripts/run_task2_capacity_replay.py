#!/usr/bin/env python3
"""Physically execute final TRAIN001-fit predictions; no new model inference."""
import argparse,hashlib,json,subprocess,sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import h5py,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from capstone_vla.grasp_evidence import evaluate_grasp,GraspEvidenceCriteria
BASE=ROOT/'data/derived/task2_multiepisode_20261026/training_runs/20261026_185108_323227'
DEFAULT=BASE/'single_episode_capacity_runs/20261027_110826_464886'
DIAG=BASE/'train_teacher_diagnostics/20261027_103424_626252'

def validate_final_stream(chunks,stream,anchors,n):
    chunks,stream,anchors=np.asarray(chunks),np.asarray(stream),np.asarray(anchors)
    if chunks.shape!=(len(range(0,n,4)),20,7) or stream.shape!=(n,7) or not np.isfinite(chunks).all() or not np.isfinite(stream).all():raise ValueError('invalid prediction arrays')
    if not np.array_equal(anchors,np.arange(0,n,4)):raise ValueError('prediction anchors changed')
    reconstructed=np.empty((n,7))
    for i,k in enumerate(anchors):
        count=min(4,n-k);reconstructed[k:k+count]=chunks[i,:count]
    reconstructed[:,6]=np.where(reconstructed[:,6]>0,1.,-1.)
    np.testing.assert_array_equal(reconstructed,stream)
    return stream.copy()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capacity-run',type=Path,default=DEFAULT)
    g=p.add_mutually_exclusive_group();g.add_argument('--run',action='store_true');g.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    capacity=args.capacity_run.resolve();report=json.loads((capacity/'report.json').read_text());fit=json.loads((capacity/'run_config.json').read_text());diagnostic=json.loads((DIAG/'report.json').read_text());prior=json.loads((Path(diagnostic['live_run'])/'run_config.json').read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if report['status']!='SINGLE_EPISODE_CAPACITY_COMPLETED' or report['normalization_unchanged'] is not True or fit['train_episodes']!=['traj_00001'] or fit['validation_opened'] or fit['reserved_test_opened']:raise ValueError('wrong capacity diagnostic')
    parent=Path(fit['initial_checkpoint'])
    if parent.resolve()!=(BASE/'checkpoint_final').resolve() or digest(parent/'model.safetensors')!=fit['initial_checkpoint_sha256']:raise ValueError('initial checkpoint changed')
    checkpoint=capacity/'checkpoint_final'
    if Path(report['checkpoint_final']).resolve()!=checkpoint or json.loads((checkpoint/'capstone_run.json').read_text())['completed_steps']!=report['optimizer_steps']:raise ValueError('final checkpoint/step mismatch')
    if json.loads((checkpoint/'capstone_normalization.json').read_text())!=json.loads((parent/'capstone_normalization.json').read_text()):raise ValueError('normalization changed')
    source=Path(prior['episode'])/'replay.hdf5'
    frozen=[source,Path(prior['task']),ROOT/'configs/franka_action_interface.json',ROOT/'src/capstone_vla/franka.py',ROOT/'src/capstone_vla/task2_bridge.py',ROOT/'src/capstone_vla/fixed_policy_replay.py']
    for path in frozen:
        if digest(path)!=prior['sha256'][str(path.relative_to(ROOT))]:raise ValueError('frozen scene/control changed: '+str(path))
    if digest(source)!=fit['source_replay_sha256']:raise ValueError('fit source mismatch')
    phase=f"step_{report['optimizer_steps']:06d}"
    with h5py.File(capacity/'generated_actions.hdf5') as f:
        commands=validate_final_stream(f[phase+'/absolute_chunks'][:],f[phase+'/queued_actions'][:],f['anchor_frames'][:],232)
        target=f['target_absolute'][:]
    with h5py.File(DIAG/'predictions.hdf5') as f:np.testing.assert_array_equal(f['target_absolute'][:],target)
    new_weight_hash=digest(checkpoint/'model.safetensors')
    print(f"CPU_PREFLIGHT_OK TRAIN001 frames=232 capacity_step={report['optimizer_steps']} final_commands=exact normalization=unchanged physical=true live_model=false source_actions_used=false",flush=True)
    if not (args.run or args.prepare_only):print('GPU_NOT_STARTED; add --run');return
    out=capacity/'physical_replay_runs'/datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');bundle=out/'command_bundle';bundle.mkdir(parents=True,exist_ok=False)
    config={**prior,'schema':'capstone_teacher_command_bundle_v1','command_origin':'recorded_expert_observation_predictions','purpose':'single_episode_fit_recorded_input_physical_diagnostic','checkpoint':str(checkpoint),'prediction_checkpoint_sha256':new_weight_hash,'capacity_training_run':str(capacity),'capacity_optimizer_steps':report['optimizer_steps'],'source_predictions':str(capacity/'generated_actions.hdf5'),'source_prediction_group':phase,'source_predictions_sha256':digest(capacity/'generated_actions.hdf5'),'source_capacity_report_sha256':digest(capacity/'report.json'),'training_eligible':False,'generalization_claim':False,'source_actions_used_for_control':False,'simulation_already_executed':False}
    (bundle/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
    with h5py.File(bundle/'commands.hdf5','x') as f:
        f.create_dataset('action',data=commands);f.create_dataset('timestamp_s',data=np.arange(232)/60)
        f.attrs['training_eligible']=False;f.attrs['source']='final capacity-fit model predictions on recorded expert RGB/proprioception; not live feedback'
    cmd=[sys.executable,str(ROOT/'scripts/run_task2_closed_loop.py'),'--episode-role','train-debug','--fixed-policy-run',str(bundle),'--frames','0','--output',str(out/'simulation')]
    if args.run:cmd.append('--run')
    print('OUTPUT',out,flush=True);subprocess.run(cmd,cwd=ROOT,check=True)
    if not args.run:return
    trace=out/'simulation/physics_trace.hdf5'
    with h5py.File(trace) as f:
        keys=['timestamp_s','ee_pos','ee_quat','object_pos','object_vel','contact_L','contact_R','impulse_L','impulse_R','grasp_assistance_audit','object_quat','object_ang_vel']
        evidence=evaluate_grasp({k:f[k][:] for k in keys if k in f},GraspEvidenceCriteria())
        if bool(f.attrs['expert_actions_used_for_control']):raise ValueError('unexpected reference-action control')
    evidence['scope']='fixed commands from fitted model using recorded expert observations; not autonomous VLA closed-loop success'
    evidence['source_trace_sha256']=digest(trace);evidence['prediction_checkpoint_sha256']=new_weight_hash
    (out/'simulation/independent_grasp_evidence.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    print('CAPACITY_PHYSICAL_REPLAY_COMPLETED engineering_grasp='+evidence['status']+' output='+str(out),flush=True)
if __name__=='__main__':main()
