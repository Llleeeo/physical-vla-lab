#!/usr/bin/env python3
"""TRAIN001 expert/model component controls; sequential CPU preflight by default."""
import argparse,hashlib,json,subprocess,sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import h5py,numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from capstone_vla.component_controls import VARIANTS,compose_commands
from capstone_vla.franka import FrankaCartesianAdapter
from capstone_vla.grasp_evidence import evaluate_grasp,GraspEvidenceCriteria
BASE=ROOT/'data/derived/task2_multiepisode_20261026/training_runs/20261026_185108_323227'
DEFAULT=BASE/'train_teacher_diagnostics/20261027_103424_626252'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostic',type=Path,default=DEFAULT)
    group=p.add_mutually_exclusive_group();group.add_argument('--run',action='store_true');group.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();d=args.diagnostic.resolve();diagnosis=json.loads((d/'report.json').read_text())
    prior=json.loads((Path(diagnosis['live_run'])/'run_config.json').read_text())
    if prior.get('episode_role')!='train-debug' or Path(prior['episode']).name!='traj_00001':raise ValueError('TRAIN001 only')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(Path(prior['checkpoint'])/'model.safetensors')!=diagnosis['checkpoint_sha256']:raise ValueError('checkpoint changed')
    # Freeze only source scene/physics and controller files here; audit-only entry edits are allowed.
    source=Path(prior['episode'])/'replay.hdf5';task=Path(prior['task'])
    frozen=[source,task,ROOT/'configs/franka_action_interface.json',ROOT/'src/capstone_vla/franka.py',ROOT/'src/capstone_vla/task2_bridge.py',ROOT/'src/capstone_vla/fixed_policy_replay.py']
    for path in frozen:
        if digest(path)!=prior['sha256'][str(path.relative_to(ROOT))]:raise ValueError('frozen environment/control changed: '+str(path))
    cfg=json.loads((ROOT/'configs/franka_action_interface.json').read_text());fk=FrankaCartesianAdapter(cfg['urdf'])
    with h5py.File(source) as f:
        joints=f['actions'][:];n=len(joints)
        if joints.shape!=(232,9):raise ValueError('source shape changed')
        fingers=joints[:,7:9]
        if not np.all(np.all(np.isclose(fingers,0,atol=1e-5),axis=1)|np.all(np.isclose(fingers,.04,atol=1e-5),axis=1)):raise ValueError('nonbinary/asymmetric source fingers')
        expert=np.stack([fk.contract.encode(*fk.pose_jacobian(q[:7])[:2],1. if q[7:9].mean()>.02 else -1.) for q in joints])
    with h5py.File(d/'predictions.hdf5') as f:model=f['original_render_live_preprocessing/queued_actions'][:];target=f['target_absolute'][:]
    np.testing.assert_allclose(expert[:,:3],target[:,:3],rtol=0,atol=1e-6)
    if np.max((Rotation.from_euler('xyz',expert[:,3:6]).inv()*Rotation.from_euler('xyz',target[:,3:6])).magnitude())>1e-5:raise ValueError('reference orientation labels changed')
    np.testing.assert_array_equal(expert[:,6],target[:,6])
    streams={variant:compose_commands(expert,model,variant) for variant in VARIANTS}
    print('CPU_PREFLIGHT_OK variants=3 frames=232 source_FK=aligned frozen_controller=true privileged_diagnostic=true train=false live_model=false',flush=True)
    for variant,(_,sources) in streams.items():print('PLAN '+variant+' '+json.dumps(sources),flush=True)
    if not (args.run or args.prepare_only):print('GPU_NOT_STARTED; add --run');return 0
    out=BASE/'component_control_runs'/datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True,exist_ok=False)
    result={'scope':'TRAIN001 privileged reference-component diagnostics; not VLA closed-loop or generalization success','status':'running' if args.run else 'prepared_only','official_benchmark_success':None,'diagnostic':str(d),'checkpoint_sha256':diagnosis['checkpoint_sha256'],'source_sha256':digest(source),'source_predictions_sha256':digest(d/'predictions.hdf5'),'reference_pose_precision':'full-precision FK of original joint targets, verified against exported labels','trials':[]}
    def save():
        tmp=out/'report.json.tmp';tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(out/'report.json')
    save();print('COMPONENT_OUTPUT',out,flush=True)
    for variant,(commands,sources) in streams.items():
        variant_root=out/variant;bundle=variant_root/'command_bundle';bundle.mkdir(parents=True,exist_ok=False)
        origin={**prior,'schema':'capstone_component_command_bundle_v1','command_origin':'expert_model_component_diagnostic','component_variant':variant,'component_sources':sources,'training_eligible':False,'source_actions_used_for_control':True,'generalization_claim':False,'purpose':'reference_component_diagnostic_only','source_predictions_sha256':result['source_predictions_sha256'],'source_sha256':result['source_sha256']}
        (bundle/'run_config.json').write_text(json.dumps(origin,indent=2)+'\n')
        with h5py.File(bundle/'commands.hdf5','x') as f:
            f.create_dataset('action',data=commands);f.create_dataset('timestamp_s',data=np.arange(n)/60)
            f.attrs['training_eligible']=False;f.attrs['diagnostic_component_sources_json']=json.dumps(sources);f.attrs['source_actions_used_for_control']=True
        simulation=variant_root/'simulation'
        cmd=[sys.executable,str(ROOT/'scripts/run_task2_closed_loop.py'),'--episode-role','train-debug','--fixed-policy-run',str(bundle),'--frames','0','--output',str(simulation)]
        if args.run:cmd.append('--run')
        trial={'variant':variant,'component_sources':sources,'simulation':str(simulation),'source_actions_used_for_control':True,'status':'started' if args.run else 'prepared_only'}
        result['trials'].append(trial);save()
        try:
            subprocess.run(cmd,cwd=ROOT,check=True)
            if not args.run:continue
            with h5py.File(simulation/'physics_trace.hdf5') as f:
                keys=['timestamp_s','ee_pos','ee_quat','object_pos','object_vel','contact_L','contact_R','impulse_L','impulse_R','grasp_assistance_audit','object_quat','object_ang_vel']
                evidence=evaluate_grasp({k:f[k][:] for k in keys if k in f},GraspEvidenceCriteria())
                if not bool(f.attrs['expert_actions_used_for_control']) or json.loads(f.attrs['diagnostic_component_sources_json'])!=sources:raise ValueError('trace source metadata incorrect')
                trial.update(left_contact_frames=int(f['contact_L'][:].sum()),right_contact_frames=int(f['contact_R'][:].sum()),tracking_mean_mm=float(f['servo_diagnostics/goal_position_error_m'][:].mean()*1000))
            evidence['scope']='reference-component diagnostic, not autonomous VLA success';evidence['component_sources']=sources;evidence['source_trace_sha256']=digest(simulation/'physics_trace.hdf5')
            (simulation/'independent_grasp_evidence.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
            trial.update(status='completed',engineering_grasp=evidence['status'],measurements=evidence['measurements']);save()
            print('COMPONENT_RESULT '+variant+' engineering_grasp='+evidence['status'],flush=True)
            if variant=='expert_all' and evidence['status']!='pass':
                result['status']='stopped_positive_control_failed';result['unrun_variants']=[v for v in VARIANTS if v!='expert_all'];save()
                print('COMPONENT_CONTROL_STOPPED expert positive control failed; remaining mixed controls not run. report='+str(out/'report.json'),flush=True);return 2
        except Exception as exc:
            trial.update(status='execution_error',error=str(exc));result['status']='execution_error';save();raise
    result['status']='completed' if args.run else 'prepared_only';save()
    print(('COMPONENT_CONTROLS_COMPLETED' if args.run else 'COMPONENT_CONTROLS_PREPARED')+' report='+str(out/'report.json'),flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
