"""Task-scoped saved-command probe through frozen RoboPhys live effort execution.

This module does not relax the upstream action-repair tool's task restrictions.
Only the independent probe worker imports these callbacks. The shadow task FSM
may produce evidence, but its motor targets and operational references are not
executed. No learned-policy evaluation is implemented here.
"""
import json
import os
from pathlib import Path

import h5py
import numpy as np

from .franka import FrankaActionContract
from .recorded_franka import RecordedFrankaAdapter
from .hf_archive import sha256
from .hgwind_cohort import TASK_ID, require, verify_files

_STATE = {}


def validate_source(plan, metadata, actions, times):
    require(plan['schema']=='physical-vla-lab.hgwind-effort-probe.v1','Unknown probe manifest')
    require(plan['mode'] in ('joint','cartesian'),'Unknown probe mode')
    steps=metadata.get('task_plan',{}).get('steps',[])
    require(metadata.get('task',{}).get('id')==TASK_ID and len(steps)==1
            and steps[0]['skill']=='Grasp' and steps[0]['target_role']=='A','Probe is HGwind Grasp only')
    require(metadata.get('robot',{}).get('type')=='franka'
            and metadata.get('arm_execution',{}).get('mode')=='operational_space_inverse_dynamics_effort','Unsupported recorded actuation')
    names=[f'panda_joint{i}' for i in range(1,8)]+['panda_finger_joint1','panda_finger_joint2']
    require(metadata.get('joint_names')==names,'Unexpected source joint order')
    require(metadata.get('success') is True and metadata.get('runtime_safety',{}).get('passed') is True,'Source is not an accepted episode')
    require(metadata.get('grasp_mode')=='physical_finger_contact_only'
            and not metadata.get('fix_joint') and not metadata.get('fixedforce'),'Assisted source is not allowed')
    require(actions.ndim==2 and actions.shape[1]==9 and len(actions)>=2
            and np.isfinite(actions).all(),'Invalid source action array')
    require(times.shape==(len(actions),) and np.isfinite(times).all()
            and np.allclose(np.diff(times),1/60,rtol=0,atol=1e-7)
            and float(metadata['control_hz'])==60,'Source clock is not 60 Hz')
    grip=actions[:,7:]
    require(np.allclose(grip[:,0],grip[:,1],atol=1e-6)
            and np.all(np.isclose(grip,0,atol=1e-5)|np.isclose(grip,.04,atol=1e-5)),'Unsupported gripper command')



def build_reference(actions, joint_positions, joint_velocities, reference):
    """Separate saved actuator references from observed-motion tracking labels.

    Row k in recorded-motion targets observed robot state k+1, while retaining
    the source gripper command at k. The absent state N is never fabricated.
    This is an offline desired trajectory, not a policy observation or recovery
    of the source controller's unrecorded reference velocity.
    """
    if reference == 'saved-targets':
        return np.asarray(actions, np.float32).copy(), np.full(np.shape(actions), np.nan)
    require(reference == 'recorded-motion', 'Unknown reference semantics')
    a=np.asarray(actions);q=np.asarray(joint_positions);v=np.asarray(joint_velocities)
    require(a.ndim==2 and a.shape[1]==9 and len(a)>=3 and q.shape==a.shape and v.shape==a.shape
            and np.isfinite(q).all() and np.isfinite(v).all(), 'Invalid recorded robot-motion arrays')
    goals=a[:-1].astype(np.float32).copy();goals[:,:7]=q[1:,:7]
    velocities=np.zeros_like(goals);velocities[:,:7]=v[1:,:7]
    return goals,velocities


def _plan():
    return json.loads(Path(os.environ['HGWIND_EFFORT_PROBE_MANIFEST']).read_text())


def load_action_reference(source, *, task_id, control_hz=None):
    plan=_plan();source=Path(source).resolve();ep=source.parent
    require(task_id==TASK_ID and source==Path(plan['source']) and control_hz==60,'Unexpected probe source/task/rate')
    verify_files(ep,plan['source_files'])
    meta=json.loads((ep/'meta.json').read_text())
    with h5py.File(source,'r') as f:
        actions=f['actions'][:];times=f['timestamp'][:];positions=f['obs/joint_pos'][:];velocities=f['obs/joint_vel'][:]
    validate_source(plan,meta,actions,times)
    actions,velocities=build_reference(actions,positions,velocities,plan.get('reference','saved-targets'))
    count=int(plan['frames']);require(2<=count<=len(actions),'Invalid probe length')
    require(plan.get('reference','saved-targets')!='recorded-motion' or plan['mode']=='joint','Recorded-motion diagnostic currently supports joint tracking only')
    keys=('ee_offset_m','damping','max_joint_target_delta_rad','finger_open_m')
    cfg=json.loads(Path(plan['controller_config']).read_text())
    require(sha256(plan['controller_config'])==plan['controller_config_sha256'],'Controller contract changed')
    adapter=RecordedFrankaAdapter(cfg['urdf'],meta['end_effector'],FrankaActionContract(**{k:cfg['action'][k] for k in keys}))
    require(adapter.urdf_sha256==plan['urdf_sha256'],'URDF changed')
    actions=actions[:count]
    poses=np.stack([adapter.contract.encode(*adapter.pose_jacobian(q[:7])[:2],1 if q[7]>.02 else -1) for q in actions])
    _STATE.clear();_STATE.update(plan=plan,adapter=adapter,poses=poses,velocities=velocities[:count],requests=[],assistance=[])
    return actions.astype(np.float32),meta,{'source_trajectory':str(source),'source_sha256':sha256(source),
        'source_joint_names':meta['joint_names'],'source_frames':count,'mode':'capstone_hgwind_effort_command_probe',
        'initialization':'same_scene_base_seed_and_attempt_index; physical first states must be compared',
        'source_success_reused':False,'exact_state_reproduction':False}


def require_repair_run(source, output, provenance):
    plan=_STATE['plan'];output=Path(output).resolve()
    require(output==Path(plan['worker_output']) and Path(source).resolve()==Path(plan['source']),'Output/source mismatch')
    require(not output.is_relative_to(Path(source).parent),'Probe output must not be in source episode')
    require(sha256(plan['task'])==plan['task_sha256'],'Task USD changed')
    require(sha256(plan['bundle_manifest'])==plan['bundle_manifest_sha256'],'Frozen bundle manifest changed')


def reference_frame(actions, index, source_names, runtime_names, *, measured_positions):
    require(list(source_names)==list(runtime_names),'Runtime joint order differs')
    measured=np.asarray(measured_positions,float)
    require(measured.shape==(9,) and np.isfinite(measured).all(),'Invalid robot proprioception')
    target=np.asarray(actions[index],np.float32).copy();mode=_STATE['plan']['mode']
    if mode=='cartesian':
        target[:7],finger=_STATE['adapter'].joint_target(measured[:7],_STATE['poses'][index])
        target[7:]=finger
    velocity=np.asarray(_STATE['velocities'][index]).copy() if 'velocities' in _STATE else np.full(target.shape,np.nan)
    _STATE['requests'].append({'requested_velocity':velocity.tolist(),'frame':int(index),'measured_joints':measured.tolist(),'requested_joints':target.tolist(),
                               'source_pose_action':_STATE['poses'][index].tolist()})
    # Saved-targets keeps the measured-state velocity fallback. Recorded-motion
    # supplies its explicit observed-velocity trajectory instead. Neither
    # reconstructs the original 180 Hz control references.
    return target,velocity


def check_assistance(stage, robot_path, object_paths, attach_state):
    from .task2_bridge import grasp_joint_candidates
    if 'joint_candidates' not in _STATE:
        _STATE['joint_candidates']=grasp_joint_candidates(stage,robot_path,object_paths)
    enabled=[p for p in _STATE['joint_candidates'] if bool(stage.GetPrimAtPath(p).GetAttribute('physics:jointEnabled').Get())]
    flags=[bool(attach_state.get('attached')),len(enabled),bool(attach_state.get('held_tool'))]
    _STATE['assistance'].append(flags)
    require(not any(flags),'Probe detected robot-object grasp assistance')


def summary():
    plan=_STATE['plan'];out=Path(plan['output'])/'command_diagnostics.npz'
    data=_STATE['requests']
    arrays={key:np.asarray([x[key] for x in data]) for key in ['frame','measured_joints','requested_joints','requested_velocity','source_pose_action']}
    arrays['assistance']=np.asarray(_STATE['assistance'])
    if out.exists():raise FileExistsError('Probe diagnostics already exist')
    np.savez(out,**arrays)
    return {'schema':plan['schema'],'mode':plan['mode'],'reference':plan.get('reference','saved-targets'),'command_frames':len(data),'assistance_checks':len(_STATE['assistance']),
        'source_reference_sha256':sha256(plan['source']),'saved_commands_own_motors':True,
        'shadow_expert_operational_reference_used':False,'shadow_expert_velocity_limit_used':False,
        'shadow_task_fsm_runs_for_evidence':True,'source_native_operational_reference_reconstructed':False,'primary_tracking_controller':'joint_inverse_dynamics',
        'training_eligible':False,'vla_policy':False,'source_actuator_mode':'operational_space_inverse_dynamics_effort',
        'velocity_reference':'next observed joint velocity' if plan.get('reference')=='recorded-motion' else 'measured-state position-error fallback',
        'reference_path':'saved/converted joint goals -> shared physics-rate viability generator -> original certified model-effort executor'}


def patch_worker(source):
    pairs=[
        ('from expert_data.action_repair import load_action_reference, require_repair_run',
         'from capstone_vla.hgwind_effort_probe import load_action_reference, require_repair_run'),
        ('from expert_data.action_repair import reference_frame',
         'from capstone_vla.hgwind_effort_probe import reference_frame'),
        ('                        robot_dof_names,\n                    )\n                # A preflight reference',
         '                        robot_dof_names, measured_positions=jp,\n                    )\n                    policy_joint_velocity_limit = None\n                # A preflight reference'),
        ('            rec["ee_pos"].append(ee_p); rec["ee_quat"].append(np.asarray(ee_q, np.float32))',
         '            rec["ee_pos"].append(ee_p); rec["ee_quat"].append(np.asarray(ee_q, np.float32))\n            from capstone_vla.hgwind_effort_probe import check_assistance\n            check_assistance(stage, robot_path, objects, attach_state)'),
        ('"skill_terminations": _jsonify(skill_terminations),',
         '"skill_terminations": _jsonify(skill_terminations),\n            "capstone_effort_probe": __import__("capstone_vla.hgwind_effort_probe", fromlist=["summary"]).summary(),'),
    ]
    # The upstream repair branch already discards the shadow operational
    # reference. Refuse a worker where that guarantee disappeared.
    required="                    if repair_actions is not None:\n                        # Consume but never execute the live planner's Cartesian\n                        # reference: the saved joint reference owns the motors.\n                        operational_reference = None"
    require(source.count(required)==1,'Shadow operational-reference suppression changed')
    for old,new in pairs:
        require(source.count(old)==1,'Probe insertion point changed: '+old[:75])
        source=source.replace(old,new)
    compile(source,'<frozen-hgwind-effort-probe>','exec')
    return source


def initial_state_differences(source, current, source_paths, current_paths):
    require(set(source_paths)==set(current_paths),'Dynamic-object identity differs')
    order=[current_paths.index(p) for p in source_paths]
    result={}
    for key in ['joint_pos','joint_vel']:
        result[key+'_max_abs']=float(np.max(np.abs(np.asarray(source[key])-np.asarray(current[key]))))
    for key in ['object_pos','object_vel','object_ang_vel']:
        result[key+'_max_norm']=float(np.max(np.linalg.norm(np.asarray(source[key])-np.asarray(current[key])[order],axis=-1)))
    a=np.asarray(source['object_quat']);b=np.asarray(current['object_quat'])[order]
    result['object_quaternion_sign_invariant_max']=float(np.max(np.minimum(np.linalg.norm(a-b,axis=-1),np.linalg.norm(a+b,axis=-1))))
    return result
