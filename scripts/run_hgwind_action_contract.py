#!/usr/bin/env python3
"""One integrated float32 action-contract gate; GPU simulation needs --run.

This reexecutes expert-labelled actions, not a trained policy. The legacy probe
runner and all saved baselines remain unchanged. Always-on geometry protection
and float32 pose/motion roundtrip are tested together, then stop micro-ablations.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
from zoneinfo import ZoneInfo
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from capstone_vla.ball_action_contract import (SCHEMA, encode_stream, decode_stream,
    specification, model_configuration, VARIABLE_INPUTS)
from capstone_vla.cartesian_packets import audit_packets
from capstone_vla.cartesian_target_replay import require
from capstone_vla.hf_archive import sha256
from capstone_vla.hgwind_cohort import verify_files

DEFAULT_BASELINE=ROOT/'data/derived/hgwind_effort_probes/20261029_202630_260880_joint_online-expert_native-expert_target-reexecution_packets_fixed-9-8_posture-latched'


def prepare(baseline):
    baseline=Path(baseline).resolve()
    report=json.loads((baseline/'report.json').read_text())
    prior=json.loads((baseline/'plan.json').read_text())
    require(report.get('packet_latched_posture_probe_passed') is True
        and report.get('packet_controller_probe_passed') is True
        and prior.get('packet_gain_profile')=='fixed-9-8'
        and prior.get('packet_posture_profile')=='latched'
        and not prior.get('action_contract'), 'Need a passing latched-posture reference')
    require(report['source_reference_sha256']==prior['source_reference_sha256']
        and report['task_sha256']==prior['task_sha256']
        and report['execution_fingerprints']=={k:prior[k] for k in [
            'worker_sha256','patch_sha256','controller_config_sha256','urdf_sha256','adapter_sha256']},
        'Baseline report/plan provenance mismatch')
    for key in ['source_reference','task','worker','bundle_manifest','controller_config',
        'cartesian_target_capture','packet_baseline_report']:
        path=prior['source'] if key=='source_reference' else prior[key]
        require(sha256(path)==prior[key+'_sha256'],'Baseline input changed: '+key)
    # Optional recording hooks may be corrected without changing commands.
    # Re-audit the parent raw links below; frozen geometry, servo, posture and
    # effort implementations must still match the parent exactly.
    trace=ROOT/'src/capstone_vla/hgwind_expert_trace.py'
    recorder=ROOT/'src/capstone_vla/cartesian_target_trace.py'
    for name,digest in prior['adapter_sha256'].items():
        path=Path(name) if Path(name).is_absolute() else ROOT/name
        if path not in (trace,recorder):
            require(sha256(path)==digest,'Baseline dependency changed: '+str(path))
    source=Path(prior['source']);verify_files(source.parent,prior['source_files'])
    cfg=json.loads(Path(prior['controller_config']).read_text())
    lab=Path(cfg['isaaclab_root'])
    require((lab/'VERSION').read_text().strip()=='2.2.1' and
        ((lab/'_isaac_sim').resolve()/'VERSION').read_text().startswith('4.5.0'),'SDK changed')
    sys.path.insert(0,str(Path(prior['worker']).parents[1]))
    from pxr import Usd
    from dataset.scene_signature import stage_physics_signature
    require(stage_physics_signature(Usd.Stage.Open(prior['task']))['sha256']==prior['physics_sha256'],
        'Physics scene changed')
    packets=json.loads((baseline/'packet_requests.json').read_text())
    payload=json.loads((baseline/'cartesian_packet_execution.json').read_text())
    segments=json.loads((baseline/'original_expert_control_references.json').read_text())
    require(audit_packets(payload,packets,segments,prior['frames'])['packet_execution_verified'],
        'Saved baseline motor-boundary audit failed')
    require(payload['posture_initialization']['configuration']==prior['robot_posture_configuration'],
        'Posture configuration mismatch')
    with h5py.File(source,'r') as handle:
        require(np.array_equal(handle['obs/joint_pos'][0,:7],
            payload['posture_initialization']['initial_joint_position_rad']), 'Baseline initial robot mismatch')
    anchor=payload['posture_initialization']['anchor_rad']
    require(all(p['inputs']['posture_target']==anchor for p in packets),'Baseline posture is not latched')
    actions=encode_stream(packets)
    decoded=decode_stream(actions,anchor)
    # Complete field accounting; no variable request is supplied by a side file.
    require(all(set(a['inputs'])==set(b['inputs']) for a,b in zip(packets,decoded)),
        'Contract omitted a native field')
    for old,new in zip(packets,decoded):
        require(old['gripper_closed'] is new['gripper_closed'],'Jaw event lost')
        for name in VARIABLE_INPUTS-{'target_quaternion'}:
            a,b=old['inputs'][name],new['inputs'][name]
            require((a is None and b is None) or (a is not None and b is not None
                and np.allclose(a,b,atol=1.e-6,rtol=1.e-6)),
                'Numeric action roundtrip mismatch: '+name)
        for name in ['target_linear_acceleration','linear_velocity_reference',
            'maximum_relative_linear_speed_mps','orientation_axis_world']:
            require((old['inputs'][name] is None)==(new['inputs'][name] is None),
                'Nullable command semantics changed: '+name)
        for name in ['preserve_orientation_during_translation','operational_orientation_task']:
            require(old['inputs'][name] is new['inputs'][name],'Controller mode changed')
    pose_error=max(np.linalg.norm(np.asarray(a['inputs']['target_position'])-
        b['inputs']['target_position']) for a,b in zip(packets,decoded))
    quaternion_error=max(min(np.linalg.norm(np.asarray(a['inputs']['target_quaternion'])-
        b['inputs']['target_quaternion']),np.linalg.norm(np.asarray(a['inputs']['target_quaternion'])+
        b['inputs']['target_quaternion'])) for a,b in zip(packets,decoded))
    require(pose_error<1.e-6 and quaternion_error<1.e-6,'Float32 pose roundtrip outside tolerance')
    raw=json.loads((ROOT/'configs/dynamicvla_dom_pretrained.json').read_text())
    model_cfg=model_configuration(raw)
    require(model_cfg['output_features']['action']['shape']==[29]
        and model_cfg['input_features']['observation.state']['shape']==[30]
        and not model_cfg['use_delta_action'],'Invalid model feature configuration')
    stamp=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
    out=ROOT/'data/derived/hgwind_action_contract_v1'/stamp
    plan=copy.deepcopy(prior)
    plan.update(output=str(out),worker_output=str(out/'worker_output'),
        action_contract=SCHEMA,action_vectors=str(out/'actions.npy'),
        action_contract_parent_report=str(baseline/'report.json'),
        action_contract_parent_report_sha256=sha256(baseline/'report.json'),
        action_contract_parent_requests=str(baseline/'packet_requests.json'),
        action_contract_parent_requests_sha256=sha256(baseline/'packet_requests.json'),
        reference_fidelity='Expert-labelled 29D float32 upper-command surrogate with fixed gains/live-latched posture and always-on geometry callbacks. No model predictions, online skill or cached inner commands.',
        training_eligible=False)
    for path in [trace,recorder,ROOT/'src/capstone_vla/ball_action_contract.py',
        Path(__file__).resolve(),ROOT/'scripts/analyze_hgwind_effort_probe.py']:
        plan['adapter_sha256'][str(path)]=sha256(path)
    diagnostics=dict(frames=len(actions),action_shape=list(actions.shape),action_dtype=str(actions.dtype),
        model_state_dim=30,pretrained_action_capacity=32,upstream_use_delta_action=False,
        maximum_position_roundtrip_error_mm=float(pose_error*1000),
        maximum_quaternion_sign_invariant_error=float(quaternion_error),
        callback_activation_changed_frames=sum(a['safety_callbacks']!=b['safety_callbacks']
            for a,b in zip(packets,decoded)),
        hidden_terminal_schedule_removed_frames=sum(p['terminal_cartesian_hold'] for p in packets),
        gripper_and_null_modes_preserved=True,
        action_array_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
        parent_report_sha256=plan['action_contract_parent_report_sha256'],
        gpu_started=False,learned_policy_evaluated=False)
    return plan,actions,model_cfg,diagnostics,lab


def command(plan,lab):
    meta=json.loads((Path(plan['source']).parent/'meta.json').read_text())
    worker=ROOT/'scripts/sim/run_hgwind_native_expert_worker.py'
    if plan.get('initialized_contract'):
        from capstone_vla.initialized_contract import validate
        validate(plan)
        worker=ROOT/'scripts/sim/run_hgwind_initialized_contract_worker.py'
    else:
        require(not plan.get('setup_force_control') and not plan.get('gripper_gain_control'),
            'Initialization hooks require the explicit initialized-contract entry')
    return [str(lab/'isaaclab.sh'),'-p',str(worker),
        '--task',plan['task'],'--out_root',plan['worker_output'],'--episodes','1','--max_attempts','1',
        '--seed',str(plan['seed']),'--attempt_offset',str(plan['attempt_offset']),
        '--variation',str(meta['variation']),'--control_hz','60','--grasp_mode','physical',
        '--arm_execution','inverse_dynamics','--keep_failed','--run_id','hgwind_action_contract_'+Path(plan['output']).name,
        '--device','cuda:0','--headless','--no-visualization','--fast_exit_after_save',
        '--experience','isaacsim.exp.base.python.kit',
        '--kit_args=--portable-root '+str(ROOT/'data/derived/.cache/isaacsim/linux')+
            ' --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/enabled=false --/renderer/multiGpu/maxGpuCount=1 --/app/extensions/fsWatcherEnabled=false',
        '--episode_s',str(plan['frames']/60),'--no-expert-config']


def run_prepared(plan,actions,model_cfg,diagnostics,lab,run=False):
    cmd=command(plan,lab);out=Path(plan['output'])
    print('CPU_PREFLIGHT_OK '+json.dumps(diagnostics),flush=True)
    print('CONTRACT absolute_world_pose=true motion_feedforward=explicit safety_callbacks=always_on hidden_expert_schedule=false',flush=True)
    print('PURPOSE expert_command_interface_validation; learned_policy=false',flush=True)
    print('OUTPUT '+str(out),flush=True)
    print('COMMAND '+shlex.join(cmd),flush=True)
    if not run:
        print('DRY_RUN_ONLY: add --run');return 0
    out.mkdir(parents=True,exist_ok=False)
    np.save(out/'actions.npy',actions,allow_pickle=False)
    plan['action_vectors_sha256']=sha256(out/'actions.npy')
    for name,value in [('plan.json',plan),('contract.json',specification()),
        ('model_config.json',model_cfg),('preflight.json',diagnostics)]:
        (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    cohort=json.loads((ROOT/'data/derived/hgwind_40_camera_v2_20261028/render_manifest.json').read_text())
    context=Path(cohort['task_context']).read_text()
    env=dict(os.environ,TERM='xterm',OMNI_KIT_ACCEPT_EULA='YES',PYTHONDONTWRITEBYTECODE='1',
        HGWIND_EFFORT_PROBE_MANIFEST=str(out/'plan.json'),ISAACLAB_PATH=str(lab),
        CAPSTONE_ISAACLAB_PATH=str(lab),ISAAC_SIM_ROOT=str((lab/'_isaac_sim').resolve()),
        ROBOPHYS_CAPSTONE_OFFLINE_TASK='1',ROBOPHYS_TASK_CONTEXT=context)
    packages=[p for p in (lab/'source').iterdir() if p.is_dir()]
    env['PYTHONPATH']=os.pathsep.join([str(ROOT/'src'),*[str(p) for p in packages]])
    with (out/'simulation.log').open('w') as log:
        process=subprocess.Popen(cmd,cwd=Path(plan['worker']).parents[2],env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',start_new_session=True)
        try:
            for line in process.stdout:
                log.write(line);log.flush()
                if any(x in line for x in ['COLLECT ATTEMPT','HGWIND_EFFORT_PROBE_ACTIVE',
                    'SAFETY ABORT','Traceback','RuntimeError','ValueError','INITIALIZED_CONTRACT',
                    'GRIPPER_GAIN_CONTROL','STAGE_LOAD_BARRIER','SETUP_FORCE_CONTROL']):
                    print(line.rstrip(),flush=True)
            code=process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            return 130
    verify_files(Path(plan['source']).parent,plan['source_files'])
    require(sha256(plan['task'])==plan['task_sha256'],'Source scene changed')
    (out/'worker_exit.json').write_text(json.dumps(dict(exit_code=code))+'\n')
    from analyze_hgwind_effort_probe import analyze,find_probe_episodes
    if not find_probe_episodes(out/'worker_output'):
        print('CONTRACT_RUNTIME_FAILED log='+str(out/'simulation.log'));return code or 1
    report=analyze(out)
    print('NEXT stop_after_integrated_gate; inspect_result_then_validate_3_to_5_distinct_episodes',flush=True)
    return code or (0 if report['ball_action_contract_passed'] else 1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sources=parser.add_mutually_exclusive_group()
    sources.add_argument('--baseline-run',type=Path)
    sources.add_argument('--teacher-run',type=Path,help='Fresh verified native expert capture; bounded cohort diagnostic only')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.teacher_run:
        from prepare_hgwind_teacher_contract import prepare_teacher
        prepared=prepare_teacher(args.teacher_run)
    else:
        prepared=prepare(args.baseline_run or DEFAULT_BASELINE)
    return run_prepared(*prepared,run=args.run)


if __name__=='__main__':raise SystemExit(main())
