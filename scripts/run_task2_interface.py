#!/usr/bin/env python3
"""Dry-run or execute one Task-2 episode under the capstone robot interface."""
import argparse
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
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from capstone_vla.task2_bridge import patch_worker


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('joint','cartesian','both'),default='both')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--episode',type=Path,help='explicit single-object Franka 1s1p source episode')
    parser.add_argument('--task',type=Path,help='isolated task copy matching episode task id')
    parser.add_argument('--enforce-frozen-controller',action='store_true')
    parser.add_argument('--velocity-reference',choices=('zero','causal','recorded-diagnostic'),default='zero')
    parser.add_argument('--actuator-profile',choices=('capstone','recorded-diagnostic'),default='capstone')
    parser.add_argument('--bounded-from-command',type=int,default=0,help='diagnostic scheduled solver switch; zero applies throughout')
    parser.add_argument('--ik-solver',choices=('clipped','bounded','adaptive'),default='clipped')
    parser.add_argument('--grasp-mode',choices=('source','physical'),default='source')
    parser.add_argument('--native-rotation-probe',action='store_true',help='record synchronized native hand/object states at every physics tick')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.ik_solver=='adaptive' and args.bounded_from_command!=0:
        parser.error('adaptive solver cannot use --bounded-from-command')
    if args.velocity_reference=='recorded-diagnostic' and args.mode!='joint':
        parser.error('recorded-diagnostic is a joint-only offline controller probe')
    repo=ROOT/'data/derived/hf_pilot_render_repo'
    episode=ROOT/'data/derived/hf_pilot_render_60hz_work/Trajectory/1s1p/grasp_sliding_motion_on_flat_surface_1s1p_g9wvkk1/traj_00001'
    episode=(args.episode or episode).resolve()
    source=episode/'replay.hdf5';meta=json.loads((episode/'meta.json').read_text())
    cfg_path=ROOT/'configs/franka_action_interface.json';cfg=json.loads(cfg_path.read_text())
    worker=repo/'Scripts/expert_data/collect_worker.py'
    patched=patch_worker(worker.read_text())
    if args.native_rotation_probe:
        from capstone_vla.native_rotation_probe import instrument_worker_source
        patched=instrument_worker_source(patched)
    task_id=meta['task']['id'];task=(args.task or repo/'Tasks/1s1p'/f'{task_id}.usd').resolve()
    if task.stem!=task_id:raise ValueError('task id mismatch')
    plan=meta['task_plan']
    if (plan['family']!='1s1p' or len(plan['steps'])!=1 or plan['steps'][0]['target_role']!='A'
            or meta['robot']['type']!='franka' or meta['action_space']!='joint_position_target'
            or meta.get('arm_execution',{}).get('mode')!='position_drive'
            or len(meta['object_paths'])!=1):
        raise ValueError('runner currently supports single-object position-drive Franka 1s1p only')
    control_fingerprint={}
    if args.enforce_frozen_controller:
        import ast
        snapshot=ROOT/'data/derived/task2_interface_checks/20261025_160909/controller_snapshot'
        for rel in ['configs/franka_action_interface.json','src/capstone_vla/franka.py']:
            if (ROOT/rel).read_bytes()!=(snapshot/rel).read_bytes():raise ValueError('frozen controller changed: '+rel)
        rel='src/capstone_vla/task2_bridge.py'
        def definitions(path):
            tree=ast.parse(path.read_text())
            return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
        old,new=definitions(snapshot/rel),definitions(ROOT/rel)
        for name in ['configure_actuators','bounded_dls','ConstraintRecovery','SceneController']:
            if old[name]!=new[name]:raise ValueError('frozen control definition changed: '+name)
            control_fingerprint[name]=hashlib.sha256(new[name].encode()).hexdigest()
    for path in (source,task,Path(cfg['urdf']),Path(cfg['isaaclab_root'])/'isaaclab.sh'):
        if not path.is_file():raise FileNotFoundError(path)
    import h5py
    import numpy as np
    from capstone_vla.franka import FrankaCartesianAdapter
    adapter=FrankaCartesianAdapter(cfg['urdf'])
    dynamic_source=meta.get('arm_execution',{}).get('dynamic_drive',{})
    source_execution_audit={
        'outer_mode':meta.get('arm_execution',{}).get('mode'),
        'nested_dynamic_drive_active':bool(dynamic_source.get('active',False)),
        'nested_physics_callback_count':dynamic_source.get('physics_callback_count'),
        'source_velocity_feedforward_applied':dynamic_source.get('velocity_feedforward_applied'),
        'source_reference_source':dynamic_source.get('reference_source'),
        'saved_command_hz':meta.get('control_hz'),
        'requested_velocity_reference':args.velocity_reference,
        'source_180hz_reference_reconstructed':False,
    }
    with h5py.File(source) as handle:
        actions=handle['actions'][:]
        if args.velocity_reference=='recorded-diagnostic':
            key='obs/joint_drive_velocity'
            if key not in handle:raise ValueError('source has no recorded velocity-reference channel')
            velocity=handle[key][:]
            if velocity.shape!=actions.shape or not np.isfinite(velocity).all():raise ValueError('invalid recorded velocity-reference channel')
    print('SOURCE_EXECUTION_AUDIT '+json.dumps(source_execution_audit),flush=True)
    if dynamic_source.get('velocity_feedforward_applied') is True and args.velocity_reference=='zero':
        print('SOURCE_FIDELITY_LIMIT zero-velocity is an ablation; source used nonzero velocity feedforward',flush=True)
    if args.velocity_reference=='recorded-diagnostic':
        print('SOURCE_FIDELITY_LIMIT saved control-rate velocity endpoints do not reconstruct every original physics substep',flush=True)
    if actions.ndim!=2 or actions.shape[1]!=9 or not np.isfinite(actions).all():
        raise ValueError('expected finite Nx9 source actions')
    if meta['joint_names']!=[f'panda_joint{i}' for i in range(1,8)]+['panda_finger_joint1','panda_finger_joint2']:
        raise ValueError('unexpected source joint order')
    for action in actions:
        adapter.state(action[:7])
        if not (np.all(np.isclose(action[7:],0,atol=1e-5)) or np.all(np.isclose(action[7:],.04,atol=1e-5))):
            raise ValueError('source gripper is not symmetric binary')
    if meta['control_hz']!=60:raise ValueError('source frequency changed')
    step=meta['task_plan']['steps'][0]
    context={'schema_version':1,'source':'capstone_saved_task_metadata_probe','family':'1s1p','row':{
        'Task_ID':task_id,'Skill_1':step['skill'],'Skill_2':'','Phenomenon_1':step['phenomenon'],
        'Phenomenon_2':'','Task_CN':meta['task_plan']['task_cn'],
        'Task_EN':meta['task'].get('instruction_en',''),'build_ok':''}}
    sys.path.insert(0,str(repo/'Scripts'))
    from dataset.dataset_diversity import metadata_grasp_mode
    stamp=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S')
    default_root=(ROOT/'data/derived/task2_interface_checks' if args.episode is None else ROOT/'data/derived/development_validation_20261025/runs'/task_id)
    output=(args.output or default_root/stamp).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('output is not empty')
    modes=('joint','cartesian') if args.mode=='both' else (args.mode,)
    print(f'CPU_PREFLIGHT_OK frames={len(actions)} patch_anchors={10 if args.native_rotation_probe else 7} gripper=binary',flush=True)
    print('OUTPUT',output,flush=True)
    if args.run:output.mkdir(parents=True,exist_ok=True)
    for mode in modes:
        trace=output/f'{mode}_trace.hdf5'
        command=[str(Path(cfg['isaaclab_root'])/'isaaclab.sh'),'-p',str(ROOT/'scripts/sim/replay_task2_interface.py'),
            '--task',str(task),'--replay_trajectories',str(source),'--replay_no_write',
            '--pilot-comparison-out',str(trace),'--out_root',str(output/f'{mode}_worker_output'),
            '--control_hz','60','--grasp_mode',(metadata_grasp_mode(meta) if args.grasp_mode=='source' else 'physical'),'--device','cuda:0',
            '--fast_exit_after_save','--no-visualization','--experience','isaacsim.exp.base.python.kit',
            '--kit_args=--portable-root '+str(ROOT/'data/derived/.cache/isaacsim/linux')
            +' --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/enabled=false'
            +' --/renderer/multiGpu/maxGpuCount=1 --/app/extensions/fsWatcherEnabled=false']
        print('COMMAND',shlex.join(command),flush=True)
        if not args.run:continue
        env=dict(os.environ,TERM='xterm',OMNI_KIT_ACCEPT_EULA='YES',
            CAPSTONE_TASK2_WORKER=str(worker),CAPSTONE_FRANKA_CONFIG=str(cfg_path),CAPSTONE_TASK2_MODE=mode,CAPSTONE_NATIVE_ROTATION_PROBE=('1' if args.native_rotation_probe else '0'),CAPSTONE_BOUNDED_FROM_COMMAND=str(args.bounded_from_command),CAPSTONE_IK_SOLVER=args.ik_solver,CAPSTONE_VELOCITY_REFERENCE=args.velocity_reference,CAPSTONE_ACTUATOR_PROFILE=args.actuator_profile,
            ROBOPHYS_CAPSTONE_OFFLINE_TASK='1',ROBOPHYS_TASK_CONTEXT=json.dumps(context,ensure_ascii=True))
        cache=ROOT/'data/derived/.cache/isaacsim/linux'
        env['OMNI_CONFIG_PATH']=str(cache/'config');env['XDG_CACHE_HOME']=str(cache/'xdg')
        lab=Path(cfg['isaaclab_root'])
        env['PYTHONPATH']=':'.join([str(ROOT/'src'),*[str(p) for p in (lab/'source').iterdir() if p.is_dir()],env.get('PYTHONPATH','')])
        with (output/f'{mode}.log').open('w') as log:
            proc=subprocess.Popen(command,cwd=repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                  text=True,errors='replace',start_new_session=True)
            try:
                for line in proc.stdout:
                    log.write(line);log.flush()
                    if any(s in line for s in ('PILOT COMPARISON','LIVE ACTUATOR-SAFETY','COLLECT ATTEMPT')):
                        print(line.rstrip(),flush=True)
                code=proc.wait()
            except KeyboardInterrupt:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                return 130
        provenance={'source_execution_audit':source_execution_audit,'mode':mode,'native_rotation_probe':args.native_rotation_probe,'grasp_mode_selection':args.grasp_mode,'velocity_reference':args.velocity_reference,'actuator_profile':args.actuator_profile,'ik_solver':args.ik_solver,'bounded_from_command':args.bounded_from_command,'command':command,'exit_code':code,'source_frames':len(actions),
            'patch_sha256':hashlib.sha256(patched.encode()).hexdigest(),
            'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (worker,cfg_path,task,source,ROOT/'src/capstone_vla/native_rotation_probe.py',ROOT/'src/capstone_vla/task2_bridge.py',ROOT/'src/capstone_vla/franka.py',
                 ROOT/'scripts/sim/replay_task2_interface.py',Path(__file__).resolve())},
            'official_success_verified':False,'frozen_control_definitions_sha256':control_fingerprint,'episode':str(episode),'source_expected_task_sha256':meta.get('provenance',{}).get('source_task',{}).get('sha256')}
        (output/f'{mode}_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        if code or not trace.exists():
            print('FAILED log:',output/f'{mode}.log');return code or 2
        with h5py.File(trace) as handle:
            if len(handle['actions'])!=len(actions) or handle.attrs['servo_calls']!=3*len(actions):
                raise RuntimeError('incomplete episode or servo cadence mismatch')
        print('TASK2_TRACE_OK',mode,trace,flush=True)
    if args.run:
        analysis_code=subprocess.run([sys.executable,str(ROOT/'scripts/analyze_task2_interface.py'),str(output),'--episode',str(episode)],cwd=ROOT).returncode
        if analysis_code:return analysis_code
        if args.native_rotation_probe:
            return subprocess.run([sys.executable,str(ROOT/'scripts/analyze_native_rotation.py'),str(output)],cwd=ROOT).returncode
        return 0
    if not args.run:print('DRY_RUN_ONLY: add --run to start simulation')
    return 0

if __name__=='__main__':raise SystemExit(main())
