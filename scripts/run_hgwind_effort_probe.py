#!/usr/bin/env python3
"""Prepare one isolated effort-command probe; simulation requires --run."""
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

import h5py

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from capstone_vla.hgwind_cohort import TASK_ID,require,verify_files
from capstone_vla.hf_archive import sha256
from capstone_vla.hgwind_effort_probe import validate_source,patch_worker,build_reference,load_logged_commands
from capstone_vla.recorded_franka import RecordedFrankaAdapter


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('joint','cartesian'),default='joint')
    parser.add_argument('--tracking',choices=('joint','task-space','native-pose','native-expert','logged-command'),default='joint')
    parser.add_argument('--reference',choices=('saved-targets','recorded-motion'),default='saved-targets')
    parser.add_argument('--host',choices=('local','hongtao'),default='local')
    parser.add_argument('--episode',default='traj_00000')
    parser.add_argument('--frames',type=int,default=0,help='0: complete source; positive values are setup smoke only')
    parser.add_argument('--reference-run',type=Path,help='Passing reproduced native-expert run with full control log')
    parser.add_argument('--baseline-run',type=Path,help='Required successful joint baseline for the Cartesian branch')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args();require(args.frames==0 or args.frames>=2,'frames must be 0 or >=2')
    manifest_path=ROOT/'data/derived/hgwind_40_camera_v2_20261028/render_manifest.json'
    cohort=json.loads(manifest_path.read_text());matches=[r for r in cohort['rows'] if r['host']==args.host and r['episode']==args.episode]
    require(len(matches)==1,'Unknown source episode');row=matches[0];source=Path(row['source'])/'replay.hdf5';meta=json.loads((source.parent/'meta.json').read_text())
    verify_files(source.parent,row['files']);cfg_path=ROOT/'configs/franka_action_interface.json';cfg=json.loads(cfg_path.read_text())
    worker=Path(cohort['bundle'])/'data/derived/hgwind_current_baseline_20261027/repo/Scripts/expert_data/collect_worker.py'
    bundle_manifest=Path(cohort['bundle'])/'bundle_manifest.json'
    require(sha256(bundle_manifest)==cohort['bundle_manifest_sha256'],'Frozen bundle changed')
    require(sha256(cohort['task_usd'])==cohort['task_sha256'],'Task USD changed')
    lab=Path(cfg['isaaclab_root']);require((lab/'VERSION').read_text().strip()=='2.2.1' and ((lab/'_isaac_sim').resolve()/'VERSION').read_text().startswith('4.5.0'),'SDK version changed')
    adapter=RecordedFrankaAdapter(cfg['urdf'],meta['end_effector'])
    patched=patch_worker(worker.read_text())
    with h5py.File(source,'r') as f:
        actions=f['actions'][:];times=f['timestamp'][:];positions=f['obs/joint_pos'][:];velocities=f['obs/joint_vel'][:]
    reference_actions,_=build_reference(actions,positions,velocities,args.reference)
    require(args.reference!='recorded-motion' or args.mode=='joint','Recorded-motion is a separate joint-tracking diagnostic; Cartesian support is not implemented')
    n=len(reference_actions) if args.frames==0 else args.frames;require(n<=len(reference_actions),'Smoke length exceeds source')
    stamp=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
    output=ROOT/'data/derived/hgwind_effort_probes'/(stamp+'_'+args.mode+'_'+({'native-expert':'online-expert','logged-command':'control-log'}.get(args.tracking,args.reference))+'_'+args.tracking)
    plan={'schema':'physical-vla-lab.hgwind-effort-probe.v1','mode':args.mode,'host':args.host,'episode':args.episode,
          'tracking':args.tracking,'reference':args.reference,'reference_frames':len(reference_actions),'frames':n,'source_frames':len(actions),'source':str(source),'source_files':row['files'],'source_reference_sha256':sha256(source),
          'task':cohort['task_usd'],'task_sha256':cohort['task_sha256'],'physics_sha256':cohort['physics_sha256'],
          'worker':str(worker),'worker_sha256':sha256(worker),'patch_sha256':hashlib.sha256(patched.encode()).hexdigest(),
          'bundle_manifest':str(bundle_manifest),'bundle_manifest_sha256':cohort['bundle_manifest_sha256'],
          'adapter_sha256':{name:sha256(ROOT/name) for name in ['src/capstone_vla/hgwind_effort_probe.py','src/capstone_vla/recorded_franka.py','src/capstone_vla/franka.py','scripts/sim/run_hgwind_effort_probe_worker.py','src/capstone_vla/task2_bridge.py']},
          'controller_config':str(cfg_path),'controller_config_sha256':sha256(cfg_path),'urdf_sha256':adapter.urdf_sha256,
          'output':str(output),'worker_output':str(output/'worker_output'),'seed':meta['base_seed'],'attempt_offset':meta['attempt_index'],
          'reference_fidelity':args.tracking+' tracking through the existing effort feasibility/limit pipeline; original expert operational references are not reconstructed.',
          'baseline_run':str(args.baseline_run.resolve()) if args.baseline_run else None,'training_eligible':False}
    if args.tracking in ('native-expert','logged-command'):
        require(args.mode=='joint' and args.reference=='saved-targets' and args.frames==0,'Native expert uses its online commands and full 12 s attempt budget')
        require(meta['capstone_expert_control']['profile']=='smooth_lift_v1','Original profile is not the frozen smooth-lift configuration')
        from capstone_vla.hgwind_expert_trace import patch_collector,patch_profile_wrapper
        bundle=Path(cohort['bundle']);wrapper=bundle/'scripts/sim/collect_hgwind_worker.py'
        patch_profile_wrapper(wrapper.read_text())
        original_text=worker.read_text();anchor='"skill_terminations": _jsonify(skill_terminations),'
        require(original_text.count(anchor)==1,'Original profile metadata insertion changed')
        original_text=original_text.replace(anchor,anchor+'\n            "capstone_expert_control": CAPSTONE_PROFILE_SUMMARY(pol),')
        patched=patch_collector(original_text) if args.tracking=='native-expert' else patch_worker(original_text)
        plan.update(reference='online-expert',reference_frames=720,frames=720,patch_sha256=hashlib.sha256(patched.encode()).hexdigest(),
            original_profile_wrapper=str(wrapper),original_depth_helper=str(ROOT/'scripts/run_hgwind_depth_worker.py'),original_depth_bias_m=meta['capstone_expert_control']['depth_bias']['requested_downward_goal_bias_m'],
            reference_fidelity='Fresh frozen online expert with source smooth-lift/depth profile; saved source actions are not executed.')
        require(0<=plan['original_depth_bias_m']<=.002,'Unexpected original depth profile')
        for path in [ROOT/'src/capstone_vla/hgwind_expert_trace.py',ROOT/'scripts/sim/run_hgwind_native_expert_worker.py',wrapper,ROOT/'scripts/run_hgwind_depth_worker.py',bundle/'src/capstone_vla/hgwind_control.py']:
            plan['adapter_sha256'][str(path)]=sha256(path)
    if args.tracking=='logged-command':
        require(args.reference_run is not None,'Provide the passing original-expert reference run')
        ref_root=args.reference_run.resolve();ref_report=ref_root/'report.json';ref=json.loads(ref_report.read_text())
        require(ref.get('native_expert_reference_passed') is True and ref['source_reference_sha256']==plan['source_reference_sha256'] and ref['task_sha256']==plan['task_sha256'] and ref['execution_fingerprints']['worker_sha256']==plan['worker_sha256'],'Native reference is not a passing matching source/scene/worker')
        for dependency in [Path(plan['original_profile_wrapper']),Path(plan['original_depth_helper']),bundle/'src/capstone_vla/hgwind_control.py']:
            require(ref['execution_fingerprints']['adapter_sha256'].get(str(dependency))==sha256(dependency),'Original profile/depth implementation differs from the logged run')
        control_log=ref_root/'original_expert_control_references.json';targets,_,_=load_logged_commands(control_log,positions,velocities)
        plan.update(reference='control-log',frames=len(targets),reference_frames=len(targets),control_log=str(control_log),control_log_sha256=sha256(control_log),control_log_report=str(ref_report),control_log_report_sha256=sha256(ref_report),reference_fidelity='Diagnostic replay of logged original q/qdot/OSC references, including cached original Jacobians; not a deployed policy controller.')
    validate_source(plan,meta,actions,times)
    if args.mode=='cartesian':
        require(args.baseline_run is not None,'First run/inspect the joint baseline; provide --baseline-run for Cartesian')
        baseline=json.loads((args.baseline_run/'report.json').read_text())
        require(baseline['mode']=='joint' and baseline['baseline_ready_for_cartesian']
                and baseline.get('reference','saved-targets')==args.reference and baseline.get('tracking','joint')==args.tracking
                and baseline['source_reference_sha256']==plan['source_reference_sha256']
                and baseline['task_sha256']==plan['task_sha256']
                and baseline['execution_fingerprints']=={k:plan[k] for k in ['worker_sha256','patch_sha256','controller_config_sha256','urdf_sha256','adapter_sha256']},'Baseline is incomplete, failed or refers to another source')
    # CPU USD/context audit; this never creates SimulationApp.
    os.environ['ROBOPHYS_CAPSTONE_OFFLINE_TASK']='1';context=Path(cohort['task_context']).read_text();os.environ['ROBOPHYS_TASK_CONTEXT']=context
    sys.path.insert(0,str(worker.parents[1]))
    from pxr import Usd
    from dataset.scene_signature import stage_physics_signature
    require(stage_physics_signature(Usd.Stage.Open(plan['task']))['sha256']==plan['physics_sha256'],'Physical scene changed')
    command=[str(lab/'isaaclab.sh'),'-p',str(ROOT/'scripts/sim/run_hgwind_effort_probe_worker.py'),
             '--task',plan['task'],'--repair-action-reference',str(source),'--out_root',plan['worker_output'],
             '--episodes','1','--max_attempts','1','--seed',str(meta['base_seed']),'--attempt_offset',str(meta['attempt_index']),
             '--variation',str(meta['variation']),'--control_hz','60','--grasp_mode','physical','--arm_execution','inverse_dynamics',
             '--keep_failed','--run_id','hgwind_effort_probe_'+stamp,'--device','cuda:0','--headless','--no-visualization','--fast_exit_after_save',
             '--experience','isaacsim.exp.base.python.kit',
             '--kit_args=--portable-root '+str(ROOT/'data/derived/.cache/isaacsim/linux')+' --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/enabled=false --/renderer/multiGpu/maxGpuCount=1 --/app/extensions/fsWatcherEnabled=false']
    if args.tracking in ('native-expert','logged-command'):
        command[2]=str(ROOT/'scripts/sim/run_hgwind_native_expert_worker.py')
        if args.tracking=='native-expert':
            i=command.index('--repair-action-reference');del command[i:i+2]
        command.extend(['--episode_s','12','--no-expert-config'])
        n=plan['frames']
    print(f'CPU_PREFLIGHT_OK mode={args.mode} reference={plan['reference']} source={args.host}/{args.episode} frames={n} effort_pipeline=true primary_tracking={args.tracking} exact_state_replay=false',flush=True)
    print('REFERENCE_LIMIT '+plan['reference_fidelity'],flush=True);print('OUTPUT '+str(output),flush=True);print('COMMAND '+shlex.join(command),flush=True)
    if not args.run:print('DRY_RUN_ONLY: add --run');return 0
    output.mkdir(parents=True,exist_ok=False);(output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    env=dict(os.environ,TERM='xterm',OMNI_KIT_ACCEPT_EULA='YES',PYTHONDONTWRITEBYTECODE='1',HGWIND_EFFORT_PROBE_MANIFEST=str(output/'plan.json'),
             ISAACLAB_PATH=str(lab),CAPSTONE_ISAACLAB_PATH=str(lab),ISAAC_SIM_ROOT=str((lab/'_isaac_sim').resolve()))
    packages=[p for p in (lab/'source').iterdir() if p.is_dir()]
    env['PYTHONPATH']=os.pathsep.join([str(ROOT/'src'),*[str(p) for p in packages]])
    with (output/'simulation.log').open('w') as log:
        process=subprocess.Popen(command,cwd=worker.parents[2],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',start_new_session=True)
        try:
            for line in process.stdout:
                log.write(line);log.flush()
                if any(x in line for x in ['COLLECT ATTEMPT','HGWIND_EFFORT_PROBE_ACTIVE','SAFETY ABORT','Traceback','RuntimeError','ValueError']):print(line.rstrip(),flush=True)
            code=process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            return 130
    verify_files(source.parent,row['files']);require(sha256(plan['task'])==plan['task_sha256'],'Source scene changed')
    (output/'worker_exit.json').write_text(json.dumps({'exit_code':code})+'\n')
    from analyze_hgwind_effort_probe import analyze
    if code:
        print('PROBE_RUNTIME_FAILED log='+str(output/'simulation.log'))
        if any((output/'worker_output').rglob('replay.hdf5')):analyze(output)
        return code
    report=analyze(output)
    if report['baseline_ready_for_cartesian']:
        print('NEXT_CARTESIAN_COMMAND '+shlex.join([sys.executable,str(Path(__file__).resolve()),'--mode','cartesian','--host',args.host,'--episode',args.episode,'--baseline-run',str(output),'--run']))
    return 0


if __name__=='__main__':raise SystemExit(main())
