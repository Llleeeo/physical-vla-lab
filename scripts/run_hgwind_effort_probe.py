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
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from capstone_vla.hgwind_cohort import TASK_ID,require,verify_files
from capstone_vla.hf_archive import sha256
from capstone_vla.hgwind_effort_probe import validate_source,patch_worker,build_reference,load_logged_commands
from capstone_vla.recorded_franka import RecordedFrankaAdapter


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('joint','cartesian'),default='joint')
    parser.add_argument('--tracking',choices=('joint','task-space','native-pose','native-expert','logged-command','live-jacobian'),default='joint')
    parser.add_argument('--jacobian-method',choices=('shifted-tool','native-pinch'),default='shifted-tool',help='Live-Jacobian geometry calculation only')
    parser.add_argument('--task-feedforward',choices=('logged','zero'),default='logged',help='One-factor task-acceleration diagnostic; gravity compensation stays unchanged')
    parser.add_argument('--kinematics-run',type=Path,help='Passing native live-kinematics baseline required for zero task feedforward')
    parser.add_argument('--trace-cartesian-targets',action='store_true',help='Read-only original upper Cartesian requests; native-expert only')
    parser.add_argument('--cartesian-target-run',type=Path,help='Passing upper-target capture; regenerate native inner commands from current robot state (expert diagnostic)')
    parser.add_argument('--packet-baseline-run',type=Path,help='Passing upper-feedback or recorded packet run; execute explicit packets without online task/skill steps')
    parser.add_argument('--packet-gain-profile',choices=('recorded','fixed-9-8'),default='recorded',help='Opt-in one-factor constant pose gains; requires a passing recorded packet baseline')
    parser.add_argument('--packet-posture-profile',choices=('recorded','measured','latched'),default='recorded',help='One-factor posture handling; measured/latched require passing fixed-9-8 packet baseline')
    parser.add_argument('--reference',choices=('saved-targets','recorded-motion'),default='saved-targets')
    parser.add_argument('--host',choices=('local','hongtao'),default='local')
    parser.add_argument('--episode',default='traj_00000')
    parser.add_argument('--frames',type=int,default=0,help='0: complete source; positive values are setup smoke only')
    parser.add_argument('--reference-run',type=Path,help='Passing reproduced native-expert run with full control log')
    parser.add_argument('--baseline-run',type=Path,help='Required successful joint baseline for the Cartesian branch')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args();require(args.frames==0 or args.frames>=2,'frames must be 0 or >=2')
    require(args.jacobian_method=='shifted-tool' or args.tracking=='live-jacobian','Native pinch calculation is only a live-Jacobian component diagnostic')
    require(args.task_feedforward!='zero' or (args.tracking=='live-jacobian' and args.jacobian_method=='native-pinch' and args.kinematics_run is not None),'First provide a passing native-pinch --kinematics-run for zero task feedforward')
    require(not args.trace_cartesian_targets or args.tracking=='native-expert','Upper target capture requires the unchanged original expert')
    require(args.cartesian_target_run is None or (args.tracking=='native-expert' and not args.trace_cartesian_targets),'Upper target replay requires native-expert and excludes read-only capture')
    require(args.packet_gain_profile=='recorded' or args.packet_baseline_run is not None,'Fixed gain probe requires a passing packet baseline')
    require(args.packet_posture_profile=='recorded' or (args.packet_baseline_run is not None and args.packet_gain_profile=='fixed-9-8'),'Measured posture requires the passing fixed-9-8 packet baseline')
    gain_reference = None
    if args.packet_baseline_run is not None:
        require(args.tracking=='native-expert' and args.cartesian_target_run is None and not args.trace_cartesian_targets,'Packet probe requires native-expert and a single passing baseline')
        packet_baseline=args.packet_baseline_run.resolve()
        packet_report=json.loads((packet_baseline/'report.json').read_text())
        packet_plan=json.loads((packet_baseline/'plan.json').read_text())
        if packet_report.get('packet_controller_probe_passed') is True:
            require(packet_plan.get('independent_packets') and packet_plan.get('packet_posture_profile','recorded')=='recorded','Reference must use recorded posture')
            if args.packet_posture_profile in ('measured','latched'):
                require(packet_report.get('packet_fixed_gains_probe_passed') is True and packet_plan.get('packet_gain_profile')=='fixed-9-8','First provide a passing fixed-gain packet baseline')
            else:
                require(packet_plan.get('packet_gain_profile','recorded')=='recorded','Gain probe requires the unchanged recorded-gain packet reference')
            gain_reference = dict(root=packet_baseline, report=packet_report, plan=packet_plan)
            args.packet_baseline_run=Path(packet_plan['packet_baseline_report']).parent
            require(sha256(args.packet_baseline_run/'report.json')==packet_plan['packet_baseline_report_sha256'],'Underlying upper-feedback report changed')
            packet_baseline=args.packet_baseline_run
            packet_report=json.loads((packet_baseline/'report.json').read_text())
            packet_plan=json.loads((packet_baseline/'plan.json').read_text())
        require(args.packet_gain_profile=='recorded' or gain_reference is not None,'Fixed gains require the physically passing recorded packet baseline')
        require(packet_report.get('cartesian_feedback_replay_passed') is True,'First supply the passing upper-feedback reference')
        args.cartesian_target_run=Path(packet_plan['cartesian_target_capture']).parent
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
    output=ROOT/'data/derived/hgwind_effort_probes'/(stamp+'_'+args.mode+'_'+({'native-expert':'online-expert','logged-command':'control-log','live-jacobian':'control-log'}.get(args.tracking,args.reference))+'_'+args.tracking)
    if args.task_feedforward=='zero':output=output.with_name(output.name+'_ff-zero')
    if args.cartesian_target_run is not None:output=output.with_name(output.name+'_target-reexecution')
    if args.packet_baseline_run is not None:output=output.with_name(output.name+'_packets')
    if args.trace_cartesian_targets:output=output.with_name(output.name+'_target-capture')
    if args.packet_gain_profile!='recorded':output=output.with_name(output.name+'_'+args.packet_gain_profile)
    if args.packet_posture_profile!='recorded':output=output.with_name(output.name+'_posture-'+args.packet_posture_profile)
    plan={'packet_posture_profile':args.packet_posture_profile,'packet_gain_profile':args.packet_gain_profile,'independent_packets':args.packet_baseline_run is not None,'trace_cartesian_targets':args.trace_cartesian_targets,'task_feedforward':args.task_feedforward,'schema':'physical-vla-lab.hgwind-effort-probe.v1','mode':args.mode,'host':args.host,'episode':args.episode,
          'tracking':args.tracking,'reference':args.reference,'reference_frames':len(reference_actions),'frames':n,'source_frames':len(actions),'source':str(source),'source_files':row['files'],'source_reference_sha256':sha256(source),
          'task':cohort['task_usd'],'task_sha256':cohort['task_sha256'],'physics_sha256':cohort['physics_sha256'],
          'worker':str(worker),'worker_sha256':sha256(worker),'patch_sha256':hashlib.sha256(patched.encode()).hexdigest(),
          'bundle_manifest':str(bundle_manifest),'bundle_manifest_sha256':cohort['bundle_manifest_sha256'],
          'adapter_sha256':{name:sha256(ROOT/name) for name in ['src/capstone_vla/hgwind_effort_probe.py','src/capstone_vla/recorded_franka.py','src/capstone_vla/franka.py','scripts/sim/run_hgwind_effort_probe_worker.py','src/capstone_vla/task2_bridge.py','src/capstone_vla/live_kinematics.py']},
          'controller_config':str(cfg_path),'controller_config_sha256':sha256(cfg_path),'urdf_sha256':adapter.urdf_sha256,
          'output':str(output),'worker_output':str(output/'worker_output'),'seed':meta['base_seed'],'attempt_offset':meta['attempt_index'],
          'reference_fidelity':args.tracking+' tracking through the existing effort feasibility/limit pipeline; original expert operational references are not reconstructed.',
          'baseline_run':str(args.baseline_run.resolve()) if args.baseline_run else None,'training_eligible':False}
    if args.tracking in ('native-expert','logged-command','live-jacobian'):
        require(args.mode=='joint' and args.reference=='saved-targets' and args.frames==0,'Native expert uses its online commands and full 12 s attempt budget')
        require(meta['capstone_expert_control']['profile']=='smooth_lift_v1','Original profile is not the frozen smooth-lift configuration')
        from capstone_vla.hgwind_expert_trace import patch_collector,patch_profile_wrapper
        bundle=Path(cohort['bundle']);wrapper=bundle/'scripts/sim/collect_hgwind_worker.py'
        patch_profile_wrapper(wrapper.read_text())
        original_text=worker.read_text();anchor='"skill_terminations": _jsonify(skill_terminations),'
        require(original_text.count(anchor)==1,'Original profile metadata insertion changed')
        original_text=original_text.replace(anchor,anchor+'\n            "capstone_expert_control": CAPSTONE_PROFILE_SUMMARY(pol),')
        patched=patch_collector(original_text) if args.tracking=='native-expert' else patch_worker(original_text)
        if args.packet_baseline_run is not None:
            from capstone_vla.cartesian_packets import patch_packet_collector
            patched=patch_packet_collector(patched)
            for name in ['src/capstone_vla/cartesian_packets.py','src/capstone_vla/cartesian_target_trace.py']:
                plan['adapter_sha256'][str(ROOT/name)]=sha256(ROOT/name)
        elif args.trace_cartesian_targets or args.cartesian_target_run is not None:
            from capstone_vla.cartesian_target_trace import patch_capture_collector
            patched=patch_capture_collector(patched)
            path=ROOT/'src/capstone_vla/cartesian_target_trace.py';plan['adapter_sha256'][str(path)]=sha256(path)
        plan.update(reference='online-expert',reference_frames=720,frames=720,patch_sha256=hashlib.sha256(patched.encode()).hexdigest(),
            original_profile_wrapper=str(wrapper),original_depth_helper=str(ROOT/'scripts/run_hgwind_depth_worker.py'),original_depth_bias_m=meta['capstone_expert_control']['depth_bias']['requested_downward_goal_bias_m'],
            reference_fidelity='Fresh frozen online expert with source smooth-lift/depth profile; saved source actions are not executed.')
        require(0<=plan['original_depth_bias_m']<=.002,'Unexpected original depth profile')
        for path in [ROOT/'src/capstone_vla/hgwind_expert_trace.py',ROOT/'scripts/sim/run_hgwind_native_expert_worker.py',wrapper,ROOT/'scripts/run_hgwind_depth_worker.py',bundle/'src/capstone_vla/hgwind_control.py']:
            plan['adapter_sha256'][str(path)]=sha256(path)
    if args.cartesian_target_run is not None:
        from capstone_vla.cartesian_target_trace import audit_capture
        from capstone_vla.cartesian_target_replay import CartesianTargetReplayer
        previous=args.cartesian_target_run.resolve();prior_path=previous/'report.json'
        prior=json.loads(prior_path.read_text());prior_plan=json.loads((previous/'plan.json').read_text())
        require(prior.get('cartesian_target_capture_verified') is True and prior.get('native_expert_reference_passed') is True,'First provide a passing original upper-target capture')
        for key in ['source_reference_sha256','task_sha256','physics_sha256','worker_sha256','controller_config_sha256','urdf_sha256']:
            require(prior_plan[key]==plan[key],'Upper capture provenance differs: '+key)
        for dependency in [wrapper,Path(plan['original_depth_helper']),bundle/'src/capstone_vla/hgwind_control.py']:
            require(prior['execution_fingerprints']['adapter_sha256'].get(str(dependency))==sha256(dependency),'Original profile/depth changed')
        target_path=previous/'cartesian_target_inputs.json';capture=json.loads(target_path.read_text())
        segments=json.loads((previous/'original_expert_control_references.json').read_text())
        require(audit_capture(capture,segments,len(capture['frames']))['capture_alignment_passed'],'Raw upper inputs no longer align')
        CartesianTargetReplayer(capture)
        require(len(capture['frames'])==len(actions),'Upper capture length differs from source')
        # Verify the actual native API and safety implementation against the
        # frozen bundle, not merely the bundle manifest file itself.
        frozen_files={entry['path']:entry['sha256'] for entry in json.loads(bundle_manifest.read_text())['files']}
        for dependency in [worker.parent/'expert.py',worker.parent/'collision_safety.py', *([worker.parent/'skills/catch.py'] if args.packet_baseline_run is not None else [])]:
            expected=frozen_files.get(str(dependency.relative_to(bundle)))
            require(expected is not None and sha256(dependency)==expected,'Frozen native feedback implementation changed: '+dependency.name)
            plan['adapter_sha256'][str(dependency)]=expected
        import ast,inspect
        tree=ast.parse((worker.parent/'expert.py').read_text())
        method=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='operational_space_pinch_action')
        stub=ast.FunctionDef(name='native_signature',args=method.args,body=[ast.Pass()],decorator_list=[])
        namespace={};exec(compile(ast.fix_missing_locations(ast.Module(body=[stub],type_ignores=[])),'<native-upper-signature>','exec'),namespace)
        parameters=set(inspect.signature(namespace['native_signature']).parameters)-{'self'}
        require(all(set(event['inputs'])==parameters for event in capture['calls']),'Captured upper parameters differ from frozen native signature')
        plan.update(cartesian_target_capture=str(target_path),cartesian_target_capture_sha256=sha256(target_path),
            cartesian_target_capture_report=str(prior_path),cartesian_target_capture_report_sha256=sha256(prior_path),
            cartesian_target_source_segments=str(previous/'original_expert_control_references.json'),
            cartesian_target_source_segments_sha256=sha256(previous/'original_expert_control_references.json'),
            reference_fidelity='Recorded upper goals/motion/settings through current-state native feedback; live native safety callbacks, expert call timing/gripper/terminal joint fallback retained. No saved inner q/qdot/Jacobian/OSC commands execute. Not a standalone VLA controller.')
        path=ROOT/'src/capstone_vla/cartesian_target_replay.py';plan['adapter_sha256'][str(path)]=sha256(path)
    if args.packet_baseline_run is not None:
        from capstone_vla.cartesian_packets import build_packets
        for key in ['source_reference_sha256','task_sha256','physics_sha256','worker_sha256','controller_config_sha256','urdf_sha256','cartesian_target_capture_sha256']:
            require(packet_plan[key]==plan[key],'Packet reference differs: '+key)
        require(not packet_plan.get('independent_packets'),'Packet baseline must be the retained-context successful comparison')
        posture_anchor = None
        if args.packet_posture_profile=='latched':
            from capstone_vla.posture_anchor import sampled_posture_configuration,compute_anchor,validate_anchor_bounds
            plan['robot_posture_configuration']=sampled_posture_configuration(meta['trajectory_variation'])
            posture_anchor=compute_anchor(positions[0,:7],plan['robot_posture_configuration'])
            validate_anchor_bounds(posture_anchor,adapter.lower,adapter.upper)
            plan['adapter_sha256'][str(ROOT/'src/capstone_vla/posture_anchor.py')]=sha256(ROOT/'src/capstone_vla/posture_anchor.py')
        packets=build_packets(capture,actions[:,7:],args.packet_gain_profile,args.packet_posture_profile,posture_anchor)
        if gain_reference is not None:
            from capstone_vla.cartesian_packets import audit_gain_profile
            reference_root=gain_reference['root']
            prior_plan=gain_reference['plan']
            for key in ['source_reference_sha256','task_sha256','physics_sha256','worker_sha256','controller_config_sha256','urdf_sha256','cartesian_target_capture_sha256']:
                require(prior_plan[key]==plan[key],'Gain probe reference differs: '+key)
            saved_packets=json.loads((reference_root/'packet_requests.json').read_text())
            require(saved_packets==build_packets(capture,actions[:,7:],prior_plan.get('packet_gain_profile','recorded'),prior_plan.get('packet_posture_profile','recorded')),'Passing packet stream differs from current recorded inputs')
            if args.packet_posture_profile in ('measured','latched'):
                from capstone_vla.cartesian_packets import audit_posture_profile
                require(audit_posture_profile(saved_packets,packets,args.packet_posture_profile,posture_anchor),'Not an isolated/effective posture change')
                if args.packet_posture_profile=='latched':
                    require(all(x['inputs']['posture_target'] is None or np.array_equal(np.asarray(x['inputs']['posture_target']),posture_anchor) for x in saved_packets),'Source posture cannot be reconstructed from the proposed initialization')
                plan.update(packet_posture_reference_report=str(reference_root/'report.json'),packet_posture_reference_report_sha256=sha256(reference_root/'report.json'),
                    packet_posture_reference_requests=str(reference_root/'packet_requests.json'),packet_posture_reference_requests_sha256=sha256(reference_root/'packet_requests.json'),
                    changed_posture_frames=sum(a!=b for a,b in zip(saved_packets,packets)))
            else:
                require(audit_gain_profile(saved_packets,packets,args.packet_gain_profile),'Not an isolated/effective gain change')
                plan.update(packet_gain_reference_report=str(reference_root/'report.json'),packet_gain_reference_report_sha256=sha256(reference_root/'report.json'),
                    packet_gain_reference_requests=str(reference_root/'packet_requests.json'),packet_gain_reference_requests_sha256=sha256(reference_root/'packet_requests.json'),
                    changed_gain_frames=sum(a!=b for a,b in zip(saved_packets,packets)))
        plan.update(frames=len(packets),reference_frames=len(packets),packet_baseline_report=str(packet_baseline/'report.json'),
            packet_baseline_report_sha256=sha256(packet_baseline/'report.json'),
            reference_fidelity='Explicit recorded upper/motion/settings and binary gripper packets; no online task/skill step or expert joint tail. Terminal pose held through native Cartesian feedback. Non-target geometry safety only; not a learned policy contract.')
    if args.packet_gain_profile!='recorded':
        plan['reference_fidelity']='Only position/orientation feedback gains are fixed at 9/8 for every packet. All other recorded numeric settings/motion, jaws, callback mode, terminal hold and native limits are retained. Not a learned-policy contract.'
    if args.packet_posture_profile=='measured':
        plan['reference_fidelity']='Only posture_target is changed to None from the passing fixed-9/8 baseline; native redundancy handling uses current measured joints. Other numeric motion/settings/jaws/callbacks/hold and native limits retained. Not a learned-policy contract.'
    if args.packet_posture_profile=='latched':
        plan['reference_fidelity']='Posture generated once from live initial robot joints plus declared sampled IK/homotopy offsets, held through all frames. Other fixed-gain baseline requests/settings/limits retained. No per-frame saved posture is executed. Still recorded target motion, not a learned policy.'
    if args.tracking in ('logged-command','live-jacobian'):
        require(args.reference_run is not None,'Provide the passing original-expert reference run')
        ref_root=args.reference_run.resolve();ref_report=ref_root/'report.json';ref=json.loads(ref_report.read_text())
        require(ref.get('native_expert_reference_passed') is True and ref['source_reference_sha256']==plan['source_reference_sha256'] and ref['task_sha256']==plan['task_sha256'] and ref['execution_fingerprints']['worker_sha256']==plan['worker_sha256'],'Native reference is not a passing matching source/scene/worker')
        for dependency in [Path(plan['original_profile_wrapper']),Path(plan['original_depth_helper']),bundle/'src/capstone_vla/hgwind_control.py']:
            require(ref['execution_fingerprints']['adapter_sha256'].get(str(dependency))==sha256(dependency),'Original profile/depth implementation differs from the logged run')
        control_log=ref_root/'original_expert_control_references.json';targets,_,_=load_logged_commands(control_log,positions,velocities)
        plan.update(reference='control-log',frames=len(targets),reference_frames=len(targets),control_log=str(control_log),control_log_sha256=sha256(control_log),control_log_report=str(ref_report),control_log_report_sha256=sha256(ref_report),reference_fidelity='Diagnostic replay of logged original q/qdot/OSC references, including cached original Jacobians; not a deployed policy controller.')
    if args.tracking=='live-jacobian':
        plan['jacobian_method']=args.jacobian_method
        if args.jacobian_method=='native-pinch':
            from capstone_vla.live_kinematics import compile_native_pinch_jacobian
            native_source=worker.parent/'expert.py'
            compile_native_pinch_jacobian(native_source.read_text())
            plan.update(native_jacobian_source=str(native_source),native_jacobian_source_sha256=sha256(native_source))
            plan['adapter_sha256'][str(native_source)]=sha256(native_source)
        require(all(x['operational_space_reference'] is None or (np.asarray(x['operational_space_reference']['jacobian']).shape==(6,7) and 'secondary_jacobian' not in x['operational_space_reference']) for x in json.loads(Path(plan['control_log']).read_text())),'Live component check requires six-dimensional task references')
        from capstone_vla.live_kinematics import validate_logged_task_metric
        # Catch explicitly passes max(r_A, 1e-6) as its coupled orientation
        # length scale. This is a frozen controller configuration, not a live
        # object observation. Require the source task metrics to agree.
        scale=max(float(meta['success_parameters']['1']['r_A']),1.e-6)
        metric_rows=json.loads(Path(plan['control_log']).read_text())
        metric_count=validate_logged_task_metric(metric_rows,scale)
        plan.update(operational_orientation_length_scale_m=scale,
            operational_metric_provenance='frozen Catch operational_orientation_length_scale_m=max(success_parameters[1].r_A,1e-6)',
            source_coupled_task_metric_rows_verified=metric_count)
        plan['reference_fidelity']='Current robot pinch Jacobian with original coupled angular length scale at each control boundary; original joint goals, twist, feed-forward and constraints retained for component diagnosis. Not a standalone policy controller.'
    if args.task_feedforward=='zero':
        previous=args.kinematics_run.resolve();previous_report=previous/'report.json';prior=json.loads(previous_report.read_text())
        prior_plan=json.loads((previous/'plan.json').read_text());prior_tag=prior['motor_reference_audit']
        require(prior.get('live_kinematics_component_passed') is True and prior_tag.get('jacobian_method')=='native-pinch'
                and prior_tag.get('task_feedforward','logged')=='logged','Kinematics baseline is not a passing native geometry case with original feedforward')
        for key in ['source_reference_sha256','task_sha256','native_jacobian_source_sha256','control_log_sha256','controller_config_sha256','worker_sha256','urdf_sha256']:
            require(prior_plan[key]==plan[key],'Kinematics baseline mismatch: '+key)
        plan.update(kinematics_baseline_report=str(previous_report),kinematics_baseline_report_sha256=sha256(previous_report),
            expected_removed_feedforward_commands=sum(bool(np.any(np.asarray(x['operational_space_reference']['feedforward_task_acceleration'])!=0)) for x in metric_rows if x['operational_space_reference'] is not None))
        plan['reference_fidelity']='Only task acceleration feedforward is zeroed, including its within-segment twist extrapolation; original effort gravity compensation, live native geometry and all other logged references/limits remain. Not a policy rollout.'
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
    if args.tracking in ('native-expert','logged-command','live-jacobian'):
        command[2]=str(ROOT/'scripts/sim/run_hgwind_native_expert_worker.py')
        if args.tracking=='native-expert':
            i=command.index('--repair-action-reference');del command[i:i+2]
        command.extend(['--episode_s',str(plan['frames']/60) if plan.get('independent_packets') else '12','--no-expert-config'])
        n=plan['frames']
    print(f'CPU_PREFLIGHT_OK mode={args.mode} reference={plan['reference']} source={args.host}/{args.episode} frames={n} effort_pipeline=true primary_tracking={args.tracking} exact_state_replay=false',flush=True)
    if args.tracking=='live-jacobian':print(f"TASK_METRIC_OK orientation_length_scale_m={plan['operational_orientation_length_scale_m']} coupled_rows={plan['source_coupled_task_metric_rows_verified']} jacobian_method={args.jacobian_method}",flush=True)
    if args.task_feedforward=='zero':print(f"TASK_FEEDFORWARD_ABLATION mode=zero nonzero_source_commands={plan['expected_removed_feedforward_commands']} gravity_compensation=unchanged",flush=True)
    if args.packet_baseline_run is not None:print('PACKET_CONTROLLER no_online_skill_step=true explicit_gripper=true terminal_cartesian_hold=true',flush=True)
    if args.packet_gain_profile!='recorded' and args.packet_posture_profile=='recorded':print('PACKET_FIXED_GAINS profile='+args.packet_gain_profile+' changed_frames='+str(plan['changed_gain_frames'])+' all_other_inputs=exact limits=unchanged',flush=True)
    if args.packet_posture_profile=='measured':print('PACKET_MEASURED_POSTURE changed_frames='+str(plan['changed_posture_frames'])+' gains=fixed-9-8 all_other_inputs=exact native_measured_state_audit=true',flush=True)
    if args.packet_posture_profile=='latched':print('PACKET_LATCHED_POSTURE changed_frames='+str(plan['changed_posture_frames'])+' initial_robot_plus_declared_variation=true one_initialization=true gains=fixed-9-8',flush=True)
    if args.cartesian_target_run is not None and args.packet_baseline_run is None:print('CARTESIAN_TARGET_REEXECUTION enabled=true saved_inner_outputs=false native_safety_and_fallback=retained',flush=True)
    if args.trace_cartesian_targets:print('CARTESIAN_TARGET_CAPTURE enabled=true motor_overrides=false callable_descriptors_only=true',flush=True)
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
