#!/usr/bin/env python3
"""Three original attempts in one worker; CPU preflight unless --run."""
import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
from zoneinfo import ZoneInfo
import h5py
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from capstone_vla.hf_archive import sha256
from capstone_vla.hgwind_cohort import TASK_ID, require, verify_files
from capstone_vla.hgwind_history import assemble, index_saved_episodes, compare_motion
from capstone_vla.contract_batch import inspect_review_videos
from diagnose_hgwind_startup_history import LOG, worker_context
FRESH = ROOT / 'data/derived/hgwind_effort_probes/20261029_225526_640181_joint_online-expert_native-expert_target-capture'
ATTEMPTS = [58, 59, 60]
HISTORY = ROOT / 'data/derived/hgwind_history_diagnostics/20261029_234931_549595'


def prepare(native_backtrace=False, stage_load_barrier=False, fresh_control=False, omit_setup_force_once=False, trace_gripper_drives=False, restore_gripper_gains=False):
    require(not fresh_control or stage_load_barrier, 'Fresh control requires the matched stage-load barrier')
    require(not omit_setup_force_once or (fresh_control and stage_load_barrier), 'Omission requires matched-startup fresh60')
    require(not restore_gripper_gains or (fresh_control and omit_setup_force_once and trace_gripper_drives), "Gain control requires observed fresh60 plus force omission")
    attempts = [60] if fresh_control else list(ATTEMPTS)
    from capstone_vla.inotify_preflight import require_headroom
    watcher_preflight = require_headroom()
    plan = json.loads((FRESH / 'plan.json').read_text())
    require(plan['attempt_offset'] == 60 and plan['seed'] == 202610280, 'Fresh reference differs')
    require(json.loads((FRESH / 'report.json').read_text())['recorded_frames'] == 169, 'Fresh reference changed')
    for name, digest in plan['adapter_sha256'].items():
        require(sha256(ROOT / name) == digest, 'Reference adapter changed: ' + name)
    for pathkey, hashkey in [('task','task_sha256'), ('worker','worker_sha256'), ('bundle_manifest','bundle_manifest_sha256'), ('controller_config','controller_config_sha256')]:
        require(sha256(plan[pathkey]) == plan[hashkey], 'Reference dependency changed: ' + pathkey)
    context = worker_context(LOG.read_text(errors='replace').splitlines(), 60)
    require(context['attempts_in_this_worker_through_source'] == ATTEMPTS, 'Original worker sequence differs')
    manifest = json.loads((ROOT / 'data/derived/hgwind_40_camera_v2_20261028/render_manifest.json').read_text())
    require(manifest['task_sha256'] == plan['task_sha256'], 'Camera scene differs')
    lab = Path(json.loads(Path(plan['controller_config']).read_text())['isaaclab_root'])
    require((lab / 'VERSION').read_text().strip() == '2.2.1' and ((lab / '_isaac_sim').resolve() / 'VERSION').read_text().startswith('4.5.0'), 'SDK changed')
    require(not os.environ.get('CONDA_PREFIX'), 'Run from the existing project interpreter outside Conda')
    stamp = datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
    out = ROOT / 'data/derived/hgwind_history_diagnostics' / (stamp + ('_fresh60' if fresh_control else '') + ('_omit1' if omit_setup_force_once else '') + ('_gains' if restore_gripper_gains else ''))
    plan.pop('native_debugger', None)
    plan.update(output=str(out), worker_output=str(out / 'worker_output'), attempt_offset=attempts[0], fresh_control=fresh_control,
        history_attempts=attempts, inotify_preflight=watcher_preflight, native_debugger_requested=native_backtrace, stage_load_barrier=stage_load_barrier,
        fresh_reference=str(FRESH), fresh_plan_sha256=sha256(FRESH/'plan.json'), fresh_report_sha256=sha256(FRESH/'report.json'),
        original_worker_context=context, original_log_sha256=sha256(LOG),
        reference_fidelity='One frozen native worker, original attempts58/59/60, normal reset and safety exits; no source actions execute.')
    for name in ['src/capstone_vla/hgwind_history.py','scripts/sim/run_hgwind_history_worker.py','scripts/run_hgwind_history_diagnostic.py']:
        plan['adapter_sha256'][str(ROOT/name)] = sha256(ROOT/name)
    if stage_load_barrier:
        name = 'src/capstone_vla/scene_load_barrier.py'
        plan['adapter_sha256'][str(ROOT/name)] = sha256(ROOT/name)
    import hashlib
    patched = assemble(plan)
    plan['patch_sha256'] = hashlib.sha256(patched.encode()).hexdigest()
    compile(patched, '<history-preflight>', 'exec')
    if fresh_control:
        matched_plan = json.loads((HISTORY/'plan.json').read_text())
        matched_report = json.loads((HISTORY/'report.json').read_text())
        require(matched_report['complete'] and matched_report['stage_load_barrier_verified'] and matched_report['target_grasp_status']=='pass' and all(row['comparison_to_original']['all_core_arrays_exact'] for row in matched_report['cases']), 'Matched history result is not verified')
        require(plan['patch_sha256']==matched_plan['patch_sha256'], 'Fresh control changes the native collector assembly')
        for name in ['task_sha256','physics_sha256','worker_sha256','controller_config_sha256','original_depth_bias_m','seed']:
            require(plan[name]==matched_plan[name], 'Fresh/history dependency differs: '+name)
        for name, expected_hash in matched_plan['adapter_sha256'].items():
            if Path(name).name in ('hgwind_history.py','run_hgwind_history_diagnostic.py'):
                continue  # Sequence selection/reporting only; assembled collector must still match above.
            require(sha256(ROOT/name)==expected_hash, 'Matched-history runtime dependency changed: '+name)
        plan['matched_history_reference'] = str(HISTORY)
        plan['matched_history_plan_sha256'] = sha256(HISTORY/'plan.json')
        plan['matched_history_report_sha256'] = sha256(HISTORY/'report.json')
        plan['reference_fidelity'] = 'Fresh native worker, original attempt60 only, same verified stage-load barrier/collector as history58/59/60; no saved source actions execute.'
    worker_entry = ROOT/'scripts/sim/run_hgwind_history_worker.py'
    if omit_setup_force_once:
        from capstone_vla.setup_force_control import prepare_intervention
        plan['setup_force_control'] = prepare_intervention(plan, ROOT)
        worker_entry = ROOT/'scripts/sim/run_hgwind_setup_force_worker.py'
        for path in [ROOT/'src/capstone_vla/setup_force_control.py', worker_entry]:
            plan['adapter_sha256'][str(path)] = sha256(path)
        plan['reference_fidelity'] = 'Matched fresh60 with one explicitly omitted pre-recording wind force call; modified forcing history, no training/VLA claim.'
    if trace_gripper_drives:
        plan['trace_gripper_drives'] = True
        worker_entry = ROOT/'scripts/sim/run_hgwind_gripper_probe_worker.py'
        for path in [ROOT/'src/capstone_vla/gripper_drive_trace.py',worker_entry]:
            plan['adapter_sha256'][str(path)] = sha256(path)
    # Resolve original failed as well as successful episodes by attempt identity.
    sources = {}
    for meta_path in Path(plan['source']).parent.parent.glob('*/meta.json'):
        meta = json.loads(meta_path.read_text())
        index = int(meta['attempt_index'])
        if index in attempts:
            require(index not in sources, 'Duplicate original attempt')
            sources[index] = (meta_path.parent, meta)
    require(sorted(sources) == attempts, 'Missing original attempt evidence')
    cases = {}
    baseline_meta = sources[60][1]
    for index in attempts:
        source, meta = sources[index]
        for key in ['base_seed','variation','grasp_mode','fix_joint','fixedforce','control_hz','object_paths','robot_limit_profile','sim','success_parameters']:
            require(meta[key] == baseline_meta[key], 'Original configuration changed across history: ' + key)
        require(meta['capstone_expert_control']['profile'] == 'smooth_lift_v1' and meta['capstone_expert_control']['depth_bias']['requested_downward_goal_bias_m'] == .002, 'Original depth/profile changed')
        require(meta['runtime_safety']['passed'] is True, 'Original prefix contains a safety abort; cannot continue same worker')
        require(meta['arm_execution']['mode'] == 'operational_space_inverse_dynamics_effort', 'Original motor mode changed')
        with h5py.File(source/'replay.hdf5','r') as f:
            t = f['timestamp'][:]
            require(len(t) == meta['num_steps'] and f['actions'].shape == (len(t),9) and np.allclose(np.diff(t),1/60,rtol=0,atol=1e-7), 'Original clock/action schema changed')
        case = copy.deepcopy(plan)
        case.pop('history_attempts')
        case.update(output=str(out / ('attempt_%05d' % index)), worker_output=str(out / ('attempt_%05d' % index) / 'worker_output'),
            source=str(source/'replay.hdf5'), source_reference_sha256=sha256(source/'replay.hdf5'), source_frames=meta['num_steps'],
            source_files=[dict(path=name,bytes=(source/name).stat().st_size,sha256=sha256(source/name)) for name in ['meta.json','replay.hdf5','trajectory.hdf5']],
            episode=source.name, attempt_offset=index, expected_episode_seed=meta['episode_seed'], expected_variation=meta['trajectory_variation'])
        cases[str(index)] = case
    plan['history_cases'] = cases
    if restore_gripper_gains:
        from capstone_vla.gripper_gain_control import prepare_control
        plan['gripper_gain_control'] = prepare_control(plan, ROOT)
        worker_entry = ROOT/'scripts/sim/run_hgwind_gripper_gain_worker.py'
        for path in [ROOT/'src/capstone_vla/gripper_gain_control.py',worker_entry]:
            plan['adapter_sha256'][str(path)] = sha256(path)
        for case in cases.values():
            case['gripper_gain_control'] = plan['gripper_gain_control']
            case['adapter_sha256'] = dict(plan['adapter_sha256'])
    os.environ['ROBOPHYS_CAPSTONE_OFFLINE_TASK'] = '1'
    os.environ['ROBOPHYS_TASK_CONTEXT'] = Path(manifest['task_context']).read_text()
    sys.path.insert(0, str(Path(plan['worker']).parents[1]))
    from pxr import Usd
    from dataset.scene_signature import stage_physics_signature
    require(stage_physics_signature(Usd.Stage.Open(plan['task']))['sha256'] == plan['physics_sha256'], 'Physical scene changed')
    command = [str(lab/'isaaclab.sh'), '-p', str(worker_entry),
        '--task', plan['task'], '--out_root', plan['worker_output'], '--episodes',str(len(attempts)),'--max_attempts',str(len(attempts)),
        '--target_successes','0','--seed','202610280','--attempt_offset',str(attempts[0]),'--variation','1.0','--control_hz','60',
        '--grasp_mode','physical','--arm_execution','inverse_dynamics','--catch-targeting-mode','auto',
        '--keep_failed','--no-code-snapshot','--run_id','hgwind_history_'+stamp,
        '--device','cuda:0','--headless','--no-visualization','--fast_exit_after_save',
        '--episode_s','12','--no-expert-config','--experience','isaacsim.exp.base.python.kit',
        '--kit_args=--portable-root '+str(ROOT/'data/derived/.cache/isaacsim/linux')+' --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/enabled=false --/renderer/multiGpu/maxGpuCount=1 --/app/extensions/fsWatcherEnabled=false']
    if native_backtrace:
        require(shutil.which('gdb') is not None, 'Existing system GDB missing')
    print('CPU_PREFLIGHT_OK attempts='+','.join(map(str,attempts))+' one_worker=true max_attempts='+str(len(attempts))+' originals='+','.join(str(sources[i][1]['num_steps']) for i in attempts)+' native_reset_and_safety=retained',flush=True)
    if fresh_control:print('MATCHED_STARTUP_CONTROL fresh60=true prior_attempts=0 collector_hash=exact history_reference=verified',flush=True)
    return plan, command, lab


def finish(out, code, render=False):
    from analyze_hgwind_effort_probe import analyze
    plan = json.loads((out/'plan.json').read_text())
    attempts = plan.get('history_attempts', ATTEMPTS)
    episodes = index_saved_episodes(plan['worker_output'], TASK_ID, expected=attempts)
    result = dict(schema='physical-vla-lab.hgwind-history.v1', worker_exit=code,
        requested_attempts=attempts, condition='fresh60_single_setup_force_omission' if plan.get('setup_force_control') else ('fresh60_matched_startup' if plan.get('fresh_control') else 'history58_59_60'), saved_attempts=sorted(episodes), cases=[], learned_policy_evaluated=False,
        complete=False, physical_success=False, render_requested=render, stage_load_barrier_requested=bool(plan.get('stage_load_barrier')), stage_load_barrier_evidence=json.loads((out/'stage_load_barrier.json').read_text()) if (out/'stage_load_barrier.json').exists() else None, native_backtrace=str(out/'native_backtrace.log') if (out/'native_backtrace.log').exists() else None, interpretation='Incomplete history cannot answer the attempt60 history hypothesis.')
    for index, episode in sorted(episodes.items()):
        case = plan['history_cases'][str(index)]; location = Path(case['output'])
        row = dict(attempt_index=index, original_episode=case['episode'], run=str(location), analysis_complete=False,render_complete=False)
        try:
            actual = json.loads((episode/'meta.json').read_text())
            target = location/'worker_output/1s1p'/TASK_ID/episode.name
            target.mkdir(parents=True,exist_ok=True)
            for name in ['meta.json','replay.hdf5','trajectory.hdf5']:
                dest = target/name
                if dest.exists(): require(sha256(dest)==sha256(episode/name), 'Existing review copy differs')
                else: shutil.copy2(episode/name,dest)
            (location/'worker_exit.json').write_text(json.dumps({'exit_code':code})+'\n')
            require(actual['episode_seed'] == case['expected_episode_seed'] and actual['trajectory_variation'] == case['expected_variation'], 'Sampled conditions differ')
            verify_files(Path(case['source']).parent, case['source_files'])
            report = analyze(location)
            row.update(analysis_complete=True, grasp=report['independent_grasp_evidence']['status'],
                initial_state_match=report['initial_state_match'], runtime_safety=report['runtime_safety_passed'],
                comparison_to_original=compare_motion(case['source'],target/'replay.hdf5'))
            if index == 60:
                fresh = json.loads((FRESH/'report.json').read_text())
                row['comparison_to_fresh'] = compare_motion(Path(fresh['probe_episode'])/'replay.hdf5',target/'replay.hdf5')
                if plan.get('fresh_control'):
                    reference = Path(plan['matched_history_reference'])
                    require(sha256(reference/'report.json')==plan['matched_history_report_sha256'], 'Matched history report changed')
                    matched = json.loads((reference/'attempt_00060/report.json').read_text())
                    row['comparison_to_matched_history'] = compare_motion(Path(matched['probe_episode'])/'replay.hdf5',target/'replay.hdf5')
        except Exception as exc:
            row['analysis_error'] = repr(exc)
        row['render_status'] = 'skipped_by_request'
        if render:
            try:
                rendered = subprocess.run([sys.executable,str(ROOT/'scripts/render_hgwind_effort_probe.py'),'--probe-run',str(location),'--run'],cwd=ROOT)
                require(rendered.returncode == 0, 'Rendering failed; use --finish --render to retry delivery without recollection')
                row['videos'] = inspect_review_videos(ROOT,location)
                row['render_complete'] = True
                row['render_status'] = 'complete'
            except Exception as exc:
                row['render_error'] = repr(exc)
                row['render_status'] = 'failed'
        trace = location/'startup_force_calls.jsonl'
        if trace.exists():
            calls = [json.loads(line) for line in trace.read_text().splitlines()]
            row['force_trace'] = dict(calls=len(calls),pre_recording_calls=sum(x['frame'] is None for x in calls),
                zero_configured_dt_calls=sum(x['physics_dt_s']==0 for x in calls),
                omitted_calls=sum(x['status']=='omitted_by_setup_force_control' for x in calls),
                pre_recording_returned_calls=sum(x['frame'] is None and x['status']=='returned' for x in calls),
                returned_native_calls=sum(x['status']=='returned' for x in calls),sha256=sha256(trace))
        if plan.get('trace_gripper_drives'):
            from capstone_vla.gripper_drive_trace import trace_status
            row['gripper_drive_observation'] = trace_status(location/'gripper_drive_trace.jsonl')
        row['raw_source_sha256'] = {name:sha256(episode/name) for name in ['meta.json','replay.hdf5','trajectory.hdf5']}
        result['cases'].append(row)
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    order_path=out/'attempt_order.jsonl'
    order=[json.loads(line)['attempt_index'] for line in order_path.read_text().splitlines()] if order_path.exists() else []
    result['observed_attempt_order']=order
    stage_evidence = result['stage_load_barrier_evidence'] or {}
    result['stage_load_barrier_verified'] = bool(stage_evidence.get('ready') is True and stage_evidence.get('held_script_instances') == 0 and stage_evidence.get('initial_timeline_time') == stage_evidence.get('final_timeline_time')) if plan.get('stage_load_barrier') else None
    startup_ok = not plan.get('stage_load_barrier') or result['stage_load_barrier_verified'] is True
    if plan.get('setup_force_control'):
        from capstone_vla.setup_force_control import audit_intervention
        result['setup_force_control'] = audit_intervention(plan)
        startup_ok = startup_ok and result['setup_force_control']['verified']
    if plan.get('gripper_gain_control'):
        from capstone_vla.gripper_gain_control import audit_control
        result['gripper_gain_control'] = audit_control(plan)
        startup_ok = startup_ok and result['gripper_gain_control']['verified']
        result['condition'] = 'fresh60_omit1_post_usd_gripper_gain_reapplication'
    result['complete'] = startup_ok and code == 0 and sorted(episodes)==attempts and order==attempts and all(x['analysis_complete'] and (not render or x['render_complete']) for x in result['cases'])
    target = next((x for x in result['cases'] if x['attempt_index']==60),{})
    result['target_grasp_status'] = target.get('grasp', 'not_evaluated')
    result['physical_success'] = bool(startup_ok and target.get('analysis_complete') and target.get('grasp')=='pass' and target.get('runtime_safety') and target.get('initial_state_match'))
    # A saved safety-aborted fresh control still measures the first-step/history
    # contrast. Keep process completion and grasp success as separate outcomes.
    comparisons = ['comparison_to_original', 'comparison_to_fresh']
    if plan.get('fresh_control'):
        comparisons.append('comparison_to_matched_history')
    result['comparison_evaluable'] = bool(
        startup_ok and sorted(episodes) == attempts and order == attempts
        and all(x.get('analysis_complete') and x.get('initial_state_match')
                for x in result['cases'])
        and all(target.get(key, {}).get('compared_native_rows', 0) >= 2
                for key in comparisons))
    if result['comparison_evaluable'] and not result['complete']:
        result['interpretation'] = (
            'Saved startup/history comparison is evaluable despite worker failure. '
            'The execution is incomplete and physical outcome remains separately reported; '
            'do not discard the matched first-step evidence or infer a unique mechanism.')
    if result['complete']:
        if plan.get('fresh_control'):
            result['interpretation']='Matched-startup fresh60 control: compare with history60 and the earlier fresh60; no learned policy is evaluated.'
        else:
            result['interpretation']=('Startup attach sequencing changed under an opt-in workaround; this is a modified-startup history experiment. ' if plan.get('stage_load_barrier') else '')+'Compare original and fresh startup/native arrays before attributing any result to history; this remains a development expert experiment.'
    if plan.get('setup_force_control'):
        result['interpretation'] = ('One intentionally modified wind-force-history experiment; '
            'require intervention verification, then compare first native ball step and full original arrays. '
            'No unique mechanism or general fix follows from a single outcome.')
        result.update(modified_forcing_history=True, training_eligible=False)
        print('SETUP_FORCE_CONTROL verified='+str(result['setup_force_control']['verified']).lower(), flush=True)
        matched = target.get('comparison_to_matched_history', {})
        print('SETUP_FORCE_COMPARISON '+json.dumps(dict(
            first_step_history_velocity_delta_m_s=matched.get('first_step_ball_velocity_delta_m_s'),
            all_history_core_arrays_exact=matched.get('all_core_arrays_exact'),
            pre_recording_native_calls=target.get('force_trace', {}).get('pre_recording_returned_calls'))), flush=True)
    if plan.get('gripper_gain_control'):
        result['interpretation'] = 'One measured warm-gain reproduction after native frame0 USD setup, combined with existing explicit setup-force omission. Both finger DOFs driven with mimic retained. Diagnostic only; require native readbacks and independent grasp, not a production fix or learned-policy result.'
        print('GRIPPER_GAIN_CONTROL verified='+str(result['gripper_gain_control']['verified']).lower()
            +' grasp='+result['target_grasp_status'],flush=True)
    if plan.get('trace_gripper_drives'):
        result['gripper_observation_available'] = bool(result['cases']) and all(
            row.get('gripper_drive_observation',{}).get('verified') for row in result['cases'])
        result['passive_observer_noninterference_verified'] = bool(result['comparison_evaluable']) and all(
            row.get('comparison_to_original',{}).get('all_core_arrays_exact') for row in result['cases']) if not plan.get('fresh_control') else None
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Worker-history video review','','Raw trajectories retained. Rendering is optional; skipped views have not been visually reviewed.','']
    for row in result['cases']:
        lines.append('## Attempt '+str(row['attempt_index'])+' — independent grasp '+row.get('grasp','unavailable'))
        lines.append('')
        for name,video in row.get('videos',{}).get('videos',{}).items(): lines.append('- ['+name+']('+video['path']+')')
        if row['render_status'] == 'skipped_by_request':lines.append('Rendering skipped by request; raw trajectory retained.')
        elif not row['render_complete']:lines.append('Video delivery incomplete: '+row.get('render_error','unknown'))
        lines.append('')
    (out/'VIDEO_REVIEW.md').write_text('\n'.join(lines))
    print('HISTORY_COMPARISON evaluable='+str(result['comparison_evaluable']).lower()
        +' execution_complete='+str(result['complete']).lower()
        +' grasp='+result['target_grasp_status'],flush=True)
    print('HISTORY_DIAGNOSTIC_'+('COMPLETED' if result['complete'] else 'INCOMPLETE')+' report='+str(out/'report.json'),flush=True)
    return 0 if result['complete'] else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--native-backtrace',action='store_true')
    parser.add_argument('--fresh-control',action='store_true',help='Run original attempt60 alone using the matched history startup barrier')
    parser.add_argument('--stage-load-barrier',action='store_true',help='Experimental stopped-timeline stage attach barrier before robot USD edits')
    parser.add_argument('--omit-setup-force-once',action='store_true',help='Experimental: omit the pinned second pre-recording wind call in fresh60 only')
    parser.add_argument('--trace-gripper-drives',action='store_true',help='Passive native/USD finger-drive observations; no setters or extra simulation steps')
    parser.add_argument('--restore-gripper-gains',action='store_true',help='Diagnostic: reapply measured warm finger gains after original frame0 setup; mimic retained')
    parser.add_argument('--render',action='store_true',help='Opt in to five-view review videos; default analyzes only')
    parser.add_argument('--finish',type=Path,help='Analyze/render a completed run without launching physics again')
    args=parser.parse_args()
    if args.finish:
        require(not args.run,'--finish never launches collection')
        out=args.finish.resolve();out.relative_to(ROOT/'data/derived/hgwind_history_diagnostics')
        return finish(out,json.loads((out/'worker_exit.json').read_text())['exit_code'],render=args.render)
    plan,command,lab=prepare(args.native_backtrace,args.stage_load_barrier,args.fresh_control,args.omit_setup_force_once,args.trace_gripper_drives,args.restore_gripper_gains)
    plan['render_requested'] = args.render
    if args.stage_load_barrier:print('STARTUP_WORKAROUND stage_load_barrier=true efficacy=unverified physics_parameters_unchanged=true',flush=True)
    if args.omit_setup_force_once:print('SETUP_FORCE_CONTROL omit_event=1 modified_forcing_history=true efficacy=unverified',flush=True)
    if args.trace_gripper_drives:print('GRIPPER_TRACE requested=true setters=false',flush=True)
    if args.restore_gripper_gains:print('GRIPPER_GAIN_CONTROL requested=true one_reapplication=true production_fix=false',flush=True)
    print('RENDER_MODE '+('enabled' if args.render else 'skipped; raw trajectories retained'),flush=True)
    print('COMMAND '+shlex.join(command),flush=True)
    if not args.run:return 0
    out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False)
    if args.native_backtrace:
        from capstone_vla.native_debugger import write_debugger
        plan['native_debugger']=write_debugger(out,(lab/'_isaac_sim').resolve()/'kit/python/bin/python3',shutil.which('gdb'))
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    env=dict(os.environ,TERM='xterm',OMNI_KIT_ACCEPT_EULA='YES',PYTHONDONTWRITEBYTECODE='1',
        HGWIND_EFFORT_PROBE_MANIFEST=str(out/'plan.json'),ISAACLAB_PATH=str(lab),CAPSTONE_ISAACLAB_PATH=str(lab),ISAAC_SIM_ROOT=str((lab/'_isaac_sim').resolve()))
    env['PYTHONPATH']=os.pathsep.join([str(ROOT/'src'),*[str(p) for p in (lab/'source').iterdir() if p.is_dir()]])
    if args.native_backtrace:env['PYTHONEXE']=plan['native_debugger']['wrapper']
    print('HISTORY_OUTPUT '+str(out),flush=True)
    with (out/'simulation.log').open('w') as log:
        process=subprocess.Popen(command,cwd=Path(plan['worker']).parents[2],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',start_new_session=True)
        try:
            for line in process.stdout:
                log.write(line);log.flush()
                if any(s in line for s in ['HISTORY_ATTEMPT','COLLECT ATTEMPT','SAFETY ABORT','Traceback','Error:','terminate called','what():','Aborted','CAPSTONE_NATIVE','STAGE_LOAD_BARRIER','SETUP_FORCE_CONTROL','GRIPPER_TRACE','GRIPPER_GAIN_CONTROL','DONE attempts','FATAL']):print(line.rstrip(),flush=True)
            code=process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            code=130
    (out/'worker_exit.json').write_text(json.dumps({'exit_code':code})+'\n')
    if code == 130:
        print('HISTORY_INTERRUPTED saved outputs retained; resume delivery with --finish '+str(out),flush=True)
        return 130
    return finish(out,code,render=args.render)


if __name__=='__main__':raise SystemExit(main())
