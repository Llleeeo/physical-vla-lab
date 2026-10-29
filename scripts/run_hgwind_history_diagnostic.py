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


def prepare(native_backtrace=False, stage_load_barrier=False):
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
    out = ROOT / 'data/derived/hgwind_history_diagnostics' / stamp
    plan.pop('native_debugger', None)
    plan.update(output=str(out), worker_output=str(out / 'worker_output'), attempt_offset=58,
        history_attempts=ATTEMPTS, native_debugger_requested=native_backtrace, stage_load_barrier=stage_load_barrier,
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
    # Resolve original failed as well as successful episodes by attempt identity.
    sources = {}
    for meta_path in Path(plan['source']).parent.parent.glob('*/meta.json'):
        meta = json.loads(meta_path.read_text())
        index = int(meta['attempt_index'])
        if index in ATTEMPTS:
            require(index not in sources, 'Duplicate original attempt')
            sources[index] = (meta_path.parent, meta)
    require(sorted(sources) == ATTEMPTS, 'Missing original attempt evidence')
    cases = {}
    baseline_meta = sources[60][1]
    for index in ATTEMPTS:
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
    os.environ['ROBOPHYS_CAPSTONE_OFFLINE_TASK'] = '1'
    os.environ['ROBOPHYS_TASK_CONTEXT'] = Path(manifest['task_context']).read_text()
    sys.path.insert(0, str(Path(plan['worker']).parents[1]))
    from pxr import Usd
    from dataset.scene_signature import stage_physics_signature
    require(stage_physics_signature(Usd.Stage.Open(plan['task']))['sha256'] == plan['physics_sha256'], 'Physical scene changed')
    command = [str(lab/'isaaclab.sh'), '-p', str(ROOT/'scripts/sim/run_hgwind_history_worker.py'),
        '--task', plan['task'], '--out_root', plan['worker_output'], '--episodes','3','--max_attempts','3',
        '--target_successes','0','--seed','202610280','--attempt_offset','58','--variation','1.0','--control_hz','60',
        '--grasp_mode','physical','--arm_execution','inverse_dynamics','--catch-targeting-mode','auto',
        '--keep_failed','--no-code-snapshot','--run_id','hgwind_history_'+stamp,
        '--device','cuda:0','--headless','--no-visualization','--fast_exit_after_save',
        '--episode_s','12','--no-expert-config','--experience','isaacsim.exp.base.python.kit',
        '--kit_args=--portable-root '+str(ROOT/'data/derived/.cache/isaacsim/linux')+' --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/enabled=false --/renderer/multiGpu/maxGpuCount=1 --/app/extensions/fsWatcherEnabled=false']
    if native_backtrace:
        require(shutil.which('gdb') is not None, 'Existing system GDB missing')
    print('CPU_PREFLIGHT_OK attempts=58,59,60 one_worker=true max_attempts=3 originals=720,363,319 native_reset_and_safety=retained',flush=True)
    return plan, command, lab


def finish(out, code, render=False):
    from analyze_hgwind_effort_probe import analyze
    plan = json.loads((out/'plan.json').read_text())
    episodes = index_saved_episodes(plan['worker_output'], TASK_ID)
    result = dict(schema='physical-vla-lab.hgwind-history.v1', worker_exit=code,
        requested_attempts=ATTEMPTS, saved_attempts=sorted(episodes), cases=[], learned_policy_evaluated=False,
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
                zero_configured_dt_calls=sum(x['physics_dt_s']==0 for x in calls),sha256=sha256(trace))
        row['raw_source_sha256'] = {name:sha256(episode/name) for name in ['meta.json','replay.hdf5','trajectory.hdf5']}
        result['cases'].append(row)
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    order_path=out/'attempt_order.jsonl'
    order=[json.loads(line)['attempt_index'] for line in order_path.read_text().splitlines()] if order_path.exists() else []
    result['observed_attempt_order']=order
    stage_evidence = result['stage_load_barrier_evidence'] or {}
    result['stage_load_barrier_verified'] = bool(stage_evidence.get('ready') is True and stage_evidence.get('held_script_instances') == 0 and stage_evidence.get('initial_timeline_time') == stage_evidence.get('final_timeline_time')) if plan.get('stage_load_barrier') else None
    startup_ok = not plan.get('stage_load_barrier') or result['stage_load_barrier_verified'] is True
    result['complete'] = startup_ok and code == 0 and sorted(episodes)==ATTEMPTS and order==ATTEMPTS and all(x['analysis_complete'] and (not render or x['render_complete']) for x in result['cases'])
    target = next((x for x in result['cases'] if x['attempt_index']==60),{})
    result['target_grasp_status'] = target.get('grasp', 'not_evaluated')
    result['physical_success'] = bool(startup_ok and target.get('analysis_complete') and target.get('grasp')=='pass' and target.get('runtime_safety') and target.get('initial_state_match'))
    if result['complete']:
        result['interpretation']=('Startup attach sequencing changed under an opt-in workaround; this is a modified-startup history experiment. ' if plan.get('stage_load_barrier') else '')+'Compare original and fresh startup/native arrays before attributing any result to history; this remains a development expert experiment.'
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
    print('HISTORY_DIAGNOSTIC_'+('COMPLETED' if result['complete'] else 'INCOMPLETE')+' report='+str(out/'report.json'),flush=True)
    return 0 if result['complete'] else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--native-backtrace',action='store_true')
    parser.add_argument('--stage-load-barrier',action='store_true',help='Experimental stopped-timeline stage attach barrier before robot USD edits')
    parser.add_argument('--render',action='store_true',help='Opt in to five-view review videos; default analyzes only')
    parser.add_argument('--finish',type=Path,help='Analyze/render a completed run without launching physics again')
    args=parser.parse_args()
    if args.finish:
        require(not args.run,'--finish never launches collection')
        out=args.finish.resolve();out.relative_to(ROOT/'data/derived/hgwind_history_diagnostics')
        return finish(out,json.loads((out/'worker_exit.json').read_text())['exit_code'],render=args.render)
    plan,command,lab=prepare(args.native_backtrace,args.stage_load_barrier)
    plan['render_requested'] = args.render
    if args.stage_load_barrier:print('STARTUP_WORKAROUND stage_load_barrier=true efficacy=unverified physics_parameters_unchanged=true',flush=True)
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
                if any(s in line for s in ['HISTORY_ATTEMPT','COLLECT ATTEMPT','SAFETY ABORT','Traceback','Error:','terminate called','what():','Aborted','CAPSTONE_NATIVE','DONE attempts','FATAL']):print(line.rstrip(),flush=True)
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
