#!/usr/bin/env python3
"""Bounded live ball policy; default CPU preflight, --run launches simulation."""
import argparse
import h5py
import numpy as np
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

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from capstone_vla.hf_archive import sha256
from capstone_vla.ball_live_protocol import patch_collector
from capstone_vla.cartesian_target_replay import require
from capstone_vla.live_policy_protocol import wait_file
from prepare_hgwind_teacher_contract import prepare_teacher
from analyze_hgwind_closed_loop import analyze

DEFAULT_RUN=ROOT/'data/derived/hgwind_traj18_development_20261031/training_runs/20261031_121355_529372'
TEACHER=ROOT/'data/derived/hgwind_effort_probes/20261029_220130_402255_joint_online-expert_native-expert_target-capture'
SURROGATE=ROOT/'data/derived/hgwind_action_contract_v1/20261029_220618_809794_traj_00018'


def prepare(run,frames,prefix,camera_protocol="legacy",diagnostic_vertical=False,approach_guard=False,approach_follow=False,near_capture=False,capture_depth_mm=0,capture_window=False):
    require(not capture_window or (near_capture and capture_depth_mm==5),'Capture window requires --near-capture --capture-depth-mm 5')
    require(not near_capture or approach_follow,'Near capture requires --approach-follow')
    require(capture_depth_mm in (0,5) and (not capture_depth_mm or near_capture),'Depth requires near capture')
    require(not approach_follow or approach_guard,'Visual follow requires --approach-guard')
    require(2<=frames<=269 and prefix in (1,4),'Frames must be2..269, prefix1 or4')
    prior,_,_,_,lab=prepare_teacher(TEACHER)
    passed=json.loads((SURROGATE/'report.json').read_text())
    require(passed['ball_action_contract_passed'] and passed['independent_grasp_evidence']['status']=='pass','Selected float32 surrogate no longer passes')
    record=json.loads((run/'run_config.json').read_text())
    dataset=Path(record['source_dataset']);meta=json.loads((dataset/'dataset.json').read_text())
    require(meta['episode']=='local/traj_00018' and meta['frames']==269 and not meta['delta_action'],'Wrong development case')
    require(meta['robot_posture_configuration']==prior['robot_posture_configuration'],'Posture configuration changed')
    checkpoint=run/'checkpoint_final';stamp=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
    out=run/'closed_loop_runs'/stamp
    # Command source files are retained only in the coordinator's provenance.
    # No saved action/capture path is passed to the live request executor.
    keys=['worker','task','task_sha256','physics_sha256','bundle_manifest','source','seed','attempt_offset',
          'original_depth_helper','original_depth_bias_m','robot_posture_configuration']
    cfg={k:prior[k] for k in keys}
    cfg.update(schema='physical-vla-lab.ball-live-policy.v1',output=str(out),worker_output=str(out/'worker_output'),
        frames=frames,prefix=prefix,model_seed=90000,control_hz=60,physics_hz=180,
        instruction=meta['instruction'],pinch_offset_local_m=meta['pinch_offset_local_m'],
        camera_config_digest=meta['camera_config_digest'],cameras=meta['cameras'],history_offsets=[-2,0],
        source_actions_used_for_control=False,learned_policy_evaluated=True,heldout_evaluation=False,
        real_time_evaluated=False,synchronous_pause_during_inference=True,
        checkpoint=str(checkpoint),training_run=str(run),pretrained=record['pretrained'],
        pretrained_weights=str(ROOT/'data/models/dynamicvla_dom'/record['pretrained']['revision']/'model.safetensors'),
        checkpoint_hashes={str(p):sha256(p) for p in [checkpoint/'model.safetensors',checkpoint/'config.json',checkpoint/'capstone_normalization.json']},
        patch_sha256=hashlib.sha256(patch_collector(Path(prior['worker']).read_text()).encode()).hexdigest())
    require(camera_protocol in ('legacy','formal-temporal'), 'Unknown camera protocol')
    cfg['camera_protocol']=camera_protocol
    if approach_guard:
        require(not diagnostic_vertical and prefix==1 and camera_protocol=='legacy',
            'Approach experiment requires prefix1/legacy and no recorded-height intervention')
    if camera_protocol=='formal-temporal':
        from capstone_vla.ball_camera_protocol import validated_clamps
        with h5py.File(meta['images'],'r') as image:
            calibration=json.loads(image.attrs['firefly_clamp_json'])
            validated_clamps(calibration)
        cfg['camera_render_calibration']=calibration
        cfg['camera_render_calibration_source']=meta['images']
    if diagnostic_vertical:
        from capstone_vla.ball_vertical_diagnostic import SCHEMA, validate_plan
        require(frames==269 and prefix==4 and camera_protocol=='legacy', 'Diagnostic requires matched full269/prefix4/legacy')
        payload=dataset/'arrays.npz'
        require(sha256(payload)==meta['payload_sha256'], 'Teacher payload changed')
        with np.load(payload,allow_pickle=False) as f:
            goals=f['actions']
            require(goals.shape==(269,29), 'Wrong recorded action shape')
            z=goals[138:180,2].astype(float).tolist()
        cfg['vertical_diagnostic']=dict(schema=SCHEMA,channel=2,coordinate_frame='world',unit='m',
            start_frame=138,stop_frame_exclusive=180,target_z_m=z,
            reference_payload=str(payload),reference_payload_sha256=meta['payload_sha256'],reference_dataset_sha256=sha256(dataset/'dataset.json'),
            rationale='Predeclared near-miss window; diagnostic recorded time, not a deployable observation')
        validate_plan(cfg['vertical_diagnostic'],frames)
        cfg.update(source_actions_used_for_control=True,learned_policy_evaluated=False,
            model_inference_used=True,standalone_policy_evaluation=False)
    # Freeze all native Python/config dependencies, without hashing binary assets again.
    manifest=Path(prior['bundle_manifest']);bundle=manifest.parent
    native=[]
    for row in json.loads(manifest.read_text())['files']:
        p=bundle/row['path']
        if p.suffix in ('.py','.json'):
            require(sha256(p)==row['sha256'],'Frozen worker dependency changed: '+row['path']);native.append(p)
    names=['scripts/run_hgwind_closed_loop.py','scripts/hgwind_policy_server.py','scripts/analyze_hgwind_closed_loop.py',
           'scripts/sim/run_hgwind_policy_worker.py','src/capstone_vla/ball_live_protocol.py',
           'src/capstone_vla/ball_live_runtime.py','src/capstone_vla/ball_vertical_diagnostic.py','src/capstone_vla/ball_action_contract.py','src/capstone_vla/ball_camera_protocol.py',
           'src/capstone_vla/cartesian_packets.py','src/capstone_vla/posture_anchor.py',
           'src/capstone_vla/hgwind_expert_trace.py','src/capstone_vla/hgwind_development_data.py',
           'src/capstone_vla/closed_loop.py','src/capstone_vla/live_policy_protocol.py',
           'src/capstone_vla/cartesian_target_replay.py','src/capstone_vla/cartesian_target_trace.py',
           'src/capstone_vla/task2_bridge.py','src/capstone_vla/franka.py']
    names+=['src/capstone_vla/ball_capture_window.py','src/capstone_vla/ball_near_capture.py','src/capstone_vla/ball_approach_guard.py','src/capstone_vla/ball_visual_geometry.py','src/capstone_vla/ball_approach_follow.py']
    paths=[ROOT/n for n in names]+native+[manifest,Path(prior['task']),Path(prior['original_depth_helper']),
        Path(prior['source']),Path(prior['source']).with_name('meta.json'),SURROGATE/'report.json',dataset/'dataset.json']
    if diagnostic_vertical:paths.append(dataset/'arrays.npz')
    cfg['runtime_hashes']={str(p):sha256(p) for p in dict.fromkeys(paths)}
    repo=Path(prior['worker']).parents[2]
    sys.path.insert(0,str(repo/'Scripts'))
    from pxr import Usd
    from camera.config import config_digest
    from capstone_vla.closed_loop import select_policy_camera_records
    require(config_digest()==meta['camera_config_digest'],'Frozen/training camera profile mismatch')
    stage=Usd.Stage.Open(prior['task']);records=select_policy_camera_records(stage)
    require({r['name'] for r in records}==set(meta['cameras']),'Missing calibrated policy camera')
    cfg['preflight_camera_records']=records
    if approach_guard:
        from capstone_vla.ball_approach_guard import calibration
        source_meta=json.loads(Path(cfg['source']).with_name('meta.json').read_text())
        cfg['approach_guard']=calibration(stage,records,source_meta['policy']['roles']['A'])
        if approach_follow:
            from capstone_vla.ball_approach_follow import FOLLOW_SETTINGS
            cfg['approach_guard']['follow']=dict(FOLLOW_SETTINGS)
        if near_capture:
            from capstone_vla.ball_near_capture import SETTINGS
            cfg['approach_guard']['near_capture']=dict(settings=dict(SETTINGS),extra_depth_m=capture_depth_mm/1000)
        if capture_window:
            from capstone_vla.ball_capture_window import SETTINGS as CAPTURE_SETTINGS
            cfg['approach_guard']['capture_window']=dict(CAPTURE_SETTINGS)
        cfg['policy_postprocessing']=True
    return cfg,lab,repo


def command(cfg,lab):
    meta=json.loads(Path(cfg['source']).with_name('meta.json').read_text())
    return [str(lab/'isaaclab.sh'),'-p',str(ROOT/'scripts/sim/run_hgwind_policy_worker.py'),
        '--task',cfg['task'],'--out_root',cfg['worker_output'],'--episodes','1','--max_attempts','1',
        '--target_successes','0','--seed',str(cfg['seed']),'--attempt_offset',str(cfg['attempt_offset']),
        '--variation',str(meta['variation']),'--control_hz','60','--grasp_mode','physical',
        '--arm_execution','inverse_dynamics','--keep_failed','--no-code-snapshot',
        '--run_id','ball_live_'+Path(cfg['output']).name,'--device','cuda:0','--headless','--no-visualization',
        '--fast_exit_after_save','--episode_s',str(cfg['frames']/60),'--no-expert-config',
        '--experience','isaacsim.exp.base.python.kit','--kit_args=--portable-root '+str(ROOT/'data/derived/.cache/isaacsim/linux')+
        ' --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/enabled=false --/renderer/multiGpu/maxGpuCount=1 --/app/extensions/fsWatcherEnabled=false']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--frames',type=int,default=8)
    p.add_argument('--prefix',type=int,choices=[1,4],default=4);p.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    p.add_argument('--check-load',action='store_true',help='Also strictly reload model on CPU during preflight')
    p.add_argument('--camera-protocol',choices=['legacy','formal-temporal'],default='legacy')
    p.add_argument('--diagnostic-vertical',action='store_true',help='Recorded target z on frames138..179; diagnostic only, not learned-policy success')
    p.add_argument('--approach-guard',action='store_true',help='Experimental RGB-guided finger-height floor and missing-view pose hold; original jaw decoding')
    p.add_argument('--approach-follow',action='store_true',help='Optional near-ball XY visual tracking; requires approach guard, preserves its height and original jaw rules')
    p.add_argument('--near-capture',action='store_true',help='Opt-in clipped-ball localization and continuous follow recovery')
    p.add_argument('--capture-depth-mm',type=int,choices=[0,5],default=0,help='Separate aligned-only depth ablation; requires near capture')
    p.add_argument('--capture-window',action='store_true',help='Finite geometric capture dwell; explicit hybrid controller, requires 5 mm depth')
    a=p.parse_args();cfg,lab,repo=prepare(a.run_dir.resolve(),a.frames,a.prefix,a.camera_protocol,a.diagnostic_vertical,a.approach_guard,a.approach_follow,a.near_capture,a.capture_depth_mm,a.capture_window)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    model_check=[str(ROOT/'.venv_dynamicvla/bin/python'),str(ROOT/'scripts/hgwind_policy_server.py'),'--preflight',str(a.run_dir.resolve())]
    if a.check_load:model_check.append('--check-load')
    subprocess.run(model_check,cwd=ROOT,env=env,check=True)
    cmd=command(cfg,lab)
    print(f'CPU_PREFLIGHT_OK source=local/traj_00018 frames={a.frames} prefix={a.prefix} absolute29=true state30=true live_RGB=true source_actions={a.diagnostic_vertical}',flush=True)
    if a.diagnostic_vertical:print('VERTICAL_DIAGNOSTIC channel=target_position_z frames=138..179 expert_reference=true standalone_policy=false',flush=True)
    if a.approach_guard:print('VISUAL_APPROACH_GUARD RGB_only=true static_radius_m='+str(cfg['approach_guard']['ball_radius_m'])+' missing=pose_hold jaw_threshold=0.5 certified_collision_avoidance=false',flush=True)
    if a.approach_follow:print('VISUAL_FOLLOW near_region_m=0.08 causal_RGB_velocity=true base_height_rule=retained jaw_rule_unchanged=true pure_VLA=false',flush=True)
    if a.near_capture:print('NEAR_CAPTURE clipped_RGB=true recovery_hold=true extra_depth_mm='+str(a.capture_depth_mm)+' native_limits=unchanged jaw=unchanged',flush=True)
    if a.capture_window:print('CAPTURE_WINDOW geometry_triggered=true maximum_frames=42 extra_depth_mm=5 jaw=unchanged grasp_confirmation=false',flush=True)
    print('CAMERA_PROTOCOL '+a.camera_protocol+' physics_and_model_unchanged=true',flush=True)
    print('SIMULATION_MODE synchronous_pause_during_inference; engineering_check_only; real_time=false',flush=True)
    print('COMMAND '+shlex.join(cmd),flush=True)
    if not a.run:print('GPU_NOT_STARTED; add --run');return
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);ipc=out/'ipc';ipc.mkdir()
    path=out/'run_config.json';path.write_text(json.dumps(cfg,indent=2,allow_nan=False)+'\n')
    print('OUTPUT '+str(out),flush=True)
    context=json.loads((ROOT/'data/derived/hgwind_40_camera_v2_20261028/render_manifest.json').read_text())['task_context']
    env.update(TERM='xterm',OMNI_KIT_ACCEPT_EULA='YES',CAPSTONE_BALL_LIVE_CONFIG=str(path),
        ISAACLAB_PATH=str(lab),CAPSTONE_ISAACLAB_PATH=str(lab),ISAAC_SIM_ROOT=str((lab/'_isaac_sim').resolve()),
        ROBOPHYS_CAPSTONE_OFFLINE_TASK='1',ROBOPHYS_TASK_CONTEXT=Path(context).read_text())
    env['PYTHONPATH']=os.pathsep.join([str(ROOT/'src'),*[str(x) for x in (lab/'source').iterdir() if x.is_dir()]])
    server=sim=None;code=-1;error=None
    try:
        with (out/'policy_server.log').open('x') as model_log,(out/'simulation.log').open('x') as sim_log:
            server=subprocess.Popen([str(ROOT/'.venv_dynamicvla/bin/python'),str(ROOT/'scripts/hgwind_policy_server.py'),'--config',str(path)],cwd=ROOT,env=env,stdout=model_log,stderr=subprocess.STDOUT,start_new_session=True)
            wait_file(ipc/'ready',ipc,timeout=180)
            sim=subprocess.Popen(cmd,cwd=repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',start_new_session=True)
            for line in sim.stdout:
                sim_log.write(line);sim_log.flush()
                if any(x in line for x in ['BALL_LIVE','COLLECT ATTEMPT','Traceback','Error','ABORT']):print(line.rstrip(),flush=True)
            code=sim.wait()
    except (Exception,KeyboardInterrupt) as exc:
        error=type(exc).__name__+': '+str(exc);(out/'coordinator_error.txt').write_text(error+'\n')
    finally:
        (ipc/'stop').write_text('stop\n')
        for process in [sim,server]:
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    (out/'worker_exit.json').write_text(json.dumps({'exit_code':code,'coordinator_error':error})+'\n')
    result=analyze(out,code)
    expected='BALL_VERTICAL_DIAGNOSTIC_PASS' if a.diagnostic_vertical else 'BALL_LIVE_INTEGRATION_PASS'
    if result['status']!=expected:raise SystemExit(1)

if __name__=='__main__':main()
