#!/usr/bin/env python3
"""Three fixed development cases, paired physics runs and all-outcome videos.

Default: CPU-only preflight. --run executes serially on the local GPU. Resume
never repeats a finished physics stage, including failed stages; renders retry.
"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from capstone_vla.contract_batch import select_cases,saved_episode,inspect_review_videos,write_review_index,classify_gripper_schema
from capstone_vla.cartesian_target_replay import require
from capstone_vla.hf_archive import sha256
from capstone_vla.hgwind_cohort import verify_files
MANIFEST=ROOT/'data/derived/hgwind_40_camera_v2_20261028/render_manifest.json'
EXISTING=[('passing_latched_baseline',ROOT/'data/derived/hgwind_effort_probes/20261029_202630_260880_joint_online-expert_native-expert_target-reexecution_packets_fixed-9-8_posture-latched'),
    ('failed_29d_contract',ROOT/'data/derived/hgwind_action_contract_v1/20261029_210114_208078')]


def teacher_command(case,run=False):
    cmd=[sys.executable,str(ROOT/'scripts/run_hgwind_effort_probe.py'),'--tracking','native-expert',
        '--trace-cartesian-targets','--host','local','--episode',case['episode']]
    return cmd+(['--run'] if run else [])


def render_command(run,execute=False):
    cmd=[sys.executable,str(ROOT/'scripts/render_hgwind_effort_probe.py'),'--probe-run',str(run)]
    return cmd+(['--run'] if execute else [])


def preflight():
    require(shutil.which('ffprobe') is not None,'ffprobe required to verify every MP4')
    manifest=json.loads(MANIFEST.read_text())
    schema_review=[]
    for row in manifest['rows']:
        if row['host']=='local' and row['episode']!='traj_00000':
            verify_files(row['source'],row['files'])
            row.update(classify_gripper_schema(row['source']))
            schema_review.append({k:row[k] for k in ['episode','attempt_index','gripper_schema_supported','schema_reason']})
    cases=select_cases(manifest['rows'])
    print('SCHEMA_SCOPE local_new='+str(len(schema_review))+' supported='+str(sum(x['gripper_schema_supported'] for x in schema_review))+' unsupported='+str(sum(not x['gripper_schema_supported'] for x in schema_review))+' selection_before_new_outcomes=true',flush=True)
    for row in cases:
        verify_files(row['source'],row['files'])
        result=subprocess.run(teacher_command(row),cwd=ROOT,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        require(result.returncode==0,'Teacher CPU preflight failed: '+result.stdout[-3000:])
        print('CPU_CASE_OK '+row['episode']+' attempt='+str(row['attempt_index'])+' frames='+str(row['frames']),flush=True)
    for label,run in EXISTING:
        saved_episode(ROOT,run)
        result=subprocess.run(render_command(run),cwd=ROOT,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        require(result.returncode==0,'Render CPU preflight failed: '+result.stdout[-3000:])
        print('CPU_RENDER_OK '+label,flush=True)
    return cases,schema_review


def persist(out,state):
    target=out/'batch.json';tmp=out/'batch.json.tmp'
    tmp.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n');tmp.replace(target)
    write_review_index(out,state)


def logged(cmd,log_path,on_output=None):
    with Path(log_path).open('w') as log:
        process=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,errors='replace',env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        try:
            for line in process.stdout:
                log.write(line);log.flush()
                if on_output and line.startswith('OUTPUT '):on_output(line.strip()[7:])
                if any(k in line for k in ['OUTPUT ','CPU_PREFLIGHT_OK','COLLECT ATTEMPT',
                    'INDEPENDENT_GRASP','BALL_ACTION_CONTRACT_PASSED','VIDEO_DIRECTORY',
                    '[render]','Traceback','Error','SAVED_STATE_VISUALIZATION']):
                    print(line.rstrip(),flush=True)
            return process.wait()
        except KeyboardInterrupt:
            if process.poll() is None:process.send_signal(signal.SIGINT)
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=10)
            raise


def outcome(record):
    path=Path(record['run'])/'report.json'
    if path.is_file():
        report=json.loads(path.read_text());grasp=report.get('independent_grasp_evidence') or {}
        record['grasp']=grasp.get('status','not_evaluated')
        record['report']=str(path)
        record['interface_passed']=report.get('ball_action_contract_audit',{}).get('decoded_actions_match_motor_boundary')
    else:record['grasp']='report_unavailable'


def render_record(out,state,record,label):
    if not record.get('run'):
        record['render_state']='no_saved_run';persist(out,state);return
    try:
        saved_episode(ROOT,record['run'])
    except Exception as exc:
        record.update(render_state='no_renderable_episode',error=str(exc));persist(out,state);return
    outcome(record)
    if record.get('render_state')=='complete':
        try:
            record['video_check']=inspect_review_videos(ROOT,record['run']);persist(out,state);return
        except Exception:pass
    record['render_state']='running';persist(out,state)
    try:
        code=logged(render_command(record['run'],True),out/(label+'_render.log'))
        require(code==0,'Renderer exited '+str(code)+'; see '+label+'_render.log')
        record['video_check']=inspect_review_videos(ROOT,record['run'])
        record['render_state']='complete';record.pop('error',None)
    except (ValueError,OSError,RuntimeError,subprocess.SubprocessError) as exc:
        record.update(render_state='failed',error=str(exc))
    persist(out,state)


def physics_stage(out,state,record,cmd,label):
    if record.get('state')=='finished':return
    if record.get('run'):
        # An interrupted run is retained, never silently repeated. If it saved
        # states they will still be rendered; the batch records interruption.
        record.update(state='interrupted_not_repeated',error='Interrupted physics stage retained; no automatic rerun')
        persist(out,state);return
    def identify(path):
        actual=Path(path).resolve();actual.relative_to(ROOT/'data/derived')
        record['run']=str(actual);persist(out,state)
    record['state']='running';persist(out,state)
    record['exit_code']=logged(cmd,out/(label+'_physics.log'),identify)
    record['state']='finished';persist(out,state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    if args.resume:
        require(args.run,'Use --resume PATH --run to resume rendering/execution')
        out=args.resume.resolve();out.relative_to(ROOT/'data/derived/hgwind_contract_batch')
        state=json.loads((out/'batch.json').read_text())
        require(state['manifest_sha256']==sha256(MANIFEST),'Cohort/camera manifest changed')
        require(state['contract_sha256']==sha256(ROOT/'src/capstone_vla/ball_action_contract.py'),
            'Contract changed; do not mix configurations in a resumed batch')
        require(state['task_sha256']==sha256(state['task']),'Task/cameras changed')
    else:
        cases,schema_review=preflight()
        print('BATCH_PLAN new_cases=3 max_new_physics_runs=6 existing_runs=2 render_all_outcomes=true cameras=5 review_fps=30 training=false',flush=True)
        if not args.run:
            print('CPU_PREFLIGHT_OK GPU_NOT_STARTED; add --run');return 0
        stamp=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
        out=ROOT/'data/derived/hgwind_contract_batch'/stamp;out.mkdir(parents=True,exist_ok=False)
        manifest=json.loads(MANIFEST.read_text())
        state=dict(schema='physical-vla-lab.bounded-contract-batch.v1',schema_eligibility=schema_review,selection_rule='first/middle/last local binary-schema supported source by attempt index; before new outcomes',manifest_sha256=sha256(MANIFEST),
            contract_sha256=sha256(ROOT/'src/capstone_vla/ball_action_contract.py'),
            task=manifest['task_usd'],task_sha256=manifest['task_sha256'],
            gpu_policy='local_serial',training=False,held_out_evaluation=False,manual_reviewed=False,
            existing=[dict(label=label,run=str(run),state='existing',render_state='pending') for label,run in EXISTING],
            cases=[dict(episode=r['episode'],attempt_index=r['attempt_index'],episode_seed=r['episode_seed'],
                source=r['source'],source_files=r['files'],numeric_fingerprint=r['numeric_fingerprint'],
                teacher=dict(state='pending'),contract=dict(state='pending')) for r in cases])
        persist(out,state)
    print('BATCH_OUTPUT '+str(out),flush=True)
    print('VIDEO_REVIEW '+str(out/'VIDEO_REVIEW.md'),flush=True)
    with (out/'coordinator.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another coordinator owns this batch')
        for row in state['existing']:render_record(out,state,row,row['label'])
        for case in state['cases']:
            verify_files(case['source'],case['source_files'])
            label=case['episode'];teacher=case['teacher'];contract=case['contract']
            physics_stage(out,state,teacher,teacher_command(case,True),label+'_teacher')
            render_record(out,state,teacher,label+'_teacher')
            if not teacher.get('run'):
                contract.update(state='skipped',render_state='not_run',error='Teacher did not save a run');persist(out,state);continue
            if contract.get('state')=='skipped':
                persist(out,state);continue
            if contract.get('state')=='pending':
                # Read-only preparation is the eligibility check. All teacher
                # failures/incompatible profiles remain listed and rendered.
                from prepare_hgwind_teacher_contract import prepare_teacher
                try:prepare_teacher(Path(teacher['run']))
                except (ValueError,OSError,KeyError) as exc:
                    contract.update(state='skipped',render_state='not_run',error=str(exc));persist(out,state);continue
            cmd=[sys.executable,str(ROOT/'scripts/run_hgwind_action_contract.py'),
                '--teacher-run',teacher['run'],'--run']
            physics_stage(out,state,contract,cmd,label+'_contract')
            render_record(out,state,contract,label+'_contract')
            verify_files(case['source'],case['source_files'])
        state['finished']=True
        records=state['existing']+[c[s] for c in state['cases'] for s in ['teacher','contract']]
        state['all_saved_runs_rendered']=all(r.get('render_state')=='complete' for r in records if r.get('run'))
        state['reviewable_run_count']=sum(r.get('render_state')=='complete' for r in records)
        persist(out,state)
        print('BATCH_COMPLETED '+json.dumps({k:state[k] for k in ['all_saved_runs_rendered','reviewable_run_count']}),flush=True)
        print('VIDEO_REVIEW '+str(out/'VIDEO_REVIEW.md'),flush=True)
    return 0 if state['all_saved_runs_rendered'] else 1


if __name__=='__main__':raise SystemExit(main())
