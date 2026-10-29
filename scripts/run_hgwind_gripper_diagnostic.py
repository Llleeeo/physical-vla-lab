#!/usr/bin/env python3
"""Two serial passive observations: history58/59/60 and fresh60+omit1."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from capstone_vla.hf_archive import sha256
from capstone_vla.hgwind_history import compare_motion
from capstone_vla.gripper_drive_trace import trace_status
REFERENCE=ROOT/'data/derived/hgwind_history_diagnostics/20261030_005620_189348_fresh60_omit1'


def command(fresh,run=False):
    cmd=[sys.executable,str(ROOT/'scripts/run_hgwind_history_diagnostic.py'),
         '--stage-load-barrier','--native-backtrace','--trace-gripper-drives']
    if fresh:cmd+=['--fresh-control','--omit-setup-force-once']
    return cmd+(['--run'] if run else [])


def property_differences(a,b,prefix=''):
    rows=[]
    if isinstance(a,dict) and isinstance(b,dict):
        for k in sorted(a.keys()|b.keys()):
            if k not in a or k not in b:rows.append({'field':prefix+'/'+k,'missing':True})
            else:rows+=property_differences(a[k],b[k],prefix+'/'+k)
    elif a!=b:rows.append({'field':prefix,'history':a,'fresh_omit':b})
    return rows


def read_snapshots(path):
    rows=[json.loads(x) for x in Path(path).read_text().splitlines()]
    selected={}
    for r in rows:
        key='usd_before_reset' if r['phase']=='usd_before_reset' else (
            'first_close' if r['phase']=='close_action_built' else (
            'frame_'+str(r['frame']) if r['phase']=='frame_before_policy' and r['frame'] in [0,153,154,155,166] else None))
        if key and key not in selected:selected[key]=r
    return selected


def finish(out,state):
    result=dict(schema='physical-vla-lab.gripper-drive-diagnostic.v1',
        jobs=state['jobs'],comparison_ready=False,learned_policy_evaluated=False,
        rendering_requested=False,drive_properties_overridden=False,errors=[],stop_reason=state.get('stop_reason'))
    try:
        runs={k:Path(v['run']) for k,v in state['jobs'].items()}
        reports={k:json.loads((v/'report.json').read_text()) for k,v in runs.items()}
        cases={k:next(x for x in v['cases'] if x['attempt_index']==60) for k,v in reports.items()}
        paths={k:Path(row['run'])/'gripper_drive_trace.jsonl' for k,row in cases.items()}
        result['observations']={k:trace_status(p) for k,p in paths.items()}
        result['history_core_arrays_match_original']=bool(reports['history']['comparison_evaluable']) and all(
            x['comparison_to_original']['all_core_arrays_exact'] for x in reports['history']['cases'])
        old=json.loads((REFERENCE/'attempt_00060/report.json').read_text())
        new=json.loads((Path(cases['fresh']['run'])/'report.json').read_text())
        result['fresh_comparison_to_previous_omission']=compare_motion(
            Path(old['probe_episode'])/'replay.hdf5',Path(new['probe_episode'])/'replay.hdf5')
        result['force_intervention_verified']=reports['fresh'].get('setup_force_control',{}).get('verified') is True
        result['comparison_ready']=bool(result['history_core_arrays_match_original']
            and result['fresh_comparison_to_previous_omission']['all_core_arrays_exact']
            and result['force_intervention_verified']
            and all(r['verified'] for r in result['observations'].values()))
        snapshots={k:read_snapshots(p) for k,p in paths.items()}
        differences={}
        for phase in sorted(snapshots['history'].keys()&snapshots['fresh'].keys()):
            a,b=snapshots['history'][phase],snapshots['fresh'][phase]
            if phase=='usd_before_reset':x,y=a.get('usd_fingers'),b.get('usd_fingers')
            else:
                fields=['live_gains','live_max_efforts','live_max_velocities','usd_fingers','expert_driven_indices','last_applied_action']
                x={k:a.get('snapshot',{}).get(k) for k in fields}
                y={k:b.get('snapshot',{}).get(k) for k in fields}
            differences[phase]=property_differences(x,y)
        result['property_differences']=differences
        result['source_hashes']={k:dict(trace=sha256(paths[k]),report=sha256(runs[k]/'report.json')) for k in paths}
        result['interpretation']='Read actual property differences only after checking comparison_ready. Missing observations and changes under instrumentation remain confounds. Equal recorded properties do not establish equality of hidden solver/mimic state.'
        for phase,rows in differences.items():
            print('GRIPPER_PROPERTY_DIFF phase='+phase+' fields='+str(len(rows)),flush=True)
            for row in rows[:8]:print(json.dumps(row),flush=True)
    except (OSError,ValueError,TypeError,KeyError,StopIteration) as exc:result['errors'].append(repr(exc))
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print('GRIPPER_DIAGNOSTIC comparison_ready='+str(result['comparison_ready']).lower()+' report='+str(out/'report.json'),flush=True)
    return 0 if result['comparison_ready'] else 2


def execute(cmd,log,on_run=None):
    run=None
    with Path(log).open('w') as handle:
        process=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,errors='replace',start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        try:
            for line in process.stdout:
                handle.write(line);handle.flush();print(line.rstrip(),flush=True)
                if line.startswith('HISTORY_OUTPUT '):
                    run=line.strip().split(' ',1)[1]
                    if on_run:on_run(run)
            code=process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid,signal.SIGINT)
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=10)
            raise
    return {'run':run,'exit_code':code}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true')
    p.add_argument('--finish',type=Path);args=p.parse_args()
    if args.finish:
        if args.run:raise ValueError('--finish does not launch simulation')
        out=args.finish.resolve();out.relative_to(ROOT/'data/derived/hgwind_gripper_diagnostics')
        return finish(out,json.loads((out/'jobs.json').read_text()))
    for fresh in [False,True]:
        proc=subprocess.run(command(fresh),cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        if proc.returncode:raise RuntimeError('CPU preflight failed: '+proc.stdout[-4000:])
    print('CPU_PREFLIGHT_OK workers=2 attempts=3+1 passive_drive_observation=true render=false',flush=True)
    if not args.run:return 0
    stamp=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
    out=ROOT/'data/derived/hgwind_gripper_diagnostics'/stamp;out.mkdir(parents=True,exist_ok=False)
    state={'jobs':{},'budget':'one history58/59/60 worker plus one fresh60+omit1 worker; no retries',
           'coordinator_sha256':sha256(Path(__file__)),
           'commands':{label:command(fresh,True) for label,fresh in [('history',False),('fresh',True)]}}
    print('GRIPPER_OUTPUT '+str(out),flush=True)
    def persist():
        temporary=out/'jobs.json.tmp'
        temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(out/'jobs.json')
    persist()
    for label,fresh in [('history',False),('fresh',True)]:
        state['jobs'][label]={'state':'running','run':None};persist()
        def identify(path):
            resolved=Path(path).resolve();resolved.relative_to(ROOT/'data/derived/hgwind_history_diagnostics')
            state['jobs'][label]['run']=str(resolved);persist()
        try:job=execute(command(fresh,True),out/(label+'.log'),identify)
        except KeyboardInterrupt:
            state['jobs'][label]['state']='interrupted';persist()
            print('GRIPPER_DIAGNOSTIC_INTERRUPTED outputs retained at '+str(out),flush=True)
            return 130
        state['jobs'][label].update(job,state='finished');persist()
        job=state['jobs'][label]
        if not job['run'] or not (Path(job['run'])/'report.json').exists():
            state['stop_reason']='no_evaluable_job_'+label;persist()
            print('GRIPPER_DIAGNOSTIC_STOP '+state['stop_reason'],flush=True)
            return finish(out,state)
        if label=='history':
            reference=json.loads((Path(job['run'])/'report.json').read_text())
            matched=bool(reference.get('comparison_evaluable')) and all(
                row.get('comparison_to_original',{}).get('all_core_arrays_exact')
                for row in reference['cases'])
            target=next((row for row in reference['cases'] if row['attempt_index']==60),{})
            if not matched or not target.get('gripper_drive_observation',{}).get('verified'):
                state['stop_reason']='history_or_drive_observation_not_verified';persist()
                print('GRIPPER_DIAGNOSTIC_STOP '+state['stop_reason'],flush=True)
                return finish(out,state)
    return finish(out,state)


if __name__=='__main__':raise SystemExit(main())
