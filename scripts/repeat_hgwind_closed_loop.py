#!/usr/bin/env python3
"""Five fixed-condition sampling repeats with all outcomes and original-speed RGB."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_hgwind_closed_loop import prepare
from capstone_vla.ball_repeat import matched_config,outcome,export_video
from capstone_vla.hf_archive import sha256

REFERENCE=ROOT/'data/derived/hgwind_traj18_development_20261031/training_runs/20261031_133555_985430/closed_loop_runs/20261031_184624_502108'
SEEDS=[90000,190000,290000,390000,490000]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true')
    a=p.parse_args();reference=json.loads((REFERENCE/'run_config.json').read_text())
    original=json.loads((REFERENCE/'report.json').read_text())
    for path,digest in original['artifacts'].items():
        if sha256(REFERENCE/path)!=digest:raise ValueError('Original evidence changed: '+path)
    run=Path(reference['training_run'])
    cfg,_,_=prepare(run,269,1,'legacy',False,True,True,True,5,True)
    matched_config(reference,cfg,90000)
    for tool in ['ffmpeg','ffprobe']:
        if not shutil.which(tool):raise ValueError('Missing '+tool)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    subprocess.run([str(ROOT/'.venv_dynamicvla/bin/python'),str(ROOT/'scripts/hgwind_policy_server.py'),
        '--preflight',str(run)],cwd=ROOT,env=env,check=True)
    print('REPEAT_PREFLIGHT_PASS seeds='+','.join(map(str,SEEDS))+' same_scene=true same_controller=true videos=60fps heldout=false',flush=True)
    if not a.run:print('GPU_NOT_STARTED; add --run');return
    stamp=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
    dest=run/'repeat_batches'/stamp;dest.mkdir(parents=True,exist_ok=False)
    summary=dict(schema='physical-vla-lab.ball-repeats.v1',reference=str(REFERENCE),
        seeds=SEEDS,planned=5,results=[],complete=False,heldout=False,pure_vla=False,
        scope='One exposed initial condition; one original-seed repeat and four other action-sampling seeds; no population success-rate claim')
    def save():
        summary['counts']={k:sum(x['outcome']==k for x in summary['results'])
            for k in ['grasp_pass','grasp_fail','runtime_abort','incomplete_or_invalid']}
        temp=dest/'summary.tmp';temp.write_text(json.dumps(summary,indent=2)+'\n');temp.replace(dest/'summary.json')
    save();print('BATCH_OUTPUT '+str(dest),flush=True)
    hashes={**cfg['runtime_hashes'],**cfg['checkpoint_hashes'],str(Path(__file__).resolve()):sha256(__file__),
        str(ROOT/'src/capstone_vla/ball_repeat.py'):sha256(ROOT/'src/capstone_vla/ball_repeat.py')}
    for seed in SEEDS:
        for path,digest in hashes.items():
            if sha256(path)!=digest:raise ValueError('Frozen batch dependency changed: '+path)
        cmd=[sys.executable,str(ROOT/'scripts/run_hgwind_closed_loop.py'),'--run-dir',str(run),
            '--frames','269','--prefix','1','--camera-protocol','legacy','--approach-guard',
            '--approach-follow','--near-capture','--capture-depth-mm','5','--capture-window',
            '--model-seed',str(seed),'--run']
        print('REPEAT_BEGIN seed='+str(seed),flush=True);output=None
        with (dest/f'seed_{seed}.log').open('x') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            try:
                for line in child.stdout:
                    log.write(line);log.flush();print(line.rstrip(),flush=True)
                    if line.startswith('OUTPUT '):output=Path(line.strip().split(' ',1)[1])
                code=child.wait()
            except KeyboardInterrupt:
                # Inherit the terminal process group: Ctrl-C also reaches the
                # child coordinator, which terminates its separate simulator/server groups.
                child.wait(timeout=60);raise
        row=dict(seed=seed,exit_code=code,output=str(output) if output else None,
            outcome='incomplete_or_invalid',video=None)
        if output and (output/'report.json').exists():
            report=json.loads((output/'report.json').read_text())
            matched_config(reference,json.loads((output/'run_config.json').read_text()),seed)
            row.update(outcome=outcome(report),report_sha256=sha256(output/'report.json'),
                completed_frames=report.get('completed_frames'),abort_reason=report.get('abort_reason'),
                grasp=report.get('independent_grasp'))
        if output and (output/'live_observations.hdf5').exists():
            try:row['video']=export_video(output)
            except Exception as exc:row['video_error']=repr(exc)
        summary['results'].append(row);save()
        print('REPEAT_RESULT '+json.dumps({k:row[k] for k in ['seed','outcome','video']}),flush=True)
    summary['complete']=True;save()
    print('REPEAT_BATCH_COMPLETED summary='+str(dest/'summary.json'),flush=True)
    print('COUNTS '+json.dumps(summary['counts']),flush=True)

if __name__=='__main__':main()
