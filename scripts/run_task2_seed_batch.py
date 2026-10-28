#!/usr/bin/env python3
"""Sequential fixed-scene TRAIN debugging across ten predeclared model seeds."""
import argparse, hashlib, json, subprocess, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import h5py
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from capstone_vla.grasp_evidence import evaluate_grasp, GraspEvidenceCriteria
BASE=ROOT/'data/derived/task2_multiepisode_20261026/training_runs/20261026_185108_323227'
SEEDS=list(range(60000,60010))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true')
    a=p.parse_args()
    runner=ROOT/'scripts/run_task2_closed_loop.py'
    common=[sys.executable,str(runner),'--episode-role','train-debug','--frames','0']
    subprocess.run(common+['--seed',str(SEEDS[0])],cwd=ROOT,check=True)
    print('BATCH_PLAN seeds='+','.join(map(str,SEEDS))+' scene=traj_00001 checkpoint=unchanged training=false generalization_claim=false',flush=True)
    if not a.run:return
    out=BASE/'seed_batches'/datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    weights=BASE/'checkpoint_final/model.safetensors'
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    report={'purpose':'TRAIN fixed-scene sampling sensitivity; not generalization or real-time evaluation','episode':'traj_00001','seeds':SEEDS,'checkpoint_sha256':digest(weights),'runner_sha256':digest(runner),'trials':[]}
    print('BATCH_OUTPUT',out,flush=True)
    def save():
        tmp=out/'report.json.tmp';tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(out/'report.json')
    save()
    for index,seed in enumerate(SEEDS):
        if digest(runner)!=report['runner_sha256']:raise RuntimeError('runner changed during batch')
        trial=out/f'seed_{seed}'
        print(f'BATCH_TRIAL {index+1}/{len(SEEDS)} seed={seed}',flush=True)
        try:
            subprocess.run(common+['--seed',str(seed),'--output',str(trial),'--run'],cwd=ROOT,check=True)
            config=json.loads((trial/'run_config.json').read_text())
            if config['seed']!=seed or config['episode_role']!='train-debug':raise RuntimeError('trial provenance mismatch')
            evidence_keys=['timestamp_s','ee_pos','ee_quat','object_pos','object_vel','contact_L','contact_R','impulse_L','impulse_R','grasp_assistance_audit','object_quat','object_ang_vel']
            with h5py.File(trial/'physics_trace.hdf5') as f:
                evidence=evaluate_grasp({k:f[k][:] for k in evidence_keys if k in f},GraspEvidenceCriteria())
            (trial/'independent_grasp_evidence.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
            result=json.loads((trial/'report.json').read_text())
            report['trials'].append({'seed':seed,'run':str(trial),'execution_completed':True,'engineering_grasp_status':evidence['status'],'bilateral_contact_frames':result['bilateral_contact_frames'],'official_success_verified':False})
        except Exception as exc:
            report['trials'].append({'seed':seed,'run':str(trial),'execution_completed':False,'error':str(exc)})
            save();raise
        save()
    if digest(weights)!=report['checkpoint_sha256']:raise RuntimeError('checkpoint changed during batch')
    report['summary']={'planned_trials':len(SEEDS),'completed_trials':len(report['trials']),'engineering_grasp_passes':sum(t.get('engineering_grasp_status')=='pass' for t in report['trials']),'official_benchmark_success':None,'generalization_claim':False}
    save();print('SEED_BATCH_COMPLETED report='+str(out/'report.json'),flush=True)
if __name__=='__main__':main()
