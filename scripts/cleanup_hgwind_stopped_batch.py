#!/usr/bin/env python3
"""Delete only the explicitly retired HGwind production raw batch, after audit."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

LOCAL_ROOT=Path('/home/robophys/Capstone_Project_Yuan_Haoming')
LOCAL_BASE=LOCAL_ROOT/'data/derived/hgwind_production_20261027'
REMOTE_ROOT=Path('/media/RoboPhys/DataCollection33/Capstone_HGwind_Worker_Yuan_Haoming')
REL=Path('data/derived/hgwind_current_baseline_20261027/repo/runs')
TASK='grasp_wind_driven_motion_1s1p_hgprgew'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',choices=('local','hongtao'),required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    run=(LOCAL_BASE/'worker_bundle'/REL/'local') if args.host=='local' else REMOTE_ROOT/'production_v1'/REL/'hongtao'
    expected_count=18 if args.host=='local' else 12
    audit_path=run/'analysis/contact_transition_audit.json'
    audit=json.loads(audit_path.read_text())
    if audit['summary']['all_attempts']!=expected_count:
        raise RuntimeError('Stopped batch differs from the audited deletion scope')
    # No matches are accepted while a real collection interpreter remains alive.
    for row in subprocess.check_output(['ps','-eo','comm,args'],text=True).splitlines():
        command=row.strip().split(None,1)
        if len(command)!=2 or not command[0].startswith(('python','isaac')):continue
        text=command[1]
        if 'run_hgwind_production_worker.py' in text or 'collect_hgwind_worker.py' in text:
            raise RuntimeError('Collection process still active; refusing cleanup: '+text[:250])
    targets=[run/'Trajectory']
    if args.host=='local':
        final=LOCAL_ROOT/'data/collected/hgwind_v1_20261027'
        targets += [final/name for name in ('local','hongtao','.incoming_local','.incoming_hongtao')]
    receipt={'scope':'User-authorized deletion of stopped production batch only; historical successful pilots and smoke results retained',
             'host':args.host,'audited_attempts':expected_count,'executed':args.execute,
             'created_at_utc':datetime.now(timezone.utc).isoformat(),'targets':[]}
    for path in targets:
        if path.is_symlink():raise RuntimeError('Refusing a symbolic-link deletion target: '+str(path))
        if not path.exists():continue
        if path.resolve()!=path:raise RuntimeError('Unexpected resolved deletion path: '+str(path))
        inventory=[{'path':p.relative_to(path).as_posix(),'bytes':p.stat().st_size}
                   for p in path.rglob('*') if p.is_file()]
        complete=list(path.glob('1s1p/'+TASK+'/*/meta.json')) if path.name=='Trajectory' else []
        if path.name=='Trajectory' and len(complete)!=expected_count:
            raise RuntimeError('Raw episode count changed after audit')
        receipt['targets'].append({'path':str(path),'files':len(inventory),
                                   'bytes':sum(r['bytes'] for r in inventory),'inventory':inventory})
    print('CLEANUP_PLAN '+json.dumps({k:v for k,v in receipt.items() if k!='targets'},ensure_ascii=False))
    for row in receipt['targets']:print('TARGET',row['path'],'files',row['files'],'bytes',row['bytes'])
    if args.execute:
        for row in receipt['targets']:shutil.rmtree(Path(row['path']))
        if any(Path(row['path']).exists() for row in receipt['targets']):raise RuntimeError('Cleanup incomplete')
        (run/'analysis/raw_cleanup_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
        print('STOPPED_RAW_TRAJECTORIES_REMOVED; audits and old approved samples retained')


if __name__=='__main__':main()
