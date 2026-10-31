#!/usr/bin/env python3
"""One fresh60 float32 mimic-follower control; no training or rendering."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from capstone_vla.hf_archive import sha256
from capstone_vla.mimic_drive_control import prepare as prepare_control, audit
from capstone_vla.initialized_contract import audit as audit_initialization


def prepare():
    from run_hgwind_initialized_contract import prepare as prepare_initialized
    plan,actions,model,diag,lab=prepare_initialized()
    out=Path(str(plan['output'])+'_single_drive')
    plan.update(output=str(out),worker_output=str(out/'worker_output'),
        action_vectors=str(out/'actions.npy'),history_cases={'60':{'output':str(out)}})
    for key,name in [('setup_force_control','setup_force_control_audit.json'),
                     ('gripper_gain_control','gripper_gain_control.json')]:
        plan[key]['audit']=str(out/name)
    plan['adapter_sha256'][str(Path(__file__).resolve())]=sha256(__file__)
    prepare_control(plan,actions,diag)
    return plan,actions,model,diag,lab


def finish(plan):
    return audit(plan,audit_initialization(plan))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true')
    p.add_argument('--finish',type=Path,help='Reanalyze an existing single-drive run without simulation')
    args=p.parse_args()
    if args.finish:
        if args.run:p.error('--finish cannot launch physics')
        out=args.finish.resolve();out.relative_to(ROOT/'data/derived/hgwind_initialized_contract')
        result=finish(json.loads((out/'plan.json').read_text()))
        return 0 if result['combined_passed'] else 2
    from run_hgwind_action_contract import run_prepared
    prepared=prepare();plan=prepared[0]
    try:code=run_prepared(*prepared,run=args.run)
    except Exception:
        if args.run and Path(plan['output']).exists():finish(plan)
        raise
    if not args.run or code==130:return code
    result=finish(plan)
    print('NEXT inspect_single_drive_comparison; no_automatic_retries=true',flush=True)
    return code or (0 if result['combined_passed'] else 2)


if __name__=='__main__':raise SystemExit(main())
