#!/usr/bin/env python3
"""One fixed29D surrogate from the exactly reproduced fresh60 native teacher."""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from capstone_vla.hf_archive import sha256
from capstone_vla.initialized_contract import SCHEMA,validate,patch_collector,audit
from capstone_vla.cartesian_target_replay import require
DEFAULT_REFERENCE=ROOT/'data/derived/hgwind_history_diagnostics/20261030_022121_219398_fresh60_omit1_gains'
SUMMARY=ROOT/'results/summaries/2026-10-30-gripper-gain-reapplication-result.json'


def prepare(precision_control=False):
    from prepare_hgwind_teacher_contract import prepare_teacher
    from capstone_vla.hgwind_expert_trace import patch_collector as native_patch
    from capstone_vla.cartesian_packets import patch_packet_collector
    from capstone_vla.inotify_preflight import require_headroom
    admission=require_headroom()
    summary=json.loads(SUMMARY.read_text())
    for name,digest in summary['source_hashes'].items():
        require(sha256(ROOT/name)==digest,'Verified initialization evidence changed: '+name)
    reference=DEFAULT_REFERENCE
    parent=json.loads((reference/'plan.json').read_text());outcome=json.loads((reference/'report.json').read_text())
    require(outcome['physical_success'] and outcome['gripper_gain_control']['verified']
        and outcome['setup_force_control']['verified']
        and outcome['cases'][0]['comparison_to_original']['all_core_arrays_exact'],
        'Need exactly reproduced fresh60 initialization')
    plan,actions,model,diag,lab=prepare_teacher(reference/'attempt_00060')
    stamp=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
    out=ROOT/'data/derived/hgwind_initialized_contract'/(stamp+('_float64' if precision_control else ''))
    plan.update(output=str(out),worker_output=str(out/'worker_output'),action_vectors=str(out/'actions.npy'),
        initialized_contract=SCHEMA,history_attempts=[60],fresh_control=True,stage_load_barrier=True,
        trace_gripper_drives=True,history_cases={'60':{'output':str(out)}},inotify_preflight=admission,
        initialization_reference_report=str(reference/'attempt_00060/report.json'),
        initialization_reference_report_sha256=sha256(reference/'attempt_00060/report.json'),
        initialization_result_summary_sha256=sha256(SUMMARY),render_requested=False)
    for key,name in [('setup_force_control','setup_force_control_audit.json'),('gripper_gain_control','gripper_gain_control.json')]:
        plan[key]=copy.deepcopy(parent[key]);plan[key]['audit']=str(out/name)
    source=Path(plan['worker']).read_text();anchor='"skill_terminations": _jsonify(skill_terminations),'
    require(source.count(anchor)==1,'Native profile hook changed')
    source=source.replace(anchor,anchor+'\n            "capstone_expert_control": CAPSTONE_PROFILE_SUMMARY(pol),')
    assembled=patch_packet_collector(patch_collector(native_patch(source)))
    compile(assembled,'<initialized-contract-preflight>','exec')
    plan['patch_sha256']=hashlib.sha256(assembled.encode()).hexdigest()
    paths=['src/capstone_vla/initialized_contract.py','src/capstone_vla/hgwind_history.py',
        'src/capstone_vla/hgwind_startup_trace.py','src/capstone_vla/setup_force_control.py',
        'src/capstone_vla/gripper_drive_trace.py','src/capstone_vla/gripper_gain_control.py',
        'src/capstone_vla/scene_load_barrier.py','scripts/sim/run_hgwind_initialized_contract_worker.py',
        'scripts/run_hgwind_initialized_contract.py']
    for path in paths:plan['adapter_sha256'][str(ROOT/path)]=sha256(ROOT/path)
    plan['reference_fidelity']='Frozen29D float32 teacher-labelled packets with explicit verified fresh60 setup, fixed9/8 gains, live-latched posture and always-on geometry; no online task/skill decisions or learned policy.'
    validate(plan)
    diag.update(initialized_contract=SCHEMA,one_attempt=True,force_omission_explicit=True,
        native_finger_gain_reapplication_explicit=True,render=False,initialization_gpu_result_pending=True)
    if precision_control:
        from capstone_vla.action_precision_control import prepare as prepare_precision
        actions=prepare_precision(plan,actions,diag)
    return plan,actions,model,diag,lab


def finish_audit(plan):
    result=audit(plan)
    if plan.get('precision_control'):
        from capstone_vla.action_precision_control import audit as precision_audit
        comparison=precision_audit(plan,result)
        result['combined_passed']=bool(result['combined_passed'] and not comparison['errors']
            and comparison['issued_prefix_inputs_match'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--precision-control',action='store_true',help='One matched float64 label diagnostic; not model training')
    parser.add_argument('--finish',type=Path,help='Reanalyze initialization only; never launch simulation')
    args=parser.parse_args()
    if args.finish:
        require(not args.run,'--finish cannot launch physics')
        out=args.finish.resolve();out.relative_to(ROOT/'data/derived/hgwind_initialized_contract')
        result=finish_audit(json.loads((out/'plan.json').read_text()))
        return 0 if result['combined_passed'] else 2
    from run_hgwind_action_contract import run_prepared
    prepared=prepare(precision_control=args.precision_control)
    try:code=run_prepared(*prepared,run=args.run)
    except Exception:
        if args.run and Path(prepared[0]['output']).exists():finish_audit(prepared[0])
        raise
    if not args.run:return code
    if code==130:return code
    result=finish_audit(prepared[0])
    return code or (0 if result['combined_passed'] else 2)


if __name__=='__main__':raise SystemExit(main())
