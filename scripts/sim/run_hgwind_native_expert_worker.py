#!/usr/bin/env python3
"""Execute the frozen collection profile and depth helper; add trace hooks only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from capstone_vla import hgwind_expert_trace as trace
plan=json.loads(Path(os.environ['HGWIND_EFFORT_PROBE_MANIFEST']).read_text())
if plan['tracking'] not in ('native-expert','logged-command','live-jacobian') or any(x in sys.argv for x in ['--replay_trajectories','--replay_state_exact']) or (plan['tracking']=='native-expert' and '--repair-action-reference' in sys.argv):
    raise RuntimeError('Original expert reference must not replace or replay its commands')
for name,digest in plan['adapter_sha256'].items():
    path=ROOT/name if not Path(name).is_absolute() else Path(name)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise RuntimeError('Native expert probe dependency changed: '+path.name)
if hashlib.sha256(Path(plan['worker']).read_bytes()).hexdigest()!=plan['worker_sha256']:raise RuntimeError('Frozen collector changed')
from capstone_vla import hgwind_effort_probe as probe
trace.configure(plan);bundle=Path(plan['bundle_manifest']).parent;repo=Path(plan['worker']).parents[2]
# Keep frozen control-profile imports, while the already loaded trace module
# remains local. This only changes this isolated simulation process.
import capstone_vla
capstone_vla.__path__.insert(0,str(bundle/'src/capstone_vla'));sys.path[:0]=[str(bundle/'src'),str(repo/'Scripts')]
from expert_data.skills import catch
from capstone_vla import hgwind_control
spec=importlib.util.spec_from_file_location('frozen_depth_helper',Path(plan['original_depth_helper']))
depth=importlib.util.module_from_spec(spec);spec.loader.exec_module(depth)
depth.install_depth_bias(catch.Catch,plan['original_depth_bias_m'])
original_summary=hgwind_control.profile_summary

def profile_summary(policy):
    result=original_summary(policy)
    result['depth_bias']={'requested_downward_goal_bias_m':plan['original_depth_bias_m'],'truth_geometry_changed':False,
        'auxiliary_grasp_added':False,'physics_validated_improvement':False,
        'primitives':[dict(p._capstone_depth_stats) for p in policy.active_primitives() if hasattr(p,'_capstone_depth_stats')]}
    return result

def fixed_trace_patch(source):
    patched=trace.patch_collector(source) if plan['tracking']=='native-expert' else probe.patch_worker(source)
    if hashlib.sha256(patched.encode()).hexdigest()!=plan['patch_sha256']:raise RuntimeError('Native trace assembly differs from preflight')
    return patched

os.environ['CAPSTONE_HGWIND_CONTROL_PROFILE']='smooth-lift';os.environ.pop('CAPSTONE_HGWIND_CAPTURE_SPEED_GATE',None)
wrapper=Path(plan['original_profile_wrapper']);source=trace.patch_profile_wrapper(wrapper.read_text())
print('HGWIND_EFFORT_PROBE_ACTIVE tracking='+plan['tracking']+' original_profile=smooth-lift depth_m='+str(plan['original_depth_bias_m'])+' source_motor_reference='+('live_expert' if plan['tracking']=='native-expert' else 'original_command_log'),flush=True)
exec(compile(source,str(wrapper),'exec'),{'__name__':'__main__','__file__':str(wrapper),
    'CAPSTONE_TRACE_PATCH':fixed_trace_patch,'CAPSTONE_TRACE_RECORD':trace.record_segment,
    'CAPSTONE_TRACE_SUMMARY':trace.summary,'CAPSTONE_DEPTH_PROFILE_SUMMARY':profile_summary})
