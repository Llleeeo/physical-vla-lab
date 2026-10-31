#!/usr/bin/env python3
"""One learned ball rollout through the unchanged native effort controller."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from capstone_vla import hgwind_expert_trace as trace
from capstone_vla.ball_live_protocol import patch_collector
from capstone_vla.ball_live_runtime import LiveExecutor
from capstone_vla.ball_vertical_diagnostic import validate_provenance
from capstone_vla.hf_archive import sha256
# Load local sensor/bridge dependencies before the frozen control namespace.
from capstone_vla.closed_loop import LiveRGB, publish_gui_physics_state

config=json.loads(Path(os.environ['CAPSTONE_BALL_LIVE_CONFIG']).read_text())
if any(x in sys.argv for x in ['--replay_trajectories','--replay_state_exact','--repair-action-reference']):
    raise RuntimeError('Learned rollout cannot replay source states or commands')
diagnostic=validate_provenance(config)
for path,digest in config['runtime_hashes'].items():
    if sha256(path)!=digest:raise RuntimeError('Runtime changed: '+Path(path).name)
worker=Path(config['worker']);source=patch_collector(worker.read_text())
if hashlib.sha256(source.encode()).hexdigest()!=config['patch_sha256']:raise RuntimeError('Collector patch changed')
executor=LiveExecutor(config)
trace._STATE.clear();trace._STATE.update(plan=config,rows=[],assistance=[],packet_executor=executor)
bundle=Path(config['bundle_manifest']).parent;repo=worker.parents[2]
import capstone_vla
capstone_vla.__path__.insert(0,str(bundle/'src/capstone_vla'))
sys.path[:0]=[str(bundle/'src'),str(repo/'Scripts')]
from expert_data.skills import catch
from capstone_vla.hgwind_control import install_profile,profile_summary
install_profile(catch)
# Retain identical frozen setup. No Catch.step or helper-generated goal executes.
spec=importlib.util.spec_from_file_location('ball_frozen_depth_helper',config['original_depth_helper'])
depth=importlib.util.module_from_spec(spec);spec.loader.exec_module(depth)
depth.install_depth_bias(catch.Catch,config['original_depth_bias_m'])
os.environ['CAPSTONE_HGWIND_CONTROL_PROFILE']='smooth-lift'
os.environ.pop('CAPSTONE_HGWIND_CAPTURE_SPEED_GATE',None)


def summary(policy):
    executor.finish()
    root=Path(config['output'])
    (root/'live_control_references.json').write_text(json.dumps(trace._STATE['rows'],indent=2,allow_nan=False)+'\n')
    import numpy as np
    np.savez(root/'command_diagnostics.npz',assistance=np.asarray(trace._STATE['assistance']))
    return dict(schema='physical-vla-lab.ball-live-policy.v1',vla_policy=True,
        source_actions_used_for_control=diagnostic,online_task_step_called=False,online_skill_step_called=False,
        learned_policy_evaluated=not diagnostic,model_inference_used=True,
        vertical_diagnostic=diagnostic,standalone_policy_evaluation=not diagnostic,
        gripper_confirmation=config.get("gripper_confirmation"),
        policy_postprocessing=config.get("gripper_confirmation") is not None,
        original_expert_used=False,packet_requests_own_motors=True,
        source_success_reused=False,saved_commands_own_motors=diagnostic,command_frames=len(trace._STATE['rows']),
        control_reference_file='live_control_references.json',native_effort_configuration_unchanged=True,
        real_time_evaluated=False,heldout_evaluation=False)

print(f'BALL_LIVE_WORKER absolute29=true source_actions={diagnostic} diagnostic={diagnostic} online_skill_step=false native_limits=unchanged',flush=True)
try:
    exec(compile(source,str(worker),'exec'),{'__name__':'__main__','__file__':str(worker),
        'CAPSTONE_LIVE_DECISION':executor.live_step,'CAPSTONE_PROFILE_SUMMARY':profile_summary,
        'CAPSTONE_RECORD_NATIVE_SEGMENT':trace.record_segment,'CAPSTONE_NATIVE_TRACE_SUMMARY':summary})
finally:
    # Persist already observed inputs/commands even if the collector cannot save an episode.
    summary(None)
