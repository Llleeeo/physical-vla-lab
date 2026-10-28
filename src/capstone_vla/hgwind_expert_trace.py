"""Read-only motor-boundary instrumentation of the frozen native expert."""
import json
from pathlib import Path
import numpy as np

_STATE={}


def _require(value,message):
    if not value:raise ValueError(message)


def configure(plan):
    _STATE.clear();_STATE.update(plan=plan,rows=[],assistance=[])
    if plan.get('cartesian_target_capture'):
        from .cartesian_target_replay import CartesianTargetReplayer
        from .hf_archive import sha256
        path=Path(plan['cartesian_target_capture'])
        _require(sha256(path)==plan['cartesian_target_capture_sha256'], 'Upper target capture changed')
        if plan.get('independent_packets'):
            import h5py
            from .cartesian_packets import PacketStreamExecutor,build_packets
            _require(sha256(plan['source'])==plan['source_reference_sha256'],'Packet jaw source changed')
            with h5py.File(plan['source'],'r') as handle:
                packets=build_packets(json.loads(path.read_text()),handle['actions'][:,7:])
            _STATE['packet_executor']=PacketStreamExecutor(packets)
            (Path(plan['output'])/'packet_requests.json').write_text(json.dumps(packets,indent=2,allow_nan=False)+'\n')
        else:
            _STATE['cartesian_replayer']=CartesianTargetReplayer(json.loads(path.read_text()))
    if plan.get('trace_cartesian_targets'):
        from .cartesian_target_trace import CartesianTargetRecorder
        _STATE['cartesian_recorder']=CartesianTargetRecorder()


def packet_decision(policy,world,index,dt):
    return _STATE['packet_executor'].step(policy,world,index,dt)


def begin_cartesian_frame(expert,index):
    key='cartesian_replayer' if 'cartesian_replayer' in _STATE else 'cartesian_recorder'
    _STATE[key].begin_frame(expert,index)


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(x) for x in value]
    return value


def record_segment(index,segment,target,q,qd,stage,robot_path,objects,attach_state):
    if 'joints' not in _STATE:
        from capstone_vla.task2_bridge import grasp_joint_candidates
        _STATE['joints']=grasp_joint_candidates(stage,robot_path,objects)
    enabled=[p for p in _STATE['joints'] if bool(stage.GetPrimAtPath(p).GetAttribute('physics:jointEnabled').Get())]
    assistance=[bool(attach_state.get('attached')),len(enabled),bool(attach_state.get('held_tool'))]
    _require(not any(assistance),'Original expert reference probe detected grasp assistance')
    _STATE['assistance'].append(assistance)
    # Copy at the command boundary; later physics mutates segment bookkeeping.
    observer=_STATE.get('packet_executor') or _STATE.get('cartesian_replayer') or _STATE.get('cartesian_recorder')
    links=observer.segment_links(index,segment) if observer is not None else {}
    _STATE['rows'].append(plain({**links,'frame':int(index),'joint_positions':q,'joint_velocities':qd,'target':target,
        **{k:segment[k] for k in ['goal_position','goal_velocity','operational_space_reference','policy_joint_velocity_limit','duration_s','episode_time_s']}}))


def summary(policy):
    plan=_STATE['plan'];root=Path(plan['output']);rows=_STATE['rows']
    out=root/'original_expert_control_references.json';_require(not out.exists(),'Original expert trace already exists')
    out.write_text(json.dumps(rows,indent=2)+'\n')
    if 'cartesian_recorder' in _STATE:
        capture=root/'cartesian_target_inputs.json';_require(not capture.exists(),'Cartesian capture already exists')
        capture.write_text(json.dumps(_STATE['cartesian_recorder'].payload(),indent=2,allow_nan=False)+'\n')
    if 'cartesian_replayer' in _STATE:
        (root/'cartesian_target_reexecution.json').write_text(json.dumps(_STATE['cartesian_replayer'].payload(),indent=2,allow_nan=False)+'\n')
    if 'packet_executor' in _STATE:
        (root/'cartesian_packet_execution.json').write_text(json.dumps(_STATE['packet_executor'].payload(),indent=2,allow_nan=False)+'\n')
    np.savez(root/'command_diagnostics.npz',requested_joints=np.asarray([x['target'] for x in rows]),assistance=np.asarray(_STATE['assistance']))
    return {'schema':plan['schema'],'tracking':'native-expert','mode':'joint','reference':'online-expert',
        'original_expert_used':'packet_executor' not in _STATE,'packet_requests_own_motors':'packet_executor' in _STATE,'source_success_reused':False,'saved_commands_own_motors':False,
        'shadow_expert_operational_reference_used':False,'shadow_expert_velocity_limit_used':False,
        'source_native_operational_reference_reconstructed':False,'original_control_configuration_used':True,
        'command_frames':len(rows),'segment_trace_count':len(rows),'source_reference_sha256':plan['source_reference_sha256'],
        'depth_bias_m':plan['original_depth_bias_m'],'control_reference_file':out.name,'cartesian_target_capture_enabled':'cartesian_recorder' in _STATE,'cartesian_upper_target_reexecution':'cartesian_replayer' in _STATE,'saved_inner_control_outputs_used':False,'standalone_policy_controller':False,'training_eligible':False,'vla_policy':False}


def patch_collector(source):
    anchors=[
      ('                    dynamic_drive_runtime["segment"] = segment\n\n                    def _hold_jaws():',
       '                    dynamic_drive_runtime["segment"] = segment\n                    CAPSTONE_RECORD_NATIVE_SEGMENT(k, segment, tgt, jp, jv, stage, robot_path, objects, attach_state)\n\n                    def _hold_jaws():'),
      ('"skill_terminations": _jsonify(skill_terminations),',
       '"skill_terminations": _jsonify(skill_terminations),\n            "capstone_effort_probe": CAPSTONE_NATIVE_TRACE_SUMMARY(pol),')]
    for old,new in anchors:
        _require(source.count(old)==1,'Native trace insertion point changed')
        source=source.replace(old,new)
    compile(source,'<original-expert-trace>','exec');return source


def patch_profile_wrapper(source):
    replacements=[('exec(compile(source,str(worker),\'exec\'),',"source=CAPSTONE_TRACE_PATCH(source)\nexec(compile(source,str(worker),'exec'),"),
        ("'CAPSTONE_PROFILE_SUMMARY':profile_summary}","'CAPSTONE_PROFILE_SUMMARY':CAPSTONE_DEPTH_PROFILE_SUMMARY, 'CAPSTONE_RECORD_NATIVE_SEGMENT':CAPSTONE_TRACE_RECORD, 'CAPSTONE_NATIVE_TRACE_SUMMARY':CAPSTONE_TRACE_SUMMARY, 'CAPSTONE_BEGIN_CARTESIAN_FRAME':CAPSTONE_BEGIN_CARTESIAN_FRAME, 'CAPSTONE_PACKET_DECISION':CAPSTONE_PACKET_DECISION}")]
    for old,new in replacements:
        _require(source.count(old)==1,'Frozen profile wrapper changed')
        source=source.replace(old,new)
    compile(source,'<original-profile-trace>','exec');return source
