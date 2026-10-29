"""Prepare the fixed 29D diagnostic directly from a verified native teacher.

Avoid repeating historical component ablations for every development episode.
Native teacher decisions generate labels offline; none run in the surrogate.
"""
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from capstone_vla.hf_archive import sha256
from capstone_vla.hgwind_cohort import verify_files
from capstone_vla.cartesian_target_replay import require
from capstone_vla.cartesian_target_trace import audit_capture
from capstone_vla.cartesian_packets import build_packets,patch_packet_collector
from capstone_vla.posture_anchor import sampled_posture_configuration,compute_anchor
from capstone_vla.ball_action_contract import SCHEMA,encode_stream,decode_stream,model_configuration


def teacher_actions(capture,jaws,initial_joints,configuration):
    """Strict profile support check; do not quietly alter incompatible labels."""
    anchor=compute_anchor(initial_joints,configuration)
    recorded=[np.asarray(c['inputs']['posture_target'],float) for c in capture['calls']
        if c['inputs']['posture_target'] is not None]
    require(recorded and all(np.allclose(q,anchor,atol=1.e-14,rtol=0.) for q in recorded),
        'Teacher uses a different posture recipe; case unsupported, no retuning')
    packets=build_packets(capture,jaws,'fixed-9-8','latched',anchor)
    actions=encode_stream(packets)
    decode_stream(actions,anchor)
    return actions


def prepare_teacher(teacher):
    from capstone_vla.hgwind_expert_trace import patch_collector,patch_profile_wrapper
    teacher=Path(teacher).resolve();teacher.relative_to(ROOT/'data/derived')
    prior=json.loads((teacher/'plan.json').read_text());report=json.loads((teacher/'report.json').read_text())
    require(prior.get('trace_cartesian_targets') is True and not prior.get('independent_packets')
        and report.get('native_expert_reference_passed') is True
        and report.get('initial_state_match') is True
        and report.get('independent_grasp_evidence',{}).get('status')=='pass',
        'Teacher did not pass native physical/initial-state checks; retain video, skip surrogate')
    require(report['execution_fingerprints']=={k:prior[k] for k in [
        'worker_sha256','patch_sha256','controller_config_sha256','urdf_sha256','adapter_sha256']},
        'Teacher report/plan fingerprints differ')
    bundle=Path(prior['bundle_manifest']).parent;worker=Path(prior['worker'])
    for key in ['worker','task','bundle_manifest','controller_config']:
        require(sha256(prior[key])==prior[key+'_sha256'],'Teacher dependency changed: '+key)
    for name,digest in prior['adapter_sha256'].items():
        path=Path(name) if Path(name).is_absolute() else ROOT/name
        if path.is_relative_to(bundle):
            require(sha256(path)==digest,'Frozen teacher implementation changed: '+path.name)
    frozen={x['path']:x['sha256'] for x in json.loads(Path(prior['bundle_manifest']).read_text())['files']}
    for path in [worker,worker.parent/'expert.py',worker.parent/'collision_safety.py',worker.parent/'skills/catch.py']:
        require(frozen[str(path.relative_to(bundle))]==sha256(path),'Frozen native source changed')
    source=Path(prior['source']);verify_files(source.parent,prior['source_files'])
    meta=json.loads((source.parent/'meta.json').read_text())
    ep=Path(report['probe_episode']).resolve();require(ep.is_relative_to(teacher/'worker_output'),'Teacher episode escaped run')
    replay=ep/'replay.hdf5';capture_path=teacher/'cartesian_target_inputs.json'
    capture=json.loads(capture_path.read_text());segments=json.loads((teacher/'original_expert_control_references.json').read_text())
    n=int(report['recorded_frames'])
    require(audit_capture(capture,segments,n)['capture_alignment_passed'], 'Teacher upper calls do not align')
    with h5py.File(replay,'r') as f,h5py.File(source,'r') as original:
        q=f['obs/joint_pos'][0,:7];jaws=f['actions'][:,7:]
        require(np.array_equal(q,original['obs/joint_pos'][0,:7]),'Teacher initial robot does not match selected source')
    configuration=sampled_posture_configuration(meta['trajectory_variation'])
    actions=teacher_actions(capture,jaws,q,configuration)
    cfg=json.loads(Path(prior['controller_config']).read_text());lab=Path(cfg['isaaclab_root'])
    require((lab/'VERSION').read_text().strip()=='2.2.1' and
        ((lab/'_isaac_sim').resolve()/'VERSION').read_text().startswith('4.5.0'),'SDK changed')
    sys.path.insert(0,str(worker.parents[1]))
    from pxr import Usd
    from dataset.scene_signature import stage_physics_signature
    require(stage_physics_signature(Usd.Stage.Open(prior['task']))['sha256']==prior['physics_sha256'], 'Physics changed')
    original=worker.read_text();anchor='"skill_terminations": _jsonify(skill_terminations),'
    require(original.count(anchor)==1,'Native profile hook moved')
    original=original.replace(anchor,anchor+'\n            "capstone_expert_control": CAPSTONE_PROFILE_SUMMARY(pol),')
    patched=patch_packet_collector(patch_collector(original))
    patch_profile_wrapper(Path(prior['original_profile_wrapper']).read_text())
    stamp=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
    out=ROOT/'data/derived/hgwind_action_contract_v1'/(stamp+'_'+prior['episode'])
    plan=copy.deepcopy(prior)
    plan.update(output=str(out),worker_output=str(out/'worker_output'),
        action_contract=SCHEMA,action_vectors=str(out/'actions.npy'),
        action_contract_reference_kind='fresh_verified_native_teacher',
        action_contract_parent_report=str(teacher/'report.json'),action_contract_parent_report_sha256=sha256(teacher/'report.json'),
        teacher_replay=str(replay),teacher_replay_sha256=sha256(replay),
        trace_cartesian_targets=False,independent_packets=True,packet_gain_profile='fixed-9-8',
        packet_posture_profile='latched',robot_posture_configuration=configuration,
        frames=n,reference_frames=n,cartesian_target_capture=str(capture_path),
        cartesian_target_capture_sha256=sha256(capture_path),
        packet_baseline_report=str(teacher/'report.json'),packet_baseline_report_sha256=sha256(teacher/'report.json'),
        patch_sha256=hashlib.sha256(patched.encode()).hexdigest(),
        reference_fidelity='Fixed 29D float32 teacher-labelled surrogate, live-latched posture and always-on non-target geometry. No online teacher decisions, saved joint/OSC commands, model training or inference.',training_eligible=False)
    paths=[ROOT/'src/capstone_vla/cartesian_packets.py',ROOT/'src/capstone_vla/cartesian_target_trace.py',
        ROOT/'src/capstone_vla/cartesian_target_replay.py',ROOT/'src/capstone_vla/posture_anchor.py',
        ROOT/'src/capstone_vla/hgwind_expert_trace.py',ROOT/'src/capstone_vla/ball_action_contract.py',
        ROOT/'scripts/run_hgwind_action_contract.py',ROOT/'scripts/analyze_hgwind_effort_probe.py',Path(__file__),
        worker.parent/'expert.py',worker.parent/'collision_safety.py',worker.parent/'skills/catch.py']
    plan['adapter_sha256']={name:sha256(Path(name) if Path(name).is_absolute() else ROOT/name)
        for name in plan['adapter_sha256']}
    for path in paths:plan['adapter_sha256'][str(path)]=sha256(path)
    model_cfg=model_configuration(json.loads((ROOT/'configs/dynamicvla_dom_pretrained.json').read_text()))
    diagnostics=dict(frames=n,action_shape=list(actions.shape),action_dtype='float32',model_state_dim=30,
        source=prior['episode'],teacher_capture_verified=True,
        original_collection_arrays_exact=report['cartesian_target_capture_audit'].get('core_array_exact_equality'),
        action_array_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
        profile='unchanged_ball_action_v1',gpu_started=False,learned_policy_evaluated=False)
    return plan,actions,model_cfg,diagnostics,lab
