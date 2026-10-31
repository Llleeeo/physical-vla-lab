"""One exposed float32 control: passive mimic follower, unchanged leader.

This changes gripper actuation, including available active drive authority.
It neither proves a mimic defect nor certifies a production controller.
"""
import copy
import json
from pathlib import Path
import h5py
import numpy as np
from .action_precision_control import BASELINE, SUMMARY, matched_plan
from .cartesian_target_replay import require
from .gripper_gain_control import GainControl
from .hf_archive import sha256

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'physical-vla-lab.mimic-follower-control.v1'
GAINS = [[[10500., 0.]], [[140., 0.]]]


def verify_mimic(rows):
    """Fail before the setter unless the expected authored follower exists."""
    require(len(rows) == 2, 'Need exactly two observed fingers')
    first = [r for r in rows if r['path'].endswith('/panda_finger_joint1')]
    second = [r for r in rows if r['path'].endswith('/panda_finger_joint2')]
    require(len(first) == len(second) == 1, 'Ambiguous mimic finger identity')
    a, b = first[0], second[0]
    key = 'physxMimicJoint:rotX:'
    require(not any('Mimic' in s for s in a['schemas'])
        and [s for s in b['schemas'] if 'Mimic' in s] == ['PhysxMimicJointAPI:rotX']
        and b['relationships'].get(key+'referenceJoint') == [a['path']]
        and b['attributes'].get(key+'gearing') == -1.
        and b['attributes'].get(key+'offset') == 0.
        and b['attributes'].get(key+'referenceJointAxis') == 'rotX',
        'Expected original Panda mimic coupling is absent or changed')


class SingleDriveGainControl(GainControl):
    def snapshot(self):
        value = super().snapshot()
        try:
            require(self.config['requested_gains'] == GAINS, 'Unexpected single-drive gains')
            verify_mimic(value['usd_fingers'])
        except (ValueError, KeyError, TypeError) as exc:
            self.fail('Mimic precondition/readback failed: '+str(exc))
        self.state['authored_mimic_checked'] = True
        self.state['follower_drive_passive'] = value['live_gains'] == GAINS
        return value


def verify_plan(plan):
    control = plan['mimic_drive_control']
    require(control['schema'] == SCHEMA and control['baseline'] == str(BASELINE)
        and control['training_eligible'] is False and not plan.get('precision_control'),
        'Only the pinned float32 follower-drive diagnostic is permitted')
    require(plan['gripper_gain_control']['requested_gains'] == GAINS,
        'Only follower stiffness/damping can change')
    baseline = json.loads((BASELINE/'plan.json').read_text())
    for name, key in [('plan.json','baseline_plan_sha256'), ('actions.npy','baseline_actions_sha256'),
                      ('report.json','baseline_report_sha256')]:
        require(sha256(BASELINE/name) == control[key], 'Baseline evidence changed: '+name)
    # Compare every existing physics/control key before masking the ONE treatment.
    comparison = copy.deepcopy(plan)
    comparison['gripper_gain_control']['requested_gains'] = baseline['gripper_gain_control']['requested_gains']
    matched_plan(comparison, baseline)
    return baseline


def prepare(plan, actions, diagnostics):
    summary = json.loads((ROOT/SUMMARY).read_text())
    for name, digest in summary['source_hashes'].items():
        require(sha256(ROOT/name) == digest, 'Pinned float32 failure evidence changed: '+name)
    require(actions.dtype == np.float32 and actions.shape == (319,29)
        and np.array_equal(actions, np.load(BASELINE/'actions.npy', allow_pickle=False)),
        'Single-drive control must use exactly the failed float32 actions')
    plan['gripper_gain_control']['requested_gains'] = copy.deepcopy(GAINS)
    plan['mimic_drive_control'] = dict(schema=SCHEMA, baseline=str(BASELINE),
        baseline_plan_sha256=sha256(BASELINE/'plan.json'),
        baseline_actions_sha256=sha256(BASELINE/'actions.npy'),
        baseline_report_sha256=sha256(BASELINE/'report.json'), training_eligible=False)
    plan['reference_fidelity'] = ('One float32 expert-label diagnostic: follower native stiffness/damping zero, '
        'leader10500/140 unchanged, original mimic retained. Grip authority changes; all arm limits and '
        'outer targets retained. Not learned policy or a proven controller fix.')
    plan['adapter_sha256'][str(Path(__file__).resolve())] = sha256(__file__)
    verify_plan(plan)
    diagnostics.update(mimic_drive_control=SCHEMA, requested_finger_gains=GAINS,
        baseline_float32_actions_exact=True, effective_grip_authority_changed=True,
        physical_outcome='pending', model_training_eligible=False)


def verify_runtime(plan):
    verify_plan(plan)
    require(sha256(plan['action_vectors']) == plan['action_vectors_sha256'], 'Action file changed')
    actions = np.load(plan['action_vectors'], allow_pickle=False)
    require(actions.dtype == np.float32 and np.array_equal(actions,
        np.load(BASELINE/'actions.npy', allow_pickle=False)), 'Float32 diagnostic stream changed')


def motion_summary(episode):
    """Compare a fixed recorded-time window; full execution is checked separately."""
    meta = json.loads((Path(episode)/'meta.json').read_text())
    dynamic = meta['arm_execution']['dynamic_drive']
    with h5py.File(Path(episode)/'replay.hdf5', 'r') as h:
        t = h['evidence/motion/timestamp'][:]
        v = h['evidence/motion/obs/joint_vel'][:]
        require(len(t) >= 2 and len(t) == len(v) and v.ndim == 2
            and v.shape[1] == 9 and np.isfinite(v).all()
            and np.isfinite(t).all() and np.all(np.diff(t) > 0), 'Invalid motion evidence')
        dt = np.diff(t)
        mask = (t[1:] >= 2.84) & (t[1:] < 2.99)
        result = dict(window_s=[2.84,2.99], window_samples=int(mask.sum()),
            window_covered=bool(t[0] <= 2.84 and t[-1] >= 2.99))
        if mask.any():
            result.update(joint5_max_abs_velocity_rad_s=float(np.max(abs(v[1:,4][mask]))),
                joint5_max_abs_secant_acceleration_rad_s2=float(np.max(abs(np.diff(v[:,4])[mask]/dt[mask]))),
                finger_max_abs_velocity_m_s=np.max(abs(v[1:,7:][mask]),axis=0).tolist())
    feasibility = dynamic['effort_feasibility']
    result.update(effort_projection_count=feasibility['projection_count'],
        effort_rejection_count=feasibility['rejection_count'],
        maximum_model_velocity_prediction_error_rad_s_per_joint=feasibility['maximum_model_velocity_prediction_error_rad_s_per_joint'])
    return result


def audit(plan, initialization):
    root = Path(plan['output'])
    result = dict(schema=SCHEMA, intervention_verified=False,
        execution_complete=initialization['execution_complete'], grasp=initialization['grasp'],
        combined_passed=False, learned_policy_evaluated=False, training_eligible=False,
        official_benchmark_success=None, errors=[])
    try:
        verify_runtime(plan)
        gain = json.loads((root/'gripper_gain_control.json').read_text())
        result['intervention_verified'] = bool(initialization['gain']['verified']
            and gain.get('authored_mimic_checked') and gain.get('follower_drive_passive')
            and gain['after']['live_gains'] == GAINS and gain['caps_targets_usd_state_preserved'])
        current = json.loads((root/'report.json').read_text())
        baseline = json.loads((BASELINE/'report.json').read_text())
        result.update(initialization_verified=initialization['initialization_verified'],
            interface_passed=initialization['interface_passed'], completed_intervals=current['completed_frames'],
            runtime_safety_passed=current['runtime_safety_passed'],
            independent_grasp_evidence=current['independent_grasp_evidence'],
            current_motion=motion_summary(current['probe_episode']),
            baseline_motion=motion_summary(baseline['probe_episode']),
            baseline_completed_intervals=baseline['completed_frames'])
        result['combined_passed'] = bool(result['intervention_verified'] and initialization['combined_passed'])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result['errors'].append(repr(exc))
    (root/'mimic_drive_comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('MIMIC_DRIVE_CONTROL verified='+str(result['intervention_verified']).lower()
        +' complete='+str(result['execution_complete']).lower()+' grasp='+str(result['grasp'])
        +' combined_passed='+str(result['combined_passed']).lower(), flush=True)
    return result
