"""Explicit recorded-height intervention; never an unassisted VLA evaluation."""
import numpy as np
from .cartesian_target_replay import require

SCHEMA = 'physical-vla-lab.ball-recorded-z.v1'


def validate_plan(plan, frames):
    require(plan['schema'] == SCHEMA and plan['channel'] == 2,
            'Only absolute world target-position z may be replaced')
    require(plan['coordinate_frame'] == 'world' and plan['unit'] == 'm', 'Wrong height units/frame')
    start, stop = plan['start_frame'], plan['stop_frame_exclusive']
    require(type(start) is int and type(stop) is int and 0 <= start < stop <= frames,
            'Invalid intervention window')
    z = np.asarray(plan['target_z_m'], dtype=float)
    require(z.shape == (stop-start,) and np.isfinite(z).all(), 'Invalid recorded height reference')
    return start, stop, z


def apply_height(action, frame, plan, frames):
    raw = np.asarray(action)
    require(raw.shape == (29,) and np.isfinite(raw).all(), 'Invalid model action')
    require(type(frame) is int and 0 <= frame < frames, 'Invalid action frame')
    # float64 preserves the saved float32 reference exactly, matching archive dtype.
    result = raw.astype(float, copy=True)
    if plan is not None:
        start, stop, z = validate_plan(plan, frames)
        if start <= frame < stop:
            result[2] = z[frame-start]
    return result


def audit_height(raw, executed, frame, plan, frames):
    np.testing.assert_array_equal(executed, apply_height(raw, frame, plan, frames))


def validate_provenance(config):
    diagnostic = config.get('vertical_diagnostic') is not None
    require(config['source_actions_used_for_control'] is diagnostic and
            config['learned_policy_evaluated'] is (not diagnostic), 'Wrong intervention provenance')
    if diagnostic:
        validate_plan(config['vertical_diagnostic'], config['frames'])
    return diagnostic
