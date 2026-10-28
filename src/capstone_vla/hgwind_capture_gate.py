"""Opt-in frictional LIFT admission using the existing relative-speed limit.

No dwell, world-speed threshold, attachment, drive gain or success change.
"""
import math


def install_capture_speed_gate(catch_class):
    if getattr(catch_class, '_capstone_capture_speed_gate_installed', False):
        raise RuntimeError('Capture speed gate already installed')
    original = catch_class._begin_lift_from_physical_grasp

    def begin(self, *, p_A, p_ee, q_ee, v_ee):
        stats = getattr(self, '_capstone_capture_speed_gate_stats', None)
        if stats is None:
            stats = dict(deferred_unsafe_calls=0, admitted_calls=0, last_relative_speed_mps=None,
                         last_limit_mps=None, required_dwell_s=0.0)
            self._capstone_capture_speed_gate_stats = stats
        # Runtime attachment has its own ARREST/load-bearing admission. This
        # diagnostic only constrains the physical frictional pinch boundary.
        if not self._uses_runtime_attachment():
            speed = float(self._capture_relative_speed_mps)
            limit = float(self._capture_speed_limit_mps())
            stats['last_relative_speed_mps'] = speed if math.isfinite(speed) else None
            stats['last_limit_mps'] = limit if math.isfinite(limit) else None
            if not (math.isfinite(speed) and math.isfinite(limit) and 0 <= speed <= limit):
                stats['deferred_unsafe_calls'] += 1
                # Capture is a current physical condition, not permission
                # that may survive an arbitrarily deferred LIFT. Revoke it
                # so TRACK must requalify fresh contact/geometry next tick.
                self._physical_grasp_confirmed = False
                return None
        stats['admitted_calls'] += 1
        return original(self, p_A=p_A, p_ee=p_ee, q_ee=q_ee, v_ee=v_ee)

    catch_class._begin_lift_from_physical_grasp = begin
    catch_class._capstone_capture_speed_gate_installed = True


def summary(policy):
    primitives = []
    for primitive in policy.active_primitives() if policy is not None else []:
        if hasattr(primitive, '_capstone_capture_speed_gate_stats'):
            primitives.append(dict(primitive._capstone_capture_speed_gate_stats))
    return {'enabled': True, 'revision': 'v4.1_requalify_after_deferral',
            'revokes_confirmation_when_deferred': True, 'mechanism': 'existing_relative_speed_limit_at_frictional_lift_entry',
            'required_dwell_s': 0.0, 'primitives': primitives,
            'scene_or_gripper_limits_changed': False, 'success_thresholds_changed': False}
