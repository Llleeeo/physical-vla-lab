#!/usr/bin/env python3
"""Optional 2mm downward approach-goal bias around the reviewed collector."""
import os
from pathlib import Path
import runpy
import sys
import numpy as np


def install_depth_bias(cls, depth):
    if not 0 <= depth <= .002:
        raise ValueError('Overnight pilot depth must lie in [0, 2mm]')
    original_approach, original_reset = cls._approach_action, cls.reset

    def reset(self, *args, **kwargs):
        result = original_reset(self, *args, **kwargs)
        self._capstone_depth_stats = {'calls': 0, 'limited_calls': 0, 'maximum_applied_goal_bias_m': 0.0}
        return result

    def approach(self, goal, quat, dt, p_ee, **kwargs):
        if not depth:
            return original_approach(self, goal, quat, dt, p_ee, **kwargs)
        up = np.asarray(self._up, dtype=float).reshape(3)
        length = float(np.linalg.norm(up))
        if not np.isfinite(length) or length < 1e-9 or not np.isfinite(dt) or dt <= 0:
            raise ValueError('Invalid depth-bias frame/cadence')
        up = up / length
        q = kwargs.get('current_pinch_quaternion')
        q = quat if q is None else q
        # Reuse the original swept-pose filter for the requested downward
        # increment; subsequent original joint/pose guards remain in charge.
        safe = np.asarray(self._environment_safe_translation(
            p_ee, q, -up * depth / dt,
            commanded_angular_velocity=np.zeros(3), dt=dt), dtype=float)
        if safe.shape != (3,) or not np.isfinite(safe).all():
            raise RuntimeError('Invalid collision-projected depth increment')
        applied = float(np.clip(-np.dot(safe, up) * dt, 0, depth))
        stats = self._capstone_depth_stats
        stats['calls'] += 1
        stats['limited_calls'] += int(applied < depth - 1e-10)
        stats['maximum_applied_goal_bias_m'] = max(stats['maximum_applied_goal_bias_m'], applied)
        shifted = np.asarray(goal, dtype=float).copy() - up * applied
        return original_approach(self, shifted, quat, dt, p_ee, **kwargs)

    cls.reset, cls._approach_action = reset, approach


def main():
    bundle = Path(os.environ['CAPSTONE_FROZEN_BUNDLE']).resolve(strict=True)
    repo = bundle / 'data/derived/hgwind_current_baseline_20261027/repo'
    sys.path[:0] = [str(bundle / 'src'), str(repo / 'Scripts')]
    from expert_data.skills import catch
    import capstone_vla.hgwind_control as control
    depth = float(os.environ['CAPSTONE_HGWIND_DEPTH_M'])
    install_depth_bias(catch.Catch, depth)
    original_summary = control.profile_summary
    def summary(policy):
        result = original_summary(policy)
        result['depth_bias'] = {'requested_downward_goal_bias_m': depth,
            'truth_geometry_changed': False, 'auxiliary_grasp_added': False,
            'physics_validated_improvement': False,
            'primitives': [dict(p._capstone_depth_stats) for p in policy.active_primitives()
                           if hasattr(p, '_capstone_depth_stats')]}
        return result
    control.profile_summary = summary
    print('CAPSTONE_DEPTH_BIAS requested_m=' + str(depth) + ' collision_projection=retained actual_depth_not_guaranteed=true', flush=True)
    runpy.run_path(str(bundle / 'scripts/sim/collect_hgwind_worker.py'), run_name='__main__')


if __name__ == '__main__':
    main()
