"""Opt-in physical-grasp transition experiment; original hard guards still act.

This changes expert command generation, not contact truth or task scoring.
It is an unvalidated simulation candidate until new physical rollouts pass.
"""
import ast
from dataclasses import asdict, dataclass
import inspect
import textwrap

import numpy as np


@dataclass(frozen=True)
class SmoothLiftConfig:
    name: str = 'smooth_lift_v1'
    settle_s: float = 0.08
    lift_speed_scale: float = 0.5
    acceleration_mps2: float = 0.5
    jerk_mps3: float = 5.0
    velocity_gain: float = 6.0
    position_gain: float = 8.0


def profile_configuration(name):
    if name == 'smooth-lift':
        return SmoothLiftConfig()
    if name == 'smooth-only':
        return SmoothLiftConfig(name='smooth_only_v1', settle_s=0.0)
    if name == 'mimic-safe':
        return SmoothLiftConfig(name='mimic_safe_v3', settle_s=0.0)
    raise ValueError('Unknown adapted HGwind profile: ' + str(name))


@dataclass
class ContactDwell:
    last_tick: int = -2
    elapsed_s: float = 0.0

    def observe(self, tick, eligible, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('positive finite contact dwell dt required')
        tick = int(tick)
        if tick == self.last_tick:
            if not eligible:
                self.elapsed_s = 0.0
            return self.elapsed_s
        consecutive = tick == self.last_tick + 1
        self.elapsed_s = (self.elapsed_s if consecutive else 0.0) + dt if eligible else 0.0
        self.last_tick = tick
        return self.elapsed_s


def _new_stats():
    return {'settle_deferred_calls': 0, 'settle_admitted_calls': 0,
            'reference_updates': 0, 'collision_projection_overrides': 0,
            'position_lead_caps': 0, 'maximum_smooth_acceleration_mps2': 0.0,
            'maximum_smooth_jerk_mps3': 0.0}


def smooth_reference(skill, goal, requested_velocity, measured_position, quaternion, dt):
    config = skill._capstone_lift_config
    stats = skill._capstone_lift_stats
    previous_position = np.asarray(skill._carry_position, dtype=float).reshape(3)
    previous_velocity = np.asarray(skill._carry_velocity, dtype=float).reshape(3)
    previous_acceleration = np.asarray(skill._carry_acceleration, dtype=float).reshape(3)
    maximum_speed = float(skill.param('center_pointing_differential_max_linear_speed', 1.7))
    position, velocity, acceleration = skill._capstone_reference_step(
        previous_position, previous_velocity, previous_acceleration, goal,
        requested_velocity, dt, position_gain=config.position_gain,
        velocity_gain=config.velocity_gain, maximum_speed=maximum_speed,
        maximum_acceleration=config.acceleration_mps2, maximum_jerk=config.jerk_mps3)
    stats['reference_updates'] += 1
    stats['maximum_smooth_acceleration_mps2'] = max(
        stats['maximum_smooth_acceleration_mps2'], float(np.linalg.norm(acceleration)))
    stats['maximum_smooth_jerk_mps3'] = max(stats['maximum_smooth_jerk_mps3'],
        float(np.linalg.norm(acceleration-previous_acceleration)/dt))
    # A filtered old velocity may differ from the newly collision-projected goal.
    # Reapply the original collision filter. Safety takes priority over smoothness;
    # report an override rather than claiming a jerk certificate in that case.
    safe = np.asarray(skill._environment_safe_translation(
        measured_position, quaternion, velocity,
        commanded_angular_velocity=np.zeros(3), dt=dt), dtype=float)
    if not np.allclose(safe, velocity, rtol=0, atol=1e-12):
        stats['collision_projection_overrides'] += 1
        velocity = safe
        acceleration = (velocity-previous_velocity)/dt
        position = np.asarray(measured_position) + velocity*dt
    lead = position-np.asarray(measured_position)
    maximum_lead = float(skill.param('fix_joint_carry_lookahead_m', 0.08))
    if maximum_lead > 0 and np.linalg.norm(lead) > maximum_lead:
        position = np.asarray(measured_position) + lead*maximum_lead/np.linalg.norm(lead)
        stats['position_lead_caps'] += 1
    skill._carry_position = position.copy()
    skill._carry_velocity = velocity.copy()
    skill._carry_acceleration = acceleration.copy()
    return position, velocity


def adapt_carry_method(original, callback):
    """Add one bounded-reference stage to the real physical carry path."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = tree.body[0]
    anchor = 'self._carry_command_velocity = command_velocity.copy()'
    matches = [i for i, node in enumerate(function.body) if ast.unparse(node) == anchor]
    if len(matches) != 1:
        raise RuntimeError('Physical carry insertion point changed; review upstream source')
    injected = ast.parse('''command_position, command_velocity = _capstone_smooth_reference(
    self, command_position, command_velocity, p_ee, q_ee, dt)
self._lift_target = command_position.copy()
''').body
    function.body[matches[0]:matches[0]] = injected
    namespace = dict(original.__globals__, _capstone_smooth_reference=callback)
    exec(compile(ast.fix_missing_locations(tree), '<capstone_smooth_physical_carry>', 'exec'), namespace)
    return namespace[original.__name__]


def install_profile(catch_module, config=None):
    config = config or SmoothLiftConfig()
    values = [config.lift_speed_scale, config.acceleration_mps2,
              config.jerk_mps3, config.velocity_gain, config.position_gain]
    if (not np.isfinite(config.settle_s) or config.settle_s < 0
            or not all(np.isfinite(v) and v > 0 for v in values) or config.lift_speed_scale > 1):
        raise ValueError('Invalid physical lift control profile')
    cls = catch_module.Catch
    if getattr(cls, '_capstone_smooth_profile_installed', False):
        raise RuntimeError('Control profile already installed')
    original_init, original_reset = cls.__init__, cls.reset
    original_begin, original_carry = cls._begin_lift_from_physical_grasp, cls._drive_carry

    def reset(self, *args, **kwargs):
        result = original_reset(self, *args, **kwargs)
        self._capstone_contact_dwell = ContactDwell()
        self._capstone_lift_stats = _new_stats()
        self._capstone_lift_config = config
        self._capstone_reference_step = catch_module._jerk_limited_cartesian_target_step
        return result

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Keep the original per-attempt speed randomization multiplier.
        self.params['lift_speed'] = float(self.param('lift_speed', 0.10))*config.lift_speed_scale

    def begin_lift(self, *, p_A, p_ee, q_ee, v_ee):
        if config.settle_s > 0 and not self._uses_runtime_attachment():
            speed = float(self._capture_relative_speed_mps)
            limit = float(self._capture_speed_limit_mps())
            centered = bool(np.linalg.norm(np.asarray(self._grasp_center_world)
                              - np.asarray(p_ee)) <= self._pinch_band())
            eligible = bool(centered and self._physical_grasp_confirmed and self._jaws_closed
                            and self._contact_pair_now and self._finger_segment_solid_ready
                            and np.isfinite(speed) and 0 <= speed <= limit)
            elapsed = self._capstone_contact_dwell.observe(
                self.step_count, eligible, float(self._dt))
            if elapsed+1e-12 < config.settle_s:
                self._capstone_lift_stats['settle_deferred_calls'] += 1
                return
            self._capstone_lift_stats['settle_admitted_calls'] += 1
        return original_begin(self, p_A=p_A, p_ee=p_ee, q_ee=q_ee, v_ee=v_ee)

    cls.reset, cls.__init__ = reset, initialize
    cls._begin_lift_from_physical_grasp = begin_lift
    cls._drive_carry = adapt_carry_method(original_carry, smooth_reference)
    cls._capstone_smooth_profile_installed = True
    return asdict(config)


def profile_summary(policy):
    rows = []
    for primitive in policy.active_primitives() if policy is not None else []:
        config = getattr(primitive, '_capstone_lift_config', None)
        if config is not None:
            rows.append({'configuration': asdict(config),
                         'applied_lift_speed_mps': float(primitive.param('lift_speed', .1)),
                         'measurements': dict(primitive._capstone_lift_stats)})
    return {'profile': rows[0]['configuration']['name'] if rows else 'not_initialized', 'primitives': rows,
            'hard_motion_guards_changed': False, 'success_thresholds_changed': False}
