"""Versioned ball-development action codec, independent of saved expert decisions.

Actions carry all varying upper-request fields. Constant native settings and live
robot posture belong to the controller. Collision callbacks are always enabled.
This is a proposed simulation contract; its physical gate is separate from VLA
training/evaluation. No object state or saved inner control output is decoded.
"""
import copy
import numpy as np
from .cartesian_target_replay import CALLBACKS, require

SCHEMA = 'physical-vla-lab.ball-action.v1'
ACTION_DIM = 29
STATE_DIM = 30
# Half-open slices. Scalar channels have width one, including nullable masks.
CHANNELS = (
    ('target_position', 3), ('target_quaternion', 4),
    ('target_linear_velocity', 3), ('target_linear_acceleration', 3),
    ('acceleration_enabled', 1), ('linear_velocity_reference', 3),
    ('velocity_reference_enabled', 1), ('maximum_relative_linear_speed_mps', 1),
    ('relative_speed_enabled', 1), ('orientation_axis_world', 3),
    ('orientation_axis_enabled', 1), ('orientation_roll_weight', 1),
    ('joint_limit_margin_rad', 1), ('preserve_orientation_during_translation', 1),
    ('operational_orientation_task', 1), ('gripper_closed', 1),
)
SLICES = {}
_offset = 0
for _name, _width in CHANNELS:
    SLICES[_name] = slice(_offset, _offset + _width)
    _offset += _width
assert _offset == ACTION_DIM

# Explicit controller constants, not estimated from a trajectory at runtime.
FIXED_INPUTS = dict(
    target_angular_velocity=[0., 0., 0.], joint_velocity_reference=None,
    grasp_weight=1., curobo_weight=0., smoothness_weight=.03, dt=1/60,
    finite_difference_rad=.0001, position_gain=9., orientation_gain=8.,
    maximum_linear_speed_mps=1.7, maximum_angular_speed_radps=2.5,
    maximum_joint_speed_radps=None, maximum_joint_acceleration_radps2=30.,
    maximum_joint_jerk_radps3=600., orientation_velocity_budget_fraction=1.,
    orientation_primary_weight=.2, orientation_error_vector_world=None,
    continuous_fusion=False, coupled_operational_pose_task=True,
    operational_orientation_length_scale_m=.03,
    use_operational_space_inverse_dynamics=True,
)
VARIABLE_INPUTS = {
    'target_position', 'target_quaternion', 'target_linear_velocity',
    'target_linear_acceleration', 'linear_velocity_reference',
    'maximum_relative_linear_speed_mps', 'orientation_axis_world',
    'orientation_roll_weight', 'joint_limit_margin_rad',
    'preserve_orientation_during_translation', 'operational_orientation_task',
}
NULLABLE = {
    'target_linear_acceleration': 'acceleration_enabled',
    'linear_velocity_reference': 'velocity_reference_enabled',
    'maximum_relative_linear_speed_mps': 'relative_speed_enabled',
    'orientation_axis_world': 'orientation_axis_enabled',
}


def specification():
    return dict(schema=SCHEMA, action_dim=ACTION_DIM, state_dim=STATE_DIM,
        control_hz=60, physics_hz=180, action_dtype='float32',
        position_frame='simulation_world', orientation='absolute_unit_quaternion_wxyz',
        velocity_frame='simulation_world', position_unit='m', velocity_unit='m/s',
        acceleration_unit='m/s^2', absolute_actions=True, upstream_use_delta_action=False,
        channels=[dict(name=name, start=SLICES[name].start, stop=SLICES[name].stop)
                  for name, _ in CHANNELS],
        binary_threshold=.5, scalar_projection=dict(orientation_roll_weight=[0.,1.],
            joint_limit_margin_rad=[0.,.08], maximum_relative_linear_speed_mps=[0.,1.7]),
        null_masks_are_actions=True, target_motion_is_predicted_not_read_from_simulator=True,
        fixed_inputs=copy.deepcopy(FIXED_INPUTS), collision_callbacks='always_enabled',
        posture='latched_from_initial_robot_plus_declared_configuration',
        state_channels=['arm_q_rad[7]', 'arm_qdot_radps[7]', 'pinch_world_position_m[3]',
            'pinch_world_quaternion_wxyz[4]', 'latched_posture_rad[7]', 'finger_q_m[2]'],
        terminal_mode='no_hidden_phase_schedule; explicit_pose_motion_each_tick',
        required_observations=['camera_history', 'robot_state_30', 'instruction'],
        source='expert_upper_command_labels_for_development', learned_policy_evaluated=False)


def _vector(value, width, name):
    result = np.asarray(value, dtype=float)
    require(result.shape == (width,) and np.isfinite(result).all(), 'Invalid '+name)
    return result


def _unit(value, width, name):
    value = _vector(value, width, name)
    norm = np.linalg.norm(value)
    require(norm > 1.e-6, 'Degenerate '+name)
    return value / norm


def encode_packet(packet):
    inputs = packet['inputs']
    require(set(inputs) == set(FIXED_INPUTS) | VARIABLE_INPUTS | {'posture_target'},
        'Unmapped/missing native command field')
    require(all(inputs[k] == v for k, v in FIXED_INPUTS.items()),
        'Native settings differ from the frozen ball contract')
    require(type(packet['gripper_closed']) is bool, 'Nonbinary jaw label')
    result = np.zeros(ACTION_DIM, dtype=np.float32)
    for name in VARIABLE_INPUTS:
        value = inputs[name]
        if name in NULLABLE:
            result[SLICES[NULLABLE[name]]] = float(value is not None)
            if value is None:
                continue
        slot = SLICES[name]
        if slot.stop-slot.start == 1:
            require(isinstance(value, (int, float, bool)) and np.isfinite(value), 'Invalid '+name)
            if name in ('preserve_orientation_during_translation','operational_orientation_task'):
                require(type(value) is bool, 'Nonbinary '+name)
            result[slot] = value
        else:
            result[slot] = _vector(value, slot.stop-slot.start, name)
    require(abs(np.linalg.norm(inputs['target_quaternion'])-1.) < 1.e-5, 'Invalid label quaternion')
    require(0 <= inputs['orientation_roll_weight'] <= 1 and 0 <= inputs['joint_limit_margin_rad'] <= .08,
        'Label soft objective outside contract')
    cap = inputs['maximum_relative_linear_speed_mps']
    require(cap is None or 0 <= cap <= 1.7, 'Label relative-speed cap outside contract')
    result[SLICES['gripper_closed']] = packet['gripper_closed']
    require(np.isfinite(result).all(), 'Nonfinite float32 action')
    return result


def encode_stream(packets):
    require(bool(packets) and [p['frame'] for p in packets] == list(range(len(packets))),
        'Noncontiguous action labels')
    # Quaternion signs are made causal/continuous; q and -q denote one rotation.
    result = np.stack([encode_packet(p) for p in packets])
    slot = SLICES['target_quaternion']
    for i in range(1, len(result)):
        if np.dot(result[i-1,slot], result[i,slot]) < 0:
            result[i,slot] *= -1
    return result


def decode_action(action, frame, anchor=None):
    action = _vector(action, ACTION_DIM, 'action')
    require(isinstance(frame, int) and frame >= 0, 'Invalid control frame')
    inputs = copy.deepcopy(FIXED_INPUTS)
    for name in VARIABLE_INPUTS:
        value = action[SLICES[name]]
        if name in NULLABLE and action[SLICES[NULLABLE[name]]][0] < .5:
            inputs[name] = None
        elif name == 'target_quaternion':
            inputs[name] = _unit(value,4,name).tolist()
        elif name == 'orientation_axis_world':
            inputs[name] = _unit(value,3,name).tolist()
        elif name in ('preserve_orientation_during_translation','operational_orientation_task'):
            inputs[name] = bool(value[0] >= .5)
        elif len(value) == 1:
            upper = {'orientation_roll_weight':1., 'joint_limit_margin_rad':.08,
                'maximum_relative_linear_speed_mps':1.7}[name]
            inputs[name] = float(np.clip(value[0],0.,upper))
        else:
            inputs[name] = value.tolist()
    inputs['posture_target'] = None if anchor is None else _vector(anchor,7,'robot posture').tolist()
    callbacks = {key:dict(callable_descriptor_only=True,module=module,qualified_name=method)
                 for key,(module,method) in CALLBACKS.items()}
    return dict(frame=frame, inputs=inputs, safety_callbacks=callbacks,
        gripper_closed=bool(action[SLICES['gripper_closed']][0] >= .5),
        # This is telemetry only. There is no phase/FSM or timer-triggered hold.
        terminal_cartesian_hold=False)


def decode_stream(actions, anchor=None):
    actions = np.asarray(actions)
    require(actions.ndim == 2 and actions.shape[1] == ACTION_DIM and len(actions) >= 2
        and np.isfinite(actions).all(), 'Invalid action stream')
    return [decode_action(action,i,anchor) for i,action in enumerate(actions)]


def robot_state(q, qdot, pinch_position, pinch_quaternion, anchor, fingers):
    values = [_vector(q,7,'robot q'), _vector(qdot,7,'robot qdot'),
        _vector(pinch_position,3,'pinch position'), _unit(pinch_quaternion,4,'pinch quaternion'),
        _vector(anchor,7,'robot anchor'), _vector(fingers,2,'finger q')]
    result = np.concatenate(values).astype(np.float32)
    require(result.shape == (STATE_DIM,) and np.isfinite(result).all(), 'Invalid robot state')
    return result


def model_configuration(pretrained):
    result = copy.deepcopy(pretrained)
    require(result['max_action_dim'] == 32 and result['max_state_dim'] == 32,
        'Unexpected upstream capacity')
    result['output_features']['action']['shape'] = [ACTION_DIM]
    result['input_features']['observation.state']['shape'] = [STATE_DIM]
    result['use_delta_action'] = False
    return result


def audit_contract(payload, actions):
    """Inspect actual motor-boundary inputs, not only the offline codec output."""
    from .posture_anchor import audit_latched_posture
    events = payload['calls']
    initialization = payload.get('posture_initialization')
    expected = decode_stream(actions, initialization['anchor_rad']) if initialization else []
    exact = len(events) == len(expected) and all(
        e['frame'] == p['frame'] and e['inputs'] == dict(p['inputs'], **p['safety_callbacks'])
        and e['gripper_closed'] is p['gripper_closed'] for e,p in zip(events,expected))
    return dict(schema=SCHEMA, action_dim=ACTION_DIM, state_dim=STATE_DIM,
        decoded_actions_match_motor_boundary=exact,
        fixed_collision_callbacks_always_enabled=bool(events) and all(
            all(e['inputs'].get(k) == decode_action(actions[0],0)['safety_callbacks'][k]
                for k in CALLBACKS) for e in events),
        live_posture_verified=audit_latched_posture(events,initialization),
        learned_policy_evaluated=False)
