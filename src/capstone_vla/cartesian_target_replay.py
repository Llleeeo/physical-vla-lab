"""Replay upper requests through live native feedback; exposed expert diagnostic only.

No saved joint output, Jacobian or OSC constraint enters the native call. Native
expert call timing, live safety callbacks, gripper and joint-only fallback remain.
"""
import copy
import functools
import inspect
import math
import numpy as np

from .cartesian_target_trace import encode, reference_matches

CALLBACKS = {
    'joint_velocity_projector': ('expert_data.collision_safety', 'CollisionSafety._project_keep_clear_joint_velocity'),
    'cartesian_velocity_projector': ('expert_data.collision_safety', 'CollisionSafety._environment_safe_translation'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_inputs(inputs):
    require(isinstance(inputs, dict), 'Upper inputs must be a dictionary')
    for key, value in inputs.items():
        if isinstance(value, dict):
            require(key in CALLBACKS and value == dict(callable_descriptor_only=True,
                module=CALLBACKS[key][0], qualified_name=CALLBACKS[key][1]), 'Unsupported upper input descriptor: '+key)
        elif value is not None:
            require(isinstance(value, (bool, int, float, list)), 'Unsupported upper input: '+key)
            array = np.asarray(value, float)
            require(np.isfinite(array).all(), 'Nonfinite upper input: '+key)
    for key, size in [('target_position', 3), ('target_quaternion', 4)]:
        require(np.asarray(inputs.get(key)).shape == (size,), 'Invalid upper goal: '+key)
    require(abs(np.linalg.norm(inputs['target_quaternion'])-1.) < 1.e-5, 'Upper quaternion is not unit wxyz')
    require(math.isclose(inputs.get('dt', 0.), 1/60, rel_tol=0., abs_tol=1.e-12), 'Upper target clock changed')


class CartesianTargetReplayer:
    def __init__(self, capture):
        require(capture.get('schema') == 'physical-vla-lab.cartesian-target-capture.v1', 'Unknown target capture')
        require(capture['frames'] == list(range(len(capture['frames']))) and len(capture['frames']) >= 2, 'Invalid target frame coverage')
        calls = capture['calls']
        require(calls and [x['call_id'] for x in calls] == list(range(len(calls))), 'Invalid target call order')
        require([x['frame'] for x in calls] == sorted(x['frame'] for x in calls), 'Target frames are not ordered')
        self.expected_frames = len(capture['frames'])
        self.by_frame = {}
        for event in calls:
            require(event['completed'] is True and 0 <= event['frame'] < self.expected_frames, 'Incomplete/out-of-range upper call')
            validate_inputs(event['inputs'])
            # Deliberately discard recorded outputs and pending references.
            self.by_frame.setdefault(event['frame'], []).append(dict(source_call_id=event['call_id'], inputs=copy.deepcopy(event['inputs'])))
        self.expert = None
        self.frame = None
        self.frames = []
        self.events = []
        self.used = 0
        self.original = None

    def begin_frame(self, expert, index):
        require(index == len(self.frames) and index < self.expected_frames, 'Upper replay clock/capture exhausted')
        if self.frame is not None:
            self.finish_frame(self.frame)
        if self.expert is None:
            self.expert = expert
            self.original = expert.operational_space_pinch_action
            signature = inspect.signature(self.original)
            original = self.original

            @functools.wraps(original)
            def replay(*args, **kwargs):
                rows = self.by_frame.get(self.frame, [])
                require(self.used < len(rows), 'Unexpected native upper call in this frame')
                source = rows[self.used]
                incoming = signature.bind(*args, **kwargs)
                incoming.apply_defaults()
                require(set(incoming.arguments) == set(source['inputs']), 'Native upper signature changed')
                applied = {}
                for key, value in source['inputs'].items():
                    if key in CALLBACKS:
                        live = incoming.arguments[key]
                        require(encode(live) == value, 'Native safety callback differs: '+key)
                        applied[key] = live
                    else:
                        applied[key] = np.asarray(value, float).copy() if isinstance(value, list) else value
                event = dict(call_id=len(self.events), source_call_id=source['source_call_id'], frame=self.frame,
                    inputs=encode(applied), completed=False, native_safety_callbacks_preserved=True)
                self.events.append(event)
                self.used += 1
                try:
                    result = original(**applied)
                except Exception as exc:
                    event['exception_type'] = type(exc).__name__
                    raise
                event['completed'] = True
                event['returned_action'] = None if result is None else encode({
                    key: getattr(result, key, None) for key in ['joint_indices', 'joint_positions', 'joint_velocities']})
                event['pending_operational_reference'] = encode(getattr(expert, '_pending_operational_space_reference', None))
                return result
            expert.operational_space_pinch_action = replay
        require(expert is self.expert, 'Upper replay expects one robot/episode')
        self.frame = int(index)
        self.frames.append(self.frame)
        self.used = 0

    def finish_frame(self, index):
        require(index == self.frame and self.used == len(self.by_frame.get(index, [])), 'Missing native upper call in this frame')

    def segment_links(self, index, segment):
        self.finish_frame(index)
        events = [x for x in self.events if x['frame'] == index]
        selected = reference_matches(events, dict(segment, frame=index))
        return dict(selected_cartesian_call_ids=selected if len(selected) == 1 else [])

    def restore(self):
        if self.expert is not None:
            self.expert.operational_space_pinch_action = self.original

    def payload(self):
        if self.frame is not None:
            self.finish_frame(self.frame)
        return dict(schema='physical-vla-lab.cartesian-target-reexecution.v1', frames=list(self.frames), calls=self.events,
            expected_frames=self.expected_frames, saved_inner_outputs_used=False,
            native_expert_call_schedule_retained=True, native_safety_callbacks_retained=True,
            native_gripper_and_joint_fallback_retained=True, source_controller_settings_schedule_retained=True,
            standalone_policy_controller=False, vla_evaluated=False)


def audit_reexecution(payload, capture, segments, frames):
    from .cartesian_target_trace import audit_capture
    alignment = audit_capture(payload, segments, frames)
    expected = capture['calls']
    events = payload['calls']
    prefix = expected[:len(events)]
    inputs_match = len(events) <= len(expected) and all(
        event.get('source_call_id') == source['call_id'] and event['frame'] == source['frame']
        and encode(event['inputs']) == source['inputs'] for event, source in zip(events, prefix))
    full = (payload['frames'] == capture['frames'] and len(events) == len(expected)
        and frames == len(capture['frames']) and len(segments) == frames)
    scope = (payload.get('saved_inner_outputs_used') is False
        and payload.get('standalone_policy_controller') is False
        and payload.get('native_safety_callbacks_retained') is True
        and all(x.get('native_safety_callbacks_preserved') is True for x in events))
    return dict(alignment=alignment, full_target_stream_completed=full, exact_applied_inputs_match=inputs_match,
        execution_scope_verified=scope, reexecution_verified=bool(full and inputs_match and scope
            and alignment['capture_alignment_passed'] and not alignment['failed_call_ids']),
        standalone_cartesian_policy_verified=False, vla_evaluated=False)
