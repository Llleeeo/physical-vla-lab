"""Explicit upper requests without calling the online task/skill decision loop.

The development stream retains recorded numeric controller settings. It is not
an approved learned action representation. Safety uses non-target scene geometry;
object motion is not an input to arm/gripper request generation.
"""
import copy
from types import SimpleNamespace
import numpy as np

from .cartesian_target_replay import CALLBACKS, require, validate_inputs
from .cartesian_target_trace import encode, reference_matches, audit_capture


class TargetBlindSafetyWorld:
    """Only geometry of declared non-target obstacles may reach safety refresh."""
    def __init__(self, world, forbidden):
        self.world = world
        self.forbidden = str(forbidden).rstrip('/')
        self.accessed_paths = set()
        self.blocked_paths = set()

    def __getattr__(self, name):
        if name not in ('pose', 'vel', 'velocity_reference', 'deformable_surface', 'emission_surface'):
            raise AttributeError('Unsupported safety world method: '+name)

        def access(path, *args, **kwargs):
            path = str(path)
            if path == self.forbidden or path.startswith(self.forbidden+'/'):
                self.blocked_paths.add(path)
                raise ValueError('Target truth access forbidden in safety adapter')
            self.accessed_paths.add(path)
            method = getattr(self.world, name, None)
            if method is None and name in ('deformable_surface', 'emission_surface'):
                return None
            require(callable(method), 'Missing geometry method: '+name)
            return method(path, *args, **kwargs)
        return access


def build_packets(capture, gripper_targets):
    from .cartesian_target_replay import CartesianTargetReplayer
    CartesianTargetReplayer(capture)  # Existing finite-input/clock validation.
    frames = capture['frames']
    jaws = np.asarray(gripper_targets, float)
    require(jaws.shape == (len(frames), 2) and np.isfinite(jaws).all(), 'Invalid explicit jaw targets')
    require(np.allclose(jaws[:, 0], jaws[:, 1], atol=1.e-6, rtol=0.)
        and np.all(np.isclose(jaws, 0., atol=1.e-5) | np.isclose(jaws, .04, atol=1.e-5)), 'Only binary Franka jaw targets are supported')
    calls = capture['calls']
    require([c['frame'] for c in calls] == list(range(len(calls))), 'Packet probe requires one source call per tick followed only by a terminal tail')
    packets = []
    for frame in frames:
        terminal = frame >= len(calls)
        source = calls[-1] if terminal else calls[frame]
        inputs = copy.deepcopy(source['inputs'])
        if terminal:
            # Explicit Cartesian hold; never copy saved joint fallback commands.
            inputs.update(target_linear_velocity=[0., 0., 0.], target_linear_acceleration=[0., 0., 0.],
                target_angular_velocity=[0., 0., 0.], linear_velocity_reference=None,
                maximum_relative_linear_speed_mps=None, joint_velocity_reference=None)
        callbacks = {key: inputs.pop(key) for key in CALLBACKS}
        packets.append(dict(frame=frame, inputs=inputs, safety_callbacks=callbacks,
            gripper_closed=bool(jaws[frame, 0] < .02), terminal_cartesian_hold=terminal))
    return packets


def execute_request(expert, packet, callbacks):
    """World meters/wxyz upper request + binary jaws -> newly computed actions."""
    inputs = copy.deepcopy(packet['inputs'])
    validate_inputs(dict(inputs, **packet['safety_callbacks']))
    for key, descriptor in packet['safety_callbacks'].items():
        live = callbacks[key] if descriptor is not None else None
        require(encode(live) == descriptor, 'Unexpected explicit safety callback: '+key)
        inputs[key] = live
    inputs = {key:np.asarray(value, float).copy() if isinstance(value, list) else value for key,value in inputs.items()}
    arm = expert.operational_space_pinch_action(**inputs)
    require(arm is not None, 'Native request returned no arm action')
    pending = encode(getattr(expert, '_pending_operational_space_reference', None))
    require(pending is not None, 'Explicit Cartesian request produced no OSC reference')
    jaw = expert.gripper_action(width=0., close=packet['gripper_closed'])
    require(jaw is not None, 'Explicit request returned no gripper action')
    return arm, jaw, pending, encode(inputs)


def refresh_robot_safety_motion(owner, expert, dt):
    """Mirror native robot-only pose differencing, without any skill decision."""
    position, quaternion = expert.pinch_pose()
    position = np.asarray(position, float).reshape(3).copy()
    quaternion = np.asarray(quaternion, float).reshape(4).copy()
    require(np.isfinite(position).all() and np.isfinite(quaternion).all()
        and abs(np.linalg.norm(quaternion)-1.) < 1.e-5, 'Invalid measured robot pinch pose')
    # Keep native arithmetic/order: the angular helper normalizes its current
    # quaternion, while native history stores the snapshot made before it.
    linear = np.asarray(owner._ee_velocity(position, dt), float).reshape(3).copy()
    owner._current_ee_q = quaternion.copy()
    angular = np.asarray(owner._ee_angular_velocity(quaternion, dt), float).reshape(3).copy()
    require(np.isfinite(linear).all() and np.isfinite(angular).all(), 'Invalid measured robot safety twist')
    owner._current_ee_linear_velocity = linear
    owner._current_ee_angular_velocity = angular
    return dict(position_m=position.copy(), quaternion_wxyz=owner._current_ee_q.copy()), dict(linear_mps=linear.copy(), angular_radps=angular.copy())


def audit_robot_safety_motion(events):
    """Check causal measured-pose history; never differentiate desired goals."""
    from scipy.spatial.transform import Rotation
    if not events:
        return False
    previous_position = previous_rotation = None
    for event in events:
        try:
            pose, twist = event['safety_robot_pose'], event['safety_robot_twist_world']
            position = np.asarray(pose['position_m'], float)
            quaternion = np.asarray(pose['quaternion_wxyz'], float)
            linear = np.asarray(twist['linear_mps'], float)
            angular = np.asarray(twist['angular_radps'], float)
            if position.shape!=(3,) or quaternion.shape!=(4,) or linear.shape!=(3,) or angular.shape!=(3,):
                return False
            if not all(np.isfinite(x).all() for x in [position,quaternion,linear,angular]) or abs(np.linalg.norm(quaternion)-1.)>1.e-5:
                return False
            rotation = Rotation.from_quat(quaternion[[1,2,3,0]])
            expected_linear = np.zeros(3) if previous_position is None else (position-previous_position)*60.
            expected_angular = np.zeros(3) if previous_rotation is None else (rotation*previous_rotation.inv()).as_rotvec()*60.
            if not np.allclose(linear,expected_linear,rtol=1.e-7,atol=1.e-8) or not np.allclose(angular,expected_angular,rtol=1.e-7,atol=1.e-8):
                return False
            previous_position, previous_rotation = position, rotation
        except (KeyError,ValueError,TypeError):
            return False
    return True


class PacketStreamExecutor:
    def __init__(self, packets):
        self.packets = copy.deepcopy(packets)
        self.events = []
        self.frames = []
        self.owner = None
        self.expert = None
        self.geometry_paths = set()

    def step(self, policy, world, index, dt):
        require(index == len(self.frames) and index < len(self.packets), 'Packet stream clock/capture exhausted')
        packet = self.packets[index]
        require(abs(dt-1/60) < 1.e-12 and packet['frame'] == index, 'Packet stream cadence changed')
        if self.owner is None:
            owners = [p for p in policy.active_primitives() if callable(getattr(p, '_update_environment_obstacles', None))]
            require(len(owners) == 1, 'Packet probe requires one initialized geometry owner')
            self.owner = owners[0]
            self.expert = policy.expert
            configure_drive=getattr(self.owner, '_ensure_gripper_drive', None)
            require(callable(configure_drive), 'Missing fixed native gripper configuration')
            configure_drive()
            require(getattr(self.owner, '_gripper_drive_configured', False), 'Native gripper drive was not configured')
        require(policy.expert is self.expert, 'Packet probe expects one robot')
        from expert_data.skills.base import role_path
        blind = TargetBlindSafetyWorld(world, role_path(self.owner.target))
        # Only the geometry refresh is reused. Neither controller.step nor a
        # Catch/Grasp step is called; target tracking/contact/phase logic stays idle.
        # This is the commanded jaw mode, not a contact/grasp certificate.
        # Native terminal safety drops its fixed fully-open finger proxy once
        # CLOSE is requested; the whole-arm model still checks live fingers.
        self.owner._jaws_closed = bool(packet['gripper_closed'])
        robot_pose, robot_twist = refresh_robot_safety_motion(self.owner, self.expert, dt)
        self.owner._update_environment_obstacles(blind)
        require(not blind.blocked_paths, 'Geometry helper attempted target truth access')
        missing_geometry=getattr(self.owner, '_environment_missing_narrow_phase', [])
        require(not missing_geometry, 'Non-target collision geometry is incomplete: '+', '.join(missing_geometry))
        self.geometry_paths.update(blind.accessed_paths)
        callbacks = {key:getattr(self.owner, CALLBACKS[key][1].split('.')[-1]) for key in CALLBACKS}
        event = dict(call_id=len(self.events), frame=index, inputs=encode(dict(packet['inputs'], **packet['safety_callbacks'])),
            safety_robot_pose=encode(robot_pose), safety_robot_twist_world=encode(robot_twist),
            gripper_closed=packet['gripper_closed'], safety_gripper_closed=bool(self.owner._jaws_closed), terminal_cartesian_hold=packet['terminal_cartesian_hold'], completed=False)
        self.events.append(event)
        self.frames.append(index)
        try:
            arm, jaw, pending, applied = execute_request(self.expert, packet, callbacks)
        except Exception as exc:
            event['exception_type'] = type(exc).__name__
            raise
        event.update(completed=True, inputs=applied, returned_action=encode({key:getattr(arm,key,None)
            for key in ['joint_indices','joint_positions','joint_velocities']}), pending_operational_reference=pending,
            jaw_action=encode({key:getattr(jaw,key,None) for key in ['joint_indices','joint_positions']}))
        # Advance history once, after this frame's servo evaluation. Calling
        # the original skill finalizer would also record privileged target data.
        self.owner._prev_ee_p = np.asarray(robot_pose['position_m'], float).copy()
        self.owner._prev_ee_q = np.asarray(robot_pose['quaternion_wxyz'], float).copy()
        return SimpleNamespace(actions=[arm,jaw], phenomena=[])

    def segment_links(self, index, segment):
        events = [x for x in self.events if x['frame'] == index]
        selected = reference_matches(events, dict(segment, frame=index))
        return dict(selected_cartesian_call_ids=selected if len(selected)==1 else [])

    def payload(self):
        return dict(schema='physical-vla-lab.cartesian-packet-execution.v1', frames=self.frames, calls=self.events,
            online_task_step_called=False, online_skill_step_called=False, expert_joint_fallback_used=False,
            target_truth_read_by_request_adapter=False, non_target_safety_geometry_paths=sorted(self.geometry_paths),
            fixed_native_gripper_drive_configuration_retained=True, source_numeric_controller_settings_retained=True, source_motion_and_gripper_stream_retained=True,
            learned_policy_evaluated=False)


def audit_packets(payload, packets, segments, frames):
    alignment = audit_capture(payload, segments, frames)
    events = payload['calls']
    exact = len(events) == len(packets) and all(event['frame']==packet['frame']
        and event['inputs']==encode(dict(packet['inputs'], **packet['safety_callbacks']))
        and event['gripper_closed'] is packet['gripper_closed']
        and event['terminal_cartesian_hold'] is packet['terminal_cartesian_hold']
        for event,packet in zip(events,packets))
    jaws = all(np.array_equal(np.asarray(event['jaw_action']['joint_indices']), [7,8])
        and np.array_equal(np.asarray(event['jaw_action']['joint_positions'],np.float32), np.full(2,0. if packet['gripper_closed'] else .04,np.float32))
        for event,packet in zip(events,packets))
    jaw_context = len(events)==len(packets) and all(event.get('safety_gripper_closed') is packet['gripper_closed'] for event,packet in zip(events,packets))
    motion_context = audit_robot_safety_motion(events)
    boundaries = all(payload.get(key) is False for key in ['online_task_step_called','online_skill_step_called',
        'expert_joint_fallback_used','target_truth_read_by_request_adapter'])
    full = payload['frames']==list(range(len(packets))) and frames==len(packets) and len(segments)==frames
    return dict(alignment=alignment, exact_requests_and_gripper=exact and jaws, safety_jaw_context_matches=jaw_context, causal_robot_safety_motion_verified=motion_context, full_stream_completed=full,
        no_online_skill_decisions_verified=boundaries,
        packet_execution_verified=bool(full and exact and jaws and jaw_context and motion_context and boundaries and alignment['capture_alignment_passed']
            and not alignment['failed_call_ids']),
        terminal_cartesian_hold_frames=sum(p['terminal_cartesian_hold'] for p in packets),
        source_controller_setting_schedule_retained=True, learned_policy_evaluated=False)


def patch_packet_collector(source):
    old='                policy_step_started = time.perf_counter()\n                decision = controller.step(frame, world, dt_control)'
    new='                policy_step_started = time.perf_counter()\n                decision = CAPSTONE_PACKET_DECISION(pol, world, k, dt_control)'
    require(source.count(old)==1, 'Packet decision boundary changed')
    source = source.replace(old,new)
    compile(source,'<independent-packet-decision>','exec')
    return source
