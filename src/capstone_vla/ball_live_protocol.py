"""Absolute29 live boundary; no simulator imports or expert action inputs."""
from collections import deque
import numpy as np
from .ball_action_contract import decode_action
from .cartesian_target_replay import require
from .pilot_dataset import CAMERAS


def validate_observation(states, images):
    states = np.asarray(states)
    require(states.shape == (2, 30) and np.isfinite(states).all(), 'Expected measured state history (2,30)')
    require(set(images) == set(CAMERAS), 'Expected exactly the two trained RGB cameras')
    for name, image in images.items():
        require(image.shape == (2,480,640,3) and image.dtype == np.uint8, 'Invalid RGB history: '+name)


def decoded_request(action, frame, anchor):
    packet = decode_action(action, frame, anchor)
    u = packet['inputs']
    require(u['maximum_relative_linear_speed_mps'] is None or u['linear_velocity_reference'] is not None,
            'Relative-speed cap requires a predicted velocity reference')
    # The frozen native consumer also rejects an impossible reference center.
    if u['maximum_relative_linear_speed_mps'] is not None:
        require(np.linalg.norm(u['linear_velocity_reference']) <= u['maximum_linear_speed_mps'] + 1e-9,
                'Predicted velocity reference exceeds native absolute speed bound')
    return packet


class ActionQueue:
    def __init__(self, prefix, frames):
        require(prefix in (1,4) and 2 <= frames <= 269, 'Unsupported bounded rollout')
        self.prefix, self.frames = prefix, frames
        self.values = deque(); self.next_frame = 0; self.anchor = None

    def accept(self, frame, chunk):
        require(not self.values and frame == self.next_frame, 'Stale or unsolicited policy response')
        chunk = np.asarray(chunk)
        require(chunk.shape == (20,29) and np.isfinite(chunk).all(), 'Expected finite absolute29 chunk')
        self.values.extend(chunk[:min(self.prefix,self.frames-frame)].copy())
        self.anchor = frame

    def pop(self, frame):
        require(frame == self.next_frame and frame < self.frames and bool(self.values), 'Policy queue clock mismatch')
        self.next_frame += 1
        return self.values.popleft().copy(), self.anchor


def patch_collector(source):
    """Reuse native motor-boundary recorder; replace exactly one decision call."""
    from .hgwind_expert_trace import patch_collector as record_patch
    anchor = '"skill_terminations": _jsonify(skill_terminations),'
    require(source.count(anchor) == 1, 'Native profile metadata boundary moved')
    source = source.replace(anchor, anchor + '\n            "capstone_expert_control": CAPSTONE_PROFILE_SUMMARY(pol),')
    source = record_patch(source)
    old = '                decision = controller.step(frame, world, dt_control)'
    new = ('                decision = CAPSTONE_LIVE_DECISION(pol, world, k, dt_control, '
           'stage, sim, robot, jp.copy(), jv.copy(), ee_p.copy(), np.asarray(ee_q, dtype=np.float32).copy())')
    require(source.count(old) == 1, 'Native decision boundary moved')
    source = source.replace(old,new)
    compile(source,'<ball-live-decision>','exec')
    return source


def audit_request_history(request, frame, states, images):
    """Bind IPC policy observations to retained pre-action frames, including RGB."""
    require(set(request)=={'frame','states',*CAMERAS} and int(request['frame'])==frame,
            'Unexpected policy request fields/frame')
    validate_observation(request['states'],{name:request[name] for name in CAMERAS})
    np.testing.assert_array_equal(request['states'],states)
    for name in CAMERAS:np.testing.assert_array_equal(request[name],images[name])
