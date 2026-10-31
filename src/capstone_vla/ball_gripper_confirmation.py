"""Causal model-output gripper confirmation; no source actions or object state."""
import numpy as np
from .cartesian_target_replay import require

PLAN = dict(schema="physical-vla-lab.gripper-confirmation.v1", close_threshold=.8,
            open_threshold=.2, consecutive_close_frames=3, initial_closed=False,
            control_hz=60, channel=28)


def validate(plan):
    require(plan == PLAN, "Unsupported gripper confirmation plan")


class GripperConfirmation:
    def __init__(self, plan):
        validate(plan)
        self.closed = False
        self.count = 0
        self.next_frame = 0

    def apply(self, raw, frame):
        raw = np.asarray(raw)
        require(type(frame) is int and frame == self.next_frame,
                "Noncontiguous gripper confirmation frame")
        require(raw.shape == (29,) and np.isfinite(raw).all(), "Invalid model action")
        score = float(raw[28])  # regression output, not a calibrated probability
        if self.closed:
            if score < PLAN["open_threshold"]:
                self.closed = False
                self.count = 0
        else:
            self.count = self.count + 1 if score >= PLAN["close_threshold"] else 0
            if self.count >= PLAN["consecutive_close_frames"]:
                self.closed = True
                self.count = 0
        self.next_frame += 1
        result = raw.astype(float, copy=True)
        result[28] = float(self.closed)
        return result


def audit_stream(raw, executed, plan):
    """Reconstruct the full causal sequence and reject any other channel change."""
    raw, executed = np.asarray(raw), np.asarray(executed)
    require(raw.ndim == 2 and len(raw) > 0 and raw.shape[1:] == (29,) and raw.shape == executed.shape,
            "Invalid gripper audit shapes")
    gate = GripperConfirmation(plan)
    expected = np.stack([gate.apply(action, i) for i, action in enumerate(raw)])
    np.testing.assert_array_equal(executed, expected)
    transitions = lambda values: (np.flatnonzero(np.diff(values >= .5)) + 1).tolist()
    return dict(verified=True, channel=28, other_channels_exact=True,
                raw_threshold_transitions=transitions(raw[:,28]),
                executed_transitions=transitions(executed[:,28]),
                changed_jaw_decision_frames=np.flatnonzero((raw[:,28]>=.5)!=(executed[:,28]>=.5)).tolist())
