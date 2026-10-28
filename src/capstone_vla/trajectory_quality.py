"""Descriptive recording-quality metrics; no success relabeling or policy inputs."""
import numpy as np
from scipy.spatial.transform import Rotation


def motion_metrics(time, ee, quat_wxyz, ball, ee_velocity, phase, close, pinch_offset,
                   first_bilateral, lift_code, hold_code, ball_velocity=None, first_contact=None):
    time, ee, quat_wxyz, ball, ee_velocity = [np.asarray(x, dtype=float) for x in (time, ee, quat_wxyz, ball, ee_velocity)]
    n = len(time)
    if time.shape != (n,) or n < 3 or not np.isfinite(time).all() or not np.all(np.diff(time) > 0):
        raise ValueError("Invalid observation clock")
    dt = float(np.median(np.diff(time)))
    if not np.allclose(np.diff(time), dt, rtol=0, atol=1e-7):
        raise ValueError("Nonuniform observation clock")
    for x, shape in [(ee, (n, 3)), (ball, (n, 3)), (quat_wxyz, (n, 4)), (ee_velocity, (n, 3))]:
        if x.shape != shape or not np.isfinite(x).all():
            raise ValueError("Invalid state array")
    if not np.allclose(np.linalg.norm(quat_wxyz, axis=1), 1, atol=1e-4):
        raise ValueError("Invalid quaternion")
    phase, close = np.asarray(phase), np.asarray(close, dtype=bool)
    if phase.shape != (n,) or close.shape != (n,):
        raise ValueError("Invalid control labels")
    if first_bilateral is not None and not 0 <= first_bilateral < n:
        raise ValueError("Invalid contact anchor")
    rotation = Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]])
    relative = rotation.inv().apply(ball - ee) - np.asarray(pinch_offset)
    tail_n = min(n, max(1, round(.5 / dt)))
    tail = relative[-tail_n:]
    tail_shift = np.linalg.norm(tail - np.median(tail, axis=0), axis=1)
    lift_entries = np.flatnonzero((phase == lift_code) & np.r_[True, phase[:-1] != lift_code])
    hold_entries = np.flatnonzero(phase == hold_code)
    lift = int(lift_entries[0]) if len(lift_entries) else None
    hold = int(hold_entries[0]) if len(hold_entries) else None
    pre = max(1, (first_contact if first_contact is not None else first_bilateral) or 1)
    acceleration = np.linalg.norm(np.diff(ee_velocity, axis=0) / dt, axis=1)
    pre_acceleration = acceleration[:max(1, pre - 1)]
    result = {
        "frames": n, "duration_s": float(n * dt), "tail_window_s": float(tail_n * dt),
        "first_bilateral_state_frame": first_bilateral, "first_lift_frame": lift,
        "first_hold_frame": hold, "lift_entries": int(len(lift_entries)),
        "tail_relative_center_jitter_p95_mm": float(np.percentile(tail_shift, 95) * 1000),
        "tail_relative_center_jitter_max_mm": float(tail_shift.max() * 1000),
        "lift_to_final_relative_center_shift_mm": None if lift is None else float(np.linalg.norm(relative[-1] - relative[lift]) * 1000),
        "minimum_tail_lift_from_first_bilateral_mm": None if first_bilateral is None else float((ball[-tail_n:, 2].min() - ball[first_bilateral, 2]) * 1000),
        "final_lift_from_first_bilateral_mm": None if first_bilateral is None else float((ball[-1, 2] - ball[first_bilateral, 2]) * 1000),
        "precontact_ee_acceleration_p95_mps2": float(np.percentile(pre_acceleration, 95)),
        "all_ee_acceleration_p95_mps2": float(np.percentile(acceleration, 95)),
        "reopen_transitions_after_first_bilateral": 0 if first_bilateral is None else int(np.sum(close[first_bilateral:-1] & ~close[first_bilateral+1:])),
    }
    if ball_velocity is not None:
        ball_velocity = np.asarray(ball_velocity, float)
        if ball_velocity.shape != (n, 3) or not np.isfinite(ball_velocity).all():
            raise ValueError("Invalid recorded ball velocity")
        omega = np.zeros((n, 3))
        omega[1:] = (rotation[1:] * rotation[:-1].inv()).as_rotvec() / dt
        relative_velocity = rotation.inv().apply(ball_velocity - ee_velocity - np.cross(omega, ball - ee))
        result["tail_relative_speed_p95_mps"] = float(np.percentile(np.linalg.norm(relative_velocity[-tail_n:], axis=1), 95))
    return result


def contact_metrics(step_time, step_dt, left, right, control_time=None):
    time, dt = np.asarray(step_time, float), np.asarray(step_dt, float)
    left, right = np.asarray(left, bool), np.asarray(right, bool)
    if time.ndim != 1 or not len(time) or any(x.shape != time.shape for x in (dt, left, right)):
        raise ValueError("Invalid physical-contact arrays")
    if not np.isfinite(time).all() or not np.isfinite(dt).all() or np.any(dt <= 0) or not np.all(np.diff(time) > 0):
        raise ValueError("Invalid physical-contact clock")
    both = left & right
    ids = np.flatnonzero(both)
    last = max(1, min(len(time), round(.5 / np.median(dt))))
    edges = np.diff(np.r_[False, both, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    result = {
        "raw_first_bilateral_time_s": float(time[ids[0]]) if len(ids) else None,
        "raw_first_any_contact_time_s": float(time[np.flatnonzero(left | right)[0]]) if np.any(left | right) else None,
        "raw_longest_bilateral_duration_s": float(max((dt[a:b].sum() for a, b in zip(starts, ends)), default=0)),
        "raw_tail_bilateral_fraction": float(dt[-last:][both[-last:]].sum() / dt[-last:].sum()),
        "raw_tail_window_s": float(dt[-last:].sum()),
        "raw_positive_left_ticks": int(left.sum()), "raw_positive_right_ticks": int(right.sum()),
    }
    if control_time is not None:
        control_time = np.asarray(control_time, float)
        if control_time.ndim != 1 or len(control_time) < 2 or not np.all(np.diff(control_time) > 0):
            raise ValueError("Invalid control clock")
        # Contacts have end-of-substep timestamps. Bin by substep midpoint.
        bins = np.searchsorted(control_time, time - dt / 2, side="right") - 1
        valid = (bins >= 0) & (bins < len(control_time))
        l, r = np.zeros(len(control_time), bool), np.zeros(len(control_time), bool)
        np.logical_or.at(l, bins[valid], left[valid]); np.logical_or.at(r, bins[valid], right[valid])
        window = min(len(control_time), max(1, round(.5 / np.median(np.diff(control_time)))))
        result["raw_tail_control_bin_bilateral_fraction"] = float(np.mean((l & r)[-window:]))
    return result


def segmentation_metrics(labels, target_id):
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Expected categorical segmentation")
    y, x = np.nonzero(labels == target_id)
    bbox = [int(x.min()), int(y.min()), int(x.max()), int(y.max())] if len(x) else None
    return {"pixels": int(len(x)), "bbox": bbox,
            "touches_edge": bool(len(x) and (x.min() <= 1 or y.min() <= 1 or x.max() >= labels.shape[1] - 2 or y.max() >= labels.shape[0] - 2))}


def describe_groups(rows, key, seed=20261028):
    groups = {host: np.asarray([r[key] for r in rows if r["host"] == host and r.get(key) is not None], float)
              for host in ("local", "hongtao")}
    if any(not len(x) or not np.isfinite(x).all() for x in groups.values()):
        raise ValueError("Missing/nonfinite comparison group")
    result = {host: {"n": len(x), "median": float(np.median(x)), "q25": float(np.percentile(x, 25)),
                     "q75": float(np.percentile(x, 75)), "min": float(x.min()), "max": float(x.max())}
              for host, x in groups.items()}
    rng = np.random.default_rng(seed)
    medians = [np.median(rng.choice(x, size=(5000, len(x)), replace=True), axis=1) for x in groups.values()]
    result["local_minus_hongtao_median"] = {"observed": result["local"]["median"] - result["hongtao"]["median"],
        "exploratory_bootstrap_95_interval": np.percentile(medians[0] - medians[1], [2.5, 97.5]).tolist(),
        "interpretation": "Unpaired selected-success groups; descriptive, not a causal test or a quality ranking."}
    return result
