"""Generated-action selection for exposed-episode refinement, not validation."""
import numpy as np
from .ball_prediction_metrics import summarize_stream
from .ball_live_protocol import decoded_request


def generated_metrics(stream, targets, anchor):
    stream=np.asarray(stream);targets=np.asarray(targets)
    if stream.shape!=targets.shape or stream.shape!=(269,29):
        raise ValueError('Expected full269-frame development streams')
    metrics=summarize_stream(stream,targets)
    closed=np.flatnonzero(targets[:,28]>=.5)
    if not len(closed) or closed[0]==0:raise ValueError('Missing approach interval')
    stop=int(closed[0]);rejects=[]
    for i,a in enumerate(stream):
        try:decoded_request(a,i,anchor)
        except ValueError as exc:rejects.append({'frame':i,'reason':str(exc)})
    return dict(approach_end_exclusive=stop,
        approach_position_mean_mm=float(np.linalg.norm(stream[:stop,:3]-targets[:stop,:3],axis=1).mean()*1000),
        orientation_p95_deg=metrics['orientation_error_deg']['p95'],
        gripper_error_frames=int(len(stream)-metrics['mode_accuracy']['gripper_closed']['correct']),
        gripper_transitions=len(metrics['predicted_gripper_transition_frames'] or []),
        nonfinite_frames=metrics['nonfinite_frames'],native_rejected_frames=len(rejects),
        native_rejections=rejects,full_stream_metrics=metrics)


def accept_candidate(candidate, baseline, best):
    names=['approach_position_mean_mm','orientation_p95_deg','gripper_error_frames','gripper_transitions','nonfinite_frames','native_rejected_frames']
    if any(candidate.get(k) is None or not np.isfinite(candidate[k]) for k in names):return False
    return bool(candidate['nonfinite_frames']==0 and candidate['native_rejected_frames']==0
        and candidate['approach_end_exclusive']==baseline['approach_end_exclusive']
        and candidate['orientation_p95_deg']<=baseline['orientation_p95_deg']
        and candidate['gripper_error_frames']<=baseline['gripper_error_frames']
        and candidate['gripper_transitions']<=baseline['gripper_transitions']
        and candidate['approach_position_mean_mm']<best['approach_position_mean_mm'])
