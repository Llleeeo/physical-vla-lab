"""Pose-aware offline action metrics, separate from task success."""
import numpy as np
from scipy.spatial.transform import Rotation

def absolute_from_delta(delta,current_state):
    out=np.asarray(delta,dtype=np.float64).copy()
    current=np.asarray(current_state,dtype=np.float64)
    if out.ndim!=3 or out.shape[-1]!=7 or current.shape!=(len(out),6):
        raise ValueError('expected NxHx7 delta and Nx6 state')
    if not np.isfinite(out).all() or not np.isfinite(current).all():raise ValueError('nonfinite action/state')
    out[...,:6]+=current[:,None,:]
    return out

def action_errors(predicted,target,padding):
    pred=np.asarray(predicted);goal=np.asarray(target);pad=np.asarray(padding)
    if pred.shape!=goal.shape or pred.ndim!=3 or pred.shape[-1]!=7 or pad.shape!=pred.shape[:2] or pad.dtype!=bool:
        raise ValueError('action/padding shape mismatch')
    if not np.isfinite(pred).all() or not np.isfinite(goal).all():raise ValueError('nonfinite pose')
    position=np.linalg.norm(pred[...,:3]-goal[...,:3],axis=-1)*1000
    a=Rotation.from_euler('xyz',pred[...,3:6].reshape(-1,3))
    b=Rotation.from_euler('xyz',goal[...,3:6].reshape(-1,3))
    angle=np.rad2deg((a.inv()*b).magnitude()).reshape(pad.shape)
    gripper=(pred[...,6]>0)==(goal[...,6]>0)
    def summary(mask):
        if not mask.any():raise ValueError('no valid targets')
        p=position[mask];o=angle[mask]
        return {'valid_targets':int(mask.sum()),'position_mean_mm':float(p.mean()),
                'position_median_mm':float(np.median(p)),'position_p95_mm':float(np.percentile(p,95)),
                'orientation_mean_deg':float(o.mean()),'orientation_p95_deg':float(np.percentile(o,95)),
                'gripper_sign_accuracy':float(gripper[mask].mean())}
    first=np.zeros_like(pad);first[:,0]=~pad[:,0]
    metrics={'first_action':summary(first),'all_valid_chunk_targets':summary(~pad)}
    return metrics,position,angle,gripper
