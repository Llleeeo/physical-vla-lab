"""Optional near-ball XY visual assistance; retains v1 height/jaw decisions.

A causal ball-velocity fit, not extrapolation of unknown future VLA robot motion.
Static geometry and RGB are explicit priors; no live simulator target state.
"""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from .ball_approach_guard import ApproachGuard
from .cartesian_target_replay import require

FOLLOW_SETTINGS = dict(schema='physical-vla-lab.visual-follow.v1',history_frames=7,
    maximum_horizontal_distance_m=.08,maximum_fit_residual_m=.004,
    maximum_ball_speed_m_s=.5,control_hz=60)


def fit_visual_velocity(positions):
    p=np.asarray(positions,float)
    require(p.shape==(7,3) and np.isfinite(p).all(),'Need seven finite visual positions')
    t=(np.arange(7)-6)/60;t=t-t.mean()
    velocity=np.sum(t[:,None]*(p-p.mean(axis=0)),axis=0)/np.sum(t*t)
    residual=float(np.max(np.linalg.norm(p-(p.mean(axis=0)+t[:,None]*velocity),axis=1)))
    return velocity,residual


class FollowingApproachGuard(ApproachGuard):
    def __init__(self,plan):
        require(plan.get('follow')==FOLLOW_SETTINGS,'Unrecognized visual follow settings')
        super().__init__(plan);self.positions=[]

    def apply(self,raw,frame,pinch_pose,finger_points,corners,detection):
        raw=np.asarray(raw,float)
        result,event=super().apply(raw,frame,pinch_pose,finger_points,corners,detection)
        follow=dict(active=False,reason='invalid_vision')
        if not event['vision_valid']:
            self.positions=[]
        else:
            ball=np.asarray(detection['position_world_m'],float)
            self.positions.append(ball.copy());self.positions=self.positions[-7:]
            follow.update(reason='warming_up',history_frames=len(self.positions))
            if len(self.positions)==7:
                velocity,residual=fit_visual_velocity(self.positions)
                follow.update(velocity_world_m_s=velocity.tolist(),maximum_fit_residual_m=residual)
                if residual>FOLLOW_SETTINGS['maximum_fit_residual_m']:
                    follow['reason']='poor_velocity_fit'
                elif np.linalg.norm(velocity)>FOLLOW_SETTINGS['maximum_ball_speed_m_s']:
                    follow['reason']='visual_speed_out_of_range'
                elif event['midpoint_horizontal_offset_m']>FOLLOW_SETTINGS['maximum_horizontal_distance_m']:
                    follow['reason']='outside_near_region'
                else:
                    pose=np.asarray(pinch_pose,float);midpoint=np.asarray(finger_points,float).mean(axis=0)
                    current=Rotation.from_quat(pose[[4,5,6,3]])
                    desired=Rotation.from_quat(raw[[4,5,6,3]])
                    # Pinch and configured finger midpoint are distinct frames.
                    offset=desired.apply(current.inv().apply(midpoint-pose[:3]))
                    result[:2]=ball[:2]-offset[:2]
                    # Match horizontal goal motion across all supported fields.
                    # Masks and model relative-speed settings remain unchanged.
                    result[7:9]=velocity[:2];result[10:12]=0.;result[14:16]=velocity[:2]
                    follow.update(active=True,reason='near_visual_follow',desired_midpoint_xy_m=ball[:2].tolist())
        event['follow']=follow
        event['changed_channels']=np.flatnonzero(result!=raw).tolist()
        require(result[28]==raw[28],'Follow must preserve jaw score')
        return result,event


def make_approach_guard(plan):
    if plan is None:return None
    if "capture_window" in plan:
        from .ball_capture_window import CaptureWindowGuard
        return CaptureWindowGuard(copy.deepcopy(plan))
    if "near_capture" in plan:
        from .ball_near_capture import NearCaptureGuard
        return NearCaptureGuard(copy.deepcopy(plan))
    return FollowingApproachGuard(copy.deepcopy(plan)) if 'follow' in plan else ApproachGuard(plan)
