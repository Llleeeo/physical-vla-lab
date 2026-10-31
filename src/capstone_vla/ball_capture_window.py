"""Finite, opt-in geometry-triggered capture dwell; not grasp confirmation.

Uses only current RGB geometry and robot state. The model still commands the
jaw. Native actuator/environment checks remain. This changes the policy.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .ball_near_capture import NearCaptureGuard
from .cartesian_target_replay import require

SETTINGS=dict(schema='physical-vla-lab.capture-window.v1',maximum_frames=42,
    entry_horizontal_m=.01,entry_height_above_center_m=.012,
    exit_horizontal_radius_margin_m=.005,minimum_bottom_clearance_m=.01)


class CaptureWindowGuard(NearCaptureGuard):
    def __init__(self,plan):
        require(plan.get('capture_window')==SETTINGS,'Unknown capture-window settings')
        require(plan['near_capture']['extra_depth_m']==.005,'Capture window requires explicit 5 mm depth')
        super().__init__(plan);self.capture_start=None;self.capture_finished=False;self.height_cap=None

    def apply(self,raw,frame,pinch_pose,finger_points,corners,detection):
        result,event=super().apply(raw,frame,pinch_pose,finger_points,corners,detection)
        pose=np.asarray(pinch_pose,float);mid=np.asarray(finger_points,float).mean(axis=0)
        state=dict(active=False,started_frame=self.capture_start,reason='not_entered',extra_depth_m=self.extra_depth)
        valid=event['vision_valid'];follow=event['follow']['active']
        if valid:
            ball=np.asarray(detection['position_world_m'],float)
            offset=float(np.linalg.norm(mid[:2]-ball[:2]));height=float(mid[2]-ball[2])
            corridor=min(.02,self.plan['ball_radius_m']-SETTINGS['exit_horizontal_radius_margin_m'])
            require(corridor>SETTINGS['entry_horizontal_m'],'Sphere too small for capture corridor')
            if (self.capture_start is None and not self.capture_finished and follow
                    and offset<=SETTINGS['entry_horizontal_m'] and -.005<=height<=SETTINGS['entry_height_above_center_m']):
                self.capture_start=frame;self.height_cap=float(pose[2]-self.extra_depth)
            if self.capture_start is not None and not self.capture_finished and offset>corridor:
                self.capture_finished=True;state['reason']='left_capture_corridor'
        if self.capture_start is not None and not self.capture_finished:
            if frame-self.capture_start>=SETTINGS['maximum_frames']:
                self.capture_finished=True;state['reason']='window_expired_without_success_claim'
            elif not valid or not follow:
                # Base hold is retained: never descend blind or invent ball velocity.
                state['reason']='paused_for_vision_or_velocity'
                if valid:
                    # In-corridor fit loss pauses at current height, rather than
                    # rearming the high approach floor while the ball is between fingers.
                    current=Rotation.from_quat(pose[[4,5,6,3]])
                    desired=Rotation.from_quat(result[[4,5,6,3]])
                    local=current.inv().apply(np.asarray(corners).reshape(-1,3)-pose[:3])
                    floor=float(ball[2]-self.plan['ball_radius_m']+SETTINGS['minimum_bottom_clearance_m']-
                        desired.apply(local)[:,2].min())
                    result[2]=max(float(pose[2]),floor);result[[9,12,16]]=0.
                    state['base_height_floor_suspended']=event['mode']=='unaligned_height_floor'
            else:
                current=Rotation.from_quat(pose[[4,5,6,3]])
                desired=Rotation.from_quat(result[[4,5,6,3]])
                local=current.inv().apply(np.asarray(corners).reshape(-1,3)-pose[:3])
                floor=float(ball[2]-self.plan['ball_radius_m']+SETTINGS['minimum_bottom_clearance_m']-
                    desired.apply(local)[:,2].min())
                requested=min(float(raw[2])-self.extra_depth,self.height_cap)
                result[2]=max(requested,floor)
                # Coherent vertical hold/descent: no conflicting positive feedforward.
                result[[9,12,16]]=0.
                self.height_cap=min(self.height_cap,float(result[2]))
                state.update(active=True,reason='bounded_capture_dwell',started_frame=self.capture_start,
                    elapsed_frames=frame-self.capture_start,maximum_frames=SETTINGS['maximum_frames'],
                    capture_corridor_m=corridor,height_cap_m=self.height_cap,
                    sphere_bottom_proxy_floor_m=floor,base_height_floor_suspended=event['mode']=='unaligned_height_floor')
        elif self.capture_finished and state['reason']=='not_entered':state['reason']='finished'
        state['started_frame']=self.capture_start;event['capture_window']=state
        event['changed_channels']=np.flatnonzero(result!=raw).tolist()
        require(result[28]==raw[28],'Capture dwell must preserve jaw score')
        self.previous_output=result.copy()
        return result,event
