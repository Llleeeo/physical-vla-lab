"""Experimental RGB-guided approach filter. Not certified collision avoidance.

No world/target-state access. Sphere size is declared static task geometry;
position comes exclusively from RGB. Original jaw scores are never rewritten.
"""
import copy
import itertools
import numpy as np
from scipy.spatial.transform import Rotation
from .cartesian_target_replay import require

SETTINGS=dict(schema='physical-vla-lab.visual-approach.v1',alignment_tolerance_m=.01,
    localization_margin_m=.015,surface_margin_m=.01,reaction_s=.12,control_hz=60,
    jaw_threshold_unchanged=.5,missing_view='latched_pose_hold',maximum_visual_jump_m=.04)


def validate(plan):
    require(plan['settings']==SETTINGS,'Unrecognized approach settings')
    require(0<plan['ball_radius_m']<.1,'Invalid declared ball radius')
    require(len(plan['finger_bounds'])==2 and len(plan['cameras'])==2,'Need two fingers and cameras')
    for bounds in plan['finger_bounds']:
        v=np.asarray(bounds['local_corners_m'],float)
        require(v.shape==(8,3) and np.isfinite(v).all(),'Invalid finger bounds')
    return plan


def calibration(stage,records,ball_path):
    """Read immutable optics, sphere SIZE and robot local bounds before rollout."""
    from pxr import Usd,UsdGeom,Gf
    mpu=float(UsdGeom.GetStageMetersPerUnit(stage))
    cache=UsdGeom.XformCache();bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy],useExtentsHint=False)
    ball=UsdGeom.Sphere(stage.GetPrimAtPath(ball_path+'/__analytic_sphere_collider'))
    require(bool(ball),'Expected authored analytic sphere')
    matrix=cache.GetLocalToWorldTransform(ball.GetPrim())
    lengths=[float(matrix.TransformDir(Gf.Vec3d(*axis)).GetLength()) for axis in np.eye(3)]
    require(max(lengths)-min(lengths)<1e-6,'Nonuniform sphere scale unsupported')
    radius=float(ball.GetRadiusAttr().Get())*lengths[0]*mpu
    fingers=[]
    for name in ('panda_leftfinger','panda_rightfinger'):
        path='/Root/Dynamics/franka/'+name;p=stage.GetPrimAtPath(path)
        require(bool(p),'Missing robot finger link')
        box=bbox.ComputeUntransformedBound(p).ComputeAlignedRange()
        low,high=np.array(box.GetMin())*mpu,np.array(box.GetMax())*mpu
        require(np.isfinite(low).all() and np.all(high>low) and np.max(high-low)<.2,'Invalid finger bound')
        corners=np.array(list(itertools.product(*zip(low,high))))
        fingers.append(dict(path=path,local_corners_m=corners.tolist()))
    cameras=[]
    for record in records:
        cam=UsdGeom.Camera(stage.GetPrimAtPath(record['prim_path']))
        require(cam.GetProjectionAttr().Get()=='perspective' and
            cam.GetPrim().GetAttribute('cameraProjectionType').Get() in (None,'pinhole'), 'Unsupported camera')
        require(not cam.GetHorizontalApertureOffsetAttr().Get() and not cam.GetVerticalApertureOffsetAttr().Get(),'Unsupported optical offsets')
        focal=float(cam.GetFocalLengthAttr().Get());aperture=float(cam.GetHorizontalApertureAttr().Get())
        require(focal>0 and aperture>0,'Invalid optics')
        f=640*focal/aperture
        cameras.append(dict(name=record['name'],prim_path=record['prim_path'],intrinsics=[[f,0,320],[0,f,240],[0,0,1]]))
    return validate(dict(settings=copy.deepcopy(SETTINGS),ball_radius_m=radius,finger_bounds=fingers,cameras=cameras))


def robot_corners(stage,plan):
    """Robot-only transforms, sampled after live RGB publishes measured poses."""
    from pxr import UsdGeom,Gf
    cache=UsdGeom.XformCache();mpu=float(UsdGeom.GetStageMetersPerUnit(stage));result=[]
    for bounds in plan['finger_bounds']:
        transform=cache.GetLocalToWorldTransform(stage.GetPrimAtPath(bounds['path']))
        result.append([np.asarray(transform.Transform(Gf.Vec3d(*(np.asarray(p)/mpu))))*mpu
                       for p in bounds['local_corners_m']])
    result=np.asarray(result,float)
    require(result.shape==(2,8,3) and np.isfinite(result).all(),'Invalid measured robot geometry')
    return result


class ApproachGuard:
    def __init__(self,plan):
        self.plan=copy.deepcopy(validate(plan));self.next_frame=0
        self.previous_min_z=None;self.previous_ball=None;self.hold=None;self.pending_ball=None

    def apply(self,raw,frame,pinch_pose,finger_points,corners,detection):
        raw=np.asarray(raw,float);pose=np.asarray(pinch_pose,float)
        fingers=np.asarray(finger_points,float);corners=np.asarray(corners,float)
        require(type(frame) is int and frame==self.next_frame,'Noncontiguous approach frame')
        require(raw.shape==(29,) and pose.shape==(7,) and fingers.shape==(2,3) and corners.shape==(2,8,3)
                and all(np.isfinite(v).all() for v in (raw,pose,fingers,corners)),'Invalid approach inputs')
        require(abs(np.linalg.norm(pose[3:])-1)<1e-5,'Invalid measured pinch quaternion')
        result=raw.copy();min_z=float(corners[:,:,2].min())
        downward=0. if self.previous_min_z is None else max(0.,(self.previous_min_z-min_z)*60)
        valid=bool(detection.get('valid'));reason=detection.get('reason','not_unique')
        ball=None
        if valid:
            ball=np.asarray(detection['position_world_m'],float)
            require(ball.shape==(3,) and np.isfinite(ball).all(),'Invalid visual ball estimate')
            if self.previous_ball is not None and np.linalg.norm(ball-self.previous_ball)>SETTINGS['maximum_visual_jump_m']:
                valid=False;reason='visual_jump'
        if valid and self.hold is not None:
            if self.pending_ball is None or np.linalg.norm(ball-self.pending_ball)>SETTINGS['maximum_visual_jump_m']:
                self.pending_ball=ball.copy();valid=False;reason='reacquiring'
            else:self.pending_ball=None
        elif not valid:self.pending_ball=None
        event=dict(frame=frame,vision_valid=valid,vision_reason=None if valid else reason,
                   mode='aligned_pass',changed_channels=[],jaw_unchanged=True)
        if not valid:
            # Never propagate stale target position through a missing camera view.
            if self.hold is None:self.hold=pose.copy()
            result[:7]=self.hold
            result[7:13]=0.;result[14:17]=0.;result[23]=0.
            event.update(mode='missing_pose_hold',hold_pose=self.hold.tolist())
            self.previous_ball=None
        else:
            self.hold=None;self.previous_ball=ball.copy()
            offset=float(np.linalg.norm(fingers.mean(axis=0)[:2]-ball[:2]))
            event['midpoint_horizontal_offset_m']=offset
            if offset>SETTINGS['alignment_tolerance_m']:
                # Rotate current full-finger bounding corners to the proposed
                # pinch orientation before setting a vertical goal floor.
                q=raw[3:7];require(np.linalg.norm(q)>1e-6,'Degenerate proposed quaternion')
                current=Rotation.from_quat(pose[[4,5,6,3]])
                desired=Rotation.from_quat(q[[1,2,3,0]])
                relative=corners.reshape(-1,3)-pose[:3]
                proposed=desired.apply(current.inv().apply(relative))
                min_offset=min(float(relative[:,2].min()),float(proposed[:,2].min()))
                reserve=downward*SETTINGS['reaction_s']
                finger_floor=float(ball[2]+self.plan['ball_radius_m']+
                    SETTINGS['localization_margin_m']+SETTINGS['surface_margin_m']+reserve)
                target_floor=finger_floor-min_offset
                result[2]=max(raw[2],target_floor)
                # Suppress downward feedforward in each supported motion field.
                # Positive/lifting components remain; masks are unchanged.
                for channel in (9,12,16):result[channel]=max(0.,raw[channel])
                event.update(mode='unaligned_height_floor',target_floor_m=target_floor,
                             finger_floor_m=finger_floor,descent_reserve_m=reserve)
        event['changed_channels']=np.flatnonzero(result!=raw).tolist()
        require(result[28]==raw[28],'Jaw score must remain unchanged')
        self.next_frame+=1;self.previous_min_z=min_z
        return result,event
