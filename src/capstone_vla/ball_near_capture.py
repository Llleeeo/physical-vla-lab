"""Opt-in near-capture recovery. RGB and declared sphere size only.

Partial silhouette fitting is an engineering estimator, not guaranteed depth.
The original stereo route remains preferred. No live object/contact truth.
"""
import copy
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .ball_visual_geometry import locate_ball, orange_candidates, world_ray
from .ball_approach_follow import FollowingApproachGuard
from .cartesian_target_replay import require

SETTINGS = dict(schema='physical-vla-lab.near-capture.v1',
    minimum_arc_degrees=210.,maximum_tangent_p90_m=.002,
    maximum_external_ray_gap_m=.005,maximum_range_m=.35,
    recovery_translation_step_m=.003,recovery_rotation_step_deg=1.,
    minimum_clearance_above_ball_bottom_m=.01)


def silhouette_candidates(image,intrinsics,radius):
    image=np.asarray(image);k=np.asarray(intrinsics,float)
    require(image.shape==(480,640,3) and image.dtype==np.uint8,'Invalid RGB image')
    require(k.shape==(3,3) and np.isfinite(k).all() and k[0,0]>0 and k[1,1]>0,'Invalid optics')
    require(0<radius<.1,'Invalid radius')
    red,green,blue=image.astype(float).transpose(2,0,1)
    mask=(red>90)&(red-green>25)&(red>1.25*green)&(green>1.12*blue)
    mask=ndimage.binary_closing(mask,iterations=2)
    labels,_=ndimage.label(mask);result=[]
    for label,region in enumerate(ndimage.find_objects(labels),1):
        yy,xx=region;area=int((labels[region]==label).sum())
        # Fallback only for a substantial component truncated by the image edge.
        if (not (xx.start<3 or yy.start<3 or xx.stop>637 or yy.stop>477)
                or not 150<=area<=30000 or min(yy.stop-yy.start,xx.stop-xx.start)<10):continue
        component=ndimage.binary_fill_holes(labels==label)
        edge=component&~ndimage.binary_erosion(component)
        edge[:8]=False;edge[-8:]=False;edge[:,:8]=False;edge[:,-8:]=False
        y,x=np.where(edge)
        if len(x)<40:continue
        uv=np.stack([(x-k[0,2])/k[0,0],-(y-k[1,2])/k[1,1]],axis=1)
        cx,cy,c=np.linalg.lstsq(np.column_stack([2*uv,np.ones(len(x))]),(uv*uv).sum(axis=1),rcond=None)[0]
        radius2=c+cx*cx+cy*cy
        if radius2<=0:continue
        depth=radius/np.sqrt(radius2)
        angles=np.sort(np.arctan2(uv[:,1]-cy,uv[:,0]-cx))
        coverage=float(360-np.degrees(np.max(np.diff(np.r_[angles,angles[0]+2*np.pi]))))
        if coverage<SETTINGS['minimum_arc_degrees']:continue
        rays=np.column_stack([uv,-np.ones(len(x))]);rays/=np.linalg.norm(rays,axis=1)[:,None]
        def residual(center):return np.linalg.norm(np.cross(rays,center),axis=1)-radius
        fit=least_squares(residual,[cx*depth,cy*depth,-depth],loss='soft_l1',f_scale=.0005,max_nfev=60)
        center=fit.x;error=float(np.quantile(abs(residual(center)),.9))
        if (not fit.success or not np.isfinite(center).all() or center[2]>=-radius
                or np.linalg.norm(center)>SETTINGS['maximum_range_m']
                or error>SETTINGS['maximum_tangent_p90_m'] or np.min(rays@center)<=0):continue
        result.append(dict(center_camera_m=center.tolist(),tangent_p90_m=error,arc_degrees=coverage))
    return result


def locate_for_plan(images,poses,intrinsics,names,plan):
    result=locate_ball(images,poses,intrinsics)
    if plan is None or 'near_capture' not in plan:return result
    require(plan['near_capture']['settings']==SETTINGS,'Unknown near-capture settings')
    if result['valid'] or result['matching_pairs']!=0:return result
    require(set(names)=={'wrist_cam','opst_cam'},'Unsupported near-capture cameras')
    wrist=names.index('wrist_cam');external=names.index('opst_cam');pose=np.asarray(poses[wrist],float)
    matches=[]
    for fitted in silhouette_candidates(images[wrist],intrinsics[wrist],plan['ball_radius_m']):
        center=Rotation.from_quat(pose[[4,5,6,3]]).apply(fitted['center_camera_m'])+pose[:3]
        for candidate in orange_candidates(images[external]):
            origin,ray=world_ray(candidate['pixel'],poses[external],intrinsics[external]);delta=center-origin
            gap=float(np.linalg.norm(np.cross(delta,ray)))
            if delta@ray>.01 and gap<=SETTINGS['maximum_external_ray_gap_m']:
                matches.append(dict(position_world_m=center.tolist(),external_ray_gap_m=gap,**fitted))
    if len(matches)==1:
        return dict(valid=True,method='clipped_sphere_with_external_check',candidate_counts=result['candidate_counts'],
            matching_pairs=1,original_stereo=result,**matches[0])
    return dict(result,partial_sphere_matches=len(matches))


class NearCaptureGuard(FollowingApproachGuard):
    def __init__(self,plan):
        require(plan['near_capture']['settings']==SETTINGS,'Unknown near-capture settings')
        self.extra_depth=float(plan['near_capture']['extra_depth_m'])
        require(self.extra_depth in (0.,.005),'Only isolated 0 or 5 mm depth experiments supported')
        super().__init__(plan)
        self.engaged=False;self.recovery_pose=None;self.previous_output=None;self.blending=False

    def apply(self,raw,frame,pinch_pose,finger_points,corners,detection):
        result,event=super().apply(raw,frame,pinch_pose,finger_points,corners,detection)
        active=event['follow']['active'];recovery=dict(holding=False,blending=False,depth_shift_m=0.)
        pose=np.asarray(pinch_pose,float)
        if active:self.engaged=True
        if self.engaged and not active:
            # A one-frame reacquisition must never release a distant raw model goal.
            if self.recovery_pose is None:self.recovery_pose=pose.copy()
            result[:7]=self.recovery_pose
            if event['mode']=='unaligned_height_floor':result[2]=max(result[2],event['target_floor_m'])
            result[7:13]=0.;result[14:17]=0.;result[23]=0.
            recovery['holding']=True;self.blending=True
        elif active:
            if event['mode']=='aligned_pass' and self.extra_depth:
                # This is a conservative sphere-bottom proxy, not a measured support plane.
                current=Rotation.from_quat(pose[[4,5,6,3]])
                desired=Rotation.from_quat(result[[4,5,6,3]])
                local=current.inv().apply(np.asarray(corners).reshape(-1,3)-pose[:3])
                minimum_offset=float(desired.apply(local)[:,2].min())
                floor=float(detection['position_world_m'][2]-self.plan['ball_radius_m']+
                    SETTINGS['minimum_clearance_above_ball_bottom_m']-minimum_offset)
                before=float(result[2]);result[2]=max(before-self.extra_depth,floor)
                if result[2]>=floor and before-self.extra_depth<floor:
                    for c in (9,12,16):result[c]=max(0.,result[c])
                recovery.update(depth_shift_m=float(before-result[2]),sphere_bottom_proxy_floor_m=floor)
            if self.blending and self.previous_output is not None:
                previous=self.previous_output;delta=result[:3]-previous[:3];distance=float(np.linalg.norm(delta))
                if distance>SETTINGS['recovery_translation_step_m']:
                    result[:3]=previous[:3]+delta*SETTINGS['recovery_translation_step_m']/distance
                qa=Rotation.from_quat(previous[[4,5,6,3]]);qb=Rotation.from_quat(result[[4,5,6,3]])
                rv=(qa.inv()*qb).as_rotvec();angle=float(np.linalg.norm(rv));limit=np.radians(SETTINGS['recovery_rotation_step_deg'])
                if angle>limit:
                    xyzw=(qa*Rotation.from_rotvec(rv*limit/angle)).as_quat();result[3:7]=xyzw[[3,0,1,2]]
                # Height protection takes precedence over smoothing an unsafe low goal.
                if event['mode']=='unaligned_height_floor':result[2]=max(result[2],event['target_floor_m'])
                result[7:13]=0.;result[14:17]=0.;result[23]=0.
                recovery['blending']=True
                self.blending=distance>SETTINGS['recovery_translation_step_m'] or angle>limit
            self.recovery_pose=None
        # Recheck the actually issued orientation after recovery interpolation.
        # Endpoint orientation bounds alone need not cover intermediate rotations.
        floor_z=None
        if event['mode']=='unaligned_height_floor':floor_z=event['finger_floor_m']
        elif active and self.extra_depth:
            floor_z=float(detection['position_world_m'][2]-self.plan['ball_radius_m']+
                SETTINGS['minimum_clearance_above_ball_bottom_m'])
        if floor_z is not None:
            current=Rotation.from_quat(pose[[4,5,6,3]])
            issued=Rotation.from_quat(result[[4,5,6,3]])
            local=current.inv().apply(np.asarray(corners).reshape(-1,3)-pose[:3])
            floor_target=float(floor_z-issued.apply(local)[:,2].min())
            if result[2]<floor_target:
                result[2]=floor_target
                for c in (9,12,16):result[c]=max(0.,result[c])
            recovery['issued_orientation_floor_m']=floor_target
        self.previous_output=result.copy();event['near_capture']=recovery
        event['changed_channels']=np.flatnonzero(result!=raw).tolist()
        require(result[28]==raw[28],'Near capture must preserve jaw score')
        return result,event
