"""Task-specific RGB ball localization prototype; never reads simulator objects.

Orange appearance is an explicit assumption. No trained detector, depth sensor,
known trajectory, ground-truth association, or temporal extrapolation is used.
Ray consistency is a rejection heuristic, not an accuracy guarantee.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation
from .cartesian_target_replay import require


def orange_candidates(image):
    image=np.asarray(image)
    require(image.shape==(480,640,3) and image.dtype==np.uint8,'Expected640x480 RGB')
    red,green,blue=image.astype(float).transpose(2,0,1)
    mask=(red>90)&(red-green>25)&(red>1.25*green)&(green>1.12*blue)
    mask=ndimage.binary_closing(mask,iterations=2)
    labels,_=ndimage.label(mask);result=[]
    for label,region in enumerate(ndimage.find_objects(labels),start=1):
        y,x=region;height=y.stop-y.start;width=x.stop-x.start
        area=int((labels[region]==label).sum())
        if (area<12 or area>30000 or min(height,width)<4 or not .5<width/height<2
                or area/(width*height)<.45 or x.start<3 or y.start<3 or x.stop>637 or y.stop>477):
            continue
        result.append(dict(pixel=[(x.start+x.stop-1)/2,(y.start+y.stop-1)/2],
                           area=area,bbox=[x.start,y.start,width,height]))
    return result


def world_ray(pixel,pose,intrinsics):
    pixel,pose,k=map(lambda a:np.asarray(a,float),(pixel,pose,intrinsics))
    require(pixel.shape==(2,) and pose.shape==(7,) and k.shape==(3,3),'Invalid calibration shapes')
    require(all(np.isfinite(a).all() for a in (pixel,pose,k)) and
            k[0,0]>0 and k[1,1]>0 and abs(np.linalg.norm(pose[3:])-1)<1e-5,'Invalid calibration values')
    require(np.array_equal(k[2],[0,0,1]) and k[0,1]==0 and k[1,0]==0,'Unsupported camera intrinsics')
    # Saved USD camera axes: +X right, +Y up, -Z forward. Pixels: +v down.
    direction=np.array([(pixel[0]-k[0,2])/k[0,0],-(pixel[1]-k[1,2])/k[1,1],-1.])
    direction/=np.linalg.norm(direction)
    return pose[:3],Rotation.from_quat(pose[[4,5,6,3]]).apply(direction)


def triangulate(pixels,poses,intrinsics):
    rays=[world_ray(p,v,k) for p,v,k in zip(pixels,poses,intrinsics)]
    require(len(rays)==2,'Need two rays')
    (origin_a,a),(origin_b,b)=rays
    angle=float(np.degrees(np.arccos(np.clip(abs(np.dot(a,b)),0,1))))
    if angle<5:return None
    da,db=np.linalg.lstsq(np.stack([a,-b],axis=1),origin_b-origin_a,rcond=None)[0]
    if min(da,db)<=.01:return None
    pa,pb=origin_a+da*a,origin_b+db*b
    gap=float(np.linalg.norm(pa-pb))
    if gap>.005:return None
    return dict(position_world_m=((pa+pb)/2).tolist(),ray_gap_m=gap,ray_angle_deg=angle)


def locate_ball(images,poses,intrinsics):
    require(len(images)==len(poses)==len(intrinsics)==2,'Need two synchronized calibrated views')
    candidates=[orange_candidates(image) for image in images];matches=[]
    for a in candidates[0]:
        for b in candidates[1]:
            value=triangulate([a['pixel'],b['pixel']],poses,intrinsics)
            if value is not None:matches.append(dict(value,pixels=[a['pixel'],b['pixel']]))
    result=dict(valid=len(matches)==1,candidate_counts=[len(c) for c in candidates],matching_pairs=len(matches))
    if len(matches)==1:result.update(matches[0])
    else:result['reason']='missing_or_inconsistent' if not matches else 'ambiguous'
    return result


def approach_geometry(ball_position,finger_points):
    """Reference-point offsets only, not finite-mesh clearance or grasp success."""
    ball=np.asarray(ball_position,float);fingers=np.asarray(finger_points,float)
    require(ball.shape==(3,) and fingers.shape==(2,3) and np.isfinite(ball).all()
            and np.isfinite(fingers).all(),'Invalid geometry')
    offsets=fingers-ball;midpoint=offsets.mean(axis=0)
    return dict(midpoint_horizontal_offset_m=float(np.linalg.norm(midpoint[:2])),
                finger_horizontal_offsets_m=np.linalg.norm(offsets[:,:2],axis=1).tolist(),
                finger_heights_above_center_m=offsets[:,2].tolist(),
                scope='configured robot contact reference points; not collision-surface clearance')
