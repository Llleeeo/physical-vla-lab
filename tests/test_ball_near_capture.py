import copy
import numpy as np
import pytest
from test_ball_approach_guard import inputs
from test_ball_approach_follow import follow_plan
from capstone_vla.ball_near_capture import SETTINGS,NearCaptureGuard,silhouette_candidates,locate_for_plan
from capstone_vla.ball_approach_follow import make_approach_guard
from capstone_vla.ball_visual_geometry import locate_ball
K=np.array([[320.,0,320],[0,320,240],[0,0,1]])


def plan(depth=0.):
    p=follow_plan();p['near_capture']=dict(settings=copy.deepcopy(SETTINGS),extra_depth_m=depth);return p


def sphere_image(center,radius=.023):
    y,x=np.mgrid[:480,:640];rays=np.stack([(x-320)/320,-(y-240)/320,-np.ones_like(x)],axis=-1)
    rays=rays/np.linalg.norm(rays,axis=-1)[...,None]
    image=np.zeros((480,640,3),np.uint8)
    image[(np.linalg.norm(np.cross(rays,center),axis=-1)<radius)&((rays@center)>0)]=[220,110,40]
    return image


def test_partial_sphere_known_radius_and_independent_camera_check():
    center=np.array([0.,-.17,-.25]);poses=np.array([[0,0,0,1,0,0,0],[.1,-.1,0,1,0,0,0]])
    images=[sphere_image(center-p[:3]) for p in poses]
    assert not locate_ball(images,poses,[K,K])['valid']
    result=locate_for_plan(images,poses,[K,K],['wrist_cam','opst_cam'],plan())
    assert result['valid'] and result['method']=='clipped_sphere_with_external_check'
    np.testing.assert_allclose(result['position_world_m'],center,atol=.005)
    images[1][:]=0
    assert not locate_for_plan(images,poses,[K,K],['wrist_cam','opst_cam'],plan())['valid']


def test_empty_tiny_or_extremely_truncated_component_rejected():
    assert not silhouette_candidates(np.zeros((480,640,3),np.uint8),K,.023)
    assert not silhouette_candidates(sphere_image([0,-.22,-.25]),K,.023)
    assert not silhouette_candidates(sphere_image([0,-.01,-.25]),K,.023) # fully visible: use stereo
    with pytest.raises(ValueError):silhouette_candidates(np.zeros((2,2,3)),K,.023)


def test_old_locator_exact_when_option_off():
    poses=np.array([[0,0,0,1,0,0,0],[.1,0,0,1,0,0,0]])
    images=[sphere_image(np.array([0,0,-.25])-p[:3]) for p in poses]
    assert locate_for_plan(images,poses,[K,K],['wrist_cam','opst_cam'],follow_plan())==locate_ball(images,poses,[K,K])


def test_reacquisition_never_releases_raw_goal_while_velocity_warms():
    raw,pose,points,corners,d=inputs();d['position_world_m'][0]=.05;g=make_approach_guard(plan())
    for i in range(7):g.apply(raw,i,pose,points,corners,d)
    raw[0]=.7;raw[3:7]=[0,0,0,1]
    held,e=g.apply(raw,7,pose,points,corners,dict(valid=False))
    for i in range(8,15):
        value,e=g.apply(raw,i,pose,points,corners,d)
        assert e['near_capture']['holding']
        np.testing.assert_array_equal(value[:7],held[:7]);assert value[28]==raw[28]
    resumed,e=g.apply(raw,15,pose,points,corners,d)
    assert e['near_capture']['blending']
    assert np.linalg.norm(resumed[:2]-held[:2])<=.003+1e-12
    assert np.all(resumed[7:13]==0) and resumed[23]==0
    assert abs(resumed[3])>=np.cos(np.radians(.5))-1e-12


def test_depth_is_aligned_only_bounded_and_preserves_jaw():
    raw,pose,points,corners,d=inputs();d['position_world_m'][0]=.05
    raw[2]=.15;g=NearCaptureGuard(plan(.005));base=NearCaptureGuard(plan())
    for i in range(7):value,e=g.apply(raw,i,pose,points,corners,d);old,_=base.apply(raw,i,pose,points,corners,d)
    assert old[2]-value[2]==pytest.approx(.005)
    np.testing.assert_array_equal(value[np.arange(29)!=2],old[np.arange(29)!=2])
    d['position_world_m'][0]=.07
    value,e=g.apply(raw,7,pose,points,corners,d)
    assert e['mode']=='unaligned_height_floor' and e['near_capture']['depth_shift_m']==0
    with pytest.raises(ValueError):NearCaptureGuard(plan(.02))


def test_depth_floor_prevents_driving_reference_bounds_below_sphere_bottom_proxy():
    raw,pose,points,corners,d=inputs();d['position_world_m'][0]=.05
    g=NearCaptureGuard(plan(.005))
    for i in range(7):value,e=g.apply(raw,i,pose,points,corners,d)
    assert value[2]+(corners-pose[:3])[:,:,2].min()>=.01-1e-12
    assert np.all(value[[9,12,16]]>=0) and value[28]==raw[28]


def test_interpolated_orientation_is_checked_against_finger_floor():
    from scipy.spatial.transform import Rotation
    raw,pose,points,corners,d=inputs();d['position_world_m'][0]=.05
    # Endpoint rotations have equal Z, but intermediate tilt lowers one corner.
    corners[0,:,0]-=.05;corners[1,:,0]+=.05
    g=NearCaptureGuard(plan(.005))
    for i in range(7):g.apply(raw,i,pose,points,corners,d)
    g.apply(raw,7,pose,points,corners,dict(valid=False))
    raw[3:7]=[0,0,1,0]
    for i in range(8,16):value,e=g.apply(raw,i,pose,points,corners,d)
    issued=Rotation.from_quat(value[[4,5,6,3]])
    world=value[:3]+issued.apply(corners.reshape(-1,3)-pose[:3])
    assert world[:,2].min()>=.01-1e-12
