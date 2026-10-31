import copy
import numpy as np
import pytest
from test_ball_approach_guard import inputs
from test_ball_near_capture import plan as near_plan
from capstone_vla.ball_capture_window import SETTINGS,CaptureWindowGuard
from capstone_vla.ball_approach_follow import make_approach_guard


def plan():
    p=near_plan(.005);p['capture_window']=copy.deepcopy(SETTINGS);return p


def entered():
    raw,pose,points,corners,d=inputs();pose[2]=.032;points[:,2]=.032;corners[:,:,2]=.032
    d['position_world_m']=[.05,0,.023];raw[2]=.07
    g=make_approach_guard(plan())
    for i in range(7):a,e=g.apply(raw,i,pose,points,corners,d)
    return g,raw,pose,points,corners,d,a,e


def test_geometry_entry_blocks_premature_lift_and_applies_five_mm():
    g,raw,pose,points,corners,d,a,e=entered()
    assert e['capture_window']['active'] and e['capture_window']['started_frame']==6
    assert a[2]==pytest.approx(.027) and a[28]==raw[28]
    np.testing.assert_array_equal(a[[9,12,16]],[0,0,0])
    raw[2]=.2
    a,e=g.apply(raw,7,pose,points,corners,d)
    assert a[2]<=.027+1e-12 # model lifting is suppressed only inside finite capture window


def test_small_alignment_drift_holds_capture_depth_instead_of_retracting():
    g,raw,pose,points,corners,d,_,_=entered()
    for i in range(7,20):
        d['position_world_m'][0]=.05+(i-6)*.001
        a,e=g.apply(raw,i,pose,points,corners,d)
    assert e['mode']=='unaligned_height_floor' and e['capture_window']['base_height_floor_suspended']
    assert a[2]<=.027+1e-12
    d['position_world_m'][0]=.075
    a,e=g.apply(raw,20,pose,points,corners,d)
    assert not e['capture_window']['active'] and e['capture_window']['reason']=='left_capture_corridor'
    assert a[2]>=e['target_floor_m']


def test_window_is_finite_no_contact_or_grasp_claim_and_no_retrigger():
    g,raw,pose,points,corners,d,_,_=entered()
    for i in range(7,49):a,e=g.apply(raw,i,pose,points,corners,d)
    assert not e['capture_window']['active'] and e['capture_window']['reason']=='window_expired_without_success_claim'
    assert a[2]>.027
    _,e=g.apply(raw,49,pose,points,corners,d)
    assert not e['capture_window']['active']


def test_missing_vision_pauses_descent_and_jaw_remains_model():
    g,raw,pose,points,corners,d,_,_=entered();raw[28]=.1
    a,e=g.apply(raw,7,pose,points,corners,{'valid':False})
    assert not e['capture_window']['active']
    assert e['capture_window']['reason']=='paused_for_vision_or_velocity'
    np.testing.assert_array_equal(a[:7],pose);assert a[28]==.1


def test_requires_near_height_and_explicit_depth():
    p=plan();p['near_capture']['extra_depth_m']=0
    with pytest.raises(ValueError):CaptureWindowGuard(p)
    raw,pose,points,corners,d=inputs();d['position_world_m'][0]=.05
    g=CaptureWindowGuard(plan())
    for i in range(7):_,e=g.apply(raw,i,pose,points,corners,d)
    assert e['capture_window']['started_frame'] is None


def test_recovery_vertical_displacement_does_not_consume_xy_budget():
    g,raw,pose,points,corners,d,_,_=entered()
    g.capture_finished=True;g.blending=True;g.previous_output=np.array(raw,float)
    g.previous_output[:3]=[.03,0,.2]
    a,e=g.apply(raw,7,pose,points,corners,d)
    assert a[0]-.03==pytest.approx(.003)
    assert e['near_capture']['blending']
