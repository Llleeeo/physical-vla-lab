import copy
import numpy as np
import pytest
from test_ball_approach_guard import plan,inputs
from capstone_vla.ball_approach_guard import ApproachGuard
from capstone_vla.ball_approach_follow import FOLLOW_SETTINGS,FollowingApproachGuard,fit_visual_velocity,make_approach_guard


def follow_plan():
    p=plan();p['follow']=copy.deepcopy(FOLLOW_SETTINGS);return p


def test_causal_velocity_fit_and_follow_preserve_height_orientation_jaw_masks():
    raw,pose,points,corners,d=inputs();g=FollowingApproachGuard(follow_plan());base=ApproachGuard(plan())
    for i in range(7):
        d['position_world_m']=[.002*i,0,.023]
        result,event=g.apply(raw,i,pose,points,corners,d)
        old,_=base.apply(raw,i,pose,points,corners,d)
        assert event['follow']['active']==(i==6)
        # Only XY goal and XY motion components differ from the height-only guard.
        keep=[j for j in range(29) if j not in [0,1,7,8,10,11,14,15]]
        np.testing.assert_array_equal(result[keep],old[keep])
    np.testing.assert_allclose(result[:2],[.012,0],atol=1e-15)
    np.testing.assert_allclose(result[7:9],[.12,0],atol=1e-15)
    np.testing.assert_allclose(result[14:16],result[7:9],atol=0)
    assert result[28]==raw[28] and event['jaw_unchanged']


def test_missing_view_clears_velocity_history_and_holds():
    raw,pose,points,corners,d=inputs();g=FollowingApproachGuard(follow_plan())
    for i in range(7):g.apply(raw,i,pose,points,corners,d)
    result,e=g.apply(raw,7,pose,points,corners,{'valid':False})
    assert not e['follow']['active'] and not g.positions
    np.testing.assert_array_equal(result[:7],pose)
    for i in range(8,15):
        _,e=g.apply(raw,i,pose,points,corners,d)
        assert not e['follow']['active']
    _,e=g.apply(raw,15,pose,points,corners,d);assert e['follow']['active']


@pytest.mark.parametrize('case,reason', [('far','outside_near_region'),('fast','visual_speed_out_of_range'),('noise','poor_velocity_fit')])
def test_unreliable_or_far_tracks_do_not_override(case,reason):
    raw,pose,points,corners,d=inputs();g=FollowingApproachGuard(follow_plan());base=ApproachGuard(plan())
    for i in range(7):
        x=-.3 if case=='far' else (i/60 if case=='fast' else (.012 if i==5 else 0.))
        d['position_world_m']=[x,0,.023]
        result,e=g.apply(raw,i,pose,points,corners,d);old,_=base.apply(raw,i,pose,points,corners,d)
    assert not e['follow']['active'] and e['follow']['reason']==reason
    np.testing.assert_array_equal(result,old)


def test_desired_finger_midpoint_not_pinch_is_targeted_and_factory_is_opt_in():
    raw,pose,points,corners,d=inputs();pose[1]=.02
    raw[3:7]=[np.sqrt(.5),0,0,np.sqrt(.5)]
    g=make_approach_guard(follow_plan())
    for i in range(7):result,e=g.apply(raw,i,pose,points,corners,d)
    np.testing.assert_allclose(result[:2],[-.02,0],atol=1e-15)
    assert make_approach_guard(None) is None
    assert type(make_approach_guard(plan())) is ApproachGuard
    bad=follow_plan();bad['follow']['control_hz']=30
    with pytest.raises(ValueError,match='settings'):make_approach_guard(bad)


def test_velocity_fit_is_metric_and_rejects_nonfinite_or_wrong_history():
    t=np.arange(7)/60;v=np.array([.16,-.02,.01]);p=np.array([-4.,-1.,.1])+t[:,None]*v
    actual,residual=fit_visual_velocity(p)
    np.testing.assert_allclose(actual,v,atol=1e-13);assert residual<1e-13
    with pytest.raises(ValueError):fit_visual_velocity(p[:6])
    p[0,0]=np.nan
    with pytest.raises(ValueError):fit_visual_velocity(p)
