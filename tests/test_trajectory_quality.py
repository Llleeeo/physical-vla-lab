import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from capstone_vla.trajectory_quality import motion_metrics, contact_metrics, segmentation_metrics, describe_groups


def sample():
    n = 120
    return dict(time=np.arange(n)/60, ee=np.zeros((n,3)), quat_wxyz=np.tile([1.,0,0,0], (n,1)),
                ball=np.tile([0.,0,.1], (n,1)), ee_velocity=np.zeros((n,3)), phase=np.r_[np.full(60,3), np.full(30,12), np.full(30,15)],
                close=np.arange(n)>=50, pinch_offset=[0,0,.02], first_bilateral=50, lift_code=12, hold_code=15)


def test_rigid_hand_rotation_is_not_relative_center_slip():
    d=sample();rot=Rotation.from_euler('z',np.linspace(0,2,120)[:,None])
    d['ee']=np.column_stack([np.arange(120)/100,np.zeros((120,2))])
    d['ball']=d['ee']+rot.apply(np.tile([.03,0,.1],(120,1)))
    q=rot.as_quat();d['quat_wxyz']=q[:,[3,0,1,2]]
    r=motion_metrics(**d)
    assert r['tail_relative_center_jitter_max_mm'] < 1e-10
    assert r['lift_to_final_relative_center_shift_mm'] < 1e-10


def test_relative_shift_and_reopening_are_measured():
    d=sample();d['ball'][-1,0]+=.01;d['close'][-1]=False
    r=motion_metrics(**d)
    assert r['tail_relative_center_jitter_max_mm']==pytest.approx(10)
    assert r['reopen_transitions_after_first_bilateral']==1


def test_spawn_height_is_not_lift_reference():
    d=sample();d['ball'][:20,2]=1.;d['ball'][50:,2]=.16
    r=motion_metrics(**d)
    assert r['final_lift_from_first_bilateral_mm']==pytest.approx(0)


def test_missing_contact_or_lift_is_explicit():
    d=sample();d['first_bilateral']=None;d['phase'][:]=3
    r=motion_metrics(**d)
    assert r['first_lift_frame'] is None and r['final_lift_from_first_bilateral_mm'] is None


def test_bad_clock_rejected():
    d=sample();d['time'][10]=d['time'][9]
    with pytest.raises(ValueError,match='clock'):motion_metrics(**d)


def test_raw_contact_duration_and_tail_gap():
    t=np.arange(1,181)/180;dt=np.full(180,1/180);left=np.ones(180,bool);right=left.copy();right[-9:]=False
    r=contact_metrics(t,dt,left,right)
    assert r['raw_tail_bilateral_fraction']==pytest.approx(.9)
    assert r['raw_longest_bilateral_duration_s']==pytest.approx(.95)


def test_segmentation_zero_and_edge_are_not_conflated():
    a=np.zeros((10,10),np.uint16)
    assert segmentation_metrics(a,16)=={'pixels':0,'bbox':None,'touches_edge':False}
    a[0:3,3:6]=16;r=segmentation_metrics(a,16)
    assert r['pixels']==9 and r['touches_edge']


def test_bootstrap_comparison_is_deterministic_and_preserves_groups():
    rows=[{'host':h,'metric':v} for h in ['local','hongtao'] for v in [1.,2.,3.]]
    a=describe_groups(rows,'metric');assert a==describe_groups(rows,'metric')
    assert a['local']['median']==2 and a['local_minus_hongtao_median']['observed']==0


def test_alternating_physics_contacts_are_distinct_from_control_bin_contacts():
    t=np.arange(1,181)/180;dt=np.full(180,1/180);left=np.arange(180)%2==0;right=~left
    r=contact_metrics(t,dt,left,right,control_time=np.arange(60)/60)
    assert r['raw_tail_bilateral_fraction']==0
    assert r['raw_tail_control_bin_bilateral_fraction']==1


def test_recorded_relative_velocity_compensates_hand_rotation():
    d=sample();angles=np.arange(120)/60*2
    rotation=Rotation.from_euler('z',angles[:,None]);q=rotation.as_quat()
    d['quat_wxyz']=q[:,[3,0,1,2]];d['ball']=rotation.apply(np.tile([.03,0,.1],(120,1)))
    d['ball_velocity']=np.cross(np.tile([0,0,2.],(120,1)),d['ball'])
    assert motion_metrics(**d)['tail_relative_speed_p95_mps'] < 1e-12


def test_precontact_acceleration_excludes_later_contact_transient():
    d=sample();d['ee_velocity'][40,0]=1.;d['first_contact']=30
    assert motion_metrics(**d)['precontact_ee_acceleration_p95_mps2']==0
