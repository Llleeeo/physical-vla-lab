"""Synthetic seven-joint fixtures check frame mathematics, not robot dynamics."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from capstone_vla.franka import FrankaCartesianAdapter, FrankaActionContract
from capstone_vla.recorded_franka import RecordedFrankaAdapter


@pytest.fixture
def urdf(tmp_path):
    parts=['<robot name="synthetic">']
    for i in range(1,8):
        axis=['1 0 0','0 1 0','0 0 1'][i%3]
        parts.append(f'<joint name="panda_joint{i}" type="revolute"><parent link="panda_link{i-1}"/><child link="panda_link{i}"/><origin xyz="0.01 0 0.08" rpy="0 0 0"/><axis xyz="{axis}"/><limit lower="-3" upper="3"/></joint>')
    parts.extend(['<joint name="panda_joint8" type="fixed"><parent link="panda_link7"/><child link="panda_link8"/><origin xyz="0 0 0.107" rpy="0 0 0"/></joint>',
                  '<joint name="panda_hand_joint" type="fixed"><parent link="panda_link8"/><child link="panda_hand"/><origin xyz="0 0 0" rpy="0 0 -0.7853981633974483"/></joint>','</robot>'])
    p=tmp_path/'model.urdf';p.write_text(''.join(parts));return p


def frame():
    q=Rotation.from_euler('z',3*np.pi/4).as_quat()
    return dict(link_path='/robot/panda_link7',offset_position_m=[0,0,.207],offset_quat_wxyz=q[[3,0,1,2]].tolist(),frame='right_gripper')


def test_declared_tool_corrects_both_rotation_and_translation(urdf):
    old=FrankaCartesianAdapter(urdf);new=RecordedFrankaAdapter(urdf,frame());q=np.linspace(-.2,.2,7)
    p0,r0,_=old.pose_jacobian(q);p1,r1,_=new.pose_jacobian(q)
    assert np.linalg.norm(p1-p0)==pytest.approx(.0034)
    assert (r1*r0.inv()).magnitude()==pytest.approx(np.pi)


def test_tool_jacobian_matches_finite_differences(urdf):
    a=RecordedFrankaAdapter(urdf,frame());q=np.linspace(-.2,.2,7);p,r,j=a.pose_jacobian(q);numerical=np.empty_like(j)
    for i in range(7):
        qq=q.copy();qq[i]+=1e-7;pp,rr,_=a.pose_jacobian(qq)
        numerical[:,i]=np.r_[(pp-p)/1e-7,(rr*r.inv()).as_rotvec()/1e-7]
    np.testing.assert_allclose(j,numerical,rtol=0,atol=1e-7)


def test_legacy_offset_cancels_from_recorded_frame(urdf):
    q=np.zeros(7);a=RecordedFrankaAdapter(urdf,frame());b=RecordedFrankaAdapter(urdf,frame(),FrankaActionContract(ee_offset_m=.08))
    pa,ra,ja=a.pose_jacobian(q);pb,rb,jb=b.pose_jacobian(q)
    np.testing.assert_allclose(pa,pb,atol=1e-14);np.testing.assert_allclose(ja,jb,atol=1e-14)
    assert (ra*rb.inv()).magnitude()<1e-14


def test_identity_goal_does_not_change_joints(urdf):
    a=RecordedFrankaAdapter(urdf,frame());q=np.linspace(-.2,.2,7);p,r,_=a.pose_jacobian(q)
    target,finger=a.joint_target(q,a.contract.encode(p,r,-1))
    np.testing.assert_allclose(target,q,atol=1e-7);assert finger==0


def test_missing_or_wrong_reference_frame_is_rejected(urdf):
    f=frame();f['link_path']='/robot/panda_hand'
    with pytest.raises(ValueError,match='last arm link'):RecordedFrankaAdapter(urdf,f)


def test_invalid_quaternion_is_rejected(urdf):
    f=frame();f['offset_quat_wxyz']=[0,0,0,0]
    with pytest.raises(ValueError,match='transform'):RecordedFrankaAdapter(urdf,f)
