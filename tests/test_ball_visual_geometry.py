import numpy as np
import pytest
from capstone_vla.ball_visual_geometry import world_ray,triangulate,locate_ball,orange_candidates,approach_geometry
K=np.array([[320.,0,320],[0,320,240],[0,0,1]])
POSES=np.array([[-.1,0,0,1,0,0,0],[.1,0,0,1,0,0,0]])


def test_triangulation_has_correct_usd_axes_and_metric_scale():
    result=triangulate([[352,224],[288,224]],POSES,[K,K])
    np.testing.assert_allclose(result['position_world_m'],[0,.05,-1],atol=1e-12)
    assert result['ray_gap_m']<1e-12
    assert triangulate([[320,240],[320,240]],POSES,[K,K]) is None
    assert triangulate([[288,240],[352,240]],POSES,[K,K]) is None
    assert triangulate([[352,224],[288,260]],POSES,[K,K]) is None


def test_orange_pair_uses_only_images_and_calibration_and_rejects_missing():
    images=[np.zeros((480,640,3),dtype=np.uint8) for _ in range(2)]
    for image,x in zip(images,[352,288]):image[215:234,x-9:x+10]=[220,110,40]
    result=locate_ball(images,POSES,[K,K]);assert result['valid']
    np.testing.assert_allclose(result['position_world_m'],[0,.05,-1],atol=1e-12)
    images[1][:]=0;assert not locate_ball(images,POSES,[K,K])['valid']
    assert orange_candidates(np.zeros((480,640,3),dtype=np.uint8))==[]
    with pytest.raises(ValueError):orange_candidates(np.zeros((1,1,3)))


def test_rejects_ambiguous_pair_instead_of_selecting_using_truth():
    images=[np.zeros((480,640,3),dtype=np.uint8) for _ in range(2)]
    images[0][215:234,343:362]=[220,110,40]
    for x in (288,240):images[1][215:234,x-9:x+10]=[220,110,40]
    result=locate_ball(images,POSES,[K,K]);assert not result['valid'] and result['reason']=='ambiguous'


def test_geometry_distinguishes_center_alignment_from_one_finger_over_ball():
    g=approach_geometry([0,0,0],[[.1,0,.025],[0,0,.025]])
    assert g['midpoint_horizontal_offset_m']==.05
    assert g['finger_horizontal_offsets_m']==[.1,0]
    assert g['finger_heights_above_center_m']==[.025,.025]
