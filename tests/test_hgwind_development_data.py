import json
import h5py
import numpy as np
import pytest
from capstone_vla.hgwind_development_data import (aligned_indices,states_from_robot,
    normalization,BallDevelopmentDataset,SCHEMA)
from capstone_vla.hf_archive import sha256


def test_alignment_rejects_shift_and_missing_frames():
    ts=np.arange(5)/60.
    aligned_indices(ts,np.arange(5),ts,60)
    for idx,clock,fps in [(np.array([0,1,1,3,4]),ts,60),
                          (np.arange(5),ts+1/60,60),(np.arange(5),ts,30)]:
        with pytest.raises(ValueError):aligned_indices(ts,idx,clock,fps)


def test_pinch_transform_uses_world_rotation_and_no_object_state():
    q=np.arange(18,dtype=float).reshape(2,9)/100
    quat=np.array([[1,0,0,0],[0,1,0,0]],float)
    states=states_from_robot(q,q,np.zeros((2,3)),quat,np.ones(7),np.array([0,0,.0123]))
    np.testing.assert_allclose(states[:,16],[.0123,-.0123],atol=1e-9)
    np.testing.assert_array_equal(states[:,:7],q[:,:7].astype('float32'))
    np.testing.assert_array_equal(states[:,21:28],np.ones((2,7)))
    np.testing.assert_array_equal(states[:,28:],q[:,7:].astype('float32'))
    with pytest.raises(ValueError):states_from_robot(q,q,np.zeros((2,3)),np.zeros((2,4)),np.ones(7),np.zeros(3))


def test_stats_exclude_padding_and_preserve_constant_modes():
    actions=np.zeros((3,29),dtype='float32');actions[:,0]=[1,2,10];actions[:,28]=[0,1,1]
    stats=normalization(np.zeros((3,30)),actions,chunk=2)
    # Valid targets are [1,2], [2,10], [10]; final repeat is masked.
    assert np.isclose(stats['action']['mean'][0],5)
    assert stats['action']['std'][1]==1
    assert stats['observation.state']['std'][0]==1
    bad=actions.copy();bad[0,0]=np.nan
    with pytest.raises(ValueError):normalization(np.zeros((3,30)),bad)


@pytest.fixture
def payload(tmp_path):
    actions=np.arange(269*29,dtype='float32').reshape(269,29)
    states=np.full((269,30),7,dtype='float32')
    np.savez(tmp_path/'arrays.npz',actions=actions,states=states)
    image=tmp_path/'images.hdf5'
    with h5py.File(image,'w') as f:
        for cam in ['opst_cam','wrist_cam']:
            f.create_dataset(cam+'_rgb',data=np.broadcast_to(np.arange(269,dtype=np.uint8)[:,None,None,None],(269,2,3,3)))
    meta=dict(schema=SCHEMA,purpose='development_overfit_only',delta_action=False,
        images=str(image),instruction='Grasp the moving ball.',source_hashes={str(image):sha256(image)},
        payload_sha256=sha256(tmp_path/'arrays.npz'))
    (tmp_path/'dataset.json').write_text(json.dumps(meta))
    return tmp_path,actions


def test_absolute_actions_history_boundaries_and_contiguous_rgb(payload):
    path,actions=payload;ds=BallDevelopmentDataset(path)
    a=ds[0];b=ds[268]
    np.testing.assert_array_equal(a['action'],actions[:20])
    assert a['observation.state_is_pad'].tolist()==[True,False]
    assert int(b['action_is_pad'].sum())==19
    np.testing.assert_array_equal(b['action'][0],actions[-1])
    assert a['observation.images.opst_cam'].flags.c_contiguous
    assert not any('object' in key or 'contact' in key for key in a)
    assert a['observation.images.opst_cam'].max()<=1
    with pytest.raises(IndexError):ds[-1]


def test_payload_and_image_tampering_rejected(payload):
    path,_=payload
    with h5py.File(path/'images.hdf5','a') as f:f['opst_cam_rgb'][0]=99
    with pytest.raises(ValueError,match='Source changed'):BallDevelopmentDataset(path)
    with (path/'arrays.npz').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='arrays changed'):BallDevelopmentDataset(path,verify=False)
