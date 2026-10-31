import numpy as np
import pytest
from capstone_vla.ball_noise_replanning import SEEDS,noise_seed,assemble_prefix,stream_metrics


def test_predeclared_seed_pairs():
    assert len(SEEDS)==8 and SEEDS[0]==90000 and SEEDS[-1]==790000
    for seed in SEEDS:
        assert noise_seed('independent',seed,0)==noise_seed('shared',seed,0)
        assert noise_seed('independent',seed,268)==seed+268
        assert noise_seed('shared',seed,268)==seed


@pytest.mark.parametrize('mode,frame',[('other',0),('shared',1),('shared',-4)])
def test_bad_schedule(mode,frame):
    with pytest.raises(ValueError):noise_seed(mode,90000,frame)


def test_terminal_padding_not_consumed():
    chunks=np.zeros((68,20,29))
    for i in range(68):chunks[i,:,0]=i*4+np.arange(20)
    stream=assemble_prefix(chunks)
    np.testing.assert_array_equal(stream[:,0],np.arange(269))
    chunks[-1,1:]=999
    np.testing.assert_array_equal(assemble_prefix(chunks),stream)


def test_bad_chunks():
    with pytest.raises(ValueError):assemble_prefix(np.zeros((67,20,29)))


def teacher():
    a=np.zeros((269,29));a[:,3]=1;a[:,0]=np.arange(269)*.001;a[144:,28]=1
    return a


def test_exact_expert_motion_has_zero_residual_not_zero_movement():
    a=teacher();m=stream_metrics(a,a,np.zeros(7))
    assert m['precontact_increment_error_mm']==0
    for group in m['increments'].values():
        assert group['expert_increment_error_mm']['mean']==0
        assert group['predicted_step_mm']['mean']==pytest.approx(1)
    assert m['first_predicted_close_frame']==144 and m['false_closed_frames']==[]
    assert m['increments']['precontact_80_119_replan_boundary']['predicted_step_mm']['count']==10
    assert m['increments']['precontact_80_119_within_chunk']['predicted_step_mm']['count']==30


def test_boundary_jumps_and_jaw_timing_identified():
    a=teacher();p=a.copy();p[:,0]+=(np.arange(269)//4)*.01;p[140:144,28]=1
    m=stream_metrics(p,a,np.zeros(7))
    assert m['increments']['full_replan_boundary']['expert_increment_error_mm']['mean']==pytest.approx(10)
    assert m['increments']['full_within_chunk']['expert_increment_error_mm']['mean']==pytest.approx(0,abs=1e-10)
    assert m['false_closed_frames']==[140,141,142,143] and m['false_open_frames']==[]
    assert m['first_predicted_close_frame']==140
