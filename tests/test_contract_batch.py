"""Bounded selection, all-outcome review and unsupported teacher guards."""
import copy
import importlib.util
import json
from pathlib import Path
import h5py
import numpy as np
import pytest
from capstone_vla.contract_batch import (select_cases,saved_episode,inspect_review_videos,write_review_index)
from capstone_vla.hgwind_cohort import TASK_ID,CAMERAS


def rows():
    return [dict(host='local',episode='traj_'+str(i).zfill(5),attempt_index=i*10,
        numeric_fingerprint=str(i),episode_seed=i,depth_bias_m=.002,gripper_schema_supported=True) for i in range(21)]


def fixture_run(tmp_path,name='failed_00000'):
    root=tmp_path;run=root/'data/derived/run';ep=run/'worker_output/1s1p'/TASK_ID/name
    ep.mkdir(parents=True);(ep/'replay.hdf5').write_bytes(b'saved')
    (ep/'meta.json').write_text(json.dumps(dict(task={'id':TASK_ID},num_steps=3)))
    (run/'plan.json').write_text(json.dumps(dict(worker_output=str(run/'worker_output'))))
    return root,run,ep


def test_fixed_selection_spans_local_attempts_and_excludes_exposed_case():
    selected=select_cases(rows())
    assert [r['episode'] for r in selected]==['traj_00001','traj_00010','traj_00020']
    assert select_cases(list(reversed(rows())))==selected
    unclassified=rows()
    for row in unclassified:row.pop('gripper_schema_supported')
    with pytest.raises(ValueError):select_cases(unclassified)


def test_selection_rejects_duplicate_conditions_and_profile_changes():
    r=rows();r[10]['numeric_fingerprint']=r[1]['numeric_fingerprint']
    with pytest.raises(ValueError):select_cases(r)
    r=rows();r[20]['depth_bias_m']=.004
    with pytest.raises(ValueError):select_cases(r)


def test_failure_without_analysis_report_is_renderable(tmp_path):
    root,run,ep=fixture_run(tmp_path)
    assert saved_episode(root,run)[1]==ep
    assert not (run/'report.json').exists()


def test_renderer_rejects_ambiguous_or_escaped_episode(tmp_path):
    root,run,ep=fixture_run(tmp_path)
    (run/'report.json').write_text(json.dumps(dict(probe_episode=str(ep.parent/'traj_00000'))))
    with pytest.raises(ValueError):saved_episode(root,run)
    (run/'report.json').unlink()
    other=ep.parent/'traj_00000';other.mkdir();(other/'replay.hdf5').write_bytes(b'saved')
    with pytest.raises(ValueError):saved_episode(root,run)


def test_video_index_lists_failure_and_never_overwrites_manual_notes(tmp_path):
    state=dict(existing=[dict(label='failed_case',grasp='fail',render_state='failed',
        error='render interrupted')],cases=[dict(episode='traj_00001',teacher=dict(grasp='pass',
        render_state='complete',video_check={'videos':{c:{'path':'/tmp/'+c+'.mp4'} for c in CAMERAS}}),
        contract=dict(grasp='fail',render_state='pending'))])
    write_review_index(tmp_path,state)
    text=(tmp_path/'VIDEO_REVIEW.md').read_text()
    assert 'failed_case' in text and 'traj_00001/contract' in text and 'fail' in text
    assert all(c+'.mp4' in text for c in CAMERAS)
    (tmp_path/'MANUAL_REVIEW.md').write_text('human annotations')
    write_review_index(tmp_path,state)
    assert (tmp_path/'MANUAL_REVIEW.md').read_text()=='human annotations'


def video_fixture(tmp_path,monkeypatch,decoded_frames=2,indices=(0,2)):
    root,run,ep=fixture_run(tmp_path)
    with h5py.File(ep/'replay.hdf5','w') as f:f['timestamp']=np.arange(3)/60
    video=run/'visualization/Trajectory/1s1p'/TASK_ID/ep.name/'video';video.mkdir(parents=True)
    with h5py.File(video/'rendered.hdf5','w') as f:
        f['source_frame_index']=indices;f['timestamp_s']=np.asarray(indices)/60;f.attrs['fps']=30
        for c in CAMERAS:f.create_dataset(c+'_rgb',shape=(2,480,640,3),dtype='u1')
    for c in CAMERAS:(video/(c+'_rgb_vis.mp4')).write_bytes(b'mocked video stream')
    def probe(*a,**kw):return json.dumps({'streams':[dict(width=640,height=480,nb_read_frames=str(decoded_frames),r_frame_rate='30/1')]})
    monkeypatch.setattr('capstone_vla.contract_batch.subprocess.check_output',probe)
    return root,run


def test_all_five_videos_and_source_timing_are_checked(tmp_path,monkeypatch):
    root,run=video_fixture(tmp_path,monkeypatch)
    result=inspect_review_videos(root,run)
    assert len(result['videos'])==5 and result['rendered_frames']==2
    assert not result['new_physics_rollout'] and not result['training_render']


def test_truncated_mp4_rejected_even_when_archive_complete(tmp_path,monkeypatch):
    root,run=video_fixture(tmp_path,monkeypatch,decoded_frames=1)
    with pytest.raises(ValueError,match='Truncated'):inspect_review_videos(root,run)


def test_wrong_source_state_indices_rejected(tmp_path,monkeypatch):
    root,run=video_fixture(tmp_path,monkeypatch,indices=(0,1))
    with pytest.raises(ValueError,match='source-state clock'):inspect_review_videos(root,run)


def teacher_fixture():
    from capstone_vla.ball_action_contract import decode_action,encode_packet
    action=np.zeros(29);action[3]=1
    p=decode_action(action,0,[0.]*7)
    calls=[dict(frame=i,call_id=i,completed=True,inputs=dict(p['inputs'],**p['safety_callbacks'])) for i in range(3)]
    capture=dict(schema='physical-vla-lab.cartesian-target-capture.v1',frames=list(range(5)),calls=calls)
    config=dict(schema='physical-vla-lab.robot-posture-configuration.v1',ik_offset_rad=[0.]*7,
        homotopy_enabled=False,homotopy_rank=0,strength=1.)
    jaws=np.full((5,2),.04);jaws[2:]=0
    return capture,jaws,config


def prep_module():
    spec=importlib.util.spec_from_file_location('teacher_prep',Path(__file__).resolve().parents[1]/'scripts/prepare_hgwind_teacher_contract.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_teacher_labels_keep_jaws_and_explicit_terminal_hold_without_online_steps():
    capture,jaws,config=teacher_fixture()
    actions=prep_module().teacher_actions(capture,jaws,np.zeros(7),config)
    assert actions.shape==(5,29) and actions.dtype==np.float32
    assert actions[:,-1].tolist()==[0.,0.,1.,1.,1.]
    np.testing.assert_array_equal(actions[-1,7:13],np.zeros(6))


def test_incompatible_teacher_posture_is_explicitly_unsupported():
    capture,jaws,config=teacher_fixture();capture['calls'][0]['inputs']['posture_target']=[1.]*7
    with pytest.raises(ValueError,match='unsupported'):
        prep_module().teacher_actions(capture,jaws,np.zeros(7),config)


def test_finished_failed_physics_is_not_repeated_on_resume(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('batch_runner',Path(__file__).resolve().parents[1]/'scripts/run_hgwind_contract_batch.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def forbidden(*a,**kw):raise AssertionError('physics repeated')
    monkeypatch.setattr(module,'logged',forbidden)
    module.physics_stage(tmp_path,{},dict(state='finished',exit_code=1),['unused'],'failed_case')


def test_source_schema_screen_never_thresholds_intermediate_widths(tmp_path):
    from capstone_vla.contract_batch import classify_gripper_schema
    with h5py.File(tmp_path/'replay.hdf5','w') as f:
        a=np.zeros((3,9));a[0,7:]=.04;a[1,7:]=.0218;f['actions']=a
    assert not classify_gripper_schema(tmp_path)['gripper_schema_supported']
    with h5py.File(tmp_path/'replay.hdf5','a') as f:f['actions'][1,7:]=0.
    assert classify_gripper_schema(tmp_path)['gripper_schema_supported']


def test_selection_uses_declared_schema_eligibility_before_spacing():
    data=rows()
    for row in data:row['gripper_schema_supported']=row['episode'] in ['traj_00001','traj_00007','traj_00018']
    assert [r['episode'] for r in select_cases(data)]==['traj_00001','traj_00007','traj_00018']
