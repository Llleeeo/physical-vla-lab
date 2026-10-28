import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from capstone_vla.hf_archive import RAW_FILES, sha256
from capstone_vla.hgwind_cohort import (TASK_ID, CAMERAS, inspect_recording, stage_episode,
                                      assert_metadata_enrichment, select_pending, verify_visual)


@pytest.fixture
def episode(tmp_path):
    ep = tmp_path / 'source' / 'traj_00000'
    ep.mkdir(parents=True)
    meta = dict(task={'id': TASK_ID}, provenance={'source_task': {'sha256': 'scene'}},
                success=True, trajectory_validity={'passed': True}, runtime_safety={'passed': True},
                grasp_mode='physical_finger_contact_only', fix_joint=False, fixedforce=False,
                capstone_expert_control={'profile': 'smooth_lift_v1'}, num_steps=3,
                control_hz=60, attempt_index=0, episode_seed=17)
    (ep / 'meta.json').write_text(json.dumps(meta))
    (ep / 'analyse.jsonl').write_text('{}\n')
    for name in ['replay.hdf5', 'trajectory.hdf5']:
        with h5py.File(ep / name, 'w') as f:
            f.attrs['recording_id'] = 'same-id'
            f['timestamp'] = np.arange(3) / 60
            f['actions'] = np.zeros((3, 9))
            for field, shape in [('joint_pos', (3, 9)), ('joint_vel', (3, 9)), ('ee_pos', (3, 3)),
                                 ('ee_quat', (3, 4)), ('object_pos', (3, 1, 3)),
                                 ('object_quat', (3, 1, 4)), ('object_vel', (3, 1, 3)),
                                 ('object_ang_vel', (3, 1, 3))]:
                values = np.zeros(shape)
                if 'quat' in field:
                    values[..., 0] = 1
                f['obs/' + field] = values
            if name == 'replay.hdf5':
                f['evidence/contacts/step_time'] = np.arange(9) / 180
                f['evidence/contacts/step_dt'] = np.full(9, 1 / 180)
                f['evidence/contacts/headers'] = np.array([[0, 0, 1, 0, 1, 2, 0, 1]])
                f['evidence/contacts/points'] = np.zeros((1, 10))
    return ep


def hashes(ep):
    return [dict(path=n, bytes=(ep / n).stat().st_size, sha256=sha256(ep / n)) for n in RAW_FILES]


def test_valid_core_replay_and_exact_clock(episode):
    assert inspect_recording(episode, 'scene')['frames'] == 3


@pytest.mark.parametrize('field,value,match', [
    ('timestamp', np.array([0., .02, .04]), 'clock'),
    ('actions', np.full((3, 9), np.nan), 'mismatch|Nonfinite'),
])
def test_bad_recordings_rejected(episode, field, value, match):
    with h5py.File(episode / 'replay.hdf5', 'r+') as f:
        f[field][:] = value
    with pytest.raises(ValueError, match=match):
        inspect_recording(episode, 'scene')


def test_assistance_and_wrong_scene_rejected(episode):
    with pytest.raises(ValueError, match='provenance'):
        inspect_recording(episode, 'different')
    m = json.loads((episode / 'meta.json').read_text())
    m['fixedforce'] = True
    (episode / 'meta.json').write_text(json.dumps(m))
    with pytest.raises(ValueError, match='Assisted'):
        inspect_recording(episode, 'scene')


def test_stage_preserves_raw_and_allows_only_camera_metadata(episode, tmp_path):
    files = hashes(episode)
    target = tmp_path / 'work' / 'traj_00000'
    stage_episode(episode, target, files)
    copied = json.loads((target / 'meta.json').read_text())
    copied['camera_render'] = {'camera_fps': 60}
    (target / 'meta.json').write_text(json.dumps(copied))
    stage_episode(episode, target, files)
    assert hashes(episode) == files
    copied['success'] = False
    (target / 'meta.json').write_text(json.dumps(copied))
    with pytest.raises(ValueError, match='Non-camera'):
        stage_episode(episode, target, files)


def test_staging_rejects_changed_replay(episode, tmp_path):
    files = hashes(episode)
    target = tmp_path / 'work'
    stage_episode(episode, target, files)
    (target / 'replay.hdf5').write_bytes(b'changed')
    with pytest.raises(ValueError, match='work copy changed'):
        stage_episode(episode, target, files)


def test_selection_covers_hosts_and_resumes_missing():
    rows = [dict(host=h, episode=str(i)) for h in ('local', 'hongtao') for i in range(3)]
    chosen = select_pending(rows, lambda r: r['host'] == 'local' and r['episode'] == '0', 2)
    assert [(r['host'], r['episode']) for r in chosen] == [('local', '1'), ('hongtao', '0')]
    assert len(select_pending(rows, lambda r: False, 0)) == 6


def test_visual_full_clock_and_partial_rejection(episode):
    (episode / 'video').mkdir()
    path = episode / 'video/rendered.hdf5'
    with h5py.File(path, 'w') as f:
        f.attrs['fps'] = 60
        f['source_frame_index'] = np.arange(3)
        f['timestamp_s'] = np.arange(3) / 60
        for name in CAMERAS:
            f.create_dataset(name + '_rgb', shape=(3, 480, 640, 3), dtype='u1', fillvalue=0)
    assert verify_visual(episode, 3)['suspicious_blank_count'] == 15
    with h5py.File(path, 'r+') as f:
        f['source_frame_index'][1] = 2
    with pytest.raises(ValueError, match='Missing visual frames'):
        verify_visual(episode, 3)
