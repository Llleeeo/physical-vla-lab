import json
from pathlib import Path
import shutil
import sys
import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_hgwind_cohort import episode, hashes
from capstone_vla.hgwind_cohort import CAMERAS, stage_episode
from capstone_vla.hf_archive import describe_episode
from capstone_vla.hgwind_render_delivery import install_visual_download


def prepared(ep, tmp_path):
    source_hashes = hashes(ep)
    incoming = tmp_path / 'incoming' / 'traj_00000'
    shutil.copytree(ep, incoming)
    m = json.loads((incoming / 'meta.json').read_text())
    m['camera_render'] = {'camera_fps': 60}
    (incoming / 'meta.json').write_text(json.dumps(m))
    (incoming / 'video').mkdir()
    pixels = np.tile(np.arange(640, dtype=np.uint8)[None, :, None], (480, 1, 3))
    with h5py.File(incoming / 'video/rendered.hdf5', 'w') as f:
        f.attrs['fps'] = 60
        f['source_frame_index'] = np.arange(3)
        f['timestamp_s'] = np.arange(3) / 60
        for name in CAMERAS:
            f.create_dataset(name + '_rgb', data=np.stack([pixels] * 3), compression='gzip')
    for camera in CAMERAS:
        for modality in ['rgb', 'depth', 'seg']:
            (incoming / 'video' / (camera + '_' + modality + '_vis.mp4')).write_bytes(b'video fixture')
    _, archive = describe_episode(incoming, cohort='camera_v2', host='hongtao', stage='rendered')
    target = tmp_path / 'work' / 'traj_00000'
    stage_episode(ep, target, source_hashes)
    row = dict(host='hongtao', episode='traj_00000', source=str(ep), staged=str(target),
               files=source_hashes, frames=3)
    return incoming, row, archive


def test_archive_retains_all_modalities_and_delivery_is_repeatable(episode, tmp_path):
    incoming, row, archive = prepared(episode, tmp_path)
    assert len([f for f in archive['files'] if f['path'].endswith('_vis.mp4')]) == 15
    before = hashes(episode)
    assert install_visual_download(incoming, row, archive)['suspicious_blank_count'] == 0
    assert install_visual_download(incoming, row, archive)['frames'] == 3
    assert hashes(episode) == before


def test_cross_host_archive_cannot_be_installed(episode, tmp_path):
    incoming, row, archive = prepared(episode, tmp_path)
    archive['host'] = 'local'
    with pytest.raises(ValueError, match='identity'):
        install_visual_download(incoming, row, archive)


def test_partial_rgb_only_package_is_rejected(episode, tmp_path):
    incoming, row, archive = prepared(episode, tmp_path)
    archive['files'] = [f for f in archive['files'] if '_depth_vis' not in f['path']]
    with pytest.raises(ValueError, match='full RGB'):
        install_visual_download(incoming, row, archive)
    assert not Path(row['staged']).joinpath('video').exists()


def test_different_existing_video_is_not_overwritten(episode, tmp_path):
    incoming, row, archive = prepared(episode, tmp_path)
    target = Path(row['staged'])
    (target / 'video').mkdir()
    (target / 'video/rendered.hdf5').write_bytes(b'previous version')
    with pytest.raises(ValueError, match='checksum'):
        install_visual_download(incoming, row, archive)
    assert (target / 'video/rendered.hdf5').read_bytes() == b'previous version'
