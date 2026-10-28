"""CPU checks and isolated staging for the stopped HGwind development cohort."""
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np

from capstone_vla.hf_archive import RAW_FILES, sha256

TASK_ID = 'grasp_wind_driven_motion_1s1p_hgprgew'
CAMERAS = ('opst_cam', 'side_left_cam', 'side_right_cam', 'overhead_cam', 'wrist_cam')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inspect_recording(episode, task_sha256):
    episode = Path(episode)
    m = json.loads((episode / 'meta.json').read_text())
    require(m['task']['id'] == TASK_ID, 'Wrong task identity')
    require(m['provenance']['source_task']['sha256'] == task_sha256, 'Wrong scene provenance')
    require(m.get('success') is True and m['trajectory_validity'].get('passed') is True,
            'Collector success or validity missing')
    require(m['runtime_safety'].get('passed') is True, 'Runtime safety failed')
    require(m.get('grasp_mode') == 'physical_finger_contact_only'
            and not m.get('fix_joint') and not m.get('fixedforce'), 'Assisted grasp')
    require(m['capstone_expert_control']['profile'] == 'smooth_lift_v1', 'Unexpected controller')
    n = int(m['num_steps'])
    require(n >= 2 and float(m['control_hz']) == 60., 'Unexpected sampling rate/length')
    fingerprint = hashlib.sha256()
    with h5py.File(episode / 'replay.hdf5', 'r') as full, h5py.File(episode / 'trajectory.hdf5', 'r') as core:
        t = full['timestamp'][:]
        require(t.shape == (n,) and np.isfinite(t).all()
                and np.allclose(np.diff(t), 1 / 60, rtol=0, atol=1e-7), 'Invalid 60 Hz clock')
        require(full['actions'].shape == (n, 9), 'Expected 9D recorded joint targets')
        require(full.attrs['recording_id'] == core.attrs['recording_id'], 'Core/replay identity mismatch')
        common = []
        def check(name, data):
            if not isinstance(data, h5py.Dataset):
                return
            values = data[:]
            require(name in full and np.array_equal(values, full[name][:]), 'Core/replay mismatch: ' + name)
            if np.issubdtype(values.dtype, np.number):
                require(np.isfinite(values).all(), 'Nonfinite data: ' + name)
            common.append(name)
        core.visititems(check)
        for name in ['timestamp', 'actions', 'obs/joint_pos', 'obs/joint_vel',
                     'obs/ee_pos', 'obs/ee_quat', 'obs/object_pos', 'obs/object_quat',
                     'obs/object_vel', 'obs/object_ang_vel']:
            data = full[name][:]
            require(data.shape[0] == n and np.isfinite(data).all(), 'Invalid state: ' + name)
            fingerprint.update(name.encode())
            fingerprint.update(str(data.shape).encode())
            fingerprint.update(data.astype('<f8').tobytes())
        for name in ['obs/ee_quat', 'obs/object_quat']:
            require(np.allclose(np.linalg.norm(full[name][:], axis=-1), 1., atol=1e-4),
                    'Invalid quaternion: ' + name)
        contacts = full['evidence/contacts']
        headers, points = contacts['headers'][:], contacts['points'][:]
        st, dt = contacts['step_time'][:], contacts['step_dt'][:]
        require(len(st) > 0 and np.isfinite(st).all() and np.all(np.diff(st) > 0)
                and np.isfinite(dt).all() and np.all(dt > 0), 'Invalid contact clock')
        require(headers.ndim == 2 and headers.shape[1] == 8 and len(headers) > 0
                and np.isfinite(headers).all() and np.isfinite(points).all(), 'Invalid raw contacts')
        require(np.all(headers[:, 0] >= 0) and np.all(headers[:, 0] < len(st))
                and np.all(headers[:, 6:8] >= 0)
                and np.all(headers[:, 6] + headers[:, 7] <= len(points)), 'Contact indexing invalid')
        return dict(frames=n, duration_s=float(t[-1] - t[0]), core_replay_equal_datasets=len(common),
                    maximum_dt_error_s=float(np.max(np.abs(np.diff(t) - 1 / 60))),
                    numeric_fingerprint=fingerprint.hexdigest(), contact_physics_steps=len(st),
                    action_representation='recorded_absolute_franka_joint_targets_9d',
                    attempt_index=int(m['attempt_index']), episode_seed=m.get('episode_seed'))


def verify_files(source, files):
    for row in files:
        path = Path(source) / row['path']
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == row['bytes']
                and sha256(path) == row['sha256'], 'Source checksum mismatch: ' + str(path))


def stage_episode(source, target, files):
    source, target = Path(source).resolve(), Path(target).resolve()
    require(source != target and not target.is_relative_to(source), 'Work copy must be outside source')
    verify_files(source, files)
    target.mkdir(parents=True, exist_ok=True)
    for row in files:
        require(row['path'] in RAW_FILES, 'Unexpected staged member')
        dst = target / row['path']
        require(not dst.is_symlink(), 'Stage must not contain symlinks')
        if dst.exists():
            # Only the copied metadata may be enriched by the renderer.
            if row['path'] != 'meta.json':
                require(sha256(dst) == row['sha256'], 'Existing work copy changed: ' + str(dst))
            else:
                assert_metadata_enrichment(source / 'meta.json', dst)
        else:
            shutil.copy2(source / row['path'], dst)
            require(sha256(dst) == row['sha256'], 'Copy checksum mismatch')
    require((source / 'meta.json').stat().st_ino != (target / 'meta.json').stat().st_ino
            or (source / 'meta.json').stat().st_dev != (target / 'meta.json').stat().st_dev,
            'Metadata must not be hardlinked')


def assert_metadata_enrichment(source, staged):
    original = json.loads(Path(source).read_text())
    copied = json.loads(Path(staged).read_text())
    mutable = {'camera_render', 'cameras', 'camera_count', 'camera_names', 'camera', 'vla_contract',
               'render_required_after_repair'}
    for key, value in original.items():
        if key not in mutable:
            require(copied.get(key) == value, 'Non-camera metadata changed: ' + key)


def select_pending(rows, complete, limit):
    """Interleave hosts so a two-episode pilot includes both configurations."""
    require(limit >= 0, 'limit must be nonnegative')
    groups = {host: [r for r in rows if r['host'] == host and not complete(r)]
              for host in ('local', 'hongtao')}
    ordered = []
    for i in range(max(map(len, groups.values()), default=0)):
        for host in groups:
            if i < len(groups[host]):
                ordered.append(groups[host][i])
    return ordered[:limit] if limit else ordered


def verify_visual(episode, frames):
    episode = Path(episode)
    with h5py.File(episode / 'replay.hdf5', 'r') as replay, h5py.File(episode / 'video/rendered.hdf5', 'r') as visual:
        require(np.array_equal(visual['source_frame_index'][:], np.arange(frames)), 'Missing visual frames')
        require(visual['timestamp_s'].shape == replay['timestamp'].shape
                and np.allclose(visual['timestamp_s'][:], replay['timestamp'][:], rtol=0, atol=1e-7),
                'Visual timestamp mismatch')
        require(float(visual.attrs['fps']) == 60., 'Wrong visual FPS')
        samples = []
        for name in CAMERAS:
            data = visual[name + '_rgb']
            require(data.shape == (frames, 480, 640, 3) and data.dtype == np.uint8, 'RGB schema mismatch')
            for i in sorted({0, frames // 2, frames - 1}):
                pixels = data[i]
                samples.append(dict(camera=name, frame=i, std=float(pixels.std()),
                                    suspicious_blank=bool(pixels.std() < 1 or np.mean(pixels.max(-1) < 3) > .99)))
        return dict(frames=frames, all_indices_and_timestamps_match=True, camera_fps=60,
                    sampled_images=samples, suspicious_blank_count=sum(s['suspicious_blank'] for s in samples),
                    manual_visual_reviewed=False)
