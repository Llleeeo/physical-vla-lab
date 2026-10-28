"""Independent private HF episode archives; no collection/render or team API calls."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile

import h5py
import numpy as np

RAW_FILES = ('meta.json', 'replay.hdf5', 'trajectory.hdf5', 'analyse.jsonl')
SCHEMA = 'capstone.hf_episode_archive.v1'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def label(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,159}', value):
        raise ValueError('Unsafe archive identity: ' + value)
    return value


def describe_episode(episode, *, cohort, host, stage):
    episode = Path(episode).resolve(strict=True)
    if stage not in ('raw', 'rendered') or not re.fullmatch(r'traj_\d+', episode.name):
        raise ValueError('Choose a completed successful traj and raw/rendered stage')
    meta_path = episode / 'meta.json'
    meta = json.loads(meta_path.read_text())
    if (meta.get('success') is not True or meta.get('trajectory_validity', {}).get('passed') is not True
            or meta.get('grasp_mode') != 'physical_finger_contact_only'
            or meta.get('fix_joint') or meta.get('fixedforce')):
        raise ValueError('Episode lacks valid, unassisted collector success')
    task_id = label(meta['task']['id'])
    with h5py.File(episode / 'replay.hdf5') as replay:
        time = replay['timestamp'][:]
        n = len(time)
        if n < 2 or n != int(meta['num_steps']) or replay['actions'].shape != (n, 9):
            raise ValueError('Unexpected recorded frame count or Franka action schema')
        if not np.isfinite(time).all() or not np.all(np.diff(time) > 0):
            raise ValueError('Invalid source timestamps')
    selected = [episode / name for name in RAW_FILES if (episode / name).is_file()]
    if len(selected) != len(RAW_FILES):
        raise ValueError('Expected complete raw episode files')
    render = None
    if stage == 'rendered':
        render = meta.get('camera_render') or {}
        fps = float(render.get('camera_fps', 0))
        if abs(fps - float(meta['control_hz'])) > 1e-6:
            raise ValueError('Rendered archive requires full control-rate visual alignment')
        path = episode / 'video/rendered.hdf5'
        with h5py.File(path) as visual:
            indices = visual['source_frame_index'][:]
            vt = visual['timestamp_s'][:]
            if not np.array_equal(indices, np.arange(n)) or vt.shape != time.shape or not np.allclose(vt, time, atol=1e-6, rtol=0):
                raise ValueError('Incomplete or mismatched image/source timestamp alignment')
            cameras = [k for k in visual if k.endswith('_rgb')]
            if not cameras or any(visual[k].shape[0] != n or len(visual[k].shape) != 4 or visual[k].shape[-1] not in (3, 4) for k in cameras):
                raise ValueError('Missing/incomplete RGB streams')
            render = {'camera_fps': fps, 'frame_count': n, 'rgb_streams': cameras,
                      'camera_config_digest': str(visual.attrs.get('camera_config_digest', ''))}
        selected += [path, *sorted((episode / 'video').glob('*_vis.mp4'))]
    for path in selected:
        if path.is_symlink() or not path.resolve().is_relative_to(episode):
            raise ValueError('Source links outside episode are not allowed')
    files = [{'path': p.relative_to(episode).as_posix(), 'bytes': p.stat().st_size,
              'sha256': sha256(p)} for p in sorted(selected)]
    prefix = '/'.join(('episodes', task_id, label(cohort), label(host), label(episode.name), stage))
    return episode, {'schema': SCHEMA, 'prefix': prefix, 'task_id': task_id,
        'cohort': cohort, 'host': host, 'trajectory': episode.name, 'stage': stage,
        'collector_success': True, 'trajectory_validity_passed': True,
        'manual_visual_review_claimed': False, 'training_ready_claimed': False,
        'frame_count': n, 'control_hz': float(meta['control_hz']),
        'action_representation': 'recorded_absolute_franka_joint_targets_9d',
        'depth_bias': meta.get('capstone_expert_control', {}).get('depth_bias'),
        'episode_seed': meta.get('episode_seed'), 'render_alignment': render, 'files': files}


def pack_episode(episode, *, cohort, host, stage, output):
    episode, manifest = describe_episode(episode, cohort=cohort, host=host, stage=stage)
    output = Path(output).resolve()
    if output.is_relative_to(episode):
        raise ValueError('Archive output must be separate from source episode')
    target = output / manifest['prefix']
    target.mkdir(parents=True, exist_ok=True)
    payload = target / 'payload.tar'
    manifest_path = target / 'manifest.json'
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved['files'] != manifest['files'] or sha256(payload) != saved['payload']['sha256']:
            raise RuntimeError('Existing immutable stage differs; choose another cohort/version')
        return target, saved
    temporary = payload.with_suffix('.tar.part')
    try:
        with tarfile.open(temporary, 'w') as archive:
            for row in manifest['files']:
                source = episode / row['path']
                info = tarfile.TarInfo(row['path'])
                info.size, info.mode, info.mtime = row['bytes'], 0o644, 0
                with source.open('rb') as stream:
                    archive.addfile(info, stream)
        if any(sha256(episode / r['path']) != r['sha256'] for r in manifest['files']):
            raise RuntimeError('Source changed during packaging; retry after its writer completes')
        temporary.replace(payload)
        manifest['payload'] = {'name': 'payload.tar', 'bytes': payload.stat().st_size,
                               'sha256': sha256(payload)}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        return target, manifest
    finally:
        temporary.unlink(missing_ok=True)


def private_repo(api, config):
    info = api.repo_info(repo_id=config['repo_id'], repo_type='dataset')
    if info.private is not True:
        raise RuntimeError('Refusing archive writes/reads against a public repository')
    return info


def upload_package(package, config, *, api=None):
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    api = HfApi() if api is None else api
    if api.whoami().get('name') != config['owner']:
        raise RuntimeError('HF login is not the configured personal account')
    info = private_repo(api, config)
    package = Path(package)
    manifest = json.loads((package / 'manifest.json').read_text())
    payload = package / 'payload.tar'
    if sha256(payload) != manifest['payload']['sha256']:
        raise RuntimeError('Local archive checksum mismatch')
    prefix = manifest['prefix']
    try:
        old = hf_hub_download(config['repo_id'], prefix + '/manifest.json', repo_type='dataset', revision=info.sha)
    except EntryNotFoundError:
        old = None
    if old is not None:
        if json.loads(Path(old).read_text()) != manifest:
            raise RuntimeError('Cloud stage already has different content; refusing overwrite')
        return {'revision': info.sha, 'skipped_existing': True, 'prefix': prefix}
    operations = [CommitOperationAdd(path_in_repo=prefix + '/payload.tar', path_or_fileobj=payload),
                  CommitOperationAdd(path_in_repo=prefix + '/manifest.json', path_or_fileobj=package / 'manifest.json')]
    commit = api.create_commit(config['repo_id'], repo_type='dataset', operations=operations,
                              commit_message='Archive ' + '/'.join((manifest['cohort'], manifest['host'], manifest['trajectory'], manifest['stage'])))
    # Manifest and payload become visible together in the same commit.
    return {'revision': commit.oid, 'skipped_existing': False, 'prefix': prefix}


def unpack_verified(payload, manifest, destination):
    if manifest.get('schema') != SCHEMA or sha256(payload) != manifest['payload']['sha256']:
        raise ValueError('Unsupported archive or payload checksum mismatch')
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Extraction destination already exists')
    expected = {r['path']: r for r in manifest['files']}
    with tarfile.open(payload, 'r:') as archive:
        members = archive.getmembers()
        paths = [m.name for m in members]
        if len(set(paths)) != len(paths) or set(paths) != set(expected):
            raise ValueError('Archive members differ from manifest')
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or '\\' in member.name or not member.isfile() or member.size != expected[member.name]['bytes']:
                raise ValueError('Unsafe archive member')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.incoming_' + destination.name + '_', dir=destination.parent) as incoming:
            for member in members:
                target = Path(incoming) / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as src, target.open('xb') as dst:
                    while chunk := src.read(1024 * 1024):
                        dst.write(chunk)
                if sha256(target) != expected[member.name]['sha256']:
                    raise ValueError('Extracted member checksum mismatch')
            Path(incoming).rename(destination)
    return destination


def download_package(prefix, config, destination, *, revision=None):
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    info = private_repo(api, config)
    revision = info.sha if revision is None else revision
    # Pin both files to one immutable commit, including while other hosts upload.
    manifest_path = hf_hub_download(config['repo_id'], prefix + '/manifest.json', repo_type='dataset', revision=revision)
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest['prefix'] != prefix:
        raise ValueError('Remote manifest identity mismatch')
    payload = hf_hub_download(config['repo_id'], prefix + '/payload.tar', repo_type='dataset', revision=revision)
    result = unpack_verified(payload, manifest, destination)
    return {'revision': revision, 'destination': str(result), 'verified_files': len(manifest['files'])}
