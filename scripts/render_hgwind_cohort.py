#!/usr/bin/env python3
"""Render verified HGwind work copies at 60Hz; default is CPU preflight."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from capstone_vla.hf_archive import sha256
from capstone_vla.hgwind_cohort import (TASK_ID, require, verify_files, assert_metadata_enrichment,
                                      select_pending, verify_visual)

OUT = ROOT / 'data/derived/hgwind_40_20261028'
LAB = Path('/home/robophys/RoboPhys/isaaclab')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    global OUT, LAB
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--limit', type=int, default=0, help='Maximum missing episodes; 2 samples both hosts')
    parser.add_argument('--manifest', type=Path, default=OUT / 'render_manifest.json')
    parser.add_argument('--lab', type=Path, default=LAB)
    parser.add_argument('--host', choices=('all', 'local', 'hongtao'), default='all')
    parser.add_argument('--audit-only', action='store_true', help='Validate existing visuals without rendering')
    args = parser.parse_args()
    OUT = args.manifest.resolve().parent
    LAB = args.lab.resolve()
    require(args.limit >= 0, 'limit must be nonnegative')
    require(not (OUT / 'camera_revision_pending.json').exists(),
            'This old-camera plan is paused or superseded; use scripts/run_hgwind_dual_render.py for camera v2.')
    manifest = json.loads((OUT / 'render_manifest.json').read_text())
    require(manifest['schema'] == 'capstone.hgwind_cohort_render.v1'
            and manifest['episode_count'] == len(manifest['rows'])
            and 0 < manifest['episode_count'] <= 40 and manifest['camera_fps'] == 60, 'Unexpected cohort manifest')
    bundle, task = Path(manifest['bundle']), Path(manifest['task_usd'])
    require(sha256(task) == manifest['task_sha256'], 'Source Task USD changed')
    require(sha256(bundle / 'bundle_manifest.json') == manifest['bundle_manifest_sha256'], 'Frozen bundle changed')
    require((LAB / 'VERSION').read_text().strip() == '2.2.1'
            and ((LAB / '_isaac_sim').resolve() / 'VERSION').read_text().startswith('4.5.0'), 'SDK mismatch')
    lock = (OUT / 'render.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for row in manifest['rows']:
        verify_files(row['source'], row['files'])
        verify_files(row['staged'], [f for f in row['files'] if f['path'] != 'meta.json'])
        assert_metadata_enrichment(Path(row['source']) / 'meta.json', Path(row['staged']) / 'meta.json')
    os.environ.update(CAPSTONE_ISAACLAB_PATH=str(LAB), ISAACLAB_PATH=str(LAB),
                      ISAAC_SIM_ROOT=str((LAB / '_isaac_sim').resolve()),
                      OMNI_KIT_ACCEPT_EULA='YES', ROBOPHYS_CAPSTONE_OFFLINE_TASK='1',
                      PYTHONDONTWRITEBYTECODE='1')
    context = Path(manifest.get('task_context', str(Path(manifest['source_root']) / 'local/task_context.json')))
    os.environ['ROBOPHYS_TASK_CONTEXT'] = context.read_text()
    cache = Path(manifest['runtime_root']) if manifest.get('runtime_root') else None
    if cache:
        for name in ('xdg', 'tmp', 'config', 'cuda', 'gl'):
            (cache / name).mkdir(parents=True, exist_ok=True)
        os.environ.update(XDG_CACHE_HOME=str(cache / 'xdg'), TMPDIR=str(cache / 'tmp'),
                          OMNI_CONFIG_PATH=str(cache / 'config'), CUDA_CACHE_PATH=str(cache / 'cuda'),
                          __GL_SHADER_DISK_CACHE_PATH=str(cache / 'gl'))
    adapter = load(bundle / 'scripts/sim/dispatch_hgwind_collection.py', 'hgwind_frozen_render_adapter')
    adapter.install_environment_adapter()
    if manifest.get('physics_sha256'):
        from dataset.scene_signature import stage_physics_signature
        from pxr import Usd
        require(stage_physics_signature(Usd.Stage.Open(str(task)))['sha256'] == manifest['physics_sha256'],
                'Camera revision changed the physical scene')
    repo = adapter.REPO
    renderer = load(repo / 'Scripts/tools/render_saved_trajectories.py', 'hgwind_cohort_renderer')
    if cache:
        renderer.SINGLE_GPU_KIT_ARGS += ' --portable-root ' + str(cache / 'kit')
    from types import SimpleNamespace
    from camera.manual_camera_render import manual_camera_signature
    from camera.config import load_camera_config, config_digest
    sensor_profile = load_camera_config()['default_sensor_profile']
    def scene(query):
        require(query in (TASK_ID, TASK_ID.rsplit('_', 1)[-1]), 'Unexpected task query')
        return SimpleNamespace(task_id=TASK_ID, scene_family='1s1p', usd_path=task, relative_usd=str(task))
    renderer.resolve_task_scene = scene
    deps = renderer.resolve_task_dependencies
    renderer.resolve_task_dependencies = lambda _repo, tasks: deps(Path(manifest.get('dependency_root', str(ROOT))), tasks)
    task_row = dict(task_id=TASK_ID, scene_family='1s1p', usd=str(task))
    signature_before = manual_camera_signature(task)
    def complete(row):
        root = OUT / 'render_work' / row['host'] / 'Trajectory'
        return renderer._render_complete(root, task_row, Path(row['staged']) / 'replay.hdf5',
                                         camera_data=True, camera_fps=60., render_quality='balanced',
                                         sensor_profile=sensor_profile)
    scope = [r for r in manifest['rows'] if args.host == 'all' or r['host'] == args.host]
    require(bool(scope), 'Empty host allocation')
    selected = [] if args.audit_only else select_pending(scope, complete, args.limit)
    contract = {'task_sha256': manifest['task_sha256'],
                'bundle_manifest_sha256': manifest['bundle_manifest_sha256'],
                'manual_camera_signature': signature_before, 'camera_fps': 60,
                'sensor_config_digest': config_digest(load_camera_config()),
                'resolution': [640, 480], 'render_quality': 'balanced'}
    if manifest.get('expected_render_contract'):
        require(contract == manifest['expected_render_contract'], 'Host camera/render contract differs')
    (OUT / 'render_contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    print('COHORT_RENDER_PREFLIGHT scope=' + args.host + ' total=' + str(len(scope)) + ' pending_selected=' + str(len(selected))
          + ' fps=60 original_records=protected training_ready=false', flush=True)
    if args.run and selected:
        require(shutil.disk_usage(OUT).free > 30 * 1024**3, 'Less than 30GiB free; render smaller batches after freeing space')
    print('OUTPUT ' + str(OUT / 'render_work'), flush=True)
    try:
        for host in ('local', 'hongtao'):
            chosen = [r for r in selected if r['host'] == host]
            if not chosen:
                continue
            arguments = [str(repo / 'Scripts/tools/render_saved_trajectories.py'),
                         '--launcher', str(LAB / 'isaaclab.sh'), '--out-root',
                         str(OUT / 'render_work' / host / 'Trajectory'), '--task-id', TASK_ID,
                         '--successful-only', '--only-missing', '--device', 'cuda:0',
                         '--render-quality', 'balanced', '--camera-fps', '60']
            for row in chosen:
                arguments += ['--trajectory-name', row['episode']]
                print('SELECTED ' + host + '/' + row['episode'] + ' frames=' + str(row['frames']), flush=True)
            if not args.run:
                arguments.append('--dry-run')
            sys.argv = arguments
            code = renderer.main()
            if code:
                return code
    finally:
        require(sha256(task) == manifest['task_sha256']
                and manual_camera_signature(task) == signature_before, 'Scene/cameras changed')
        for row in manifest['rows']:
            verify_files(row['source'], row['files'])
            verify_files(row['staged'], [f for f in row['files'] if f['path'] != 'meta.json'])
            assert_metadata_enrichment(Path(row['source']) / 'meta.json', Path(row['staged']) / 'meta.json')
        print('RAW_SOURCE_AND_STAGED_PHYSICS_HASHES_UNCHANGED', flush=True)
    if args.run or args.audit_only:
        visuals = []
        for row in scope:
            if complete(row):
                visuals.append(dict(host=row['host'], episode=row['episode'], path=row['staged'],
                                    **verify_visual(row['staged'], row['frames'])))
        ready = {(r['host'], r['episode']) for r in visuals}
        require(all((r['host'], r['episode']) in ready for r in selected), 'Selected render incomplete')
        report = dict(ready=len(visuals), total=len(scope), rows=visuals, host=args.host, training_ready=False,
                      manual_visual_reviewed=False)
        (OUT / ('visual_audit.json' if args.host == 'all' else 'visual_audit_' + args.host + '.json')).write_text(json.dumps(report, indent=2) + '\n')
        print('COHORT_VISUAL_ALIGNMENT ready=' + str(len(visuals)) + '/' + str(len(scope))
              + ' suspicious_blank_samples=' + str(sum(v['suspicious_blank_count'] for v in visuals)), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
