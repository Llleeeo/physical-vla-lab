#!/usr/bin/env python3
"""Render saved successful HGwind states against their recorded scene version."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'data/derived/hgwind_current_baseline_20261027/repo'
TASK_ID = 'grasp_wind_driven_motion_1s1p_hgprgew'
DEFAULT_RUN = REPO / 'runs/20261027_185825_637620'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='GPU render; default CPU preflight')
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--episode', action='append', help='exact successful directory name; repeatable')
    args = parser.parse_args()
    run = args.run_dir.expanduser().resolve()
    run.relative_to(REPO / 'runs')
    record = json.loads((run / 'run.json').read_text())
    task = Path(record['scene_selection']['task_path']).resolve()
    task.relative_to(REPO)
    if task.stem != TASK_ID or digest(task) != record['task_sha256']:
        raise RuntimeError('Recorded task version/hash mismatch; refusing a different scene')
    task_root = run / 'Trajectory/1s1p' / TASK_ID
    episodes = []
    for episode in sorted(task_root.glob('traj_*')):
        metadata = json.loads((episode / 'meta.json').read_text())
        if metadata.get('success') is True and metadata.get('trajectory_validity', {}).get('passed') is True:
            if metadata['task']['id'] != TASK_ID or not (episode / 'replay.hdf5').is_file():
                raise RuntimeError('Invalid saved success: ' + str(episode))
            episodes.append((episode, int(metadata['num_steps'])))
    if args.episode:
        selected = set(args.episode)
        if not selected.issubset({path.name for path, _ in episodes}):
            raise ValueError('Requested episode is not an eligible success')
        episodes = [(path, frames) for path, frames in episodes if path.name in selected]
    if not episodes:
        raise RuntimeError('No eligible saved successes')
    # Preserve source evidence even when the renderer creates sibling camera/video files.
    protected = [task, *(path / name for path, _ in episodes
                          for name in ('meta.json', 'replay.hdf5', 'trajectory.hdf5')
                          if (path / name).is_file())]
    before = {path: digest(path) for path in protected}
    sys.path.insert(0, str(ROOT / 'scripts/sim'))
    from dispatch_hgwind_collection import install_environment_adapter, LAB
    install_environment_adapter()
    os.environ['ROBOPHYS_CAPSTONE_OFFLINE_TASK'] = '1'
    os.environ['ROBOPHYS_TASK_CONTEXT'] = json.dumps(json.loads((run / 'task_context.json').read_text()))
    os.environ['ISAACLAB_PATH'] = str(LAB)
    os.environ['ISAAC_SIM_ROOT'] = str((LAB / '_isaac_sim').resolve())
    os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
    path = REPO / 'Scripts/tools/render_saved_trajectories.py'
    spec = importlib.util.spec_from_file_location('hgwind_success_renderer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from core.task_resolver import ResolvedTaskScene

    def recorded_scene(query):
        if query not in (TASK_ID, TASK_ID.rsplit('_', 1)[-1]):
            raise ValueError('Unexpected task requested: ' + str(query))
        return ResolvedTaskScene(TASK_ID, '1s1p', task)

    module.resolve_task_scene = recorded_scene
    dependencies = module.resolve_task_dependencies
    module.resolve_task_dependencies = lambda _repo, tasks: dependencies(ROOT, tasks)
    sys.argv = [str(path), '--launcher', str(LAB / 'isaaclab.sh'),
                '--out-root', str(run / 'Trajectory'), '--task-id', TASK_ID,
                '--successful-only', '--only-missing', '--device', 'cuda:0',
                '--render-quality', 'balanced', '--camera-fps', '30']
    for episode, frames in episodes:
        sys.argv += ['--trajectory-name', episode.name]
        print(f'SAVED_SUCCESS {episode.name} frames={frames} video={episode / "video"}', flush=True)
    if not args.run:
        sys.argv.append('--dry-run')
    print('RECORDED_SCENE ' + str(task), flush=True)
    print('VISUALIZATION saved_state_replay=true expert=true new_physics_rollout=false', flush=True)
    try:
        return module.main()
    finally:
        if any(digest(path) != old for path, old in before.items()):
            raise RuntimeError('Protected source evidence changed during rendering')
        print('SOURCE_EVIDENCE_HASHES_UNCHANGED', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
