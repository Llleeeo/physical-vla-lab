#!/usr/bin/env python3
"""Render the saved HG-wind failed expert attempt without re-executing actions."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'data/derived/hgwind_current_baseline_20261027/repo'
RUN = REPO / 'runs/20261027_161211_358940'
TASK_ID = 'grasp_wind_driven_motion_1s1p_hgprgew'
EPISODE = RUN / 'Trajectory/1s1p' / TASK_ID / 'failed_00000'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='render; default is CPU dry-run')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / 'scripts/sim'))
    from dispatch_hgwind_collection import install_environment_adapter, LAB
    install_environment_adapter()
    context = json.loads((RUN / 'task_context.json').read_text())
    os.environ['ROBOPHYS_CAPSTONE_OFFLINE_TASK'] = '1'
    os.environ['ROBOPHYS_TASK_CONTEXT'] = json.dumps(context, ensure_ascii=True)
    os.environ['ISAACLAB_PATH'] = str(LAB)
    os.environ['ISAAC_SIM_ROOT'] = str((LAB / '_isaac_sim').resolve())
    os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
    if not (EPISODE / 'replay.hdf5').is_file():
        raise FileNotFoundError(EPISODE / 'replay.hdf5')
    meta = json.loads((EPISODE / 'meta.json').read_text())
    if meta['task']['id'] != TASK_ID or meta['num_steps'] != 720:
        raise RuntimeError('Unexpected saved episode; review source before rendering')
    path = REPO / 'Scripts/tools/render_saved_trajectories.py'
    spec = importlib.util.spec_from_file_location('hgwind_saved_renderer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Shared assets resolve outside the private repo but remain under capstone.
    canonical_dependencies = module.resolve_task_dependencies
    def capstone_dependencies(_repo, tasks):
        return canonical_dependencies(ROOT, tasks)
    module.resolve_task_dependencies = capstone_dependencies
    sys.argv = [str(path), '--launcher', str(LAB / 'isaaclab.sh'),
                '--out-root', str(RUN / 'Trajectory'), '--task-id', TASK_ID,
                '--trajectory-name', 'failed_00000', '--only-missing',
                '--device', 'cuda:0', '--render-quality', 'balanced', '--camera-fps', '30']
    if not args.run:
        sys.argv.append('--dry-run')
    print('SAVED_STATE_VISUALIZATION frames=720 source=failed_00000 policy_training=false', flush=True)
    print('VIDEO_DIRECTORY ' + str(EPISODE / 'video'), flush=True)
    return module.main()


if __name__ == '__main__':
    raise SystemExit(main())
