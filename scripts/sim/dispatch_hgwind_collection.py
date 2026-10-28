#!/usr/bin/env python3
"""Run the unchanged private RoboPhys dispatcher with local Lab source paths.

The canonical child environment removes inherited PYTHONPATH deliberately.
Reintroduce only this project's pinned Isaac Lab packages AFTER that cleanup.
This CPU adapter does not instantiate SimulationApp or bypass collector gates.
"""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / 'data/derived/hgwind_current_baseline_20261027/repo'
LAB = Path('/home/robophys/RoboPhys/isaaclab')


def isaaclab_source_paths():
    paths = sorted(path for path in (LAB / 'source').iterdir()
                   if path.is_dir() and (path / path.name / '__init__.py').is_file())
    if not any(path.name == 'isaaclab' for path in paths):
        raise FileNotFoundError('Isaac Lab source package is absent: ' + str(LAB / 'source'))
    return paths


def install_environment_adapter():
    sys.path.insert(0, str(REPO / 'Scripts'))
    from sim_common import launcher
    canonical = launcher.isaac_process_environment

    def child_environment(base=None):
        environment = canonical(base)
        if environment.get('TERM') in (None, '', 'dumb'):
            environment['TERM'] = 'xterm'
        environment['PYTHONPATH'] = os.pathsep.join(map(str, isaaclab_source_paths()))
        return environment

    launcher.isaac_process_environment = child_environment
    return child_environment


def main():
    install_environment_adapter()
    dispatcher = REPO / 'Scripts/expert_data/collect.py'
    print('ISAACLAB_SOURCE_ADAPTER_OK ' + str(LAB / 'source'), flush=True)
    sys.argv[0] = str(dispatcher)
    namespace = runpy.run_path(str(dispatcher), run_name='capstone_canonical_collect')
    profile = os.environ.get('CAPSTONE_HGWIND_CONTROL_PROFILE', 'baseline')
    if profile not in ('baseline', 'smooth-lift', 'smooth-only', 'mimic-safe'):
        raise ValueError('Unknown HGwind control profile: ' + profile)
    if profile != 'baseline':
        namespace['main'].__globals__['WORKER'] = ROOT / 'scripts/sim/collect_hgwind_worker.py'
    print('EXPERT_CONTROL_DISPATCH profile=' + profile, flush=True)
    from sim_common.diagnostics import run_cli
    return run_cli(namespace['main'], str(dispatcher))


if __name__ == '__main__':
    raise SystemExit(main())
