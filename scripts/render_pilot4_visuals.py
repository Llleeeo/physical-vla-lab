#!/usr/bin/env python3
"""Render measured pilot4 states; never resimulate or certify success."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'data/derived/hf_pilot_render_repo'
RUN=ROOT/'data/derived/development_validation_20261025/runs/grasp_fixed_axis_rotation_1s1p_akrsd68/20261025_165142'
PREPARED=ROOT/'data/derived/development_validation_20261025/pilot4_workspace'
TASK_ID='grasp_fixed_axis_rotation_1s1p_akrsd68'
LAUNCHER=Path('/home/robophys/RoboPhys/isaaclab/isaaclab.sh')

def ensure_environment_cameras(task):
    """Repair only the visualization USD and reject physical scene changes."""
    sys.path.insert(0,str(REPO/'Scripts'))
    from pxr import Usd
    from fix_task.fix_task_sensor_rig import repair_stage
    from dataset.scene_signature import stage_physics_signature
    from camera.manual_calibration import rgb_cameras, ENVIRONMENT
    stage=Usd.Stage.Open(str(task))
    environment=stage.GetPrimAtPath(ENVIRONMENT)
    if environment and environment.IsValid():
        rgb_cameras(stage,ENVIRONMENT)
        return
    layer=stage.GetRootLayer()
    original=layer.ExportToString()
    original_bytes=task.read_bytes()
    before=stage_physics_signature(stage)['sha256']
    try:
        report=repair_stage(stage,apply=True)
        rgb_cameras(stage,ENVIRONMENT)
        after=stage_physics_signature(stage)['sha256']
        if before!=after:
            raise RuntimeError('Camera setup changed physical scene; refusing to save')
        if task.read_bytes()!=original_bytes:
            raise RuntimeError('Visualization Task changed during camera setup')
        backup=task.with_suffix('.before-sensor-rig.usd')
        if backup.exists():
            raise FileExistsError(backup)
        backup.write_bytes(original_bytes)
        layer.Save()
    except Exception:
        layer.ImportFromString(original)
        raise
    (task.parent/'sensor_rig_setup.json').write_text(json.dumps({
        'physics_sha256_before':before,'physics_sha256_after':after,
        'physics_unchanged':True,'repair':report},indent=2)+'\n')
    print('ENVIRONMENT_CAMERAS_READY physics_unchanged=true',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--calibrate',action='store_true')
    selection.add_argument('--render',action='store_true')
    selection.add_argument('--dry-run',action='store_true',help='Run all renderer CPU preflights without launching Isaac')
    args=parser.parse_args()
    preflight=json.loads((PREPARED/'preflight.json').read_text())
    source=Path(preflight['episode'])
    manifest=RUN/'visualization_replays/manifest.json'
    if not manifest.exists():
        subprocess.run([sys.executable,str(ROOT/'scripts/prepare_task2_visuals.py'),str(RUN),'--source',str(source)],check=True)
    record=json.loads(manifest.read_text())['cartesian']
    episode=Path(record['render_episode'])
    assert record['frames']==185 and Path(record['source_trace'])==RUN/'cartesian_trace.hdf5'
    workspace=RUN/'visualization_workspace'
    task=workspace/'Tasks/1s1p'/f'{TASK_ID}.usd'
    task.parent.mkdir(parents=True,exist_ok=True)
    if not task.exists():
        shutil.copy2(PREPARED/'Tasks/1s1p'/task.name,task)
    for name in ['Backgrounds','Robots','SolidObjects','Sensors','InteractableObjects','Scripts']:
        link=workspace/name
        if not link.exists():
            link.symlink_to((PREPARED/name).resolve(),target_is_directory=True)
    ensure_environment_cameras(task)
    print('VISUALIZATION_ONLY frames=185 source=measured_PhysX_states training_eligible=false',flush=True)
    print('VIDEO_DIRECTORY',episode/'video',flush=True)
    if args.calibrate:
        return subprocess.call([sys.executable,str(REPO/'Scripts/manual_adjust_camera.py'),str(task),'--traj',str(episode),'--isaaclab',str(LAUNCHER)],cwd=ROOT)
    sys.path.insert(0,str(REPO/'Scripts'))
    from camera.manual_camera_render import require_manual_camera_file
    try:
        require_manual_camera_file(task)
    except ValueError as exc:
        print(str(exc),flush=True)
        print('NEXT: python scripts/render_pilot4_visuals.py --calibrate',flush=True)
        return 2 if args.render else 0
    if args.render or args.dry_run:
        import core.task_resolver as resolver
        resolver.REPO_ROOT=workspace
        resolver.TASKS_ROOT=workspace/'Tasks'
        path=REPO/'Scripts/tools/render_saved_trajectories.py'
        spec=importlib.util.spec_from_file_location('pilot4_renderer',path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.REPO=workspace
        # The scene copy is isolated; its linked assets belong to the parent
        # capstone workspace. Keep complete dependency/missing-file validation.
        dependency_resolver=module.resolve_task_dependencies
        def capstone_dependencies(_scene_root,tasks):
            return dependency_resolver(ROOT,tasks)
        module.resolve_task_dependencies=capstone_dependencies
        sys.argv=[str(path),'--launcher',str(LAUNCHER),'--out-root',str(RUN/'visualization_replays/Trajectory'),
                  '--task-id',TASK_ID,'--all-trajectories','--only-missing',
                  '--device','cuda:0','--render-quality','balanced','--camera-fps','60']
        if args.dry_run:
            sys.argv.append('--dry-run')
        return module.main()
    print('CPU_PREFLIGHT_OK cameras=ready',flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
