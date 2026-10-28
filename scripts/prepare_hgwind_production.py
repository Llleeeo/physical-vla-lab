#!/usr/bin/env python3
"""Freeze the exact reviewed HGwind control and minimal scene dependencies."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/derived/hgwind_current_baseline_20261027/repo'
REVIEWED = SOURCE / 'runs/20261027_185825_637620'
BUNDLE = ROOT / 'data/derived/hgwind_production_20261027/worker_bundle'
REPO_REL = SOURCE.relative_to(ROOT)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (BUNDLE / 'bundle_manifest.json').exists():
        sys.path.insert(0, str(BUNDLE / 'scripts'))
        from run_hgwind_production_worker import verify_bundle
        verify_bundle()
        print('EXISTING_FROZEN_BUNDLE_OK ' + str(BUNDLE))
        return
    if BUNDLE.exists():
        raise RuntimeError('Unsealed preparation already exists; inspect before continuing')
    provenance = json.loads((REVIEWED / 'run.json').read_text())
    if sha(SOURCE / 'Scripts/expert_data/collect_worker.py') != provenance['collector_sha256']:
        raise RuntimeError('Canonical collector changed since the reviewed run')
    task = Path(provenance['scene_selection']['task_path'])
    if sha(task) != provenance['task_sha256']:
        raise RuntimeError('Reviewed scene changed')
    for rel, key in [('src/capstone_vla/hgwind_control.py', 'control_adapter_sha256'),
                     ('scripts/sim/collect_hgwind_worker.py', 'control_worker_sha256'),
                     ('scripts/sim/dispatch_hgwind_collection.py', 'environment_adapter_sha256')]:
        if sha(REVIEWED / 'control_snapshot' / rel) != provenance[key]:
            raise RuntimeError('Reviewed control snapshot changed: ' + rel)
    sys.path.insert(0, str(SOURCE / 'Scripts'))
    from remote.dispatch import resolve_task_dependencies
    files, dependencies = resolve_task_dependencies(ROOT, [task])
    BUNDLE.mkdir(parents=True)
    shutil.copytree(SOURCE / 'Scripts', BUNDLE / REPO_REL / 'Scripts',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'logs', '.resources'))
    # The verifier imports shared asset-side Python helpers even for a ball scene.
    # Preserve its complete distribution reference so provenance checks remain active.
    shutil.copytree(SOURCE / 'Scripts/project_update/.resources',
                    BUNDLE / REPO_REL / 'Scripts/project_update/.resources',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for category in ('SolidObjects', 'InteractableObjects', 'Robots', 'Sensors'):
        for helper in (SOURCE / category).rglob('*'):
            if not helper.is_file() or helper.suffix not in ('.py', '.json', '.yaml', '.yml', '.toml') or '__pycache__' in helper.parts:
                continue
            destination = BUNDLE / helper.resolve().relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(helper, destination)
    for source in files:
        destination = BUNDLE / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for rel in ('src/capstone_vla/hgwind_control.py', 'scripts/sim/collect_hgwind_worker.py',
                'scripts/sim/dispatch_hgwind_collection.py'):
        destination = BUNDLE / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REVIEWED / 'control_snapshot' / rel, destination)
    shutil.copy2(ROOT / 'src/capstone_vla/__init__.py', BUNDLE / 'src/capstone_vla/__init__.py')
    shutil.copy2(ROOT / 'scripts/run_hgwind_production_worker.py', BUNDLE / 'scripts/run_hgwind_production_worker.py')
    dispatcher = BUNDLE / 'scripts/sim/dispatch_hgwind_collection.py'
    original = dispatcher.read_text()
    anchor = "LAB = Path('/home/robophys/RoboPhys/isaaclab')"
    if original.count(anchor) != 1:
        raise RuntimeError('Unexpected dispatcher Lab path')
    dispatcher.write_text(original.replace(anchor, "LAB = Path(os.environ['CAPSTONE_ISAACLAB_PATH'])"))
    # Reproduce the existing relative asset layout using links entirely within the snapshot.
    version_relative = task.parents[2].relative_to(ROOT)
    links = []
    for name in ('Backgrounds', 'Robots', 'SolidObjects', 'InteractableObjects', 'Sensors'):
        target = BUNDLE / (SOURCE / name).resolve().relative_to(ROOT)
        target.mkdir(parents=True, exist_ok=True)
        link = BUNDLE / REPO_REL / name
        link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=True)
    # Asset behavior scripts load sibling Scripts/ relative to their real path.
    asset_scripts = BUNDLE / (SOURCE / 'SolidObjects').resolve().parent.relative_to(ROOT) / 'Scripts'
    asset_scripts.symlink_to(os.path.relpath(BUNDLE / REPO_REL / 'Scripts', asset_scripts.parent), target_is_directory=True)
    for name in ('Backgrounds', 'Robots', 'SolidObjects', 'InteractableObjects', 'Sensors', 'Scripts'):
        link = BUNDLE / version_relative / name
        link.symlink_to(os.path.relpath(BUNDLE / REPO_REL / name, link.parent), target_is_directory=True)
    # MDL default textures can be plain strings and absent from UsdUtils assets.
    mdl_texture = SOURCE / 'Backgrounds/Home_agphpxv_ultra/Materials/Textures/white.png'
    destination = BUNDLE / mdl_texture.resolve().relative_to(ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mdl_texture, destination)
    context = json.loads((REVIEWED / 'task_context.json').read_text())
    config = {'schema': 'capstone_hgwind_production_v1', 'task_id': task.stem,
              'task_relative': task.relative_to(ROOT).as_posix(), 'task_sha256': provenance['task_sha256'],
              'source_reviewed_run': REVIEWED.name, 'source_reviewed_control': provenance['control_adapter_sha256'],
              'control_profile': 'smooth-lift', 'task_context': context,
              'target_successes': 100, 'host_target_successes': 50, 'host_attempt_cap': 200,
              'seeds': {'local': 202610280, 'hongtao': 202710280},
              'portability_change': 'dispatcher SDK path supplied by CAPSTONE_ISAACLAB_PATH only; control source/collector unchanged',
              'dependencies': dependencies}
    (BUNDLE / 'production_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    payload = []
    for p in sorted(BUNDLE.rglob('*')):
        if p.is_symlink():
            links.append({'path': p.relative_to(BUNDLE).as_posix(), 'target': os.readlink(p)})
        elif p.is_file():
            payload.append({'path': p.relative_to(BUNDLE).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)})
    manifest = {'schema': 'capstone_frozen_bundle_v1', 'files': payload, 'links': links}
    (BUNDLE / 'bundle_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'FROZEN_BUNDLE_READY files={len(payload)} bytes={sum(r["bytes"] for r in payload)} dependencies={dependencies["files"]} scene={config["task_sha256"]}')
    print('BUNDLE ' + str(BUNDLE))


if __name__ == '__main__':
    main()
