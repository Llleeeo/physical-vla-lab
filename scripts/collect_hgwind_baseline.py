#!/usr/bin/env python3
"""Prepare, then optionally collect bounded fresh HG-wind physical expert attempts.

Uses a private copy of the existing RoboPhys collector and its original gates.
No historical replay is used as an action stream. No VLA training is performed.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
import re
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/derived/hf_pilot_render_repo'
SCREEN = ROOT / 'data/derived/ball_baseline_screen_20261027'
BASE = ROOT / 'data/derived/hgwind_current_baseline_20261027'
REPO = BASE / 'repo'
TASK_ID = 'grasp_wind_driven_motion_1s1p_hgprgew'
TASK_REL = Path('Tasks/1s1p') / (TASK_ID + '.usd')
INPUT = SCREEN / 'user_download_check/received' / (TASK_ID + '.usd')
INPUT_SHA = 'f13ede0143a8b614df318a90db8de9e3cdb7c29823f71c868bfd4fd0e98ce43a'
LAB = Path('/home/robophys/RoboPhys/isaaclab')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    if digest(INPUT) != INPUT_SHA:
        raise RuntimeError('Archived current Task has changed; review provenance first')
    marker = REPO / 'capstone_workspace.json'
    if not marker.exists():
        if REPO.exists():
            raise RuntimeError('Incomplete preparation exists: ' + str(REPO))
        REPO.mkdir(parents=True)
        # Copy executable code; the large distribution reference is read only here.
        resources = SOURCE / 'Scripts/project_update/.resources'
        shutil.copytree(SOURCE / 'Scripts', REPO / 'Scripts',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'logs', '.resources'))
        if resources.is_dir():
            (REPO / 'Scripts/project_update/.resources').symlink_to(resources, target_is_directory=True)
        for name in ('Backgrounds', 'Robots', 'SolidObjects', 'InteractableObjects', 'Sensors'):
            path = SOURCE / name
            if not path.is_dir():
                raise FileNotFoundError(path)
            (REPO / name).symlink_to(path, target_is_directory=True)
        (REPO / TASK_REL).parent.mkdir(parents=True)
        shutil.copy2(INPUT, REPO / TASK_REL)
        marker.write_text(json.dumps({'task_id': TASK_ID, 'input_sha256': INPUT_SHA,
            'collector_source': str(SOURCE), 'historical_scene_equivalence': False}, indent=2) + '\n')
    for name, target in (('isaaclab', LAB), ('isaacsim', (LAB / '_isaac_sim').resolve())):
        link = BASE / name
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
        if link.resolve() != target.resolve():
            raise RuntimeError('Existing Isaac link points elsewhere: ' + str(link))
    if not (REPO / '.venv').exists():
        (REPO / '.venv').symlink_to(SOURCE / '.venv', target_is_directory=True)
    if digest(REPO / TASK_REL) != INPUT_SHA:
        raise RuntimeError('Private Task changed; do not silently reuse a modified scene')


def prepare_scene(selection):
    task = REPO / TASK_REL
    info = {'selection': selection, 'original_scene_sha256': INPUT_SHA}
    if selection == 'edited':
        source = ROOT / 'data/derived/hgwind_scene_edit_20261027' / TASK_REL
        sha = digest(source)
        version = REPO / 'SceneVariants' / sha[:16]
        task = version / TASK_REL
        if task.exists():
            if digest(task) != sha:
                raise RuntimeError('Archived edited scene has changed: ' + str(task))
        else:
            task.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, task)
        for name in ('Backgrounds', 'Robots', 'SolidObjects', 'InteractableObjects', 'Sensors', 'Scripts'):
            link = version / name
            if not link.exists():
                link.symlink_to(REPO / name, target_is_directory=True)
        info.update(edited_source=str(source), version_root=str(version))
        previous = REPO / 'artifacts/collect_task_verify/1s1p' / (TASK_ID + '.json')
        saved = version / 'previous_task_verify_receipt.json'
        if previous.is_file() and not saved.exists():
            shutil.copy2(previous, saved)
        manifest = version / 'scene_version.json'
        if not manifest.exists():
            manifest.write_text(json.dumps({'edited_source': str(source), 'sha256': sha,
                'original_sha256': INPUT_SHA, 'task': str(task),
                'description': 'user-edited ball, fan and sample-region development scene; not historical reproduction'}, indent=2) + '\n')
    info.update(task_path=str(task), sha256=digest(task))
    return task, info


def context_and_preflight(task=None, scene_sha=None):
    task = REPO / TASK_REL if task is None else task
    scene_sha = INPUT_SHA if scene_sha is None else scene_sha
    meta_path = SCREEN / 'samples/RoboPhys001/Trajectory-008/Trajectory/1s1p' / TASK_ID / 'traj_00000/meta.json'
    meta = json.loads(meta_path.read_text())
    instruction = meta['task_plan']['task_cn']
    # Reuse the saved linguistic task, NOT the historical trajectory's physics state.
    context = {'schema_version': 1, 'source': 'capstone_local_hgwind_fresh_collection',
               'family': '1s1p', 'row': {'Task_ID': TASK_ID, 'Skill_1': 'Grasp',
               'Skill_2': '', 'Phenomenon_1': 'Wind-driven motion', 'Phenomenon_2': '',
               'Task_CN': instruction, 'Task_EN': '', 'build_ok': '完成'}}
    os.environ['ROBOPHYS_CAPSTONE_OFFLINE_TASK'] = '1'
    os.environ['ROBOPHYS_TASK_CONTEXT'] = json.dumps(context, ensure_ascii=True)
    sys.path.insert(0, str(REPO / 'Scripts'))
    from core.task_resolver import resolve_task
    from core.task_plan import build_execution_plan
    resolved = resolve_task(str(task))
    plan = build_execution_plan(resolved.spec).as_dict()
    if len(plan['steps']) != 1 or plan['steps'][0]['skill'] != 'Grasp' or plan['steps'][0]['target_role'] != 'A':
        raise RuntimeError('Unexpected local execution plan')
    from pxr import Usd
    stage = Usd.Stage.Open(str(task))
    if not stage or not stage.GetPrimAtPath('/Root/Dynamics/ball_y9bssvn').IsValid():
        raise RuntimeError('Expected ball is absent')
    if not stage.GetPrimAtPath('/Root/Dynamics/fan_a43dj8q').IsValid():
        raise RuntimeError('Expected wind source is absent')
    from remote.dispatch import resolve_task_dependencies
    _, dependencies = resolve_task_dependencies(ROOT, [task])
    return context, {'local_plan': plan, 'dependencies': dependencies,
        'instruction_source': str(meta_path), 'historical_plan': meta['task_plan'],
        'current_scene_sha256': scene_sha, 'historical_scene_equivalence': False,
        'gate_note': 'build_ok marks an available authored input; original Task verification remains required'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--scene', choices=('original', 'edited'), default='original',
                        help='edited snapshots the saved GUI scene into an independent version')
    parser.add_argument('--experiment-label', default='', help='safe label for a separate experiment directory')
    parser.add_argument('--baseline-report', type=Path, default=None,
                        help='compact retained baseline audit; no raw trajectory input')
    parser.add_argument('--seed', type=int, default=20261027)
    parser.add_argument('--attempt-offset', type=int, default=0, help='source attempt index for fresh-worker controls')
    parser.add_argument('--control-profile', choices=('baseline', 'smooth-lift', 'smooth-only', 'mimic-safe'), default='baseline',
                        help='baseline, smooth-lift, smooth-only, or independent-actuator mimic-safe diagnostic')
    parser.add_argument('--attempts', type=int, default=1,
                        help='total attempt budget; retain successes and failures (default: 1)')
    parser.add_argument('--capture-speed-gate', action='store_true', help='mimic-safe only: require the existing relative-speed bound before physical LIFT')
    parser.add_argument('--targeting-mode', choices=('auto', 'track', 'intercept'), default='auto',
                        help='explicit expert reference; track is a same-seed diagnostic against baseline auto/intercept')
    args = parser.parse_args()
    if not 0 <= args.seed < 2**31:
        parser.error('seed must be in [0, 2**31)')
    if args.attempts < 1:
        parser.error('attempts must be a positive integer')
    if args.experiment_label and not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,39}', args.experiment_label):
        parser.error('experiment-label must contain lowercase letters, numbers, underscores or hyphens (max 40)')
    if args.baseline_report is not None:
        args.baseline_report = args.baseline_report.resolve(strict=True)
        json.loads(args.baseline_report.read_text())
    if args.capture_speed_gate and args.control_profile != 'mimic-safe':
        parser.error('capture-speed-gate requires mimic-safe profile')
    if args.attempt_offset < 0:
        parser.error('attempt-offset must be nonnegative')
    prepare()
    task, scene_selection = prepare_scene(args.scene)
    context, audit = context_and_preflight(task, scene_selection['sha256'])
    audit['scene_selection'] = scene_selection
    if args.control_profile != 'baseline':
        baseline = json.loads((REPO / 'runs/20261027_165207_242720/run.json').read_text())
        if scene_selection['sha256'] != baseline['task_sha256']:
            raise ValueError('Smooth-lift comparison requires the frozen ten-attempt scene; review the scene selection')
    stamp = datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
    out = REPO / 'runs' / ((args.experiment_label + '_' if args.experiment_label else '') + stamp)
    cache = ROOT / 'data/derived/.cache/isaacsim/linux'
    command = [sys.executable, str(ROOT / 'scripts/sim/dispatch_hgwind_collection.py'),
        '--task', str(task), '--no-cloud', '--physics-check',
        '--episodes', str(args.attempts), '--target-all', str(args.attempts), '--max_attempts', str(args.attempts),
        '--seed', str(args.seed), '--attempt_offset', str(args.attempt_offset), '--run_id', 'hgwind_' + stamp,
        '--out_root', str(out / 'Trajectory'), '--grasp_mode', 'physical',
        '--catch-targeting-mode', args.targeting_mode,
        '--control_hz', '60', '--episode_s', '12', '--no-expert-config',
        '--keep_failed', '--preserve-failed-on-start', '--no-code-snapshot',
        '--no-visualization', '--headless', '--device', 'cuda:0',
        '--fast_exit_after_save', '--experience', 'isaacsim.exp.base.python.kit',
        '--kit_args=--portable-root ' + str(cache)
        + ' --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/enabled=false'
        + ' --/renderer/multiGpu/maxGpuCount=1 --/app/extensions/fsWatcherEnabled=false']
    print(f'CPU_PREFLIGHT_OK task=hgprgew attempts={args.attempts} physical_grasp=true offline_task=true', flush=True)
    print('SCENE selection=' + args.scene + ' sha256=' + scene_selection['sha256'], flush=True)
    print('EXPERT_CONTROL profile=' + args.control_profile + ' targeting_mode=' + args.targeting_mode + ' seed=' + str(args.seed) + ' success_thresholds=unchanged', flush=True)
    print('CAPTURE_SPEED_GATE enabled=' + str(args.capture_speed_gate).lower() + ' extra_dwell=false', flush=True)
    if args.experiment_label:
        print('EXPERIMENT label=' + args.experiment_label + ' baseline_report=' + str(args.baseline_report), flush=True)
    print('PLAN task_verifier -> passive_physics_check -> fresh_expert_collection', flush=True)
    print('OUTPUT ' + str(out), flush=True)
    print('COMMAND ' + shlex.join(command), flush=True)
    if not args.run:
        print('PREPARED_ONLY GPU_started=false full_task_verification=pending', flush=True)
        return 0
    env = dict(os.environ, CAPSTONE_HGWIND_CONTROL_PROFILE=args.control_profile, ISAACLAB_PATH=str(LAB), ISAAC_SIM_ROOT=str((LAB / '_isaac_sim').resolve()),
               OMNI_KIT_ACCEPT_EULA='YES', PYTHONDONTWRITEBYTECODE='1',
               OMNI_CONFIG_PATH=str(cache / 'config'), XDG_CACHE_HOME=str(cache / 'xdg'))
    env['CAPSTONE_HGWIND_CAPTURE_SPEED_GATE'] = '1' if args.capture_speed_gate else '0'
    out.mkdir(parents=True, exist_ok=False)
    (out / 'preflight.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    (out / 'task_context.json').write_text(json.dumps(context, ensure_ascii=False, indent=2) + '\n')
    provenance = {'command': command, 'seed': args.seed, 'attempt_budget': args.attempts,
                  'save_all_outcomes': True, 'purpose': 'fresh physical expert development collection',
                  'task_sha256': scene_selection['sha256'], 'scene_selection': scene_selection, 'launcher_sha256': digest(Path(__file__)),
                  'collector_sha256': digest(REPO / 'Scripts/expert_data/collect_worker.py'),
                  'environment_adapter_sha256': digest(ROOT / 'scripts/sim/dispatch_hgwind_collection.py'),
                  'historical_scene_equivalence': False, 'assisted_grasp_requested': False,
                  'expert_targeting_mode_requested': args.targeting_mode,
                  'expert_control_profile': args.control_profile,
                  'control_adapter_sha256': digest(ROOT / 'src/capstone_vla/hgwind_control.py'),
                  'control_worker_sha256': digest(ROOT / 'scripts/sim/collect_hgwind_worker.py'),
                  'diagnostic_baseline_run': str(REPO / ('runs/20261027_165207_242720' if args.control_profile != 'baseline' else 'runs/20261027_161211_358940')),
                  'control_change_scope': ('opt-in contact dwell, reduced lift speed and smooth physical carry references; original hard guards/scoring retained' if args.control_profile != 'baseline' else ('user-edited scene and explicitly recorded targeting mode; contact/success thresholds unchanged' if args.scene == 'edited' else 'free-space targeting mode only; no scene, contact or success-threshold edits'))}
    provenance['attempt_offset'] = args.attempt_offset
    provenance['experiment_label'] = args.experiment_label
    provenance['diagnostic_baseline_report'] = str(args.baseline_report) if args.baseline_report else None
    if args.control_profile == 'smooth-only':
        provenance['control_change_scope'] = 'Remove added contact dwell/duplicate admission; preserve upstream confirmed-capture transition, half lift speed, carry smoothing and all original hard/scoring rules'
    if args.control_profile == 'mimic-safe':
        provenance['mimic_actuation_adapter_sha256'] = digest(ROOT / 'src/capstone_vla/hgwind_mimic_actuation.py')
        provenance['control_change_scope'] = 'Same motion/reference configuration as smooth-only; configure and command only independent jaw actuators, preserve both physical jaw measurements and authored mimic coupling'
    provenance['capture_speed_gate_enabled'] = args.capture_speed_gate
    if args.capture_speed_gate:
        provenance['capture_speed_gate_adapter_sha256'] = digest(ROOT / 'src/capstone_vla/hgwind_capture_gate.py')
        provenance['control_change_scope'] += '; physical LIFT additionally requires existing relative-speed bound, no fixed dwell'
    (out / 'run.json').write_text(json.dumps(provenance, indent=2) + '\n')
    if args.control_profile != 'baseline':
        snapshot = out / 'control_snapshot'
        for rel in ['src/capstone_vla/hgwind_control.py', 'scripts/sim/collect_hgwind_worker.py',
                    'scripts/sim/dispatch_hgwind_collection.py', 'scripts/collect_hgwind_baseline.py']:
            target = snapshot / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, target)
        if args.control_profile == 'mimic-safe':
            rel = Path('src/capstone_vla/hgwind_mimic_actuation.py')
            target = snapshot / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, target)
    if args.capture_speed_gate:
        target = out / 'control_snapshot/src/capstone_vla/hgwind_capture_gate.py'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'src/capstone_vla/hgwind_capture_gate.py', target)
    # Inherit the terminal so the canonical small-region prompt remains human-operated.
    result = subprocess.run(command, cwd=REPO, env=env)
    provenance['exit_code'] = result.returncode
    provenance['task_sha256_after'] = digest(task)
    (out / 'run.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('COLLECTION_EXIT code=' + str(result.returncode) + ' output=' + str(out), flush=True)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
