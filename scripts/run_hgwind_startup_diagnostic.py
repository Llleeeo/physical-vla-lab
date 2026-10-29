#!/usr/bin/env python3
"""One passive original-expert startup trace, then render its saved outcome."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from capstone_vla.contract_batch import saved_episode, inspect_review_videos


def diagnostic_status(result):
    """Classify evidence availability separately from physical grasp outcome."""
    if not result.get('episode_saved'):
        return 'no_evaluable_episode'
    trace = result.get('force_trace') or {}
    if not trace.get('calls'):
        return 'force_observation_missing'
    if trace.get('unsupported_signatures'):
        return 'force_signature_unsupported'
    if not result.get('render_complete'):
        return 'review_incomplete'
    return 'measurement_recorded'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', choices=['traj_00001', 'traj_00007', 'traj_00018'], default='traj_00001')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--native-backtrace', action='store_true', help='Capture native C++ crash stacks with the installed debugger')
    args = parser.parse_args()
    command = [sys.executable, str(ROOT / 'scripts/run_hgwind_effort_probe.py'),
        '--tracking', 'native-expert', '--trace-cartesian-targets', '--trace-startup-forces', '--episode', args.episode]
    if args.native_backtrace:command.append('--native-backtrace')
    if not args.run:
        return subprocess.run(command, cwd=ROOT).returncode
    stamp = datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
    out = ROOT / 'data/derived/hgwind_startup_diagnostics' / stamp
    out.mkdir(parents=True)
    print('DIAGNOSTIC_OUTPUT ' + str(out), flush=True)
    run = None
    with (out / 'simulation_launcher.log').open('w') as log:
        process = subprocess.Popen(command + ['--run'], cwd=ROOT, text=True, errors='replace',
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        try:
            for line in process.stdout:
                print(line, end='', flush=True); log.write(line); log.flush()
                if line.startswith('OUTPUT '):
                    run = Path(line.strip()[7:]).resolve()
                    run.relative_to(ROOT / 'data/derived')
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait(timeout=30)
            raise
    result = dict(episode=args.episode, physics_launcher_exit=code, run=str(run) if run else None,
        episode_saved=False, render_complete=False, force_trace=None, learned_policy_evaluated=False,
        native_debugger_requested=args.native_backtrace)
    if run:
        backtrace = run / 'native_backtrace.log'
        result['native_backtrace'] = str(backtrace) if backtrace.is_file() else None
        trace = run / 'startup_force_calls.jsonl'
        if trace.exists():
            rows = [json.loads(line) for line in trace.read_text().splitlines()]
            result['force_trace'] = dict(path=str(trace), calls=len(rows),
                zero_physics_dt_calls=sum(x.get('physics_dt_s') == 0 for x in rows),
                pre_recording_calls=sum(x['frame'] is None for x in rows),
                unsupported_signatures=sum(bool(x.get('unsupported_signature')) for x in rows),
                event_limit_reached=len(rows) >= 2048)
        try:
            saved_episode(ROOT, run)
        except (ValueError, OSError) as exc:
            result['render_error'] = str(exc)
        else:
            result['episode_saved'] = True
            rendered = subprocess.run([sys.executable, str(ROOT / 'scripts/render_hgwind_effort_probe.py'),
                '--probe-run', str(run), '--run'], cwd=ROOT)
            if rendered.returncode == 0:
                try:
                    result['videos'] = inspect_review_videos(ROOT, run)
                    result['render_complete'] = True
                except (ValueError, OSError, RuntimeError) as exc:
                    result['render_error'] = str(exc)
            else:
                result['render_error'] = 'Renderer exit ' + str(rendered.returncode)
    result['status'] = diagnostic_status(result)
    result['force_observation_recorded'] = bool((result['force_trace'] or {}).get('calls'))
    result['measurement_recorded'] = result['status'] == 'measurement_recorded'
    (out / 'diagnostic.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    label = 'STARTUP_DIAGNOSTIC_COMPLETED' if result['measurement_recorded'] else 'STARTUP_DIAGNOSTIC_FAILED'
    print(label + ' status=' + result['status'] + ' report=' + str(out / 'diagnostic.json'), flush=True)
    if result['force_trace']:
        print('FORCE_TRACE ' + json.dumps(result['force_trace']), flush=True)
    return 0 if result['measurement_recorded'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
