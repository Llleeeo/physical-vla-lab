"""Bounded same-process history recording around the frozen native expert."""
import copy
import json
from pathlib import Path

_PLAN = None
_SEEN = []


def configure(plan):
    global _PLAN, _SEEN
    if plan.get('history_attempts') != [58, 59, 60]:
        raise ValueError('Only the declared original history58/59/60 is supported')
    _PLAN, _SEEN = plan, []


def begin_attempt(index):
    from . import hgwind_expert_trace as trace
    from . import hgwind_startup_trace as force
    index = int(index)
    expected = _PLAN['history_attempts']
    if len(_SEEN) >= len(expected) or index != expected[len(_SEEN)]:
        raise ValueError('Unexpected attempt order; do not relabel or restart history')
    case = copy.deepcopy(_PLAN['history_cases'][str(index)])
    output = Path(case['output'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'plan.json').write_text(json.dumps(case, indent=2) + '\n')
    # Retire the old episode proxy rather than wrapping it again. This changes
    # observation ownership only; no force clearing or native call is added.
    if force._ACTIVE is not None:
        force._ACTIVE.restore()
    force._ACTIVE = force.StartupForceRecorder(output / 'startup_force_calls.jsonl')
    previous = trace._STATE.get('cartesian_recorder')
    if previous is not None:
        previous.restore()
    trace.configure(case)
    _SEEN.append(index)
    with (Path(_PLAN['output']) / 'attempt_order.jsonl').open('a') as handle:
        handle.write(json.dumps({'attempt_index': index, 'ordinal': len(_SEEN)}) + '\n')
    print('HISTORY_ATTEMPT index=%d ordinal=%d/3' % (index, len(_SEEN)), flush=True)


def patch_history_collector(source, stage_load_barrier=None):
    from .hgwind_startup_trace import patch_startup_collector
    source = patch_startup_collector(source)
    if stage_load_barrier is None:
        stage_load_barrier = bool(_PLAN and _PLAN.get('stage_load_barrier'))
    if stage_load_barrier:
        from .scene_load_barrier import patch_scene_load_barrier
        source = patch_scene_load_barrier(source)
    anchor = '        attempt_index = int(args.attempt_offset) + int(ep)'
    if source.count(anchor) != 1:
        raise ValueError('History attempt boundary changed')
    source = source.replace(anchor, anchor + '\n        CAPSTONE_HISTORY_BEGIN(attempt_index)')
    prefix = 'from capstone_vla.hgwind_history import begin_attempt as CAPSTONE_HISTORY_BEGIN\n'
    anchor = 'def _inspect_loaded_scripts(_st0, robot_script_removed, allow_no_scripts):'
    if source.count(anchor) != 1:
        raise ValueError('History import boundary changed')
    source = source.replace(anchor, prefix + '\n' + anchor)
    compile(source, '<history-recording>', 'exec')
    return source


def assemble(plan):
    from .hgwind_expert_trace import patch_collector
    from .cartesian_target_trace import patch_capture_collector
    source = Path(plan['worker']).read_text()
    anchor = '"skill_terminations": _jsonify(skill_terminations),'
    if source.count(anchor) != 1:
        raise ValueError('Frozen profile insertion changed')
    source = source.replace(anchor, anchor + '\n            "capstone_expert_control": CAPSTONE_PROFILE_SUMMARY(pol),')
    return patch_capture_collector(patch_history_collector(patch_collector(source), bool(plan.get('stage_load_barrier'))))


def index_saved_episodes(worker_output, task_id):
    found = {}
    for replay in Path(worker_output).glob('1s1p/' + task_id + '/*/replay.hdf5'):
        meta = json.loads((replay.parent / 'meta.json').read_text())
        index = int(meta['attempt_index'])
        if index not in (58, 59, 60) or index in found:
            raise ValueError('Unexpected or duplicate saved attempt')
        found[index] = replay.parent
    # Partial runs are evidence, but must be a prefix of the declared order.
    if sorted(found) != [58, 59, 60][:len(found)]:
        raise ValueError('Saved history has a missing earlier attempt')
    return dict(sorted(found.items()))


def compare_motion(source, actual):
    import h5py
    import numpy as np
    with h5py.File(source, 'r') as old, h5py.File(actual, 'r') as new:
        keys = ['actions', 'timestamp', 'is_first', 'is_last', 'is_terminal'] + ['obs/' + k for k in old['obs']]
        exact = {k: bool(np.array_equal(old[k][:], new[k][:], equal_nan=True)) for k in keys}
        prefix = 'evidence/native/'
        ball = '/Root/Dynamics/ball_y9bssvn'
        def velocity(handle):
            paths = handle[prefix + 'inventory/rigid_body/prim_paths'].asstr()[:].tolist()
            if paths.count(ball) != 1:
                raise ValueError('Ambiguous ball identity')
            return handle[prefix + 'rigid_body/get_velocities/values'][:7, paths.index(ball)]
        a, b = velocity(old), velocity(new)
        count = min(len(a), len(b))
        differences = np.flatnonzero(np.any(a[:count] != b[:count], axis=1))
        return dict(core_array_exact=exact, all_core_arrays_exact=all(exact.values()),
            compared_native_rows=count,
            first_native_velocity_difference=int(differences[0]) if len(differences) else None,
            first_step_ball_velocity_delta_m_s=(b[1,:3]-a[1,:3]).tolist() if count > 1 else None)
