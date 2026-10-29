"""Opt-in pre-physics scene-attach workaround, scoped to the history probe."""
import json
from pathlib import Path
import time


def wait_for_stage(context, app, manager, barrier, timeline, output, *, max_updates=1200, timeout_s=60):
    """Finish attach with scripts deferred; never start/step the timeline."""
    if barrier is None or barrier._released or barrier._manager is not manager:
        raise RuntimeError('Stage load barrier requires the active native script barrier')
    if not timeline.is_stopped() or manager._prim_to_scripts:
        raise RuntimeError('Stage load barrier must precede timeline play and script construction')
    if getattr(manager._load_all_scripts, '__self__', None) is not barrier:
        raise RuntimeError('Native script loader is no longer deferred')
    original_apply = manager._apply_scripts
    initial_time = float(timeline.get_current_time())
    started = time.monotonic()
    output = Path(output)
    evidence = dict(schema='physical-vla-lab.stage-load-barrier.v1', ready=False,
        updates=0, deferred_apply_calls=0, initial_timeline_time=initial_time,
        final_timeline_time=initial_time, remaining=None, workaround_efficacy='unverified',
        timeline_advanced=False, held_script_instances=0)

    def save():
        temporary = output.with_name(output.name+'.tmp')
        temporary.write_text(json.dumps(evidence, indent=2)+'\n')
        temporary.replace(output)

    def hold_apply(*args, **kwargs):
        evidence['deferred_apply_calls'] += 1

    def check_stopped():
        evidence['final_timeline_time'] = float(timeline.get_current_time())
        evidence['timeline_advanced'] = evidence['final_timeline_time'] != initial_time
        evidence['held_script_instances'] = sum(len(scripts) for scripts in manager._prim_to_scripts.values())
        if not timeline.is_stopped() or evidence['final_timeline_time'] != initial_time or manager._prim_to_scripts:
            raise RuntimeError('Timeline or script state changed while waiting for stage attach')

    manager._apply_scripts = hold_apply
    try:
        print('STAGE_LOAD_BARRIER begin timeline=stopped scripts=deferred', flush=True)
        save()
        stable = 0
        for _ in range(max_updates):
            check_stopped()
            if time.monotonic()-started >= timeout_s:
                raise TimeoutError('Stage attach exceeded the diagnostic wall-time budget')
            # Record entry before update so a native abort leaves a useful boundary.
            evidence['phase'] = 'kit_update'
            save()
            app.update()
            evidence['updates'] += 1
            check_stopped()
            status = context.get_stage_loading_status()
            if len(status) != 3:
                raise RuntimeError('Unrecognized stage-loading status')
            remaining = int(status[2])
            if remaining < 0:
                raise RuntimeError('Invalid pending stage-asset count')
            evidence['remaining'] = remaining
            stable = stable+1 if remaining == 0 else 0
            if evidence['updates'] >= 2 and stable >= 2:
                evidence['ready'] = True
                evidence['phase'] = 'stage_loaded_before_robot_edits'
                break
        else:
            raise TimeoutError('Stage attach exceeded the diagnostic update budget')
    except Exception as exc:
        evidence['error'] = type(exc).__name__+': '+str(exc)
        raise
    finally:
        manager._apply_scripts = original_apply
        evidence['elapsed_s'] = time.monotonic()-started
        save()
    print('STAGE_LOAD_BARRIER ready updates=%d timeline_advanced=false script_instances=0' % evidence['updates'], flush=True)
    return evidence


def patch_scene_load_barrier(source):
    anchor = '    _st0 = omni.usd.get_context().get_stage()\n    if _st0 is None:'
    if source.count(anchor) != 1:
        raise ValueError('Stage-attach boundary changed')
    replacement = '''    CAPSTONE_WAIT_STAGE_LOAD(
        omni.usd.get_context(), simulation_app, _script_manager_instance(),
        script_load_barrier, CAPSTONE_TIMELINE.get_timeline_interface(),
        os.path.join(os.path.dirname(os.environ['HGWIND_EFFORT_PROBE_MANIFEST']), 'stage_load_barrier.json'),
    )
'''+anchor
    source = source.replace(anchor, replacement)
    anchor = 'def _inspect_loaded_scripts(_st0, robot_script_removed, allow_no_scripts):'
    if source.count(anchor) != 1:
        raise ValueError('Stage barrier import boundary changed')
    source = source.replace(anchor, 'from capstone_vla.scene_load_barrier import wait_for_stage as CAPSTONE_WAIT_STAGE_LOAD\nfrom omni import timeline as CAPSTONE_TIMELINE\n\n'+anchor)
    compile(source, '<stage-load-barrier>', 'exec')
    return source
