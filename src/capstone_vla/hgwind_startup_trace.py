"""Passive fan force-call recording; all native calls are forwarded unchanged."""
import json
from pathlib import Path

_ACTIVE = None


class ForceProxy:
    def __init__(self, native, recorder, script):
        self.native, self.recorder, self.script = native, recorder, script

    def __bool__(self):
        return bool(self.native)

    def __getattr__(self, key):
        return getattr(self.native, key)

    def apply_force_at_pos(self, *args, **kwargs):
        # Preserve the native arguments, return value and exception. No gating,
        # force clearing, injected call or extra script update is performed.
        try:
            result = self.native.apply_force_at_pos(*args, **kwargs)
        except Exception:
            self.recorder.record(self.script, args, kwargs, 'raised')
            raise
        self.recorder.record(self.script, args, kwargs, 'returned')
        return result


class StartupForceRecorder:
    def __init__(self, output, frames=6, max_events=2048):
        self.path = Path(output)
        self.frame = None
        self.sim = None
        self.events = 0
        self.frames = frames
        self.max_events = max_events
        self.namespaces = []
        if self.path.exists():
            raise ValueError('Startup force trace already exists')
        self.path.write_text('')

    def record(self, script, args, kwargs, status):
        if self.events >= self.max_events or (self.frame is not None and self.frame >= self.frames):
            return
        row = dict(event=self.events, frame=self.frame, script=script, status=status,
            physics_dt_s=None if self.sim is None else float(self.sim.get_physics_dt()),
            simulation_time_s=None if self.sim is None else float(self.sim.current_time))
        # Current fan signature is four positional arguments; never reinterpret
        # another signature silently. Calls have already been forwarded.
        if len(args) == 4 and not kwargs:
            row.update(body=str(args[0]), force=[float(args[1][i]) for i in range(3)],
                position=[float(args[2][i]) for i in range(3)], mode=str(args[3]))
        else:
            row.update(unsupported_signature=True, positional_count=len(args), keywords=sorted(kwargs))
        with self.path.open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        self.events += 1

    def install(self, rows):
        installed = 0
        for prim, script, instance in rows:
            if str(script).replace('\\', '/').rsplit('/', 1)[-1] != 'fan_script.py':
                continue
            namespace = type(instance).on_update.__globals__
            getter = namespace.get('get_physx_interface')
            if getattr(getter, '_startup_recorder', None) is self:
                continue
            if not callable(getter):
                raise ValueError('Fan PhysX interface getter unavailable')
            def wrapped(*args, _native=getter, _script=str(script), **kwargs):
                native = _native(*args, **kwargs)
                return native if native is None else ForceProxy(native, self, _script)
            wrapped._startup_recorder = self
            namespace['get_physx_interface'] = wrapped
            self.namespaces.append((namespace, getter, wrapped))
            installed += 1
        return installed

    def restore(self):
        for namespace, original, wrapped in reversed(self.namespaces):
            if namespace.get('get_physx_interface') is wrapped:
                namespace['get_physx_interface'] = original


def configure(output):
    global _ACTIVE
    if _ACTIVE is not None:
        raise ValueError('Startup recorder already configured')
    _ACTIVE = StartupForceRecorder(output)


def startup_install(manager):
    from expert_data.behavior_script_runtime import iter_script_instances
    _ACTIVE.install(iter_script_instances(manager))


def startup_context(sim, manager):
    _ACTIVE.sim = sim
    startup_install(manager)


def startup_frame(index):
    _ACTIVE.frame = int(index)


def patch_startup_collector(source):
    anchors = [
        ('def _inspect_loaded_scripts(_st0, robot_script_removed, allow_no_scripts):',
         'from capstone_vla.hgwind_startup_trace import startup_install, startup_context, startup_frame\n\n'
         'def _inspect_loaded_scripts(_st0, robot_script_removed, allow_no_scripts):'),
        ('    validate_physics_control_bindings(_st0, loaded)',
         '    startup_install(mgr)\n    validate_physics_control_bindings(_st0, loaded)'),
        ('    start_playback_without_physics(\n',
         '    startup_context(sim, _script_manager_instance())\n    start_playback_without_physics(\n'),
        ('        prepolicy_mass_guard.prepare_reset()',
         '        startup_context(sim, _script_manager_instance())\n        prepolicy_mass_guard.prepare_reset()'),
        ('                    CAPSTONE_RECORD_NATIVE_SEGMENT(k, segment, tgt, jp, jv, stage, robot_path, objects, attach_state)',
         '                    startup_frame(k)\n                    CAPSTONE_RECORD_NATIVE_SEGMENT(k, segment, tgt, jp, jv, stage, robot_path, objects, attach_state)'),
    ]
    for old, new in anchors:
        if source.count(old) != 1:
            raise ValueError('Startup recording insertion point changed: ' + old[:70])
        source = source.replace(old, new)
    compile(source, '<passive-startup-trace>', 'exec')
    return source
