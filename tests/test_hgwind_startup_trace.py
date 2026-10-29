"""Verify passive tracing preserves native behavior and command arguments."""
import json
from types import SimpleNamespace
import pytest
from capstone_vla.hgwind_startup_trace import StartupForceRecorder, ForceProxy, patch_startup_collector


class Native:
    def __init__(self):
        self.calls = []
        self.token = object()
        self.value = 7
    def apply_force_at_pos(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.token


def test_force_arguments_and_native_return_are_unchanged_at_zero_dt(tmp_path):
    recorder = StartupForceRecorder(tmp_path / 'trace.jsonl')
    recorder.sim = SimpleNamespace(current_time=0., get_physics_dt=lambda: 0.)
    native = Native(); proxy = ForceProxy(native, recorder, 'fan_script.py')
    force = [1., 2., 3.]; position = [4., 5., 6.]
    assert proxy.apply_force_at_pos('/ball', force, position, 'Force') is native.token
    assert native.calls[0][0][1] is force and native.calls[0][0][2] is position
    row = json.loads(recorder.path.read_text())
    assert row['physics_dt_s'] == 0 and row['frame'] is None and row['force'] == force
    assert bool(proxy) and proxy.value == 7


def test_event_limit_and_frame_limit_do_not_suppress_force_calls(tmp_path):
    recorder = StartupForceRecorder(tmp_path / 'trace.jsonl', frames=1, max_events=1)
    native = Native(); proxy = ForceProxy(native, recorder, 'fan_script.py')
    for frame in [0, 0, 1, 2]:
        recorder.frame = frame
        assert proxy.apply_force_at_pos('/ball', [1, 0, 0], [0, 0, 0], 'Force') is native.token
    assert len(native.calls) == 4 and len(recorder.path.read_text().splitlines()) == 1


def test_original_native_exception_is_preserved(tmp_path):
    recorder = StartupForceRecorder(tmp_path / 'trace.jsonl')
    error = RuntimeError('native force error')
    class Failing:
        def apply_force_at_pos(self, *args):
            raise error
    with pytest.raises(RuntimeError) as caught:
        ForceProxy(Failing(), recorder, 'fan_script.py').apply_force_at_pos('/ball', [1, 0, 0], [0, 0, 0], 'Force')
    assert caught.value is error
    assert json.loads(recorder.path.read_text())['status'] == 'raised'


def test_only_fan_getter_is_wrapped_once_and_restorable(tmp_path):
    recorder = StartupForceRecorder(tmp_path / 'trace.jsonl')
    native = Native(); namespace = {'get_physx_interface': lambda: native}
    exec('def on_update(self): return get_physx_interface()', namespace)
    fan = type('Fan', (), {'on_update': namespace['on_update']})()
    original = namespace['get_physx_interface']
    assert recorder.install([('/fan', '/a/fan_script.py', fan)]) == 1
    assert recorder.install([('/fan', '/a/fan_script.py', fan)]) == 0
    assert recorder.install([('/other', '/a/other_script.py', fan)]) == 0
    assert isinstance(fan.on_update(), ForceProxy)
    recorder.restore(); assert namespace['get_physx_interface'] is original


def test_unknown_native_signature_still_forwards_and_is_explicit(tmp_path):
    recorder = StartupForceRecorder(tmp_path / 'trace.jsonl'); native = Native()
    assert ForceProxy(native, recorder, 'fan').apply_force_at_pos(body='ball') is native.token
    assert native.calls == [((), {'body': 'ball'})]
    assert json.loads(recorder.path.read_text())['unsupported_signature']


def test_patch_rejects_wrong_collector_instead_of_silent_partial_logging():
    with pytest.raises(ValueError, match='insertion point'):
        patch_startup_collector('print("different collector")')


def history_module():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'scripts/diagnose_hgwind_startup_history.py'
    spec = importlib.util.spec_from_file_location('startup_history_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_source_process_ordinal_resets_only_at_worker_marker():
    lines = ['CAPSTONE_HGWIND_CONTROL_ACTIVE profile=smooth',
        'LIVE SAMPLE-REGION attempt=58 x', 'LIVE SAMPLE-REGION attempt=59 x',
        'LIVE SAMPLE-REGION attempt=60 x', 'CAPSTONE_HGWIND_CONTROL_ACTIVE profile=smooth',
        'LIVE SAMPLE-REGION attempt=143 x', 'LIVE SAMPLE-REGION attempt=144 x']
    module = history_module()
    assert module.worker_context(lines, 60)['ordinal_in_worker'] == 3
    assert module.worker_context(lines, 144)['ordinal_in_worker'] == 2
    with pytest.raises(ValueError):
        module.worker_context(lines + ['LIVE SAMPLE-REGION attempt=60 x'], 60)


def test_native_body_identity_and_contact_impulse_sign_are_not_positional(tmp_path):
    import h5py
    import numpy as np
    module = history_module()
    with h5py.File(tmp_path / 'replay.hdf5', 'w') as handle:
        root = 'evidence/native/'
        handle[root + 'inventory/rigid_body/prim_paths'] = np.asarray(['/other', module.BALL], dtype=h5py.string_dtype())
        handle[root + 'timestamp'] = np.arange(7) / 180
        handle[root + 'step_dt'] = np.r_[0, np.full(6, 1/180)]
        for key, width in [('get_velocities', 6), ('get_accelerations', 6), ('get_masses', 1), ('get_inertias', 9), ('get_coms', 7), ('get_transforms', 7)]:
            handle[root + 'rigid_body/' + key + '/values'] = np.zeros((7, 2, width))
        contacts = handle.create_group('evidence/contacts')
        contacts['paths'] = np.asarray(['/other', module.BALL], dtype=h5py.string_dtype())
        contacts['headers'] = [[0, 0, 1, 0, 1, 2, 0, 1], [1, 1, 0, 1, 0, 2, 1, 1]]
        points = np.zeros((2, 10)); points[0, 6:9] = [1, 2, 3]; points[1, 6:9] = [4, 5, 6]
        contacts['points'] = points
        handle['actions'] = np.zeros((7, 9)); handle['obs/object_vel'] = np.zeros((7, 1, 3)); handle['obs/ee_pos'] = np.zeros((7, 3))
    result = module.snapshot(tmp_path)
    assert result['first_three_recorded_contact_impulses_Ns'] == [[-1., -2., -3.], [4., 5., 6.], [0., 0., 0.]]
