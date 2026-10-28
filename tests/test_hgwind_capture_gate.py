from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from capstone_vla.hgwind_capture_gate import install_capture_speed_gate, summary


def probe(speed, limit=.0875, attached=False):
    class Capture:
        def __init__(self):
            self.calls = []
            self._capture_relative_speed_mps = speed
        def _uses_runtime_attachment(self):
            return attached
        def _capture_speed_limit_mps(self):
            return limit
        def _begin_lift_from_physical_grasp(self, **kwargs):
            self.calls.append(kwargs)
            return 'original_transition'
    install_capture_speed_gate(Capture)
    return Capture()


def call(p):
    return p._begin_lift_from_physical_grasp(p_A=[1, 2, 3], p_ee=[4, 5, 6],
                                           q_ee=[1, 0, 0, 0], v_ee=[.3, 0, 0])


@pytest.mark.parametrize('speed', [.0892780553, .192933618, float('nan'), float('inf'), -.01])
def test_unsafe_or_invalid_speed_never_enters_lift(speed):
    p = probe(speed)
    assert call(p) is None and p.calls == []
    assert p._capstone_capture_speed_gate_stats['deferred_unsafe_calls'] == 1


@pytest.mark.parametrize('speed', [0., .0349575248, .0479192205, .0236263797, .0875])
def test_safe_speed_immediately_preserves_original_transition(speed):
    p = probe(speed)
    assert call(p) == 'original_transition'
    assert len(p.calls) == 1
    assert p.calls[0]['v_ee'] == [.3, 0, 0]  # common world motion is permitted
    assert p._capstone_capture_speed_gate_stats['deferred_unsafe_calls'] == 0


def test_recovers_next_call_without_a_hidden_dwell():
    p = probe(.2)
    assert call(p) is None
    p._capture_relative_speed_mps = .04
    assert call(p) == 'original_transition'
    receipt = summary(SimpleNamespace(active_primitives=lambda: [p]))
    assert receipt['required_dwell_s'] == 0 and receipt['primitives'][0]['admitted_calls'] == 1


def test_runtime_attachment_keeps_its_original_arrest_boundary():
    assert call(probe(.5, attached=True)) == 'original_transition'


def test_actual_old_helper_ignores_computed_speed_flag():
    import ast
    path = ROOT / 'data/derived/hgwind_current_baseline_20261027/repo/Scripts/expert_data/skills/catch.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_capture_contact_transition_ready')
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    assert namespace[node.name](jaws_closed=True, bilateral_contact=True,
        capture_speed_safe=False, finger_drives_blocked=True, runtime_attachment=False)


@pytest.mark.parametrize('enabled', [False, True])
def test_real_worker_hook_preserves_default_and_compiles_enabled_metadata(monkeypatch, enabled):
    import ast
    import os
    path = ROOT / 'scripts/sim/collect_hgwind_worker.py'
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.If) and 'CAPSTONE_HGWIND_CAPTURE_SPEED_GATE' in ast.unparse(n.test))
    class Original:
        def _begin_lift_from_physical_grasp(self, **kwargs):
            return 'original'
    method = Original._begin_lift_from_physical_grasp
    monkeypatch.setenv('CAPSTONE_HGWIND_CAPTURE_SPEED_GATE', '1' if enabled else '0')
    monkeypatch.setenv('CAPSTONE_HGWIND_CONTROL_PROFILE', 'mimic-safe')
    source_path = ROOT / 'data/derived/hgwind_current_baseline_20261027/repo/Scripts/expert_data/collect_worker.py'
    source = source_path.read_text()
    namespace = dict(os=os, catch=SimpleNamespace(Catch=Original), source=source,
                     anchor='"skill_terminations": _jsonify(skill_terminations),', extra_globals={})
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    compile(namespace['source'], str(source_path), 'exec')
    if enabled:
        assert Original._begin_lift_from_physical_grasp is not method
        assert '"capstone_capture_speed_gate": CAPSTONE_CAPTURE_GATE_SUMMARY(pol)' in namespace['source']
        assert 'CAPSTONE_CAPTURE_GATE_SUMMARY' in namespace['extra_globals']
    else:
        assert Original._begin_lift_from_physical_grasp is method and namespace['source'] == source


def real_capture_phase_transition(**kwargs):
    import ast
    path = ROOT / 'data/derived/hgwind_current_baseline_20261027/repo/Scripts/expert_data/skills/catch.py'
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_capture_phase_transition')
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[node.name](**kwargs)


def phase_for(p):
    return real_capture_phase_transition(physical_grasp_confirmed=p._physical_grasp_confirmed,
        runtime_attachment=False, logical_attachment_active=False,
        handoff_after_capture=False, require_lift=True)


def test_deferral_revokes_old_capture_so_slow_empty_hand_cannot_enter_lift():
    p = probe(.193)
    p._physical_grasp_confirmed = True  # original current-contact admission
    assert phase_for(p) == 'lift'
    assert call(p) is None
    assert p._physical_grasp_confirmed is False
    p._capture_relative_speed_mps = .04  # speed fell after target left jaws
    assert phase_for(p) == 'none'  # real upstream caller cannot reuse the latch
    assert p.calls == []
    # A genuinely new current-contact confirmation can admit immediately.
    p._physical_grasp_confirmed = True
    assert phase_for(p) == 'lift' and call(p) == 'original_transition'
    assert len(p.calls) == 1


def test_safe_or_attached_capture_keeps_its_confirmation():
    for p in (probe(.04), probe(.4, attached=True)):
        p._physical_grasp_confirmed = True
        assert call(p) == 'original_transition'
        assert p._physical_grasp_confirmed is True
