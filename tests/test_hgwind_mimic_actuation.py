"""Exercise the real collector/Expert methods without importing Isaac Sim."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from pxr import Usd, UsdPhysics, Sdf

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'data/derived/hgwind_current_baseline_20261027/repo/Scripts'
sys.path[:0] = [str(ROOT / 'src'), str(SCRIPTS)]
from capstone_vla import hgwind_mimic_actuation as adapter
from expert_data.robot_control import actuator_properties

EXPERT = SCRIPTS / 'expert_data/expert.py'
TREE = ast.parse(EXPERT.read_text())
METHODS = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'Expert')


def actual_method(name):
    node = next(n for n in METHODS.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {'np': np, 'Usd': Usd, 'UsdPhysics': UsdPhysics}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(EXPERT), 'exec'), namespace)
    return namespace[name]


class Controller:
    def __init__(self):
        self.kp = np.r_[np.zeros(7), 10000., 0.]
        self.kd = np.r_[np.zeros(7), 1000., 0.]
    def get_gains(self):
        return self.kp.copy(), self.kd.copy()
    def set_gains(self, kp, kd):
        self.kp, self.kd = kp.copy(), kd.copy()


def mechanism(mimic=True):
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim('/franka', 'Xform')
    one = UsdPhysics.PrismaticJoint.Define(stage, '/franka/panda_finger_joint1').GetPrim()
    two = UsdPhysics.PrismaticJoint.Define(stage, '/franka/panda_finger_joint2').GetPrim()
    UsdPhysics.DriveAPI.Apply(one, 'linear')
    if mimic:
        two.SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit(['PhysxMimicJointAPI:rotX']))
        two.CreateRelationship('physxMimicJoint:rotX:referenceJoint').SetTargets([one.GetPath()])
        two.CreateAttribute('physxMimicJoint:rotX:gearing', Sdf.ValueTypeNames.Float).Set(-1.)
    else:
        UsdPhysics.DriveAPI.Apply(two, 'linear')
    class ExpertProbe:
        ensure_gripper_force = actual_method('ensure_gripper_force')
        _gripper_dof_indices = actual_method('_gripper_dof_indices')
        gripper_opening_width = actual_method('gripper_opening_width')
        gripper_opening_velocity = actual_method('gripper_opening_velocity')
    expert = ExpertProbe()
    expert._stage, expert._prim = stage, root
    expert._hand_runtime, expert._gripper_dof = None, None
    expert._finger_force_forced = False
    expert._gcfg = dict(kind='parallel', joints=['panda_finger_joint1', 'panda_finger_joint2'],
                        open=.04, closed=0., max_width=.08, maximum_force=70., maximum_speed=.05)
    controller = Controller()
    if not mimic:
        controller.kp[8], controller.kd[8] = 10000., 1000.
    expert._robot = SimpleNamespace(get_articulation_controller=lambda: controller,
        get_dof_index=lambda name: {'panda_finger_joint1': 7, 'panda_finger_joint2': 8}[name],
        get_joint_positions=lambda: np.r_[np.zeros(7), .02, .03],
        get_joint_velocities=lambda: np.r_[np.zeros(7), -.01, -.03])
    return expert, one, two


def caps(robot, controller, indices, value, **kwargs):
    return {'indices': list(indices), 'value': value, **kwargs}


def test_original_method_reproduces_second_drive_defect():
    expert, _, passive = mechanism()
    assert not UsdPhysics.DriveAPI.Get(passive, 'linear')
    with patch.object(actuator_properties, 'set_live_joint_cap', side_effect=caps):
        assert expert.ensure_gripper_force(force=70., speed=.05)
    assert UsdPhysics.DriveAPI.Get(passive, 'linear')
    np.testing.assert_array_equal(expert._robot.get_articulation_controller().kp[7:], [10500., 10500.])


@pytest.mark.parametrize('explicit', [None, .005])
def test_adapted_real_method_preserves_mimic_and_caps(explicit):
    expert, active, passive = mechanism()
    original_relationship = passive.GetRelationship('physxMimicJoint:rotX:referenceJoint').GetTargets()
    adapter.install_expert_patch(type(expert))
    with patch.object(actuator_properties, 'set_live_joint_cap', side_effect=caps) as call:
        assert expert.ensure_gripper_force(force=10000., speed=1., force_deflection_m=explicit)
    assert len(call.call_args_list) == 2
    assert all(list(c.args[2]) == [7] for c in call.call_args_list)
    assert [c.args[3] for c in call.call_args_list] == [70., .05]
    kp, kd = expert._robot.get_articulation_controller().get_gains()
    assert kp[7] > 0 and kd[7] > 0 and kp[8] == kd[8] == 0
    assert not UsdPhysics.DriveAPI.Get(passive, 'linear')
    assert UsdPhysics.DriveAPI.Get(active, 'linear').GetMaxForceAttr().Get() == 70.
    assert passive.GetRelationship('physxMimicJoint:rotX:referenceJoint').GetTargets() == original_relationship
    assert passive.GetAttribute('physxMimicJoint:rotX:gearing').Get() == -1.
    assert expert._gripper_dof_indices() == [7, 8]
    assert expert.gripper_opening_width() == pytest.approx(.05)
    assert expert.gripper_opening_velocity() == pytest.approx(-.04)
    receipt = adapter.summary(SimpleNamespace(expert=expert))
    assert receipt['actuated_indices'] == [7] and receipt['passive_indices'] == [8]
    assert receipt['live_readback']['verified'] and receipt['force_configuration_calls'] == 1


def test_independent_jaws_still_both_actuated():
    expert, _, _ = mechanism(mimic=False)
    adapter.install_expert_patch(type(expert))
    with patch.object(actuator_properties, 'set_live_joint_cap', side_effect=caps):
        assert expert.ensure_gripper_force(force=70., speed=.05)
    assert adapter.actuated_finger_indices(expert) == [7, 8]
    assert adapter.actuation_plan(expert)['passive_indices'] == []


def test_real_collector_target_indices_filter_only_passive_jaw():
    source = adapter.patch_worker_source((SCRIPTS / 'expert_data/collect_worker.py').read_text())
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == '_collection_drive_indices')
    namespace = {'np': np, '_mounted_hand_descriptor': lambda *a: None,
                 '_physical_finger_dof_indices': lambda *a: np.array([7, 8]),
                 '_kinematic_cspace_indices': lambda *a: np.arange(7),
                 'CAPSTONE_ACTUATED_FINGER_INDICES': adapter.actuated_finger_indices}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<real-collector-drive-indices>', 'exec'), namespace)
    expert, _, _ = mechanism()
    arm, jaws = namespace['_collection_drive_indices'](SimpleNamespace(expert=expert), expert._robot, 9)
    np.testing.assert_array_equal(arm, np.arange(7))
    np.testing.assert_array_equal(jaws, [7])
    assert expert._gripper_dof_indices() == [7, 8]
    assert 'CAPSTONE_INSTALL_MIMIC_ACTUATION()' in source
    assert '"capstone_gripper_actuation": CAPSTONE_MIMIC_SUMMARY(pol)' in source


def test_unsafe_passive_gain_is_rejected():
    expert, _, _ = mechanism()
    expert._robot.get_articulation_controller().kp[8] = 1.
    with pytest.raises(RuntimeError, match='Passive mimic jaw'):
        adapter.verify_passive_gains(expert)


def test_invalid_self_mimic_is_rejected():
    expert, _, passive = mechanism()
    passive.GetRelationship('physxMimicJoint:rotX:referenceJoint').SetTargets([passive.GetPath()])
    with pytest.raises(ValueError, match='different physical jaw'):
        adapter.actuation_plan(expert)


def test_profiles_have_identical_motion_settings():
    from dataclasses import asdict
    from capstone_vla.hgwind_control import profile_configuration
    old, new = [asdict(profile_configuration(p)) for p in ('smooth-only', 'mimic-safe')]
    old.pop('name'); new.pop('name')
    assert old == new


def test_actual_frozen_scene_has_expected_passive_topology():
    scene = ROOT / 'data/derived/hgwind_current_baseline_20261027/repo/SceneVariants/df11a22a0656b6e2/Tasks/1s1p/grasp_wind_driven_motion_1s1p_hgprgew.usd'
    stage = Usd.Stage.Open(str(scene))
    expert = SimpleNamespace(_prim=stage.GetPrimAtPath('/Root/Dynamics/franka'),
        _gcfg={'joints': ['panda_finger_joint1', 'panda_finger_joint2']},
        _gripper_dof_indices=lambda: [7, 8])
    plan = adapter.actuation_plan(expert)
    assert plan['actuated_indices'] == [7] and plan['passive_indices'] == [8]
    passive = stage.GetPrimAtPath('/Root/Dynamics/franka/panda_hand/panda_finger_joint2')
    assert not UsdPhysics.DriveAPI.Get(passive, 'linear')


def test_prestart_adapter_import_and_source_patch_do_not_require_pxr():
    """CPU env already has USD; explicitly deny it in an isolated interpreter."""
    import subprocess
    import textwrap
    program = textwrap.dedent('''
        import builtins
        import pathlib
        import sys
        sys.path.insert(0, str(pathlib.Path.cwd() / 'src'))
        original_import = builtins.__import__
        def without_usd(name, *args, **kwargs):
            if name == 'pxr' or name.startswith('pxr.'):
                raise ModuleNotFoundError('USD unavailable before SimulationApp')
            return original_import(name, *args, **kwargs)
        builtins.__import__ = without_usd
        from capstone_vla.hgwind_mimic_actuation import patch_worker_source
        worker = pathlib.Path('data/derived/hgwind_current_baseline_20261027/repo/Scripts/expert_data/collect_worker.py')
        source = patch_worker_source(worker.read_text())
        compile(source, str(worker), 'exec')
        assert 'pxr' not in sys.modules
        print('PRESTART_IMPORT_WITHOUT_USD_OK')
    ''')
    result = subprocess.run([sys.executable, '-c', program], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PRESTART_IMPORT_WITHOUT_USD_OK' in result.stdout
