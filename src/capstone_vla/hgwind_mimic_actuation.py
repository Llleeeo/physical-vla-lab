"""Respect authored passive mimic jaws without changing grasp physics or limits.

Applied only by the opt-in HGwind worker. Measurement retains both physical
fingers; gains, caps and executable targets address only independent actuators.
"""
import ast
import inspect
import textwrap

import numpy as np


def actuation_plan(expert):
    # Isaac exposes USD only after SimulationApp starts. This module is also
    # imported by the pre-start worker source adapter, so keep USD lazy.
    from pxr import Usd, UsdPhysics

    physical = tuple(int(i) for i in expert._gripper_dof_indices())
    config = expert._gcfg or {}
    names = tuple(config.get('joints', ()))
    if len(physical) != len(names):
        raise ValueError('Gripper names/physical DOFs do not match')
    old = getattr(expert, '_capstone_mimic_actuation_plan', None)
    if old is not None:
        if tuple(old['physical_indices']) != physical or tuple(old['physical_names']) != names:
            raise RuntimeError('Gripper topology changed after actuation discovery')
        return old
    root = getattr(expert, '_prim', None)
    if not root or not root.IsValid():
        raise ValueError('Authored robot prim is required for mimic actuation')
    joints = {p.GetName(): p for p in Usd.PrimRange(root) if p.IsA(UsdPhysics.Joint)}
    active, passive, references = [], [], {}
    for name, index in zip(names, physical):
        joint = joints.get(name)
        if joint is None:
            raise ValueError('Configured gripper joint missing: ' + name)
        links = [r for r in joint.GetRelationships()
                 if r.GetName().startswith('physxMimicJoint:')
                 and r.GetName().endswith(':referenceJoint') and r.GetTargets()]
        if links:
            if len(links) != 1 or len(links[0].GetTargets()) != 1:
                raise ValueError('Ambiguous authored mimic dependency: ' + name)
            target = links[0].GetTargets()[0]
            if target.name not in names or target != joints[target.name].GetPath() or target == joint.GetPath():
                raise ValueError('Mimic reference must name a different physical jaw in this mechanism')
            passive.append(index)
            references[name] = str(target)
        else:
            active.append(index)
    if not active:
        raise ValueError('Gripper has no independent actuator')
    if any(target.rsplit('/', 1)[-1] in references for target in references.values()):
        raise ValueError('Nested/cyclic mimic mechanisms require a dedicated adapter')
    plan = {'physical_indices': list(physical), 'physical_names': list(names),
            'actuated_indices': active, 'passive_indices': passive,
            'actuated_names': [name for name, index in zip(names, physical) if index in active],
            'passive_names': [name for name, index in zip(names, physical) if index in passive],
            'mimic_references': references, 'mechanism_changed': False,
            'physical_measurement_indices_changed': False}
    expert._capstone_mimic_actuation_plan = plan
    return plan


def actuated_finger_indices(expert):
    return actuation_plan(expert)['actuated_indices']


def select_drive_config(expert, config, indices):
    plan = actuation_plan(expert)
    if list(indices) != plan['physical_indices']:
        raise ValueError('Force configuration received unexpected physical indices')
    result = dict(config)
    result['joints'] = list(plan['actuated_names'])
    return result, list(plan['actuated_indices'])


def adapt_force_method(original):
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = tree.body[0]
    matches = [i for i, node in enumerate(function.body)
               if ast.unparse(node) == 'idxs = self._gripper_dof_indices()']
    if len(matches) != 1:
        raise RuntimeError('Gripper force configuration insertion point changed')
    function.body[matches[0] + 1:matches[0] + 1] = ast.parse(
        'gcfg, idxs = CAPSTONE_SELECT_DRIVE_CONFIG(self, gcfg, idxs)').body
    namespace = dict(original.__globals__, CAPSTONE_SELECT_DRIVE_CONFIG=select_drive_config)
    exec(compile(ast.fix_missing_locations(tree), '<capstone_independent_jaw_drives>', 'exec'), namespace)
    return namespace[function.name]


def verify_passive_gains(expert):
    plan = actuation_plan(expert)
    kp, kd = expert._robot.get_articulation_controller().get_gains()
    kp, kd = np.asarray(kp).reshape(-1), np.asarray(kd).reshape(-1)
    passive = plan['passive_indices']
    values = np.r_[kp[passive], kd[passive]]
    if not np.all(np.isfinite(values)) or np.any(abs(values) > 1e-8):
        raise RuntimeError('Passive mimic jaw has a nonzero or unverifiable independent drive gain')
    expert._capstone_mimic_drive_readback = {
        'passive_stiffness': kp[passive].tolist(), 'passive_damping': kd[passive].tolist(),
        'verified': True, 'hard_force_or_speed_limits_raised': False}


def install_expert_patch(expert_class):
    if getattr(expert_class, '_capstone_mimic_actuation_installed', False):
        raise RuntimeError('Mimic actuation patch already installed')
    adapted = adapt_force_method(expert_class.ensure_gripper_force)

    def force(self, *args, **kwargs):
        result = adapted(self, *args, **kwargs)
        if result:
            verify_passive_gains(self)
            self._capstone_mimic_configuration_calls = getattr(self, '_capstone_mimic_configuration_calls', 0) + 1
        return result

    expert_class.ensure_gripper_force = force
    expert_class._capstone_mimic_actuation_installed = True


def install_after_policy_import():
    from expert_data.expert import Expert
    install_expert_patch(Expert)


def patch_worker_source(source):
    anchor = 'from expert_data.policy import UniversalPolicy  # noqa: E402'
    drive = 'jaws = np.asarray(expert._gripper_dof_indices(), dtype=int)'
    metadata = '"skill_terminations": _jsonify(skill_terminations),'
    for text in (anchor, drive, metadata):
        if source.count(text) != 1:
            raise RuntimeError('Mimic actuation source insertion point changed: ' + text)
    source = source.replace(anchor, anchor + '\nCAPSTONE_INSTALL_MIMIC_ACTUATION()')
    source = source.replace(drive, 'jaws = np.asarray(CAPSTONE_ACTUATED_FINGER_INDICES(expert), dtype=int)')
    return source.replace(metadata, metadata + '\n            "capstone_gripper_actuation": CAPSTONE_MIMIC_SUMMARY(pol),')


def summary(policy):
    expert = getattr(policy, 'expert', None)
    plan = actuation_plan(expert)
    verify_passive_gains(expert)
    return dict(plan, live_readback=expert._capstone_mimic_drive_readback,
                force_configuration_calls=getattr(expert, '_capstone_mimic_configuration_calls', 0))
