"""Robot-only live kinematics; no trajectory, task or object-state inputs."""
import numpy as np


def live_pinch_jacobian(expert):
    """World spatial Jacobian at the current calibrated pinch center.

    RoboPhys exposes the tool Jacobian and the tool-to-pinch conversion. Shift
    the linear rows using omega cross offset; angular rows are unchanged.
    Calls use only the simulated robot's current state and gripper geometry.
    """
    position,quaternion=expert.ee_pose()
    position=np.asarray(position,float);quaternion=np.asarray(quaternion,float)
    jacobian=np.asarray(expert.measured_ee_jacobian(),float)
    if (position.shape!=(3,) or quaternion.shape!=(4,) or jacobian.shape!=(6,7)
            or not np.isfinite(position).all() or not np.isfinite(quaternion).all()
            or not np.isclose(np.linalg.norm(quaternion),1.,atol=1e-5) or not np.isfinite(jacobian).all()):
        raise ValueError('Invalid current robot pose/Jacobian')
    pinch=np.asarray(expert.ee_target_to_pinch(position.copy(),quaternion.copy()),float)
    if pinch.shape!=(3,) or not np.isfinite(pinch).all():raise ValueError('Invalid current pinch calibration')
    shifted=jacobian.copy()
    shifted[:3]+=np.cross(jacobian[3:].T,pinch-position).T
    return shifted


def operational_task_jacobian(physical_jacobian, orientation_length_scale_m):
    """Match RoboPhys's coupled SE(3) metric: [J_v; length_scale * J_w].

    The same metric already applies to the logged angular task velocity.
    The scale is explicit controller configuration, not inferred online from
    an object pose or a stored reference Jacobian.
    """
    jacobian=np.asarray(physical_jacobian,float)
    scale=float(orientation_length_scale_m)
    if (jacobian.shape!=(6,7) or not np.isfinite(jacobian).all()
            or not np.isfinite(scale) or scale<=0):
        raise ValueError('Invalid coupled task Jacobian/length scale')
    result=jacobian.copy();result[3:]*=scale
    return result


def validate_logged_task_metric(rows, orientation_length_scale_m):
    """Franka revolute axes have unit angular norm before task weighting.

    Check all logged coupled references against the explicit metric before
    launching the simulator. This is a source-specific CPU compatibility
    check, not a runtime policy input or a generic Jacobian identity.
    """
    scale=float(orientation_length_scale_m)
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid task length scale')
    count=0
    for row in rows:
        ref=row['operational_space_reference']
        if ref is None:continue
        j=np.asarray(ref['jacobian'],float)
        if (j.shape!=(6,7) or not np.isfinite(j).all()
                or not np.allclose(np.linalg.norm(j[3:],axis=0),scale,atol=1e-7,rtol=0)):
            raise ValueError('Logged coupled task metric differs from configured angular length scale')
        count+=1
    if not count:raise ValueError('No coupled task references to validate')
    return count


def compile_native_pinch_jacobian(source, namespace=None):
    """Reuse only the frozen robot-geometry block, excluding its servo solve.

    This preserves its finite-difference point, step, clipping, quaternion
    conversion and operation order. No upstream source is edited or vendored.
    """
    import ast
    tree=ast.parse(source)
    methods=[x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='operational_space_pinch_action']
    if len(methods)!=1:raise ValueError('Native pinch method anchor changed')
    method=methods[0]
    defaults=dict(zip([x.arg for x in method.args.args][-len(method.args.defaults):],method.args.defaults))
    defaults.update(zip([x.arg for x in method.args.kwonlyargs],method.args.kw_defaults))
    if ast.literal_eval(defaults.get('finite_difference_rad',ast.Constant(None)))!=1.e-4:
        raise ValueError('Native finite-difference step changed')
    blocks=[x for x in method.body if isinstance(x,ast.Try)]
    if len(blocks)!=1:raise ValueError('Native pinch geometry scope changed')
    body=blocks[0].body
    def assigns(node,name):
        return isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id==name for x in node.targets)
    starts=[i for i,x in enumerate(body) if assigns(x,'offset')]
    ends=[i for i,x in enumerate(body) if isinstance(x,ast.Assign) and any(isinstance(y,ast.Attribute) and y.attr=='_last_differential_jacobian' for y in x.targets)]
    if len(starts)!=1 or len(ends)!=1 or starts[0]>=ends[0]:raise ValueError('Native geometry block anchors changed')
    geometry=body[starts[0]:ends[0]]
    attrs={x.attr for n in geometry for x in ast.walk(n) if isinstance(x,ast.Attribute) and isinstance(x.value,ast.Name) and x.value.id=='self'}
    if attrs!={'_pinch_offset_local','_kinematics','_ee_frame'}:
        raise ValueError('Native block contains non-geometry robot inputs')
    template=ast.parse("""def recompute(self):
    state=self.arm_motion_state()
    q=np.asarray(state.position,dtype=float).reshape(-1)
    if q.shape!=(7,) or not np.isfinite(q).all():raise ValueError('Invalid live arm state')
    base_t,base_r=self._base_pose()
    self._kinematics.set_robot_base_pose(base_t,base_r)
    finite_difference_rad=1.e-4
    forward_kinematics_wall_s=0.0
    forward_kinematics_call_count=0
""")
    template.body[0].body.extend(geometry)
    template.body[0].body.append(ast.Return(value=ast.Name(id='jacobian',ctx=ast.Load())))
    code=compile(ast.fix_missing_locations(template),'<frozen-robot-pinch-geometry>','exec')
    # Preflight compiles without importing Isaac. Runtime supplies only the
    # original math helpers; policy, target and object state stay unavailable.
    if namespace is None:return code
    scope={key:namespace[key] for key in ['np','time','utils','shortest_quaternion_rotation_vector']}
    exec(code,scope)
    return scope['recompute']
