"""Bounded passive finger-drive observation; never sets simulator properties."""
import json
import math
from pathlib import Path
import numpy as np

FRAMES = {0, 150, 153, 154, 155, 156, 160, 166, 167, 170, 173}
FINGERS = ('panda_finger_joint1', 'panda_finger_joint2')


def plain(value):
    if isinstance(value, dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):return [plain(x) for x in value]
    if hasattr(value, 'detach'):value=value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):return plain(value.tolist())
    if isinstance(value, np.generic):return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):return str(value)
    if value is None or isinstance(value,(str,int,float,bool)):return value
    try:return plain(list(value))
    except TypeError:return str(value)


def usd_fingers(stage, robot_path=None):
    # Only inspect authored properties. Do not Apply a schema or create an attr.
    rows=[]
    if robot_path:
        from pxr import Usd
        prims=Usd.PrimRange(stage.GetPrimAtPath(robot_path))
    else:prims=stage.Traverse()
    for prim in prims:
        path=str(prim.GetPath())
        if prim.GetName() not in FINGERS or (robot_path and not path.startswith(str(robot_path)+'/')):continue
        rows.append(dict(path=path,schemas=list(prim.GetAppliedSchemas()),
            attributes={str(a.GetName()):plain(a.Get()) for a in prim.GetAttributes()
                if str(a.GetName()).startswith(('drive:','physxJoint:','physxMimicJoint:'))},
            relationships={str(r.GetName()):[str(x) for x in r.GetTargets()]
                for r in prim.GetRelationships() if 'mimic' in str(r.GetName()).lower()}))
    if len(rows)!=2:raise ValueError('Expected two uniquely resolved Panda finger joints')
    return sorted(rows,key=lambda r:r['path'])


def read_drive(expert):
    result={'errors':{}}
    robot=expert._robot
    names=list(robot.dof_names)
    if any(names.count(name)!=1 for name in FINGERS):
        raise ValueError('Finger DOF identity is ambiguous')
    indices=[names.index(name) for name in FINGERS]
    result.update(joint_names=list(FINGERS),joint_indices=indices,
        units={'live_gains':'native stiffness/damping; interpret with authored drive type','live_max_efforts':'N',
               'live_max_velocities':'m/s','usd_attributes':'raw authored USD values'})
    view=robot._articulation_view
    def read(name, callback):
        try:
            value=callback()
            if value is None:raise ValueError('getter returned None')
            result[name]=plain(value)
            if name in ['live_gains','live_max_efforts','live_max_velocities',
                        'measured_positions_m','measured_velocities_m_s']:
                if not np.isfinite(np.asarray(result[name],dtype=float)).all():
                    raise ValueError('Nonfinite live drive readback')
        except Exception as exc:result['errors'][name]=repr(exc)
    valid=bool(view.is_physics_handle_valid())
    result['live_physics_handle_valid']=valid
    if valid:
        read('live_gains',lambda:view.get_gains(joint_indices=indices))
        read('live_max_efforts',lambda:view.get_max_efforts(joint_indices=indices))
        read('live_max_velocities',lambda:view.get_joint_max_velocities(joint_indices=indices))
    else:result['errors']['live_properties']='No live physics handle; USD fallback deliberately not used'
    read('measured_positions_m',lambda:np.asarray(robot.get_joint_positions())[indices])
    read('measured_velocities_m_s',lambda:np.asarray(robot.get_joint_velocities())[indices])
    def action():
        a=robot.get_articulation_controller().get_applied_action()
        if a is None:raise ValueError('No applied action')
        return {name:None if getattr(a,name,None) is None else np.asarray(getattr(a,name)).reshape(-1)[indices]
                for name in ['joint_positions','joint_velocities','joint_efforts']}
    read('last_applied_action',action)
    read('expert_driven_indices',lambda:expert._gripper_dof_indices())
    read('usd_fingers',lambda:usd_fingers(expert._prim.GetStage(),expert._prim.GetPath()))
    result['expert_state']=plain({k:getattr(expert,k,None) for k in [
        '_last_gripper_close','_finger_force_forced','_last_gripper_drive',
        '_parallel_gripper_stiffness_n_per_m','_last_gripper_target','_gcfg']})
    return result


class DriveRecorder:
    def __init__(self, path, max_events=512):
        self.path=Path(path);self.max_events=max_events;self.events=0
        self.expert=None;self.frame=None;self.originals=[];self.limit=False
        if self.path.exists():raise ValueError('Preserve existing drive observation')
        self.path.write_text('')

    def emit(self, phase, payload):
        # Observer failures are visible but must not replace native calls/errors.
        if self.limit:return
        if self.events>=self.max_events:
            self.limit=True;phase='event_limit_reached';payload={'incomplete':True}
        try:
            row=dict(event=self.events,frame=self.frame,phase=phase,**plain(payload))
            with self.path.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            self.events+=1
        except Exception as exc:print('GRIPPER_TRACE_WARNING '+repr(exc),flush=True)

    def capture(self, phase, **extra):
        if self.limit:return
        try:payload=read_drive(self.expert)
        except Exception as exc:payload={'errors':{'snapshot':repr(exc)}}
        self.emit(phase,dict(snapshot=payload,**extra))

    def bind(self, expert):
        if self.expert is expert:return
        if self.expert is not None:raise ValueError('One drive observer cannot mix experts')
        self.expert=expert
        ensure=expert.ensure_gripper_force;grip=expert.gripper_action
        def force(*args,**kwargs):
            self.capture('before_ensure_force',arguments={'args':args,'kwargs':kwargs})
            try:result=ensure(*args,**kwargs)
            except Exception as exc:
                self.emit('ensure_force_raised',{'exception':repr(exc)});raise
            self.capture('after_ensure_force',returned=plain(result))
            return result
        def action(*args,**kwargs):
            was_closed=bool(getattr(expert,'_last_gripper_close',False))
            result=grip(*args,**kwargs)
            if not was_closed and bool(getattr(expert,'_last_gripper_close',False)):
                self.capture('close_action_built',arguments={'args':args,'kwargs':kwargs})
            return result
        for name,original,wrapped in [('ensure_gripper_force',ensure,force),('gripper_action',grip,action)]:
            setattr(expert,name,wrapped);self.originals.append((name,original,wrapped))

    def observe(self, expert, frame):
        self.frame=int(frame);self.bind(expert)
        if self.frame in FRAMES:self.capture('frame_before_policy')

    def restore(self):
        for name,original,wrapped in self.originals:
            if getattr(self.expert,name) is wrapped:setattr(self.expert,name,original)


def install(plan):
    from . import hgwind_history as history, hgwind_expert_trace as trace
    original_begin=history.begin_attempt;original_frame=trace.begin_cartesian_frame
    active=[None]
    def begin(index):
        if active[0] is not None:active[0].restore()
        original_begin(index)
        folder=Path(plan['history_cases'][str(index)]['output'])
        recorder=DriveRecorder(folder/'gripper_drive_trace.jsonl');active[0]=recorder
        try:
            import omni.usd
            recorder.emit('usd_before_reset',{'usd_fingers':usd_fingers(omni.usd.get_context().get_stage(),'/Root/Dynamics/franka')})
        except Exception as exc:recorder.emit('usd_before_reset',{'errors':{'usd':repr(exc)}})
    def frame(expert,index):
        original_frame(expert,index)
        if active[0] is None:raise ValueError('Drive recorder has no declared attempt')
        active[0].observe(expert,index)
    history.begin_attempt=begin;trace.begin_cartesian_frame=frame


def trace_status(path):
    path=Path(path)
    if not path.exists():return {'available':False,'verified':False,'reason':'missing_trace'}
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    frames={r['frame']:r for r in rows if r['phase']=='frame_before_policy'}
    close=[r for r in rows if r['phase']=='close_action_built']
    def usable(row):
        s=row.get('snapshot',{})
        required=['live_gains','live_max_efforts','live_max_velocities','usd_fingers',
                  'measured_positions_m','measured_velocities_m_s','last_applied_action']
        return (s.get('live_physics_handle_valid') is True
            and all(key in s and key not in s.get('errors',{}) for key in required))
    before=[r for r in rows if r['phase']=='usd_before_reset']
    verified=bool(0 in frames and usable(frames[0]) and close and usable(close[0])
        and before and 'errors' not in before[0]
        and not any(r['phase'] in ['event_limit_reached','ensure_force_raised'] for r in rows))
    return dict(available=True,verified=verified,events=len(rows),frame_samples=sorted(frames),
        first_close_frame=close[0]['frame'] if close else None,
        reason='observations_recorded' if verified else 'missing_or_invalid_drive_evidence')
