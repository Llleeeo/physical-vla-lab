"""Read-only capture of original Cartesian servo requests, not a policy adapter."""
import functools
import inspect
import math
import numpy as np


def encode(value):
    if value is None or isinstance(value,(str,bool,int)):return value
    if isinstance(value,np.ndarray):return encode(value.tolist())
    if isinstance(value,np.generic):return encode(value.item())
    if isinstance(value,float):return value if math.isfinite(value) else {'unsupported_type':'nonfinite_float'}
    if isinstance(value,(tuple,list)):return [encode(x) for x in value]
    if isinstance(value,dict):return {str(k):encode(v) for k,v in value.items()}
    if callable(value):return {'callable_descriptor_only':True,'qualified_name':getattr(value,'__qualname__',type(value).__qualname__),'module':getattr(value,'__module__',type(value).__module__)}
    return {'unsupported_type':type(value).__module__+'.'+type(value).__qualname__}


def match_calls(events,segment):
    """Link successful raw arm outputs to the selected control segment."""
    matches=[]
    for event in events:
        output=event.get('returned_action')
        if not event.get('completed') or not isinstance(output,dict):continue
        try:
            indices=np.asarray(output['joint_indices']);q=np.asarray(output['joint_positions'],float);v=np.asarray(output['joint_velocities'],float)
            if indices.shape!=(7,) or q.shape!=(7,) or v.shape!=(7,) or set(indices.tolist())!=set(range(7)):continue
            order=np.argsort(indices)
            if (np.array_equal(q[order].astype(np.float32),np.asarray(segment['goal_position'],np.float32))
                    and np.array_equal(v[order].astype(np.float32),np.asarray(segment['goal_velocity'],np.float32))):
                matches.append(event['call_id'])
        except (TypeError,ValueError,KeyError):continue
    return matches


class CartesianTargetRecorder:
    def __init__(self):
        self.events=[];self.pending_references=[];self.frame=None;self.frames=[];self.expert=None;self.original=None

    def begin_frame(self,expert,index):
        if self.expert is None:
            self.expert=expert;self.original=expert.operational_space_pinch_action
            original=self.original;signature=inspect.signature(original)
            @functools.wraps(original)
            def observed(*args,**kwargs):
                # Bind/copy for observation; call the original with untouched
                # original objects and argument spelling, exactly once.
                bound=signature.bind(*args,**kwargs);bound.apply_defaults()
                event={'call_id':len(self.events),'frame':self.frame,'inputs':encode(bound.arguments),'completed':False}
                self.events.append(event);self.pending_references.append(None)
                try:result=original(*args,**kwargs)
                except Exception as exc:
                    event['exception_type']=type(exc).__name__;raise
                event['completed']=True
                event['returned_action']=None if result is None else encode({key:getattr(result,key,None) for key in ['joint_indices','joint_positions','joint_velocities']})
                pending=getattr(expert,'_pending_operational_space_reference',None)
                self.pending_references[event['call_id']]=pending
                event['pending_operational_reference']=encode(pending)
                return result
            expert.operational_space_pinch_action=observed
        if expert is not self.expert:raise ValueError('Target recorder expects one episode/robot')
        self.frame=int(index);self.frames.append(self.frame)

    def restore(self):
        if self.expert is not None:self.expert.operational_space_pinch_action=self.original

    def segment_links(self,index,segment):
        events=[x for x in self.events if x['frame']==int(index)]
        matches=match_calls(events,segment)
        selected=[i for i in matches if self.pending_references[i] is segment['operational_space_reference']] if segment.get('operational_space_reference') is not None else []
        return {'cartesian_call_ids':[x['call_id'] for x in events],'matching_cartesian_call_ids':matches,'selected_cartesian_call_ids':selected}

    def payload(self):
        return {'schema':'physical-vla-lab.cartesian-target-capture.v1','frames':list(self.frames),'calls':self.events,
            'motor_commands_overridden':False,'callables_serialized_as_descriptors_only':True,'replay_contract_verified':False}


def audit_capture(payload,segments,frames):
    calls=payload['calls'];by_id={x['call_id']:x for x in calls}
    coverage=payload['frames']==list(range(frames)) and [x['frame'] for x in segments]==list(range(frames))
    call_order=[x['call_id'] for x in calls]==list(range(len(calls))) and all(isinstance(x['frame'],int) and 0<=x['frame']<frames for x in calls)
    unmatched=[];bad_links=[];osc=0
    for row in segments:
        if row['operational_space_reference'] is None:continue
        osc+=1;ids=row.get('selected_cartesian_call_ids',[])
        if len(ids)!=1:unmatched.append(row['frame'])
        for i in ids:
            event=by_id.get(i)
            if event is None or event['frame']!=row['frame'] or i not in match_calls([event],row):bad_links.append(i)
    callable_fields=sorted({key for event in calls for key,value in event['inputs'].items() if isinstance(value,dict) and value.get('callable_descriptor_only')})
    failed=[x['call_id'] for x in calls if not x['completed']]
    return {'frame_coverage_complete':coverage,'call_count':len(calls),'osc_segment_count':osc,'unmatched_osc_frames':unmatched,
        'invalid_call_links':bad_links,'failed_call_ids':failed,'callable_input_fields':callable_fields,
        'call_order_valid':call_order,'capture_alignment_passed':bool(coverage and call_order and calls and osc and not unmatched and not bad_links),
        'standalone_cartesian_replay_verified':False,'vla_evaluated':False}


def patch_capture_collector(source):
    old='                policy_step_started = time.perf_counter()\n                decision = controller.step(frame, world, dt_control)'
    new='                CAPSTONE_BEGIN_CARTESIAN_FRAME(pol.expert, k)\n'+old
    if source.count(old)!=1:raise ValueError('Cartesian capture policy boundary changed')
    source=source.replace(old,new);compile(source,'<cartesian-input-capture>','exec');return source
