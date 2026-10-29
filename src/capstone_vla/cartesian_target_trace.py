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


def reference_content_equal(left,right):
    """Exact finite contents; native consumption copies dictionaries/arrays."""
    if isinstance(left,dict) or isinstance(right,dict):
        if not isinstance(left,dict) or not isinstance(right,dict) or set(left)!=set(right):return False
        if {'unsupported_type','callable_descriptor_only'} & set(left):return False
        return all(reference_content_equal(left[key],right[key]) for key in left)
    if left is None or right is None:return left is None and right is None
    try:
        a,b=np.asarray(left,float),np.asarray(right,float)
        return bool(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.array_equal(a,b))
    except (TypeError,ValueError):return False


def reference_matches(events,segment):
    candidates=set(match_calls(events,segment));reference=segment.get('operational_space_reference')
    if reference is None:return []
    return [event['call_id'] for event in events if event['call_id'] in candidates
        and event['frame']==segment['frame'] and reference_content_equal(event.get('pending_operational_reference'),reference)]


class CartesianTargetRecorder:
    def __init__(self):
        self.events=[];self.frame=None;self.frames=[];self.expert=None;self.original=None

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
                self.events.append(event)
                try:result=original(*args,**kwargs)
                except Exception as exc:
                    event['exception_type']=type(exc).__name__;raise
                event['completed']=True
                event['returned_action']=None if result is None else encode({key:getattr(result,key,None) for key in ['joint_indices','joint_positions','joint_velocities']})
                pending=getattr(expert,'_pending_operational_space_reference',None)
                event['pending_operational_reference']=encode(pending)
                return result
            expert.operational_space_pinch_action=observed
        if expert is not self.expert:raise ValueError('Target recorder expects one episode/robot')
        self.frame=int(index);self.frames.append(self.frame)

    def restore(self):
        if self.expert is not None:self.expert.operational_space_pinch_action=self.original

    def segment_links(self,index,segment):
        # Native segments do not carry the recording frame. The explicit
        # callback index supplies it; do not mutate the live control segment.
        if 'frame' in segment and segment['frame']!=int(index):
            raise ValueError('Segment frame differs from recorder callback index')
        framed=dict(segment,frame=int(index))
        events=[x for x in self.events if x['frame']==int(index)]
        matches=match_calls(events,framed)
        full=reference_matches(events,framed);selected=full if len(full)==1 else []
        return {'cartesian_call_ids':[x['call_id'] for x in events],'matching_cartesian_call_ids':matches,
            'reference_matching_cartesian_call_ids':full,'selected_cartesian_call_ids':selected,
            'selection_rule':'unique_same_frame_arm_and_full_reference_content'}

    def payload(self):
        return {'schema':'physical-vla-lab.cartesian-target-capture.v1','frames':list(self.frames),'calls':self.events,
            'motor_commands_overridden':False,'callables_serialized_as_descriptors_only':True,'replay_contract_verified':False}


def audit_capture(payload,segments,frames):
    calls=payload['calls']
    coverage=payload['frames']==list(range(frames)) and [x['frame'] for x in segments]==list(range(frames))
    call_order=[x['call_id'] for x in calls]==list(range(len(calls))) and all(isinstance(x['frame'],int) and 0<=x['frame']<frames for x in calls)
    unmatched=[];bad_links=[];recovered=[];selected_by_frame={};osc=0
    for row in segments:
        if row['operational_space_reference'] is None:continue
        osc+=1
        candidates=reference_matches([x for x in calls if x['frame']==row['frame']],row)
        ids=candidates if len(candidates)==1 else []
        if not ids:unmatched.append(row['frame'])
        else:selected_by_frame[str(row['frame'])]=ids[0]
        persisted=row.get('selected_cartesian_call_ids',[])
        if persisted and persisted!=ids:bad_links.extend(persisted)
        elif not persisted and ids:recovered.append(row['frame'])
    callable_fields=sorted({key for event in calls for key,value in event['inputs'].items() if isinstance(value,dict) and value.get('callable_descriptor_only')})
    failed=[x['call_id'] for x in calls if not x['completed']]
    return {'frame_coverage_complete':coverage,'call_count':len(calls),'osc_segment_count':osc,'unmatched_osc_frames':unmatched,
        'invalid_call_links':bad_links,'failed_call_ids':failed,'recovered_selection_frames':recovered,
        'selection_rule':'unique_same_frame_arm_and_full_reference_content','selected_call_ids_by_frame':selected_by_frame,'callable_input_fields':callable_fields,
        'call_order_valid':call_order,'capture_alignment_passed':bool(coverage and call_order and calls and osc and not unmatched and not bad_links),
        'standalone_cartesian_replay_verified':False,'vla_evaluated':False}


def patch_capture_collector(source):
    old='                policy_step_started = time.perf_counter()\n                decision = controller.step(frame, world, dt_control)'
    new='                CAPSTONE_BEGIN_CARTESIAN_FRAME(pol.expert, k)\n'+old
    if source.count(old)!=1:raise ValueError('Cartesian capture policy boundary changed')
    source=source.replace(old,new);compile(source,'<cartesian-input-capture>','exec');return source
