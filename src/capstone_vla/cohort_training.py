"""Train-only statistics and periodic Euler-delta labels for the fixed cohort."""
from pathlib import Path
import json
import numpy as np
import pyarrow.parquet as pq

def wrap_delta_euler(values):
    source=np.asarray(values)
    if source.shape[-1]!=7 or not np.isfinite(source).all():raise ValueError('expected finite 7D action deltas')
    result=source.copy()
    result[...,3:6]=(np.remainder(source[...,3:6].astype(np.float64)+np.pi,2*np.pi)-np.pi).astype(source.dtype)
    return result

def read_split_arrays(root):
    root=Path(root);identity=json.loads((root/'capstone_cohort.json').read_text())
    info=json.loads((root/'meta/info.json').read_text());rows=[]
    if len(identity['episode_order'])!=info['total_episodes']:raise ValueError('incomplete cohort export')
    for i in range(info['total_episodes']):
        path=root/info['data_path'].format(episode_chunk=i//info['chunks_size'],episode_index=i)
        data=pq.read_table(path,columns=['observation.state','action'])
        state=np.asarray(data['observation.state'].to_pylist(),dtype=np.float32)
        action=np.asarray(data['action'].to_pylist(),dtype=np.float32)
        if state.shape!=(len(action),6) or action.shape[1:]!=(7,) or not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError('invalid policy state/absolute target arrays')
        rows.append({'episode':identity['episode_order'][i],'state':state,'action':action})
    return rows

def train_normalization(root,chunk_size=20):
    root=Path(root)
    if json.loads((root/'capstone_cohort.json').read_text())['split']!='train':
        raise ValueError('normalization may only be fitted on TRAIN')
    rows=read_split_arrays(root);states=[];targets=[];wrap_count=0;total=0
    for row in rows:
        s=row['state'];a=row['action'];n=len(a)
        indices=np.arange(n)[:,None]+np.arange(chunk_size);valid=indices<n
        delta=a[np.minimum(indices,n-1)].copy();delta[...,:6]-=s[:,None,:]
        raw=delta[valid];wrap_count+=int((np.abs(raw[:,3:6])>np.pi).any(axis=1).sum());total+=len(raw)
        states.append(s);targets.append(wrap_delta_euler(raw))
    def summary(x):
        x=x.astype(np.float64);std=x.std(axis=0);std[std<1e-6]=1.
        return {key:value.astype(np.float32) for key,value in {
            'mean':x.mean(axis=0),'std':std,'min':x.min(axis=0),'max':x.max(axis=0)}.items()}
    stats={'observation.state':summary(np.concatenate(states)),'action':summary(np.concatenate(targets))}
    audit={'fit_split':'train','episodes':len(rows),'state_frames':sum(len(x) for x in states),
       'valid_chunk_targets':total,'raw_euler_delta_over_pi_count':wrap_count,
       'raw_euler_delta_over_pi_fraction':wrap_count/total,'wrapped_abs_euler_max_rad':float(np.abs(np.concatenate(targets)[:,3:6]).max()),
       'label_convention':'componentwise periodic Euler delta in [-pi,pi), not SO(3) rotation vector',
       'state_convention':'unchanged pretrained absolute Euler state convention','validation_used_for_fitting':False}
    return stats,audit

class CyclingFrameSampler:
    def __init__(self,length,seed):
        if length<1:raise ValueError('empty dataset')
        self.length=length;self.rng=np.random.default_rng(seed);self.order=self.rng.permutation(length);self.cursor=0
    def draw(self,count):
        if count<1:raise ValueError('positive batch size required')
        result=[]
        while len(result)<count:
            if self.cursor==self.length:self.order=self.rng.permutation(self.length);self.cursor=0
            take=min(count-len(result),self.length-self.cursor)
            result.extend(self.order[self.cursor:self.cursor+take].tolist());self.cursor+=take
        return result
