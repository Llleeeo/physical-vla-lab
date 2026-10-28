"""CPU development payload for DynamicVLA; this is not a LeRobot dataset."""
from pathlib import Path
import h5py
import numpy as np

CAMERAS=('opst_cam','wrist_cam')

def temporal_indices(frame, length, history=(-2,0), chunk_size=20):
    if not isinstance(frame,(int,np.integer)) or frame<0 or frame>=length:
        raise IndexError(frame)
    if chunk_size<1 or not history or history[-1]!=0 or any(x>0 for x in history):
        raise ValueError('history must end at current frame and contain no future frames')
    obs=np.asarray(history,dtype=np.int64)+frame
    actions=np.arange(chunk_size,dtype=np.int64)+frame
    return (np.clip(obs,0,length-1),obs<0,
            np.clip(actions,0,length-1),actions>=length)


def action_chunk(absolute_actions, states, frame, chunk_size=20, delta=True):
    _,_,indices,padding=temporal_indices(frame,len(states),chunk_size=chunk_size)
    values=np.asarray(absolute_actions[indices],dtype=np.float32).copy()
    # Match the pinned upstream: componentwise Euler subtraction, not SO(3)
    # increments, and every target is relative to the CURRENT observed state.
    if delta:
        values[:,:6]-=states[frame,:6]
    return values,padding


class PilotH5Dataset:
    """Numpy reader with model-facing keys and explicit padding; no GPU imports."""
    def __init__(self,path,chunk_size=20,history=(-2,0),delta=True):
        self.path=Path(path)
        self.chunk_size=chunk_size
        self.history=history
        self.delta=delta
        with h5py.File(self.path,'r') as f:
            if f.attrs.get('schema')!='capstone_dynamicvla_development_payload_v1':
                raise ValueError('unexpected payload schema')
            self.states=f['observation.state'][:]
            self.actions=f['action'][:]
            self.instruction=str(f.attrs['instruction'])
            self.fps=float(f.attrs['fps'])
            if self.states.shape!=(len(self.actions),6) or self.actions.shape[1:]!=(7,):
                raise ValueError('invalid state/action dimensions')
            for name in CAMERAS:
                d=f['observation.images/'+name]
                if len(d)!=len(self.actions) or d.dtype!=np.uint8 or d.shape[-1]!=3:
                    raise ValueError('invalid RGB image sequence')
        temporal_indices(0,len(self),history,chunk_size)

    def __len__(self):return len(self.actions)

    def __getitem__(self,frame):
        obs,obs_pad,_,act_pad=temporal_indices(frame,len(self),self.history,self.chunk_size)
        actions,_=action_chunk(self.actions,self.states,frame,self.chunk_size,self.delta)
        result={'observation.state':self.states[obs].copy(),
                'observation.state_is_pad':obs_pad.copy(),
                'action':actions,'action_is_pad':act_pad,'task':self.instruction}
        with h5py.File(self.path,'r') as f:
            for name in CAMERAS:
                key='observation.images.'+name
                # Per-frame indexing supports repeated boundary indices.
                images=np.stack([f['observation.images/'+name][int(i)] for i in obs])
                result[key]=images.transpose(0,3,1,2).astype(np.float32)/255.
                result[key+'_is_pad']=obs_pad.copy()
        return result
