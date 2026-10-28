#!/usr/bin/env python3
"""Compare Task-2 live rollouts without assigning a benchmark success label."""
import argparse
import json
from pathlib import Path
import sys
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from capstone_vla.franka import FrankaCartesianAdapter


def stats(values):
    return {k:float(fn(values)) for k,fn in [('median',np.median),('p95',lambda x:np.percentile(x,95)),('max',np.max)]}


def contacts(left,right):
    ids=np.flatnonzero(left & right)
    return {'left_frames':int(left.sum()),'right_frames':int(right.sum()),
            'bilateral_frames':len(ids),'first_bilateral_frame':int(ids[0]) if len(ids) else None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--episode',type=Path)
    args=parser.parse_args()
    cfg=json.loads((ROOT/'configs/franka_action_interface.json').read_text())
    adapter=FrankaCartesianAdapter(cfg['urdf'])
    episode=ROOT/'data/derived/hf_pilot_render_60hz_work/Trajectory/1s1p/grasp_sliding_motion_on_flat_surface_1s1p_g9wvkk1/traj_00001'
    episode=args.episode or episode
    def positions(q):return np.stack([adapter.pose_jacobian(row[:7])[0] for row in q])
    reference_file=episode/'trajectory.hdf5'
    if not reference_file.is_file():reference_file=episode/'replay.hdf5'
    with h5py.File(reference_file,'r') as ref:
        actions=ref['actions'][:];reference=positions(ref['obs/joint_pos'][:])
        objects=ref['obs/object_pos'][:]
        source_contacts=contacts(ref['obs/contact_L'][:],ref['obs/contact_R'][:])
    end=source_contacts['first_bilateral_frame']
    result={'schema':'capstone_task2_interface_comparison_v1','official_success_verified':False,
            'interpretation':'diagnostic replay; contact alone does not certify grasp success',
            'source_contact':source_contacts,'modes':{}}
    curves={}
    for mode in ('joint','cartesian'):
        path=args.output/f'{mode}_trace.hdf5'
        if not path.exists():continue
        with h5py.File(path) as trace:
            if len(trace['joint_pos'])!=len(actions):raise ValueError('incomplete trace')
            if not np.allclose(trace['expert_actions'][:],actions,atol=1e-7,rtol=0):
                raise ValueError('source actions differ')
            measured=positions(trace['joint_pos'][:])
            error=np.linalg.norm(measured-reference,axis=1)
            obj_error=np.linalg.norm(trace['object_pos'][:]-objects,axis=2).max(axis=1)
            curves[mode]=error
            item={'source_state_tracking_error_m':stats(error),
                  'object_source_deviation_m':stats(obj_error),
                  'contact':contacts(trace['contact_L'][:],trace['contact_R'][:]),
                  'servo_calls':int(trace.attrs['servo_calls']),
                  'final_object_positions_world_m':trace['object_pos'][-1].tolist()}
            if end is not None and end>0:
                item['before_original_first_bilateral_contact_error_m']=stats(error[:end])
            result['modes'][mode]=item
    if not result['modes']:raise ValueError('no traces found')
    (args.output/'comparison_report.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4))
    for mode,values in curves.items():ax.plot(np.arange(len(values))/60,values*1000,label=mode)
    if end is not None:ax.axvline(end/60,color='gray',linestyle='--',label='source first bilateral contact')
    ax.set(xlabel='Simulation time (s)',ylabel='EE difference from recorded state (mm)')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(args.output/'tracking_comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
