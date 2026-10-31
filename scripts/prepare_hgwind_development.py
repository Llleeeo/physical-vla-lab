#!/usr/bin/env python3
"""CPU-only preparation for one passed ball development episode; no rendering."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from capstone_vla.hgwind_development_data import inspect_source,DESTINATION,BallDevelopmentDataset
from capstone_vla.hf_archive import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/DESTINATION)
    parser.add_argument('--run',action='store_true',help='Write only small CPU arrays and provenance')
    args=parser.parse_args()
    states,actions,ts,meta=inspect_source(ROOT)
    print('CPU_ALIGNMENT_PASS episode=local/traj_00018 frames=269 state=30 action=29 RGB=60Hz absolute_actions=true',flush=True)
    if not args.run:return
    out=args.output.resolve();out.relative_to(ROOT/'data/derived')
    out.mkdir(parents=True,exist_ok=False)
    np.savez(out/'arrays.npz',states=states,actions=actions,timestamp=ts)
    meta['payload_sha256']=sha256(out/'arrays.npz')
    (out/'dataset.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
    ds=BallDevelopmentDataset(out)
    for i in [0,1,154,162,163,268]:
        item=ds[i]
        if item['action'].shape!=(20,29) or item['observation.state'].shape!=(2,30):raise ValueError('Reader shape mismatch')
    if int(ds[268]['action_is_pad'].sum())!=19:raise ValueError('End padding mismatch')
    print('BALL_DEVELOPMENT_DATA_READY '+str(out/'dataset.json'),flush=True)

if __name__=='__main__':main()
