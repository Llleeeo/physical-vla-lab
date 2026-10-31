#!/usr/bin/env python3
"""CPU saved-observation preview only; no object truth, model call or physics."""
import argparse,collections,datetime,json,sys
from pathlib import Path
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from capstone_vla.ball_approach_guard import ApproachGuard
from capstone_vla.ball_approach_follow import FOLLOW_SETTINGS,FollowingApproachGuard
from capstone_vla.ball_visual_geometry import locate_ball
from capstone_vla.ball_live_protocol import decoded_request
from capstone_vla.hf_archive import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-run',type=Path,required=True);a=p.parse_args();source=a.source_run.resolve()
    cfg=json.loads((source/'run_config.json').read_text());plan=cfg['approach_guard']
    if 'follow' in plan:raise ValueError('Preview requires the original height-only source')
    original_report=json.loads((source/'report.json').read_text())
    for name in ['live_observations.hdf5','approach_guard_events.json']:
        if sha256(source/name)!=original_report['artifacts'][name]:raise ValueError('Source artifact changed: '+name)
    candidate_plan=dict(plan,follow=dict(FOLLOW_SETTINGS));base=ApproachGuard(plan);candidate=FollowingApproachGuard(candidate_plan)
    source_events=json.loads((source/'approach_guard_events.json').read_text());events=[];actions=[]
    from analyze_hgwind_closed_loop import audit_approach_value
    with h5py.File(source/'live_observations.hdf5') as f:
        if not f.attrs['complete']:raise ValueError('Incomplete source observations')
        order=json.loads(f.attrs['camera_pose_order_json']);optics={c['name']:c['intrinsics'] for c in plan['cameras']}
        for i in range(len(f['state'])):
            with np.load(source/'ipc'/f'response_{i:06d}.npz',allow_pickle=False) as response:
                raw=response['absolute'][0];np.testing.assert_array_equal(raw,f['model_action'][i])
            detection=locate_ball([f[n][i] for n in order],f['camera_pose_world'][i],[optics[n] for n in order])
            args=(raw,i,f['state'][i,14:21],f['robot_finger_points'][i],f['robot_finger_corners'][i],detection)
            previous,prior_event=base.apply(*args);filtered,event=candidate.apply(*args)
            np.testing.assert_allclose(previous,f['action'][i],rtol=0,atol=1e-12)
            audit_approach_value(dict(prior_event,detection=detection),source_events[i])
            unchanged=[j for j in range(29) if j not in [0,1,7,8,10,11,14,15]]
            np.testing.assert_array_equal(previous[unchanged],filtered[unchanged])
            decoded_request(filtered,i,f['state'][i,21:28])
            events.append(event);actions.append(filtered)
    active=[e['frame'] for e in events if e['follow']['active']]
    output=source/'follow_previews'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');output.mkdir(parents=True)
    np.savez(output/'candidate_actions.npz',actions=np.asarray(actions))
    (output/'decisions.json').write_text(json.dumps(events,indent=2,allow_nan=False)+'\n')
    report=dict(status='CPU_FOLLOW_PREVIEW_PASS',frames=len(events),plan=candidate_plan,
        active_frames=len(active),first_active_frame=active[0] if active else None,last_active_frame=active[-1] if active else None,
        reasons=dict(collections.Counter(e['follow']['reason'] for e in events)),native_decode_rejections=0,
        original_height_and_jaw_preserved=True,raw_model_responses_bound=True,physics_launched=False,
        live_improvement=None,scope='Frozen-observation command preview; future motion/images and grasp cannot be inferred',
        source_hashes={name:sha256(source/name) for name in ['live_observations.hdf5','approach_guard_events.json','run_config.json','report.json']},
        implementation_sha256=sha256(ROOT/'src/capstone_vla/ball_approach_follow.py'))
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('CPU_FOLLOW_PREVIEW_PASS '+json.dumps({k:report[k] for k in ['active_frames','first_active_frame','last_active_frame','reasons']}),flush=True)
    print('REPORT '+str(output/'report.json'),flush=True)

if __name__=='__main__':main()
