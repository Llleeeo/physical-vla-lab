#!/usr/bin/env python3
"""CPU-only comparison of original and regenerated startup physics evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import h5py
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from capstone_vla.hgwind_cohort import verify_files
BALL = '/Root/Dynamics/ball_y9bssvn'
BATCH = ROOT / 'data/derived/hgwind_contract_batch/20261029_215343_401088'
LOG = ROOT / 'data/derived/hgwind_overnight_20261027_230043/logs/collect_local_1793113836669755100.log'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker_context(lines, attempt):
    recent = []; matches = []; worker_line = None
    for number, line in enumerate(lines, 1):
        if 'CAPSTONE_HGWIND_CONTROL_ACTIVE' in line:
            recent = []; worker_line = number
        match = re.search(r'LIVE SAMPLE-REGION attempt=(\d+) ', line)
        if match:
            recent.append(int(match[1]))
            if int(match[1]) == attempt:
                matches.append(dict(worker_marker_line=worker_line, sample_line=number,
                    attempts_in_this_worker_through_source=list(recent),
                    ordinal_in_worker=len(recent) if worker_line else None))
    if len(matches) != 1:
        raise ValueError('Original attempt context is not uniquely recoverable')
    return matches[0]


def snapshot(episode):
    with h5py.File(Path(episode) / 'replay.hdf5') as handle:
        root = 'evidence/native/'
        paths = handle[root + 'inventory/rigid_body/prim_paths'].asstr()[:].tolist()
        if paths.count(BALL) != 1:
            raise ValueError('Native ball identity is ambiguous')
        index = paths.index(BALL); body = handle[root + 'rigid_body']
        def native(name, count=7):
            return body[name + '/values'][:count, index].astype(float)
        times = handle[root + 'timestamp'][:7].astype(float)
        times -= times[0]
        dt = handle[root + 'step_dt'][:7]
        if not np.allclose(dt[1:], 1/180, atol=1e-9, rtol=0):
            raise ValueError('Unexpected startup physics cadence')
        contacts = handle['evidence/contacts']; cp = contacts['paths'].asstr()[:].tolist()
        headers = contacts['headers'][:]
        impulse = np.zeros((3, 3))
        for row in headers[headers[:, 0] < 3]:
            step, a, b, _, _, _, start, count = map(int, row)
            if cp[a] == BALL or cp[b] == BALL:
                direction = 1 if cp[a] == BALL else -1
                impulse[step] += direction * contacts['points'][start:start+count, 6:9].sum(axis=0)
        return dict(physics_time_s=times.tolist(), native_velocity=native('get_velocities').tolist(),
            native_acceleration=native('get_accelerations', 6).tolist(),
            initial_mass=native('get_masses', 1)[0].tolist(),
            initial_inertia=native('get_inertias', 1)[0].tolist(),
            initial_com=native('get_coms', 1)[0].tolist(),
            initial_transform=native('get_transforms', 1)[0].tolist(),
            first_three_recorded_contact_impulses_Ns=impulse.tolist(),
            control_actions=handle['actions'][:7].tolist(),
            control_ball_velocity=handle['obs/object_vel'][:7, 0].tolist(),
            control_ee_position=handle['obs/ee_pos'][:7].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, default=BATCH)
    args = parser.parse_args(); batch = json.loads((args.batch / 'batch.json').read_text())
    lines = LOG.read_text(errors='replace').splitlines()
    result = dict(schema='physical-vla-lab.startup-physics-diagnosis.v1',
        batch=args.batch.name, gpu_launched=False, learned_policy_evaluated=False,
        source_log_sha256=digest(LOG), batch_sha256=digest(args.batch / 'batch.json'), cases=[])
    for case in batch['cases']:
        verify_files(case['source'], case['source_files'])
        report = Path(case['teacher']['report']); data = json.loads(report.read_text())
        original = snapshot(case['source']); fresh = snapshot(data['probe_episode'])
        first = np.flatnonzero(np.any(np.asarray(original['native_velocity']) != np.asarray(fresh['native_velocity']), axis=1))
        row = dict(episode=case['episode'], attempt_index=case['attempt_index'],
            source_worker_context=worker_context(lines, case['attempt_index']),
            original=original, fresh=fresh, teacher_report_sha256=digest(report),
            fresh_replay_sha256=digest(Path(data['probe_episode']) / 'replay.hdf5'),
            initial_native_properties_exact={k:original[k] == fresh[k] for k in
                ['initial_mass','initial_inertia','initial_com','initial_transform']},
            first_native_velocity_discrepancy_index=int(first[0]) if len(first) else None,
            first_ball_linear_velocity_delta_m_s=(np.asarray(fresh['native_velocity'])[1,:3]-np.asarray(original['native_velocity'])[1,:3]).tolist(),
            first_command_exact=original['control_actions'][0] == fresh['control_actions'][0],
            first_control_ee_positions_exact=original['control_ee_position'][1] == fresh['control_ee_position'][1])
        result['cases'].append(row)
        print('STARTUP_EVIDENCE',case['episode'],'first_native_velocity_difference=',row['first_native_velocity_discrepancy_index'],
            'source_worker_ordinal=',row['source_worker_context']['ordinal_in_worker'], flush=True)
    result['interpretation'] = [
        'In both aborted teachers, ball native velocity differs after the first180Hz physical step, before hand contact and before meaningful arm divergence; initial native properties and first motor action match.',
        'The first three raw contact-step records contain no nonzero ball contact impulse. This excludes a difference in these recorded impulses, not every unrecorded solver/contact state.',
        'Historical cases1/7 originate from the third/second attempt in their workers; case18 is a first attempt. The regenerated probes all start fresh workers. Association is not proof of a cache/reset root cause.',
        'Fan.on_update calls apply_force_at_pos; native startup Play pumps updates while physics_dt is zero. Force-call timing and reset history are plausible causes but are not recorded in original source and remain unproven.',
        'One passive trace of teacher1 is prepared to measure native force calls and actual physics_dt, without suppressing or injecting forces. No GPU job started by this analysis.',
    ]
    target = args.batch / 'startup_physics_diagnosis.json'
    if target.exists():
        if json.loads(target.read_text()) != result:
            raise ValueError('Different diagnosis already exists; preserve prior artifact')
    else:
        target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('STARTUP_DIAGNOSIS ' + str(target), flush=True)


if __name__ == '__main__':
    main()
