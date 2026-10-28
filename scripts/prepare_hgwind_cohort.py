#!/usr/bin/env python3
"""Audit the stopped 40-success cohort and prepare separate render copies; CPU only."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from capstone_vla.hf_archive import RAW_FILES, sha256
from capstone_vla.hgwind_cohort import TASK_ID, inspect_recording, require, stage_episode
from analyze_hgwind_batch import audit_episode

OUT = ROOT / 'data/derived/hgwind_40_20261028'


def main():
    config = json.loads((ROOT / 'configs/hgwind_overnight_20261027.json').read_text())
    raw = ROOT / config['local_final']
    bundle = ROOT / config['bundle']
    production = json.loads((bundle / 'production_config.json').read_text())
    task = bundle / production['task_relative']
    scene_hash = sha256(task)
    require(scene_hash == production['task_sha256'], 'Frozen scene changed')
    rows, audits = [], []
    expected = {'local': 21, 'hongtao': 19}
    for host, expected_count in expected.items():
        run = raw / host
        record = json.loads((run / 'cohort_summary.json').read_text())
        require(record['task_sha256'] == scene_hash, 'Different source scenes')
        receipt = {f['path']: f for f in json.loads((run / 'transport_manifest.json').read_text())['files']}
        episodes = sorted((run / 'Trajectory/1s1p' / TASK_ID).glob('traj_*'))
        require(len(episodes) == expected_count, 'Stopped cohort count changed')
        for ep in episodes:
            basic = inspect_recording(ep, scene_hash)
            physics = audit_episode(ep)
            require(physics['no_assistance_observed'] and physics['bilateral_frames'] > 0
                    and all(v > 0 for v in physics['raw_positive_impulse_ticks'].values()),
                    'Missing unassisted bilateral contact evidence: ' + str(ep))
            require(physics['post_contact_final_lift_m'] > 0, 'No retained lift')
            files = []
            for name in RAW_FILES:
                f = ep / name
                h = sha256(f)
                old = receipt[f.relative_to(run).as_posix()]
                require(h == old['sha256'] and f.stat().st_size == old['bytes'], 'Transport source changed')
                files.append(dict(path=name, bytes=f.stat().st_size, sha256=h))
            target = OUT / 'render_work' / host / 'Trajectory/1s1p' / TASK_ID / ep.name
            stage_episode(ep, target, files)
            row = dict(host=host, episode=ep.name, source=str(ep), staged=str(target),
                       depth_bias_m=record['depth_bias_m'], files=files, **basic)
            rows.append(row)
            audits.append({'host': host, 'depth_bias_m': record['depth_bias_m'], **physics, **basic})
            print('AUDITED ' + host + '/' + ep.name + ' frames=' + str(basic['frames']), flush=True)
    require(len({r['numeric_fingerprint'] for r in rows}) == len(rows), 'Duplicate numeric trajectories')
    require(len({(r['host'], r['attempt_index']) for r in rows}) == len(rows), 'Duplicate attempts')
    manifest = dict(schema='capstone.hgwind_cohort_render.v1', cohort=config['batch_id'],
                    source_root=str(raw), bundle=str(bundle), task_usd=str(task), task_sha256=scene_hash,
                    bundle_manifest_sha256=sha256(bundle / 'bundle_manifest.json'),
                    camera_fps=60, control_hz=60, resolution=[640, 480], render_quality='balanced',
                    episode_count=len(rows), total_frames=sum(r['frames'] for r in rows),
                    rows=rows, split_status='not assigned; no model training or held-out evaluation yet',
                    training_ready=False, manual_visual_reviewed=False)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'render_manifest.json'
    if path.exists():
        require(json.loads(path.read_text()) == manifest, 'Existing frozen render manifest differs')
    else:
        path.write_text(json.dumps(manifest, indent=2) + '\n')
    report = dict(status='RAW_SCHEMA_AND_CONTACT_EVIDENCE_PASS', episode_count=len(rows),
                  total_frames=manifest['total_frames'], numeric_duplicates=0, rows=audits,
                  interpretation='Collector labels and independent recording/contact checks pass. This is expert development data; visual content and VLA action conversion are not certified.',
                  raw_velocity_ratio_note='A few accepted real-contact transients can exceed ratio 1 slightly under the unchanged collector recovery rule. Values are retained for review, not relabeled.')
    (OUT / 'raw_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print('RAW_COHORT_PASS episodes=' + str(len(rows)) + ' frames=' + str(manifest['total_frames']), flush=True)
    print('MANIFEST ' + str(path), flush=True)


if __name__ == '__main__':
    main()
