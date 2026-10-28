"""Verify cloud-rendered work copies before installing them beside local raw copies."""
import json
from pathlib import Path
import shutil

from capstone_vla.hf_archive import sha256
from capstone_vla.hgwind_cohort import require, verify_files, assert_metadata_enrichment, verify_visual


def install_visual_download(incoming, row, archive):
    incoming, target = Path(incoming), Path(row['staged'])
    require(archive['host'] == row['host'] and archive['trajectory'] == row['episode']
            and archive['stage'] == 'rendered', 'Cloud episode identity mismatch')
    verify_files(incoming, archive['files'])
    verify_files(incoming, [f for f in row['files'] if f['path'] != 'meta.json'])
    assert_metadata_enrichment(Path(row['source']) / 'meta.json', incoming / 'meta.json')
    qa = verify_visual(incoming, row['frames'])
    require(qa['suspicious_blank_count'] == 0, 'Blank sampled images in cloud render')
    camera_files = [f for f in archive['files'] if f['path'].startswith('video/')]
    require(len([f for f in camera_files if f['path'].endswith('_vis.mp4')]) == 15,
            'Cloud archive lacks full RGB/depth/seg video set')
    target.mkdir(parents=True, exist_ok=True)
    if (target / 'video').exists():
        verify_files(target, camera_files)
    else:
        pending = target / '.incoming_cloud_video'
        shutil.copytree(incoming / 'video', pending, dirs_exist_ok=True)
        for f in camera_files:
            require(sha256(pending / Path(f['path']).name) == f['sha256'], 'Video copy checksum mismatch')
        pending.rename(target / 'video')
    meta_tmp = target / '.cloud_meta.json.tmp'
    shutil.copy2(incoming / 'meta.json', meta_tmp)
    meta_tmp.replace(target / 'meta.json')
    verify_files(target, [f for f in row['files'] if f['path'] != 'meta.json'])
    assert_metadata_enrichment(Path(row['source']) / 'meta.json', target / 'meta.json')
    return qa
