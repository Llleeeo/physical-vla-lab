from pathlib import Path
import io
import json
import sys
import tarfile
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from capstone_vla.hf_archive import pack_episode, unpack_verified, sha256, private_repo, upload_package


@pytest.fixture
def episode(tmp_path):
    p=tmp_path/'source/traj_00000';p.mkdir(parents=True)
    meta={'success':True,'trajectory_validity':{'passed':True},'grasp_mode':'physical_finger_contact_only',
          'fix_joint':False,'fixedforce':False,'task':{'id':'grasp_ball_1s1p_test'},'num_steps':3,'control_hz':60}
    (p/'meta.json').write_text(json.dumps(meta))
    (p/'analyse.jsonl').write_text('{"type":"test"}\n')
    with h5py.File(p/'replay.hdf5','w') as f:
        f['timestamp']=np.arange(3)/60;f['actions']=np.zeros((3,9))
    with h5py.File(p/'trajectory.hdf5','w') as f:f['test']=[1]
    return p


def pack(episode,tmp_path,stage='raw',folder='archive'):
    return pack_episode(episode,cohort='night1',host='local',stage=stage,output=tmp_path/folder)


def add_render(episode,fps=60,indices=None):
    m=json.loads((episode/'meta.json').read_text());m['camera_render']={'camera_fps':fps};(episode/'meta.json').write_text(json.dumps(m))
    (episode/'video').mkdir(exist_ok=True)
    with h5py.File(episode/'video/rendered.hdf5','w') as f:
        f['source_frame_index']=np.arange(3) if indices is None else indices
        f['timestamp_s']=np.arange(3)/60;f['front_rgb']=np.zeros((3,2,2,3),dtype=np.uint8)


def test_deterministic_pack_and_verified_roundtrip_preserve_source(episode,tmp_path):
    before={p.name:sha256(p) for p in episode.iterdir()}
    a,ma=pack(episode,tmp_path)
    b,mb=pack(episode,tmp_path,folder='archive2')
    assert ma==mb and sha256(a/'payload.tar')==sha256(b/'payload.tar')
    assert not ma['training_ready_claimed'] and ma['render_alignment'] is None
    target=unpack_verified(a/'payload.tar',ma,tmp_path/'download')
    assert {p.name:sha256(p) for p in target.iterdir()}==before
    assert {p.name:sha256(p) for p in episode.iterdir()}==before
    assert pack(episode,tmp_path)[1]==ma


@pytest.mark.parametrize('field,value',[('success',False),('fix_joint',True),('grasp_mode','fixedforce')])
def test_failed_or_assisted_episode_rejected(episode,tmp_path,field,value):
    m=json.loads((episode/'meta.json').read_text());m[field]=value;(episode/'meta.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='unassisted'):pack(episode,tmp_path)


def test_unsafe_identity_and_source_links_rejected(episode,tmp_path):
    with pytest.raises(ValueError):pack_episode(episode,cohort='../secret',host='local',stage='raw',output=tmp_path/'archive')
    target=tmp_path/'other';target.write_text('something')
    (episode/'analyse.jsonl').unlink();(episode/'analyse.jsonl').symlink_to(target)
    with pytest.raises(ValueError,match='links'):pack(episode,tmp_path)


def test_modified_stage_never_silently_overwrites(episode,tmp_path):
    pack(episode,tmp_path)
    (episode/'analyse.jsonl').write_text('changed')
    with pytest.raises(RuntimeError,match='immutable'):pack(episode,tmp_path)


def test_only_full_rate_render_is_accepted(episode,tmp_path):
    add_render(episode,fps=30)
    with pytest.raises(ValueError,match='control-rate'):pack(episode,tmp_path,stage='rendered')
    add_render(episode,indices=[0,0,2])
    with pytest.raises(ValueError,match='alignment'):pack(episode,tmp_path,stage='rendered')
    add_render(episode)
    _,m=pack(episode,tmp_path,stage='rendered')
    assert m['render_alignment']['camera_fps']==60
    assert m['render_alignment']['rgb_streams']==['front_rgb']
    assert 'video/rendered.hdf5' in [x['path'] for x in m['files']]
    assert not m['training_ready_claimed']  # alignment is not manual quality acceptance


def test_corrupt_archive_rejected_before_extraction(episode,tmp_path):
    p,m=pack(episode,tmp_path)
    with (p/'payload.tar').open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):unpack_verified(p/'payload.tar',m,tmp_path/'download')
    assert not (tmp_path/'download').exists()


def test_path_traversal_is_rejected(tmp_path):
    p=tmp_path/'bad.tar'
    with tarfile.open(p,'w') as f:
        info=tarfile.TarInfo('../outside');info.size=1;f.addfile(info,io.BytesIO(b'x'))
    m={'schema':'capstone.hf_episode_archive.v1','files':[{'path':'../outside','bytes':1,'sha256':'x'}],
       'payload':{'sha256':sha256(p)}}
    with pytest.raises(ValueError,match='Unsafe'):unpack_verified(p,m,tmp_path/'download')
    assert not (tmp_path/'outside').exists()


def test_public_repo_or_wrong_account_is_rejected():
    with pytest.raises(RuntimeError,match='public'):
        private_repo(SimpleNamespace(repo_info=lambda **kw:SimpleNamespace(private=False)),{'repo_id':'user/repo'})
    with pytest.raises(RuntimeError,match='personal account'):
        upload_package('unused',{'owner':'LeeOoOo3'},api=SimpleNamespace(whoami=lambda:{'name':'other'}))


def test_failed_member_verification_never_leaves_a_completed_directory(episode,tmp_path):
    p,m=pack(episode,tmp_path)
    m['files'][0]['sha256']='0'*64
    with pytest.raises(ValueError,match='member checksum'):
        unpack_verified(p/'payload.tar',m,tmp_path/'download')
    assert not (tmp_path/'download').exists()
    assert not list(tmp_path.glob('.incoming_download_*'))
