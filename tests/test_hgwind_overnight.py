from pathlib import Path
import importlib.util
import shlex
from unittest.mock import Mock

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('overnight_entry',ROOT/'scripts/collect_hgwind_overnight.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


def test_local_depth_and_remote_reviewed_allocations_are_explicit():
    local=entry.worker_command('local',run=True)
    remote=shlex.split(entry.worker_command('hongtao',run=True)[-1])
    assert local[local.index('--depth-m')+1]=='0.002'
    assert remote[remote.index('--depth-m')+1]=='0.0'
    for cmd in (local,remote):
        assert cmd[cmd.index('--target-successes')+1]=='50'
        assert cmd[cmd.index('--max-attempts')+1]=='500'
        assert '--run' in cmd
    assert 'DataCollection28' not in ' '.join(remote)
    assert '/media/RoboPhys/DataCollection33/' in ' '.join(remote)


def test_background_detaches_and_refuses_duplicate_launch(tmp_path,monkeypatch):
    monkeypatch.setattr(entry,'BASE',tmp_path)
    monkeypatch.setattr(entry,'validate_smoke',lambda:None)
    popen=Mock(return_value=Mock(pid=43210));monkeypatch.setattr(entry.subprocess,'Popen',popen)
    assert entry.start_background()==0
    assert popen.call_args.kwargs['start_new_session'] is True
    assert popen.call_args.kwargs['stdin']==entry.subprocess.DEVNULL
    assert popen.call_args.args[0][-1]=='--run'
    monkeypatch.setattr(entry.os,'kill',lambda pid,sig:None)
    with pytest.raises(RuntimeError,match='already running'):entry.start_background()
    assert popen.call_count==1


def test_saved_smoke_receipts_still_match_exact_frozen_base():
    entry.validate_smoke()
