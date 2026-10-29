"""Watcher shortages must fail before an expensive native process is launched."""
import ctypes
import errno
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
from capstone_vla import inotify_preflight as watchers


@pytest.mark.parametrize('failure_at',[None,2])
def test_probe_releases_all_watches_and_temporary_files(monkeypatch,failure_at):
    actual_read,actual_write=os.pipe();paths=[];calls=[];closed=[]
    def add(fd,path,mask):
        assert fd==actual_read and mask==4
        paths.append(Path(os.fsdecode(path)))
        assert paths[-1].is_file()
        calls.append(1)
        if failure_at is not None and len(calls)>failure_at:
            ctypes.set_errno(errno.ENOSPC);return -1
        return len(calls)
    monkeypatch.setattr(watchers,'_syscalls',lambda:SimpleNamespace(inotify_init1=lambda flags:actual_read,inotify_add_watch=add))
    original_close=os.close
    def close(fd):closed.append(fd);original_close(fd)
    monkeypatch.setattr(watchers.os,'close',close)
    try:
        result=watchers.probe_watches(4)
        assert result['passed'] is (failure_at is None)
        assert result['allocated']==(4 if failure_at is None else failure_at)
        assert result['errno']==(None if failure_at is None else errno.ENOSPC)
        assert actual_read in closed and all(not p.exists() for p in paths)
        with pytest.raises(OSError):os.fstat(actual_read)
    finally:original_close(actual_write)


def test_init_failure_and_invalid_probe_do_not_allocate_files(monkeypatch):
    def init(flags):ctypes.set_errno(errno.EMFILE);return -1
    monkeypatch.setattr(watchers,'_syscalls',lambda:SimpleNamespace(inotify_init1=init))
    result=watchers.probe_watches(1)
    assert result['allocated']==0 and result['phase']=='init' and result['errno']==errno.EMFILE
    with pytest.raises(ValueError):watchers.probe_watches(0)


def test_shortage_raises_actionable_prelaunch_error(monkeypatch):
    monkeypatch.setattr(watchers,'probe_watches',lambda n:{'passed':False,'errno':errno.ENOSPC,'allocated':12})
    with pytest.raises(RuntimeError,match='no simulator launched'):watchers.require_headroom()
