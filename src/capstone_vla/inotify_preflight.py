"""Bounded Linux watcher admission check before launching a native SDK child."""
import ctypes
import errno
import json
import os
from pathlib import Path
import tempfile


def _syscalls():
    library = ctypes.CDLL(None, use_errno=True)
    library.inotify_init1.argtypes = [ctypes.c_int]
    library.inotify_init1.restype = ctypes.c_int
    library.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    library.inotify_add_watch.restype = ctypes.c_int
    return library


def probe_watches(reserve=512):
    if not isinstance(reserve, int) or not 1 <= reserve <= 16384:
        raise ValueError('Watcher probe must request 1..16384 watches')
    result = dict(schema='physical-vla-lab.inotify-preflight.v1',requested=reserve,allocated=0,
        passed=False,errno=None,phase='init',scope='Admission probe only; not an Isaac stability guarantee')
    for key in ['max_user_watches','max_user_instances']:
        path=Path('/proc/sys/fs/inotify')/key
        result[key]=int(path.read_text()) if path.is_file() else None
    library=_syscalls()
    fd=library.inotify_init1(os.O_CLOEXEC|os.O_NONBLOCK)
    if fd < 0:
        result['errno']=ctypes.get_errno()
        return result
    try:
        with tempfile.TemporaryDirectory(prefix='physical-vla-watches-') as directory:
            result['phase']='add_watch'
            for index in range(reserve):
                path=Path(directory)/str(index)
                path.touch()
                # IN_ATTRIB: no file contents are read and no source paths change.
                if library.inotify_add_watch(fd,os.fsencode(path),0x00000004) < 0:
                    result['errno']=ctypes.get_errno()
                    break
                result['allocated']+=1
            result['passed']=result['allocated']==reserve
            result['phase']='complete' if result['passed'] else 'add_watch'
    finally:
        os.close(fd)  # Releases every probe watch on success and failure.
    return result


def require_headroom(reserve=512):
    result=probe_watches(reserve)
    print('INOTIFY_PREFLIGHT '+json.dumps(result,sort_keys=True),flush=True)
    if not result['passed']:
        detail='inotify watch quota exhausted' if result['errno']==errno.ENOSPC else 'inotify resources unavailable'
        raise RuntimeError(detail+'; no simulator launched. Apply project files.watcherExclude and reload the VS Code window, then retry this preflight.')
    return result
