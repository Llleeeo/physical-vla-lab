"""Generate a process-local GDB wrapper; never attach to other processes."""
import hashlib
import json
from pathlib import Path
import shlex


def _path(value):
    text = str(Path(value).resolve())
    if any(c in text for c in ['\n', '\r', '\x00']):
        raise ValueError('Debugger paths must be single-line paths')
    return text


def write_debugger(output, python_executable, gdb_executable):
    out = Path(_path(output))
    python = _path(python_executable); debugger = _path(gdb_executable)
    if not Path(python).is_file() or not Path(debugger).is_file():
        raise ValueError('Debugger or child interpreter missing')
    commands = out / 'native_debugger.gdb'
    wrapper = out / 'native_debugger_python.sh'
    backtrace = out / 'native_backtrace.log'
    if any(p.exists() for p in [commands, wrapper, backtrace]):
        raise ValueError('Native debugger output already exists')
    text = r"""set pagination off
set confirm off
set print thread-events off
set disable-randomization off
set debuginfod enabled off
handle SIGPIPE nostop noprint pass
handle SIGUSR1 nostop noprint pass
handle SIGUSR2 nostop noprint pass
handle SIGABRT stop print pass
python
import os
for native_key, saved_key in [('LD_PRELOAD', 'CAPSTONE_NATIVE_CHILD_PRELOAD'), ('LD_LIBRARY_PATH', 'CAPSTONE_NATIVE_CHILD_LIBRARY_PATH'), ('PYTHONPATH', 'CAPSTONE_NATIVE_CHILD_PYTHONPATH'), ('PYTHONHOME', 'CAPSTONE_NATIVE_CHILD_PYTHONHOME')]:
    value = os.environ.get(saved_key, '')
    if '\n' in value or '\r' in value:
        raise ValueError('Multiline child library environment')
    if value:
        gdb.execute('set environment ' + native_key + ' ' + value)
    else:
        gdb.execute('unset environment ' + native_key)
end
run
python
if gdb.selected_inferior().pid:
    text = gdb.execute('bt 40', to_string=True)
    text += '\nALL_THREADS\n' + gdb.execute('thread apply all bt 4', to_string=True)
    with open(BACKTRACE_PATH, 'w') as handle:
        handle.write(text)
    print('CAPSTONE_NATIVE_BACKTRACE path=' + BACKTRACE_PATH)
    print(text)
else:
    print('CAPSTONE_NATIVE_DEBUGGER child_exited')
end
quit
""".replace('BACKTRACE_PATH', json.dumps(str(backtrace)))
    commands.write_text(text)
    # python.sh establishes SDK libraries first. Keep those for the inferior,
    # while the system debugger itself uses its own normal system libraries.
    shell = ('#!/bin/sh\nexec /usr/bin/env -u LD_PRELOAD -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME '
        'CAPSTONE_NATIVE_CHILD_PRELOAD="${LD_PRELOAD-}" '
        'CAPSTONE_NATIVE_CHILD_LIBRARY_PATH="${LD_LIBRARY_PATH-}" '
        'CAPSTONE_NATIVE_CHILD_PYTHONPATH="${PYTHONPATH-}" '
        'CAPSTONE_NATIVE_CHILD_PYTHONHOME="${PYTHONHOME-}" '
        + shlex.quote(debugger) + ' --batch --return-child-result --command='
        + shlex.quote(str(commands)) + ' --args ' + shlex.quote(python) + ' "$@"\n')
    wrapper.write_text(shell); wrapper.chmod(0o700)
    return dict(wrapper=str(wrapper), commands=str(commands), backtrace=str(backtrace),
        python_executable=python, gdb_executable=debugger,
        wrapper_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest(),
        commands_sha256=hashlib.sha256(commands.read_bytes()).hexdigest())
