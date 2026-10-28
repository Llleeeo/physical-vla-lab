#!/usr/bin/env python3
"""Use the frozen SDK adapter and dispatcher, replacing only the worker entry."""
import importlib.util
import os
from pathlib import Path
import runpy
import sys


def main():
    bundle = Path(os.environ['CAPSTONE_FROZEN_BUNDLE']).resolve(strict=True)
    path = bundle / 'scripts/sim/dispatch_hgwind_collection.py'
    spec = importlib.util.spec_from_file_location('capstone_frozen_dispatch_adapter', path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    adapter.install_environment_adapter()
    dispatcher = adapter.REPO / 'Scripts/expert_data/collect.py'
    sys.argv[0] = str(dispatcher)
    namespace = runpy.run_path(str(dispatcher), run_name='capstone_depth_collect')
    namespace['main'].__globals__['WORKER'] = Path(__file__).with_name('run_hgwind_depth_worker.py')
    from sim_common.diagnostics import run_cli
    return run_cli(namespace['main'], str(dispatcher))


if __name__ == '__main__':
    raise SystemExit(main())
