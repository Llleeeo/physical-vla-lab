#!/usr/bin/env python3
"""Execute the isolated collector with a capstone-owned robot control bridge."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from capstone_vla.task2_bridge import patch_worker
worker=Path(os.environ['CAPSTONE_TASK2_WORKER'])
source=patch_worker(worker.read_text())
if os.environ.get('CAPSTONE_NATIVE_ROTATION_PROBE')=='1':
    from capstone_vla.native_rotation_probe import instrument_worker_source
    source=instrument_worker_source(source)
# Retain original __file__ so bootstrap, script resources and relative paths
# resolve exactly as in the isolated worker. No worker file is rewritten.
exec(compile(source,str(worker),'exec'),{'__name__':'__main__','__file__':str(worker)})
