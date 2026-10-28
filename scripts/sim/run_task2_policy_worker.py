#!/usr/bin/env python3
import os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from capstone_vla.closed_loop_patch import patch_policy_worker
worker=Path(os.environ['CAPSTONE_TASK2_WORKER']);source=patch_policy_worker(worker.read_text())
if os.environ.get('CAPSTONE_DIAGNOSTIC_COMPONENT_SOURCES'):
 from capstone_vla.component_controls import annotate_component_worker
 source=annotate_component_worker(source,json.loads(os.environ['CAPSTONE_DIAGNOSTIC_COMPONENT_SOURCES']))
exec(compile(source,str(worker),'exec'),{'__name__':'__main__','__file__':str(worker)})
