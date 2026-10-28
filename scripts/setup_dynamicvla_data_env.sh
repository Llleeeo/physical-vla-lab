#!/usr/bin/env bash
# Isolated model/data environment; never invokes Isaac or a real robot.
set -euo pipefail
capstone_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
capstone_env="$capstone_root/.venv_dynamicvla"
capstone_upstream="$capstone_root/third_party/DynamicVLA"
capstone_pin=434dc7105cf6d559018d15af5fecbcdad44f036e
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
if [[ ! -d "$capstone_upstream" ]]; then
    mkdir -p "$capstone_root/third_party"
    git init "$capstone_upstream"
    git -C "$capstone_upstream" remote add origin https://github.com/hzxie/DynamicVLA.git
    git -C "$capstone_upstream" fetch --depth 1 origin "$capstone_pin"
    git -C "$capstone_upstream" checkout --detach FETCH_HEAD
fi
if [[ "$(git -C "$capstone_upstream" rev-parse HEAD)" != "$capstone_pin" ]]; then
    echo 'Existing upstream checkout has a different revision; refusing to change it.' >&2
    exit 2
fi
if [[ ! -x "$capstone_env/bin/python" ]]; then
    "$capstone_root/data/derived/hf_pilot_render_repo/.venv/bin/python" -m venv "$capstone_env"
fi
"$capstone_env/bin/python" -m pip install --upgrade pip
"$capstone_env/bin/python" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
capstone_requirements="$capstone_root/data/derived/task2_dynamicvla_development_20261025/requirements_without_memcached.txt"
"$capstone_env/bin/python" - "$capstone_upstream/requirements.txt" "$capstone_requirements" <<'FILTER_REQUIREMENTS'
from pathlib import Path
import re
import sys
source=Path(sys.argv[1]).read_text().splitlines()
selected=[line for line in source if re.split(r'[<>=!~\[;\s]',line.strip(),maxsplit=1)[0].lower()!='pylibmc']
if len(source)-len(selected)!=1:
    raise RuntimeError('Expected exactly one optional pylibmc requirement in pinned upstream')
Path(sys.argv[2]).write_text('# Capstone cache-disabled profile; upstream requirements unchanged.\n'+'\n'.join(selected)+'\n')
FILTER_REQUIREMENTS
"$capstone_env/bin/python" -m pip install -c "$capstone_root/configs/dynamicvla_data_constraints.txt" -r "$capstone_requirements" torchcodec==0.5.0
"$capstone_env/bin/python" -m pip check
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
"$capstone_env/bin/python" "$capstone_root/scripts/smoke_task2_upstream_loader.py"
"$capstone_env/bin/python" -m pip freeze > "$capstone_root/data/derived/task2_dynamicvla_development_20261025/model_env_freeze.txt"
echo 'DONE: isolated environment + actual upstream data-reader smoke; no model weights downloaded, no training started.'
