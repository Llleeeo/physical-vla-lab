#!/usr/bin/env bash
set -euo pipefail
umask 077
CAPSTONE_PRIVATE_HF_ROOT=/media/RoboPhys/DataCollection33/Capstone_HGwind_Worker_Yuan_Haoming/hf_private_archive
CAPSTONE_PRIVATE_HF_PYTHON=/media/RoboPhys/DataCollection33/Capstone_HGwind_Worker_Yuan_Haoming/cpu_env/bin/python
export HF_HOME=/home/RoboPhys/.config/capstone_hf_archive
export HF_HUB_CACHE="$CAPSTONE_PRIVATE_HF_ROOT/runtime/hub_cache"
export HF_XET_CACHE="$CAPSTONE_PRIVATE_HF_ROOT/runtime/xet_cache"
export HF_TOKEN_PATH="$HF_HOME/token"
export PYTHONDONTWRITEBYTECODE=1
# Keep the other student's login/cache and any inherited tokens untouched.
# These overrides affect only this project's child process.
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
mkdir -p "$HF_HOME"
case "${1:-help}" in
  login)
    exec "$CAPSTONE_PRIVATE_HF_PYTHON" -c '
import getpass
from huggingface_hub import HfApi, login
capstone_token = getpass.getpass("HF token for LeeOoOo3 (hidden): ")
capstone_name = HfApi(token=capstone_token).whoami().get("name")
if capstone_name != "LeeOoOo3":
    raise SystemExit("Wrong HF account; project login was not saved")
login(token=capstone_token, add_to_git_credential=False)
print("PROJECT_HF_LOGIN_OK account=LeeOoOo3 git_credentials_changed=false")
'
    ;;
  whoami)
    exec "$CAPSTONE_PRIVATE_HF_PYTHON" -c 'from huggingface_hub import HfApi; print(HfApi().whoami()["name"])'
    ;;
  help)
    printf '%s\n' 'login | whoami | pack ... | upload ... --run | download ...'
    ;;
  *)
    exec "$CAPSTONE_PRIVATE_HF_PYTHON" "$CAPSTONE_PRIVATE_HF_ROOT/scripts/hf_private_archive.py" "$@"
    ;;
esac
