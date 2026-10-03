#!/bin/bash
set -eu
task_user=mhanaa_vision_eng_1987
task_root=/home/$task_user/klara-training-1000
task_python=/home/$task_user/catvton-env/bin/python
cd "$task_root"
# Data acquisition is CPU/network only; leave the studio online during preparation.
runuser -u "$task_user" -- timeout --kill-after=30s 15m "$task_python" -u prepare_final_1000_data.py
runuser -u "$task_user" -- "$task_python" evaluate_final_1000.py --preflight
studio_was_active=0
if systemctl is-active --quiet klara-studio; then studio_was_active=1; fi
restore_studio() {
  result=$?
  trap - EXIT
  printf '%s\n' "$result" > "$task_root/final-evaluation-exit-code.txt"
  if [ "$studio_was_active" = 1 ]; then systemctl start klara-studio; fi
  exit "$result"
}
trap restore_studio EXIT
systemctl stop klara-studio
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo 'Another GPU process is running; final evaluation not started.' >&2
  exit 1
fi
runuser -u "$task_user" -- env HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
  timeout --signal=TERM --kill-after=30s 30m "$task_python" -u evaluate_final_1000.py
