#!/bin/bash
# Launch the 12 Table-4 CI repeats with bounded concurrency (default 4)
# to keep load on the shared :8001 controller modest while the SpAgent
# campaign is active. Each run resumes/retries failed items on relaunch
# (run_blink.py is resumable), so re-executing this script backfills.
set -uo pipefail
ROOT=/raid/icy/jarvis-cathy
CONC=${1:-4}
JOBS=()
for arm in singleround det_image_only det_image_and_text det_text_only; do
  for k in 1 2 3; do
    JOBS+=("$arm $k")
  done
done
printf '%s\n' "${JOBS[@]}" | xargs -P "$CONC" -I{} bash -c '
  set -- {}
  arm=$1; k=$2
  d='"$ROOT"'/experiments/sandboxes/${arm}_r${k}
  echo "[$(date +%H:%M:%S)] start ${arm}_r${k}"
  bash '"$ROOT"'/experiments/run_ci.sh "$arm" "$k" >> "$d/logs/run.log" 2>&1
  echo "[$(date +%H:%M:%S)] done ${arm}_r${k} (exit $?)"
'
echo "fleet finished"
