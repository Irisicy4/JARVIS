#!/bin/bash
# Deadline helper: every 5 min regenerate the board CSV and push if it
# changed, so each arm lands on the board as soon as it completes.
# Usage: watch_and_push.sh <token> [minutes]
set -uo pipefail
TOKEN=$1
MINS=${2:-180}
REPO=/raid/icy/exp-record
DATA=$REPO/projects/hugginggpt-encoding/data
END=$(( $(date +%s) + MINS*60 ))
while [ "$(date +%s)" -lt "$END" ]; do
  python3 "$DATA/gen_results.py" >/dev/null 2>&1
  cd "$REPO"
  if ! git diff --quiet -- projects/hugginggpt-encoding/data/eval_results.csv; then
    git add projects/hugginggpt-encoding/data/eval_results.csv
    git -c user.name=IcyWang -c user.email=irisicy@outlook.com \
        commit -q -m "hugginggpt-encoding: board refresh ($(date -u +%H:%MZ)) — new arm results"
    git pull --rebase --autostash "https://irisicy4:${TOKEN}@github.com/williamium3000/exp-record.git" main >/dev/null 2>&1
    git push "https://irisicy4:${TOKEN}@github.com/williamium3000/exp-record.git" main >/dev/null 2>&1 \
      && echo "$(date -u +%H:%M:%SZ) pushed board update" \
      || echo "$(date -u +%H:%M:%SZ) push failed"
    rm -f .git/FETCH_HEAD
  fi
  sleep 300
done
echo "watcher finished"
