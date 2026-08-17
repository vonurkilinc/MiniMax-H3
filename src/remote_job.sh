#!/usr/bin/env bash
# Runs ON the server for a single job. Pushed to /dev/shm/h3-job/code/ by run-remote.ps1.
# Everything lives under /dev/shm/h3-job; run-remote.ps1 deletes it after the run.
set -uo pipefail

JOB_DIR=/dev/shm/h3-job
PY="$HOME/h3-env/bin/python"

mkdir -p "$JOB_DIR/out"
export HF_HUB_OFFLINE=1
export H3_MODEL_DIR="${H3_MODEL_DIR:-$HOME/models/minimax-h3}"
export PYTHONUNBUFFERED=1

if [ ! -x "$PY" ]; then
    echo "ERROR: $PY not found. Run: .\\run-remote.ps1 -Provision" | tee "$JOB_DIR/out/run.log"
    exit 2
fi
if [ ! -d "$H3_MODEL_DIR/transformer" ]; then
    echo "ERROR: model not found at $H3_MODEL_DIR. Run: .\\run-remote.ps1 -DownloadModel" | tee "$JOB_DIR/out/run.log"
    exit 2
fi

"$PY" "$JOB_DIR/code/generate.py" --config "$JOB_DIR/in/config.json" --job-dir "$JOB_DIR" 2>&1 | tee "$JOB_DIR/out/run.log"
rc=${PIPESTATUS[0]}
echo "$rc" > "$JOB_DIR/out/exit_code.txt"
exit "$rc"
