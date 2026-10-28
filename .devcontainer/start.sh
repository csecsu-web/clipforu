#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
if pgrep -f '[.]venv/bin/python app.py' >/dev/null; then
  exit 0
fi
nohup .venv/bin/python app.py >/tmp/clipforu.log 2>&1 </dev/null &
