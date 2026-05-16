#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  /Users/adb/.local/bin/uv venv .venv
fi

/Users/adb/.local/bin/uv pip install -r requirements.txt --python .venv/bin/python

.venv/bin/python scripts/autonomous_research_system.py \
  --out-dir results \
  --entire-explain inputs/entire_checkpoint_explain.txt \
  --trials 6 \
  --epochs 6
