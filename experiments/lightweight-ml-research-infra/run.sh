#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  /Users/adb/.local/bin/uv venv .venv
fi

/Users/adb/.local/bin/uv pip install -r requirements.txt --python .venv/bin/python

.venv/bin/python scripts/lightweight_ml_suite.py --out-dir results --epochs 12
.venv/bin/python scripts/torch_ml_suite.py --out-dir results --epochs 8
