#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 scripts/join_logs.py \
  --inputs inputs \
  --results results
