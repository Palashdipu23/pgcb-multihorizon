#!/usr/bin/env bash
# Reproduces every table and figure of the paper.
#   data audit -> train all models (5 seeds, 3 test years, 4 horizons) -> ablation -> long-lag switch
#   -> tables / MCS / DM tests / figures -> grouped importance -> parameter counts
# Usage:  bash scripts/run_all.sh                (full run, GPU recommended)
#         bash scripts/run_all.sh --smoke        (quick pipeline check)
#         bash scripts/run_all.sh --horizons 24  (one horizon only)
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/run_all.py "$@"
