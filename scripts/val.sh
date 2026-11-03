#!/usr/bin/env bash
set -euo pipefail
project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$project/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=131072
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec /usr/bin/python3 -m perception.evaluation.offline "$@"
