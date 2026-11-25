#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash portable_eval_bundle/run_eval_sweep.sh /path/to/repo [extra args...]
#
# Example:
#   bash portable_eval_bundle/run_eval_sweep.sh . --max-samples 50 --visualize

REPO_ROOT="${1:-$(pwd)}"
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" "${BUNDLE_DIR}/scripts/run_eval_sweep.py" \
  --repo-root "${REPO_ROOT}" \
  "${@:2}"

