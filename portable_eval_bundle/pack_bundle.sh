#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "${THIS_DIR}/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_TAR="${PARENT_DIR}/portable_eval_bundle_${STAMP}.tar.gz"

tar -czf "${OUT_TAR}" -C "${PARENT_DIR}" "portable_eval_bundle"
echo "Created: ${OUT_TAR}"
