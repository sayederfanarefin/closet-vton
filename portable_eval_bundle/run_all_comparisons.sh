#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash portable_eval_bundle/run_all_comparisons.sh /path/to/repo
# If repo path is omitted, current working directory is used.

REPO_ROOT="${1:-$(pwd)}"
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="${BUNDLE_DIR}/scripts"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d "${REPO_ROOT}/runs" ]]; then
  echo "ERROR: '${REPO_ROOT}/runs' not found. Run this from your project root or pass it as arg 1."
  exit 1
fi

# Default run selectors (override with env vars if needed).
RUN_PAIRED="${RUN_PAIRED:-20260131_055842_a14987_vitonhd_test_paired}"
RUN_UNPAIRED="${RUN_UNPAIRED:-20260131_125119_2a93d2_vitonhd_test_unpaired}"
RUN_MYTEST="${RUN_MYTEST:-20260205_053833_e3eb3e_mytest}"
RUN_MYTEST2="${RUN_MYTEST2:-20260205_120832_46c2e4_mytest2}"
RUN_TRIPLET="${RUN_TRIPLET:-20260205_184710_7968b6_mytest3curt}"

DEEPFAKE_DETECTORS="${DEEPFAKE_DETECTORS:-fft_peakiness,jpeg_blockiness,noise_inconsistency,chroma_misalignment}"
MAX_SAMPLES="${MAX_SAMPLES:-0}" # 0 means full overlap.
USE_MEDIAPIPE="${USE_MEDIAPIPE:-0}" # set 1 to use --use-mediapipe-face/pose.

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${REPO_ROOT}/runs/eval_transfer_${STAMP}"
mkdir -p "${OUT_ROOT}"

MAX_ARGS=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then
  MAX_ARGS=(--max-samples "${MAX_SAMPLES}")
fi

FIDELITY_EXTRA=()
if [[ "${USE_MEDIAPIPE}" == "1" ]]; then
  FIDELITY_EXTRA=(--use-mediapipe-face --use-mediapipe-pose)
fi

echo "Repo root: ${REPO_ROOT}"
echo "Output root: ${OUT_ROOT}"
echo "Python: ${PYTHON_BIN}"

echo ""
echo "1) Building overlap manifests..."
"${PYTHON_BIN}" "${SCRIPTS_DIR}/build_overlap_manifest.py" \
  --runs-root "${REPO_ROOT}/runs" \
  --run-a "${RUN_PAIRED}" \
  --run-b "${RUN_MYTEST}" \
  --out "${OUT_ROOT}/overlap_paired_vs_mytest.csv"

"${PYTHON_BIN}" "${SCRIPTS_DIR}/build_overlap_manifest.py" \
  --runs-root "${REPO_ROOT}/runs" \
  --run-a "${RUN_UNPAIRED}" \
  --run-b "${RUN_MYTEST2}" \
  --out "${OUT_ROOT}/overlap_unpaired_vs_mytest2.csv"

"${PYTHON_BIN}" "${SCRIPTS_DIR}/build_overlap_manifest.py" \
  --runs-root "${REPO_ROOT}/runs" \
  --run-a "${RUN_TRIPLET}" \
  --run-b "${RUN_UNPAIRED}" \
  --out "${OUT_ROOT}/overlap_triplet_vs_unpaired.csv"

run_pair_eval() {
  local name="$1"
  local overlap_csv="$2"
  local out_dir="${OUT_ROOT}/${name}"
  mkdir -p "${out_dir}"

  echo ""
  echo "2) Running deepfake detectors for ${name}..."
  if [[ "${#MAX_ARGS[@]}" -gt 0 ]]; then
    "${PYTHON_BIN}" "${SCRIPTS_DIR}/run_deepfake_comparison.py" \
      --overlap-csv "${overlap_csv}" \
      --outdir "${out_dir}/deepfake" \
      --detectors "${DEEPFAKE_DETECTORS}" \
      "${MAX_ARGS[@]}"
  else
    "${PYTHON_BIN}" "${SCRIPTS_DIR}/run_deepfake_comparison.py" \
      --overlap-csv "${overlap_csv}" \
      --outdir "${out_dir}/deepfake" \
      --detectors "${DEEPFAKE_DETECTORS}"
  fi

  echo "3) Running cloth/face/posture fidelity for ${name}..."
  if [[ "${#MAX_ARGS[@]}" -gt 0 && "${#FIDELITY_EXTRA[@]}" -gt 0 ]]; then
    "${PYTHON_BIN}" "${SCRIPTS_DIR}/run_fidelity_comparison.py" \
      --overlap-csv "${overlap_csv}" \
      --outdir "${out_dir}/fidelity" \
      "${MAX_ARGS[@]}" \
      "${FIDELITY_EXTRA[@]}"
  elif [[ "${#MAX_ARGS[@]}" -gt 0 ]]; then
    "${PYTHON_BIN}" "${SCRIPTS_DIR}/run_fidelity_comparison.py" \
      --overlap-csv "${overlap_csv}" \
      --outdir "${out_dir}/fidelity" \
      "${MAX_ARGS[@]}"
  elif [[ "${#FIDELITY_EXTRA[@]}" -gt 0 ]]; then
    "${PYTHON_BIN}" "${SCRIPTS_DIR}/run_fidelity_comparison.py" \
      --overlap-csv "${overlap_csv}" \
      --outdir "${out_dir}/fidelity" \
      "${FIDELITY_EXTRA[@]}"
  else
    "${PYTHON_BIN}" "${SCRIPTS_DIR}/run_fidelity_comparison.py" \
      --overlap-csv "${overlap_csv}" \
      --outdir "${out_dir}/fidelity"
  fi
}

run_pair_eval "paired_vs_mytest" "${OUT_ROOT}/overlap_paired_vs_mytest.csv"
run_pair_eval "unpaired_vs_mytest2" "${OUT_ROOT}/overlap_unpaired_vs_mytest2.csv"
run_pair_eval "triplet_vs_unpaired" "${OUT_ROOT}/overlap_triplet_vs_unpaired.csv"

echo ""
echo "Done."
echo "All outputs are in: ${OUT_ROOT}"
