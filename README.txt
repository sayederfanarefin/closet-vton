ClosetVTON: reproducible two-photo virtual try-on

Code: https://github.com/sayederfanarefin/closet-vton
Data: https://www.kaggle.com/datasets/sayederfanarefin/closetvton-final-cascade-evaluation
Use Kaggle version 2. Visibility remains private unless changed by the owner.

Contents
  serve_workflow.py, workflow_api.json   Final API and evaluated workflow
  comfy_custom_nodes/crop_integrity/    Mask cascade, safe composition, crop validation
  evaluation/                          Paired scoring and API evaluation runner
  portable_eval_bundle/scripts/        Original pairwise and four-model statistics
  scripts/                             Data verification, preparation, summaries and plots
  tests/                               Reported-metric and mask/crop regression tests
  provenance/                          Recorded versions, policy and compatibility patch
  artifact_maps/                       Figure/table/result crosswalk

1. Install
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.dev.txt
  python -m pytest tests -q
Choose torch/torchvision builds appropriate for your GPU. Recorded Python, torch,
CUDA, package inventory and custom-node commits are in provenance/. The dependency
lists are not a complete frozen environment lock.

2. Obtain data
Download the Kaggle dataset. Its extracted data root is
ClosetVTON_Data_v2/Data/. In commands below replace /path/to/Data with that folder.
The data contains accepted predictions, exact pair manifests, 5823 attempt records,
per-pair metrics and figure assets. Source photos and weights are obtained separately.
  python scripts/verify_release.py --data-root /path/to/Data

3. Reproduce numerical summaries and quantitative figures without new inference
  python scripts/reproduce_summary.py --data-root /path/to/Data --outdir results/summary
  python scripts/plot_results.py --data-root /path/to/Data --outdir results/figures
  python portable_eval_bundle/scripts/generate_nonstable_paper_report.py --eval-root /path/to/Data/metrics/comparison --outdir results/diagnostics --top-k-qualitative 0
The first command checks completion, all five paired means, and client latency against
released values. The last command uses the original four-model normalization/statistics.
Use a new output directory for each run. Data and archived results are never overwritten.

4. Rescore the saved paired images
Obtain VITON-HD separately; /path/to/zalando-hd-resized must contain test/.
  python scripts/prepare_data.py --data-root /path/to/Data --source-root /path/to/zalando-hd-resized --output-dir prepared_manifests
  python -m evaluation.score_folder --manifest prepared_manifests/paired.jsonl --pred-dir /path/to/Data/predictions/closetvton/paired --outdir results/rescored --lpips --lpips-net alex
The prepared overlap CSVs can be passed to run_fidelity_comparison.py and
run_deepfake_comparison.py (--overlap-csv and --outdir). Both use the original
metric implementations. Preserve the original optional detector settings when comparing
scores; changing face/pose detection changes the proxy protocol.

5. Generate new predictions (GPU and separate model downloads required)
  docker build -f Dockerfile.gpu -t closetvton .
  mkdir -p models input output
  docker run --rm -v "$PWD/models:/opt/ComfyUI/models" --entrypoint python3 closetvton /app/prepare_models.py
Set API_USER and API_PASS in your shell, then:
  docker run --rm --gpus all -p 8000:8000 -p 8188:8188 -e API_USER -e API_PASS -v "$PWD/models:/opt/ComfyUI/models" -v "$PWD/input:/opt/ComfyUI/input" -v "$PWD/output:/opt/ComfyUI/output" closetvton
  python -m evaluation.run --help
Use the released manifests and save every new evaluation to a new output directory.
The Dockerfile pins the recorded source commits and applies the recorded compatibility
patch. This cleanup does not rebuild the GPU image or promise pixel-identical fresh renders.

Interpretation and limits
Final accepted outputs = 2032 paired + 2032 unpaired + 1000 worn-reference.
These retain accepted outputs and add successful retries, not a fresh first-pass cohort.
Acceptance checks structure/person/reference mismatch; it does not prove garment fidelity.
Latency tables separate client and service measurements, accepted and cumulative attempts.
Do not mix pairwise and four-model pooled artifact z-scores.
Qualitative reconstruction requires external VITON-HD source photographs:
  python scripts/build_paired_figure.py --data-root /path/to/Data --source-root /path/to/zalando-hd-resized --outdir results/paired_figure
  python scripts/build_worn_figure.py --data-root /path/to/Data --source-root /path/to/zalando-hd-resized --outdir results/worn_figure
The ten outside-cohort predictions needed by the worn illustration are provided in
Data/qualitative_archive/. They are not counted as evaluation cases. Both rebuilt figures
have been checked for exact pixel equality with the released images.
For the architecture drawing, install TeX Live with TikZ and the standalone class:
  mkdir -p results/diagram
  pdflatex -output-directory results/diagram diagrams/architecture.tex
Detailed figure/table/result mapping: artifact_maps/README_RESULTS_MAP.txt (also CSV).
Project code uses the existing Apache-2.0 LICENSE. Third-party code, models and dataset
imagery retain their own terms; see THIRD_PARTY_NOTICES.txt and the dataset TERMS.txt.
