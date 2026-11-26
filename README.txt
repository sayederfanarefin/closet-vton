ClosetVTON — standalone code accompanying the paper
Code repository: https://github.com/sayederfanarefin/closet-vton
Paper repository: https://github.com/sayederfanarefin/Saadman-Paper-tryout

This folder contains renderer/API, final workflow, local adaptive segmentation/crop nodes,
evaluation scripts and focused tests. It excludes model weights, dataset images, credentials,
unrelated RAG/application code, raw outputs and build caches.

Layout:
  serve_workflow.py, workflow_api.json: final two-photo service and cascade.
  comfy_custom_nodes/crop_integrity/: guards, fallback masks, vendored SCHP (with license).
  evaluation/: paired PSNR/SSIM/LPIPS and masked metrics, API evaluation runner.
  portable_eval_bundle/: original pairwise fidelity and artifact statistical code.
  provenance/: exact observed commits, compatibility patch, package inventory and policy.
  tests/: focused guard/fallback/LPIPS checks.

Data lives separately in the Kaggle Data export; keep it next to Code or pass its root explicitly.
  python scripts/prepare_data.py --data-root ../Data --output-dir prepared_manifests
  python scripts/verify_release.py --release-root .
  python -m pytest tests/test_crop_integrity.py tests/test_segmentation_fallback.py tests/test_segmentation_rescue.py -q

For mask/LPIPS scoring: install a compatible torch/torchvision pair, then requirements.txt.
Pairwise comparison dependencies: portable_eval_bundle/requirements.txt.
The observed Python/torch/CUDA versions and full package inventory are in provenance/;
requirements.txt alone is not a frozen environment lock.

GPU service:
  docker build -f Dockerfile.gpu -t closetvton-paper .
  mkdir -p models input output
  docker run --rm -v "$PWD/models:/opt/ComfyUI/models" --entrypoint python3 closetvton-paper /app/prepare_models.py
  docker run --rm --gpus all -p 8000:8000 -p 8188:8188 -e API_USER -e API_PASS -v "$PWD/models:/opt/ComfyUI/models" -v "$PWD/input:/opt/ComfyUI/input" -v "$PWD/output:/opt/ComfyUI/output" closetvton-paper
Set API_USER and API_PASS in your shell before starting the service. No secrets are supplied.
Checkpoints are fetched separately; comply with their individual access and license terms.
The exported Dockerfile pins all six recorded git checkouts and applies the exact recorded BERT patch.
It has not been rebuilt as part of packaging, and fresh dependency resolution is not guaranteed bit-identical.

Original scoring commands:
  python -m evaluation.score_folder --help
  python -m evaluation.run --help
  python portable_eval_bundle/scripts/run_fidelity_comparison.py --help
  python portable_eval_bundle/scripts/run_deepfake_comparison.py --help
  python portable_eval_bundle/scripts/generate_nonstable_paper_report.py --help
Historical bulk shell runners reference original run layouts; use prepared manifests and explicit paths on a new machine.

No inference is performed by export/verification. The final reported cohort is retained outputs plus successful retries.
The existing repository LICENSE specifies Apache-2.0 for project code; third-party components retain their own terms. See THIRD_PARTY_NOTICES.txt.

Detailed paper-to-data/code README: artifact_maps/README_RESULTS_MAP.txt
Machine-readable crosswalk: artifact_maps/artifact_map.csv
C: paths in that README refer to this repository root; P: paths refer to the paper repository.
