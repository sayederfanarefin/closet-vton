# Portable VTON Eval Bundle

This folder is self-contained for transfer to another machine.

## Contents

- `run_all_comparisons.sh`: sequential runner (all commands in one script).
- `requirements.txt`: Python dependencies.
- `scripts/`: evaluation Python scripts copied from this repo.

## Transfer

Copy the entire `portable_eval_bundle/` folder to your other PC.

Put it inside your project root (same level as `runs/` and `data/`), for example:

```text
your-project/
  runs/
  data/
  portable_eval_bundle/
```

## Setup On GPU PC

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r portable_eval_bundle/requirements.txt
```

## Run Everything

```bash
bash portable_eval_bundle/run_all_comparisons.sh .
```

Outputs will be created under:

`runs/eval_transfer_YYYYMMDD_HHMMSS/`

## Eval Sweep (All Runs)

Run overlap + deepfake + fidelity for *all* run folders found under `runs/`:

```bash
bash portable_eval_bundle/run_eval_sweep.sh . --max-samples 0
```

Add `--visualize` to also create a Markdown report + plots under the sweep output folder.

## Visualize Results

Generate a Markdown report + PNG plots (no pandas/matplotlib required):

```bash
# Pick the latest runs/eval_transfer_* automatically:
python3 portable_eval_bundle/scripts/visualize_eval_transfer.py

# Or point to a specific eval root:
python3 portable_eval_bundle/scripts/visualize_eval_transfer.py \
  --eval-root runs/eval_transfer_YYYYMMDD_HHMMSS
```

Open:

`runs/eval_transfer_YYYYMMDD_HHMMSS/viz/report.md`

## Useful Overrides

Limit sample count (quick smoke run):

```bash
MAX_SAMPLES=50 bash portable_eval_bundle/run_all_comparisons.sh .
```

Enable MediaPipe-based face/pose scoring:

```bash
USE_MEDIAPIPE=1 bash portable_eval_bundle/run_all_comparisons.sh .
```

Override any run selector:

```bash
RUN_TRIPLET=20260205_184710_7968b6_mytest3curt \
RUN_UNPAIRED=20260131_125119_2a93d2_vitonhd_test_unpaired \
bash portable_eval_bundle/run_all_comparisons.sh .
```
