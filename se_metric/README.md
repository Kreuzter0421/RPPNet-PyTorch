# SE Metric

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**[English](README.md) | [中文](README.zh.md)**

Structure Error (SE) metric for symbolic music evaluation.

## Installation

```bash
pip install -e .
```

Requirements: Python >= 3.8, `miditoolkit`, `numpy`, `tqdm`.

## Data Preparation

Place MIDI files (`.mid`) in the `data/` directory:

```
data/
├── ground_truth/     # Reference melodies
└── generated/        # Model-generated melodies
```

For batch evaluation (multiple generation runs):

```
data/
├── ground_truth/
└── generated_model/
    ├── batch_0/
    │   ├── 0.mid
    │   ├── 1.mid
    │   └── ...
    ├── batch_1/
    │   └── ...
    └── ...
```

When evaluating **multiple models** against the same ground truth, place each model under `data/generated/`:

```
data/
├── ground_truth/
└── generated/
    ├── model_A/
    │   ├── batch_0/
    │   ├── batch_1/
    │   └── ...
    └── model_B/
        ├── batch_0/
        ├── batch_1/
        └── ...
```

The first instrument track (`midi.instruments[0]`) is used for evaluation.
Default settings assume 4/4 time at 1920 ticks per bar.

## Usage

### Basic evaluation

```python
from se_metric import compute_structure_error, list_midi_files

gt_files  = list_midi_files("data/ground_truth")
gen_files = list_midi_files("data/generated")

se = compute_structure_error(gt_files, gen_files)
print(f"SE = {se:.6f}")   # lower is better
```

Run the example script:

```bash
python examples/example.py
```

### Batch evaluation

When evaluating multiple batches against the same ground truth, precompute the
GT similarity curve to avoid redundant computation:

```python
from se_metric.core import compute_similarity_curve
from se_metric import compute_structure_error, list_midi_files

gt_files = list_midi_files("data/ground_truth")
gen_files = list_midi_files("data/generated_model/batch_0")

L_gt = compute_similarity_curve(gt_files)

for batch in ["batch_0", "batch_1", "batch_2"]:
    gen_files = list_midi_files(f"data/generated_model/{batch}")
    se = compute_structure_error(gt_files, gen_files, precomputed_gt_curve=L_gt)
    print(f"{batch}: SE = {se:.6f}")
```

Run the batch example:

```bash
python examples/evaluate_batches.py
```

### Compare multiple models

To evaluate several models at once (each model may contain multiple `batch_*` subdirectories), use the provided comparison script:

```bash
python examples/compare_models.py
```

This script will:
1. Compute the ground-truth similarity curve once.
2. For every model directory under `data/generated/`, compute SE for each `batch_*` subdirectory.
3. Print per-batch SE and summarize each model as **Mean SE ± Std**.

Example output:

```
Ground truth: 302 files
Ground-truth curve computed.

[model_A] 20 batches
  SE = 0.013348 ± 0.000830

[model_B] 20 batches
  SE = 0.017522 ± 0.001175

============================================================
Model                                         SE
------------------------------------------------------------
model_A                                   0.013348 ± 0.000830
model_B                                   0.017522 ± 0.001175
============================================================
Lower SE is better.
```

## API

### `compute_structure_error(gt_files, gen_files, max_bars=32, ticks_per_bar=1920, num_workers=None, precomputed_gt_curve=None, verbose=True)`

Compute SE between two sets of MIDI files.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gt_files` | `List[str]` | — | Paths to reference MIDI files. |
| `gen_files` | `List[str]` | — | Paths to generated MIDI files. |
| `max_bars` | `int` | `32` | Number of bars to evaluate. |
| `ticks_per_bar` | `int` | `1920` | MIDI ticks per bar. |
| `num_workers` | `int` | `None` | Parallel workers (default: CPU count). |
| `precomputed_gt_curve` | `List[float]` | `None` | Cached reference curve. |
| `verbose` | `bool` | `True` | Print progress. |

**Returns:** `float` — Structure Error (lower is better).

### `compute_similarity_curve(midi_files, max_bars=32, ticks_per_bar=1920, num_workers=None, verbose=True)`

Compute the bar-level self-similarity curve for a set of MIDI files.

**Returns:** `List[float]` of length `max_bars`.

### `list_midi_files(directory, recursive=True)`

List all `.mid` files in a directory.

**Returns:** sorted `List[str]`.
