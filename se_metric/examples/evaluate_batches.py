"""
Example: evaluate SE across multiple batches (e.g., for a generative model
with multiple independent generation runs).

This is useful when your model outputs are organized as:
    model_dir/
        batch_0/
            0.mid
            1.mid
            ...
        batch_1/
            ...

Usage:
    python examples/evaluate_batches.py
"""

import os
import statistics
from glob import glob
from typing import List, Optional

from se_metric import compute_structure_error, list_midi_files
from se_metric.core import compute_similarity_curve


def evaluate_model_batches(
    model_dir: str,
    gt_dir: str,
    max_bars: int = 32,
    num_workers: Optional[int] = None,
) -> List[float]:
    """
    Evaluate SE for each batch subdirectory under ``model_dir``.

    Args:
        model_dir: Root directory containing ``batch_*`` subfolders.
        gt_dir: Directory containing ground-truth MIDI files.
        max_bars: Maximum number of bars to evaluate.
        num_workers: Number of parallel workers.

    Returns:
        List of SE values, one per batch.
    """
    gt_files = list_midi_files(gt_dir)
    print(f"Ground-truth: {len(gt_files)} files\n")

    # Precompute the ground-truth similarity curve once and reuse it
    # across all batches to save computation.
    L_gt = compute_similarity_curve(
        gt_files, max_bars=max_bars, num_workers=num_workers, verbose=True
    )
    print("Ground-truth curve computed.\n")

    batch_dirs = sorted(glob(os.path.join(model_dir, "batch_*")))
    se_values: List[float] = []

    for batch_dir in batch_dirs:
        if not os.path.isdir(batch_dir):
            continue

        batch_files = list_midi_files(batch_dir)
        if not batch_files:
            continue

        batch_name = os.path.basename(batch_dir)
        print(f"Evaluating {batch_name} ({len(batch_files)} files)...")

        se = compute_structure_error(
            gt_files=gt_files,
            gen_files=batch_files,
            max_bars=max_bars,
            num_workers=num_workers,
            precomputed_gt_curve=L_gt,
            verbose=False,
        )
        se_values.append(se)
        print(f"  SE = {se:.6f}\n")

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    if len(se_values) > 1:
        mean_se = statistics.mean(se_values)
        std_se = statistics.stdev(se_values)
        print(f"{'=' * 50}")
        print(f"Mean SE across {len(se_values)} batches: {mean_se:.6f} ± {std_se:.6f}")
        print(f"{'=' * 50}")
    elif se_values:
        print(f"SE: {se_values[0]:.6f}")
    else:
        print("No valid batches found.")

    return se_values


if __name__ == "__main__":
    model_dir = os.path.join("data", "generated_model")
    gt_dir = os.path.join("data", "ground_truth")
    evaluate_model_batches(model_dir, gt_dir)
