"""
Compare multiple models under data/generated/ against the same ground truth.

Each model directory should contain batch_*/ subdirectories.
SE is computed per-batch and then aggregated as mean ± std.

Expected structure:
    data/
    ├── ground_truth/
    │   └── GT_302/           (or directly *.mid files)
    └── generated/
        ├── model_A/
        │   ├── batch_0/
        │   └── batch_1/
        └── model_B/
            └── ...

Usage:
    python examples/compare_models.py
"""

import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from se_metric import compute_structure_error, list_midi_files
from se_metric.core import compute_similarity_curve


def get_batch_dirs(model_path: str):
    """Return sorted list of batch_* directories."""
    batches = sorted(
        d for d in os.listdir(model_path)
        if os.path.isdir(os.path.join(model_path, d)) and d.startswith("batch_")
    )
    return [os.path.join(model_path, d) for d in batches]


def main():
    gt_dir = os.path.join("data", "ground_truth")
    gen_root = os.path.join("data", "generated")

    max_bars = 32
    num_workers = 4

    # ------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------
    gt_files = list_midi_files(gt_dir)
    if not gt_files:
        print(f"Error: no MIDI files found in {gt_dir}")
        return

    print(f"Ground truth: {len(gt_files)} files")
    L_gt = compute_similarity_curve(
        gt_files, max_bars=max_bars, num_workers=num_workers, verbose=False
    )
    print("Ground-truth curve computed.\n")

    # ------------------------------------------------------------------
    # Evaluate each model
    # ------------------------------------------------------------------
    model_names = sorted(
        d for d in os.listdir(gen_root)
        if os.path.isdir(os.path.join(gen_root, d))
    )

    if not model_names:
        print(f"Error: no model directories found in {gen_root}")
        return

    results = []
    for model_name in model_names:
        model_path = os.path.join(gen_root, model_name)
        batch_dirs = get_batch_dirs(model_path)

        if not batch_dirs:
            # Fallback: treat whole directory as flat files
            gen_files = list_midi_files(model_path)
            if not gen_files:
                print(f"[{model_name}] skipped: no MIDI files")
                continue
            print(f"[{model_name}] {len(gen_files)} files (flat)")
            se = compute_structure_error(
                gt_files, gen_files, max_bars=max_bars,
                num_workers=num_workers, precomputed_gt_curve=L_gt, verbose=False
            )
            results.append((model_name, se, 0.0))
            print(f"  SE = {se:.6f}\n")
            continue

        # Per-batch evaluation
        se_list = []
        for batch_dir in batch_dirs:
            batch_files = list_midi_files(batch_dir)
            if not batch_files:
                continue
            se = compute_structure_error(
                gt_files, batch_files, max_bars=max_bars,
                num_workers=num_workers, precomputed_gt_curve=L_gt, verbose=False
            )
            se_list.append(se)

        if not se_list:
            print(f"[{model_name}] skipped: no valid batches")
            continue

        mean_se = statistics.mean(se_list)
        std_se = statistics.stdev(se_list) if len(se_list) > 1 else 0.0
        results.append((model_name, mean_se, std_se))
        print(f"[{model_name}] {len(se_list)} batches")
        print(f"  SE = {mean_se:.6f} ± {std_se:.6f}\n")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"{'Model':<45} {'SE':>12}")
    print("-" * 60)
    for name, mean_se, std_se in results:
        print(f"{name:<45} {mean_se:>8.6f} ± {std_se:.6f}")
    print("=" * 60)
    print("Lower SE is better.")


if __name__ == "__main__":
    main()
