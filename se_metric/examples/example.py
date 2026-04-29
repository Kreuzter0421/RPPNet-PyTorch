"""
Example: compute the Structure Error (SE) between a generated melody set
and a ground-truth human melody set.

Usage:
    python examples/example.py

Expected directory structure:
    data/
    ├── ground_truth/
    │   └── *.mid
    └── generated/
        └── *.mid
"""

import os

from se_metric.examples.se_metric import compute_structure_error, list_midi_files


def main() -> None:
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    gt_dir = os.path.join("data", "ground_truth")
    gen_dir = os.path.join("data", "generated")

    max_bars = 32          # Evaluate first 32 bars
    ticks_per_bar = 1920   # 480 ticks/beat * 4 beats (4/4 time)
    num_workers = 4        # Parallel workers; set to 1 to disable

    # ------------------------------------------------------------------
    # List MIDI files
    # ------------------------------------------------------------------
    gt_files = list_midi_files(gt_dir)
    gen_files = list_midi_files(gen_dir)

    print(f"Ground-truth files: {len(gt_files)}")
    print(f"Generated files:    {len(gen_files)}")

    if not gt_files or not gen_files:
        print("\nError: Please place MIDI files in data/ground_truth/ and data/generated/")
        return

    # ------------------------------------------------------------------
    # Compute SE
    # ------------------------------------------------------------------
    se = compute_structure_error(
        gt_files=gt_files,
        gen_files=gen_files,
        max_bars=max_bars,
        ticks_per_bar=ticks_per_bar,
        num_workers=num_workers,
    )

    print(f"\n{'=' * 50}")
    print(f"Structure Error (SE) = {se:.6f}")
    print(f"{'=' * 50}")
    print("Interpretation: lower is better (closer to reference structure).")


if __name__ == "__main__":
    main()
