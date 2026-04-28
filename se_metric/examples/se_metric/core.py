"""
Core implementation of the Structure Error (SE) metric.

SE measures the structural difference between a generated melody set and a
ground-truth human melody set by comparing their bar-level self-similarity
curves. For each interval t (in bars), we compute the average Jaccard
similarity between bars separated by t positions. The error is then defined
as the mean absolute difference between the ground-truth curve and the
generated curve.
"""

import os
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import miditoolkit
from tqdm import tqdm

DEFAULT_TICKS_PER_BAR = 1920
DEFAULT_MAX_BARS = 32

NoteTuple = Tuple[int, int, int]
"""Type alias for a note represented as (pitch, duration, onset_within_bar)."""


def group_notes_to_bars(
    notes: List[miditoolkit.Note],
    max_bars: int,
    ticks_per_bar: int = DEFAULT_TICKS_PER_BAR,
) -> Dict[int, List[NoteTuple]]:
    """
    Group notes into bars based on their onset times.

    Each note is represented as a tuple of (pitch, duration, onset_within_bar).
    Notes whose onset falls beyond ``max_bars`` are ignored.

    Args:
        notes: List of ``miditoolkit.Note`` objects. Should be sorted by
            ``start`` time before calling this function.
        max_bars: Maximum number of bars to consider (1-based indexing).
        ticks_per_bar: Number of MIDI ticks per bar. Default is 1920,
            which corresponds to 480 ticks/beat × 4 beats in 4/4 time.

    Returns:
        Dictionary mapping 1-based bar indices to lists of note tuples.
    """
    grouped: Dict[int, List[NoteTuple]] = {bar: [] for bar in range(1, max_bars + 1)}

    for note in notes:
        bar_idx = note.start // ticks_per_bar + 1
        if bar_idx > max_bars:
            break

        pitch = note.pitch
        duration = note.end - note.start
        onset = note.start % ticks_per_bar
        grouped[bar_idx].append((pitch, duration, onset))

    return grouped


def _compute_file_similarity(
    midi_path: str,
    interval: int,
    max_bars: int,
    ticks_per_bar: int = DEFAULT_TICKS_PER_BAR,
) -> float:
    """
    Compute the average Jaccard similarity for a single MIDI file at a given
    bar interval.

    For each valid starting bar ``i``, we compare the set of notes in bar ``i``
    with the set of notes in bar ``i + interval`` using Jaccard similarity:
    ``|intersection| / |union|``.

    Args:
        midi_path: Path to the MIDI file.
        interval: Bar interval (t) for comparison.
        max_bars: Maximum number of bars to evaluate.
        ticks_per_bar: Number of ticks per bar.

    Returns:
        Mean Jaccard similarity across all valid bar pairs. Returns 0.0 if
        no valid pairs exist.
    """
    midi = miditoolkit.MidiFile(midi_path)
    notes = sorted(midi.instruments[0].notes, key=lambda n: n.start)
    grouped = group_notes_to_bars(notes, max_bars, ticks_per_bar)

    bar_keys = list(grouped.keys())
    similarities: List[float] = []

    for idx, start in enumerate(bar_keys):
        if idx + interval >= len(bar_keys):
            break

        bar1 = set(grouped[start])
        bar2 = set(grouped[start + interval])
        union = bar1 | bar2

        if union:
            similarities.append(len(bar1 & bar2) / len(union))
        else:
            similarities.append(0.0)

    return sum(similarities) / len(similarities) if similarities else 0.0


def compute_similarity_for_interval(
    midi_files: List[str],
    interval: int,
    max_bars: int = DEFAULT_MAX_BARS,
    ticks_per_bar: int = DEFAULT_TICKS_PER_BAR,
    num_workers: Optional[int] = None,
) -> float:
    """
    Compute the average bar similarity across a list of MIDI files at a
    specific interval.

    Args:
        midi_files: List of paths to MIDI files.
        interval: Bar interval for comparison (e.g., 1 means adjacent bars).
        max_bars: Maximum number of bars to evaluate.
        ticks_per_bar: Number of ticks per bar.
        num_workers: Number of parallel worker processes. Defaults to the
            value of the ``N_PROC`` environment variable, or the CPU count.

    Returns:
        Mean similarity across all files.
    """
    if not midi_files:
        return 0.0

    if num_workers is None:
        num_workers = int(os.getenv("N_PROC", os.cpu_count() or 1))

    worker = partial(
        _compute_file_similarity,
        interval=interval,
        max_bars=max_bars,
        ticks_per_bar=ticks_per_bar,
    )

    if num_workers <= 1 or len(midi_files) == 1:
        results = [worker(f) for f in midi_files]
    else:
        with Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(worker, midi_files),
                    total=len(midi_files),
                    desc=f"Interval {interval}",
                    leave=False,
                )
            )

    return sum(results) / len(results) if results else 0.0


def compute_similarity_curve(
    midi_files: List[str],
    max_bars: int = DEFAULT_MAX_BARS,
    ticks_per_bar: int = DEFAULT_TICKS_PER_BAR,
    num_workers: Optional[int] = None,
    verbose: bool = True,
) -> List[float]:
    """
    Compute the structural similarity curve ``L`` for a set of MIDI files.

    ``L[t]`` (0-indexed) represents the average Jaccard similarity between
    bars separated by ``t + 1`` positions, computed over all files.

    Args:
        midi_files: List of paths to MIDI files.
        max_bars: Maximum number of bars.
        ticks_per_bar: Number of ticks per bar.
        num_workers: Number of parallel workers.
        verbose: Whether to display a progress bar.

    Returns:
        Similarity curve as a list of length ``max_bars``.
    """
    curve: List[float] = []
    iterator = range(1, max_bars + 1)
    if verbose:
        iterator = tqdm(iterator, desc="Computing similarity curve")

    for t in iterator:
        sim = compute_similarity_for_interval(
            midi_files, t, max_bars, ticks_per_bar, num_workers
        )
        curve.append(sim)

    return curve


def compute_structure_error(
    gt_files: List[str],
    gen_files: List[str],
    max_bars: int = DEFAULT_MAX_BARS,
    ticks_per_bar: int = DEFAULT_TICKS_PER_BAR,
    num_workers: Optional[int] = None,
    precomputed_gt_curve: Optional[List[float]] = None,
    verbose: bool = True,
) -> float:
    """
    Compute the Structure Error (SE) between generated melodies and
    ground-truth human melodies.

    The metric is defined as:

    .. math::
        SE = \\frac{1}{T} \\sum_{t=1}^{T} |L_{gt}[t] - L_{gen}[t]|

    where :math:`L[t]` is the average Jaccard similarity between bars
    separated by :math:`t` positions, and :math:`T` is ``max_bars``.

    Lower SE indicates better structural alignment with human melodies.

    Args:
        gt_files: List of paths to ground-truth MIDI files.
        gen_files: List of paths to generated MIDI files.
        max_bars: Maximum number of bars to evaluate.
        ticks_per_bar: Number of ticks per bar.
        num_workers: Number of parallel worker processes.
        precomputed_gt_curve: Optional precomputed ground-truth similarity
            curve. This is useful when evaluating multiple models against
            the same ground truth to avoid redundant computation.
        verbose: Whether to print progress messages.

    Returns:
        Structure Error as a scalar float.
    """
    if precomputed_gt_curve is not None:
        L_gt = precomputed_gt_curve
    else:
        if verbose:
            print(
                f"Computing ground-truth similarity curve "
                f"({len(gt_files)} files)..."
            )
        L_gt = compute_similarity_curve(
            gt_files, max_bars, ticks_per_bar, num_workers, verbose
        )

    if verbose:
        print(
            f"Computing generated similarity curve "
            f"({len(gen_files)} files)..."
        )
    L_gen = compute_similarity_curve(
        gen_files, max_bars, ticks_per_bar, num_workers, verbose
    )

    se = sum(abs(x - y) for x, y in zip(L_gt, L_gen)) / max_bars

    if verbose:
        print(f"Structure Error (SE) = {se:.6f}")

    return se
