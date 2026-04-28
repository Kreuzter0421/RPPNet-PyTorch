"""
Utility functions for MIDI file I/O.
"""

import os
from glob import glob
from multiprocessing import Pool
from typing import List, Optional, Tuple

import miditoolkit
from tqdm import tqdm


def load_midi(midi_path: str) -> Optional[miditoolkit.MidiFile]:
    """
    Load a single MIDI file using miditoolkit.

    Args:
        midi_path: Path to the MIDI file.

    Returns:
        A ``MidiFile`` object if loading succeeds, otherwise ``None``.
    """
    try:
        return miditoolkit.MidiFile(midi_path)
    except Exception as e:
        print(f"Failed to load {midi_path}: {e}")
        return None


def list_midi_files(directory: str, recursive: bool = True) -> List[str]:
    """
    List all MIDI files (``.mid``) in a directory.

    Args:
        directory: Path to the directory to search.
        recursive: If ``True``, search recursively through subdirectories.

    Returns:
        Alphabetically sorted list of MIDI file paths.
    """
    if recursive:
        pattern = os.path.join(directory, "**", "*.mid")
        files = glob(pattern, recursive=True)
    else:
        pattern = os.path.join(directory, "*.mid")
        files = glob(pattern)
    return sorted(files)


def load_midi_files(
    directory: str,
    num_workers: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[List[miditoolkit.MidiFile], List[str]]:
    """
    Load all MIDI files from a directory in parallel.

    Args:
        directory: Path to the directory containing MIDI files.
        num_workers: Number of parallel worker processes. Defaults to the
            value of the ``N_PROC`` environment variable, or the CPU count.
        verbose: Whether to display a progress bar.

    Returns:
        Tuple of (list of successfully loaded ``MidiFile`` objects,
        list of their corresponding file paths).
    """
    file_paths = list_midi_files(directory)

    if num_workers is None:
        num_workers = int(os.getenv("N_PROC", os.cpu_count() or 1))

    if num_workers <= 1 or len(file_paths) <= 1:
        midis: List[miditoolkit.MidiFile] = []
        valid_paths: List[str] = []
        iterator = tqdm(file_paths, desc="Loading MIDI files") if verbose else file_paths
        for path in iterator:
            midi = load_midi(path)
            if midi is not None:
                midis.append(midi)
                valid_paths.append(path)
        return midis, valid_paths

    with Pool(processes=num_workers) as pool:
        futures = [pool.apply_async(load_midi, args=(p,)) for p in file_paths]

        midis, valid_paths = [], []
        iterator = zip(futures, file_paths)
        if verbose:
            iterator = tqdm(iterator, total=len(futures), desc="Loading MIDI files")

        for future, path in iterator:
            try:
                midi = future.get()
                if midi is not None:
                    midis.append(midi)
                    valid_paths.append(path)
            except Exception as e:
                print(f"Failed to process {path}: {e}")

    return midis, valid_paths
