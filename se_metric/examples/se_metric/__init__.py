"""
SE Metric: Structure Error evaluation for symbolic music generation.

This package implements the Structure Error (SE) metric for evaluating
the structural similarity between generated melodies and reference
melodies.
"""

from .core import (
    compute_structure_error,
    compute_similarity_curve,
    compute_similarity_for_interval,
    group_notes_to_bars,
)
from .utils import list_midi_files, load_midi, load_midi_files

__all__ = [
    "compute_structure_error",
    "compute_similarity_curve",
    "compute_similarity_for_interval",
    "group_notes_to_bars",
    "list_midi_files",
    "load_midi",
    "load_midi_files",
]

__version__ = "0.1.0"
