"""Sorting utilities for filenames and frames."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple


def numeric_frame_sort_key(fname: str) -> Tuple[float, str]:
    """Ensure filenames sort by embedded integer before lexical fallback.

    This function extracts the first numeric substring from a filename
    and uses it as the primary sort key, with the full filename as fallback.

    Args:
        fname: Filename to extract sort key from

    Returns:
        Tuple of (numeric_value, filename) for sorting

    Examples:
        >>> files = ["frame_100.jpg", "frame_20.jpg", "frame_3.jpg"]
        >>> sorted(files, key=numeric_frame_sort_key)
        ['frame_3.jpg', 'frame_20.jpg', 'frame_100.jpg']
    """
    stem = Path(fname).stem
    match = re.search(r"\d+", stem)
    if match:
        try:
            return float(int(match.group())), fname
        except ValueError:
            pass
    return float("inf"), fname
