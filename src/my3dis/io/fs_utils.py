"""Filesystem utilities for My3DIS.

This module provides functions for directory creation, file operations,
and frame subset management.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple


def ensure_dir(path: str | Path) -> str:
    """Create directory if missing and return the absolute string path.

    Args:
        path: Directory path to create

    Returns:
        Absolute path string
    """
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return str(resolved)


def build_subset_video(
    frames_dir: str,
    selected: Sequence[str],
    selected_indices: Sequence[int],
    out_root: str,
    folder_name: str = "selected_frames",
) -> Tuple[str, dict[int, str]]:
    """Symlink/copy selected frames into a compact folder for later stages.

    This function creates a compact subset of frames for SAM2 tracking,
    preserving frame relationships while avoiding full dataset copies.

    Args:
        frames_dir: Source directory containing frame images
        selected: List of selected frame filenames
        selected_indices: Corresponding absolute frame indices
        out_root: Output root directory
        folder_name: Name of subfolder to create (default: "selected_frames")

    Returns:
        Tuple of (subset_dir_path, index_to_filename_mapping)
    """
    subset_dir = ensure_dir(Path(out_root) / folder_name)
    index_to_subset: dict[int, str] = {}

    for local_idx, (abs_idx, fname) in enumerate(zip(selected_indices, selected)):
        src = Path(frames_dir) / fname
        dst_name = f"{local_idx:06d}.jpg"
        dst = Path(subset_dir) / dst_name
        index_to_subset[int(abs_idx)] = dst_name

        if dst.exists():
            # Preserve existing symlink/copy when it already matches the source.
            try:
                if dst.is_symlink() and dst.resolve() == src.resolve():
                    continue
            except FileNotFoundError:
                pass
            dst.unlink()

        try:
            dst.symlink_to(src)
        except OSError:
            from shutil import copy2

            copy2(src, dst)

    return subset_dir, index_to_subset
