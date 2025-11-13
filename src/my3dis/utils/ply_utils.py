"""
Utility helpers for loading scene point clouds and exporting predictions to PLY.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Iterable, Sequence

import numpy as np


def _resolve_scene_ply(scene_path: Path) -> Path:
    """
    Resolve the PLY file for a MultiScan scene.

    The standard layout stores either scene_<id>.ply or scene_<id>_converted.ply
    under the scene directory.
    """
    scene_path = scene_path.expanduser().resolve()
    if scene_path.is_file():
        return scene_path

    scene_dir = scene_path
    scene_name = scene_dir.name
    candidates = [
        scene_dir / f"{scene_name}.ply",
        scene_dir / f"{scene_name}_converted.ply",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Point cloud PLY not found under {scene_dir}")


def load_scene_pointcloud(scene_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load (points, colors) for a MultiScan scene.

    Returns:
        points: float32 [N, 3]
        colors: uint8 [N, 3]
    """
    ply_path = _resolve_scene_ply(Path(scene_path))

    try:
        from plyfile import PlyData  # type: ignore
    except ImportError as exc:
        raise ImportError("plyfile package is required for PLY export. Install via `pip install plyfile`.") from exc

    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"]

    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    if {"red", "green", "blue"}.issubset(vertex.data.dtype.names):
        colors = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.uint8)
    else:
        colors = np.ones_like(points, dtype=np.uint8) * 255

    return points, colors


def save_mask_as_ply(
    points: np.ndarray,
    colors: np.ndarray,
    mask: np.ndarray,
    output_path: str | Path,
    highlight_color: Sequence[int] | None = None,
    include_full_scene: bool = False,
    background_color: Sequence[int] | None = (80, 80, 80),
) -> bool:
    """
    Save predictions as PLY files for visualization.

    When ``include_full_scene`` is False (default), only the masked points are exported
    (legacy behavior). When ``include_full_scene`` is True, the full point cloud is
    written and masked points are highlighted while all other points are painted with
    ``background_color`` (or original colors if the background color is None).

    Args:
        points: [N, 3] float32 array
        colors: [N, 3] uint8 array
        mask: [N] boolean or {0,1}
        output_path: destination PLY path
        highlight_color: optional RGB color to override for the masked points
        include_full_scene: export the entire scene instead of just the masked subset
        background_color: RGB color for background points when exporting the full scene

    Returns:
        True if any points were written, False if mask was empty.
    """
    mask = np.asarray(mask).astype(bool)
    if mask.ndim != 1 or mask.shape[0] != points.shape[0]:
        raise ValueError("Mask must be 1D with length equal to number of points.")

    if not np.any(mask):
        return False

    if include_full_scene:
        sel_points = points
        sel_colors = colors.copy()
        if background_color is not None:
            bg_color = np.array(background_color, dtype=np.uint8)
            if bg_color.shape != (3,):
                raise ValueError("background_color must be a sequence of 3 integers.")
            sel_colors[:, :] = bg_color
        if highlight_color is not None:
            hl_color = np.array(highlight_color, dtype=np.uint8)
            sel_colors[mask, :] = hl_color
    else:
        sel_points = points[mask]
        sel_colors = colors[mask].copy()
        if highlight_color is not None:
            sel_colors[:, :] = np.array(highlight_color, dtype=np.uint8)

    vertex_dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    vertex_array = np.empty(sel_points.shape[0], dtype=vertex_dtype)
    vertex_array["x"] = sel_points[:, 0]
    vertex_array["y"] = sel_points[:, 1]
    vertex_array["z"] = sel_points[:, 2]
    vertex_array["red"] = sel_colors[:, 0]
    vertex_array["green"] = sel_colors[:, 1]
    vertex_array["blue"] = sel_colors[:, 2]

    try:
        from plyfile import PlyElement, PlyData  # type: ignore
    except ImportError as exc:
        raise ImportError("plyfile package is required for PLY export. Install via `pip install plyfile`.") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex_array, "vertex")], text=False).write(str(output_path))
    return True
