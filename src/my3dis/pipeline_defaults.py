"""Centralised default paths and constants for the My3DIS pipeline."""

from __future__ import annotations

from pathlib import Path

# Semantic-SAM assets -----------------------------------------------------
_DEFAULT_SEMANTIC_SAM_ROOT = Path("/media/Pluto/richkung/Semantic-SAM")
DEFAULT_SEMANTIC_SAM_ROOT = _DEFAULT_SEMANTIC_SAM_ROOT
DEFAULT_SEMANTIC_SAM_CKPT = DEFAULT_SEMANTIC_SAM_ROOT / "checkpoints" / "swinl_only_sam_many2many.pth"

# SAM2 assets -------------------------------------------------------------
_DEFAULT_SAM2_ROOT = Path("/media/Pluto/richkung/SAM2")
DEFAULT_SAM2_ROOT = _DEFAULT_SAM2_ROOT
DEFAULT_SAM2_CFG = DEFAULT_SAM2_ROOT / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml"
DEFAULT_SAM2_CKPT = DEFAULT_SAM2_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"

# Example dataset + output layout ----------------------------------------
DEFAULT_DATA_PATH = Path("/media/public_dataset2/multiscan/scene_00065_00/outputs/color")
DEFAULT_OUTPUT_ROOT = Path("/media/Pluto/richkung/My3DIS/outputs/scene_00065_00")


def expand_default(path: Path) -> str:
    """Return an absolute string path for defaults, preserving user overrides."""

    try:
        return str(path.expanduser().resolve())
    except FileNotFoundError:
        return str(path.expanduser().absolute())
