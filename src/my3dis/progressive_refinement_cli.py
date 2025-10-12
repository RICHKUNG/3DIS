"""
Backward-compatible CLI wrapper for the old progressive refinement entrypoint.
"""
from __future__ import annotations

import warnings

warnings.warn(
    "progressive_refinement_cli is deprecated. "
    "Use semantic_refinement_cli instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .semantic_refinement_cli import main, parse_args  # noqa: F401

__all__ = ["main", "parse_args"]


if __name__ == "__main__":
    main()
