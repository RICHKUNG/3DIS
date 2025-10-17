"""Compatibility wrapper exposing Semantic-SAM candidate generation helpers."""

from __future__ import annotations

from . import generate_candidates as _ssam_impl
from .generate_candidates import *  # noqa: F401,F403

__all__ = getattr(_ssam_impl, "__all__", [name for name in globals() if not name.startswith("_")])
