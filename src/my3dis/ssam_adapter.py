"""Compatibility adapter exposing the legacy Semantic-SAM progressive helpers."""

from __future__ import annotations

from . import ssam_progressive_adapter as _adapter_impl
from .ssam_progressive_adapter import *  # noqa: F401,F403

__all__ = getattr(_adapter_impl, "__all__", [name for name in globals() if not name.startswith("_")])
