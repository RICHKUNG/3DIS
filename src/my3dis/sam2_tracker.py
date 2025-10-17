"""Compatibility wrapper exposing SAM2 tracking helpers under the new module name."""

from __future__ import annotations

from . import track_from_candidates as _track_impl
from .track_from_candidates import *  # noqa: F401,F403

__all__ = getattr(_track_impl, "__all__", [name for name in globals() if not name.startswith("_")])
