"""Compatibility wrapper that keeps the older candidate_filter entrypoint working."""

from __future__ import annotations

from . import filter_candidates as _filter_impl
from .filter_candidates import *  # noqa: F401,F403

__all__ = getattr(_filter_impl, "__all__", [name for name in globals() if not name.startswith("_")])
