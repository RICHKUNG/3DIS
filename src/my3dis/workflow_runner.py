#!/usr/bin/env python3
"""Compatibility entrypoint delegating to the maintained run_workflow module."""

from __future__ import annotations

from . import run_workflow as _workflow_impl
from .run_workflow import *  # noqa: F401,F403

__all__ = getattr(_workflow_impl, "__all__", [name for name in globals() if not name.startswith("_")])
