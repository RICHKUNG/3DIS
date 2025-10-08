#!/usr/bin/env python3
"""Compatibility wrapper to launch the My3DIS workflow from the repo root."""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

# Ensure package modules are discoverable when running from repo checkout.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from my3dis.run_workflow import main  # noqa: E402  (import after sys.path tweak)


if __name__ == "__main__":
    # Mirror module entrypoint behaviour.
    sys.exit(main())
