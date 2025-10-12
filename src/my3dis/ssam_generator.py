```python
"""
SSAM 候選生成器
重新命名自 generate_candidates.py，提供更清晰的模組名稱

Author: Rich Kung
Updated: 2025-01-XX
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

# Ensure package modules are discoverable when running from repo checkout.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from my3dis.ssam_generator import main  # noqa: E402  (import after sys.path tweak)


def run_generation(...):
    """SSAM 候選生成主流程"""
    # ...existing code...


def main():
    """CLI 入口，解析參數後呼叫 run_generation"""
    # ...existing code...


if __name__ == "__main__":
    main()
```