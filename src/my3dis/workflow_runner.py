```python
#!/usr/bin/env python3
"""工作流程執行器
重新命名自 run_workflow.py，提供更清晰的模組名稱

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

from my3dis.run_workflow import main  # noqa: E402  (import after sys.path tweak)


def main():
    """解析 --config 等參數、載入 YAML、呼叫 workflow.execute_workflow"""
    # ...existing code...


if __name__ == "__main__":
    main()
```