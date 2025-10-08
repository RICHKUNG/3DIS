"""IO 與設定檔讀取相關輔助函式。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .errors import WorkflowConfigError


def load_yaml(path: Path) -> Dict[str, Any]:
    """讀取 YAML 並檢查最上層為 mapping。"""
    try:
        with path.open('r') as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise WorkflowConfigError(f'Config file not found: {path}') from exc
    if not isinstance(data, dict):
        raise WorkflowConfigError(f'Config file {path} must define a mapping at the top level')
    return data


__all__ = ['load_yaml']
