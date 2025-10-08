"""場景相關的設定解析與輔助函式。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .errors import WorkflowConfigError, WorkflowRuntimeError

_ALL_SCENE_TOKENS = {'all', '*', '__all__'}


def expand_output_path_template(path_value: Any, experiment_cfg: Dict[str, Any]) -> str:
    """依據實驗名稱展開 output 路徑模板。"""
    if path_value is None:
        raise WorkflowConfigError('experiment.output_root is required (or override via --override-output)')

    path_str = str(path_value)
    if '{name}' in path_str:
        experiment_name = experiment_cfg.get('name')
        if not experiment_name:
            raise WorkflowConfigError(
                'experiment.output_root 使用了 {name} 但未提供 experiment.name'
            )
        path_str = path_str.replace('{name}', str(experiment_name))

    return path_str


def discover_scene_names(dataset_root: Path) -> List[str]:
    """在資料集根目錄底下列出場景資料夾。"""
    if not dataset_root.exists():
        raise WorkflowConfigError(f'experiment.dataset_root does not exist: {dataset_root}')

    try:
        entries = sorted(p.name for p in dataset_root.iterdir() if p.is_dir())
    except OSError as exc:
        raise WorkflowRuntimeError(f'failed to list dataset_root={dataset_root}: {exc}') from exc

    preferred = [name for name in entries if name.startswith('scene_')]
    return preferred or entries


def normalize_scene_list(
    raw_scenes: Any,
    dataset_root: Path,
    *,
    scene_start: Optional[str] = None,
    scene_end: Optional[str] = None,
) -> List[str]:
    """解析 config 中的場景選擇設定。"""
    discovered_cache: Optional[List[str]] = None

    def ensure_discovered() -> List[str]:
        nonlocal discovered_cache
        if discovered_cache is None:
            discovered_cache = discover_scene_names(dataset_root)
        return discovered_cache

    if raw_scenes is None:
        scenes_iterable: List[str] = ensure_discovered()
    elif isinstance(raw_scenes, str):
        token = raw_scenes.strip()
        if token.lower() in _ALL_SCENE_TOKENS:
            scenes_iterable = ensure_discovered()
        else:
            scenes_iterable = [token]
    elif isinstance(raw_scenes, (list, tuple)):
        result: List[str] = []
        seen: Set[str] = set()
        for entry in raw_scenes:
            token = str(entry).strip()
            if not token:
                continue
            if token.lower() in _ALL_SCENE_TOKENS:
                for name in ensure_discovered():
                    if name not in seen:
                        result.append(name)
                        seen.add(name)
                continue
            if token not in seen:
                result.append(token)
                seen.add(token)
        scenes_iterable = result
    else:
        raise WorkflowConfigError('experiment.scenes must be a list, string, or null when provided')

    if not scenes_iterable:
        raise WorkflowConfigError('No scenes resolved for experiment (empty list after processing)')

    missing = [scene for scene in scenes_iterable if not (dataset_root / scene).exists()]
    if missing:
        raise WorkflowConfigError(
            'The following scenes were not found under dataset_root '
            f"{dataset_root}: {', '.join(missing)}"
        )

    if scene_start is not None or scene_end is not None:
        ordered = ensure_discovered()
        order_map = {name: idx for idx, name in enumerate(ordered)}

        if scene_start is not None:
            start_token = str(scene_start).strip()
            if start_token not in order_map:
                raise WorkflowConfigError(
                    f'scene_start {scene_start!r} not found under dataset_root {dataset_root}'
                )
            start_idx = order_map[start_token]
        else:
            start_idx = 0

        if scene_end is not None:
            end_token = str(scene_end).strip()
            if end_token not in order_map:
                raise WorkflowConfigError(
                    f'scene_end {scene_end!r} not found under dataset_root {dataset_root}'
                )
            end_idx = order_map[end_token]
        else:
            end_idx = len(ordered) - 1

        if end_idx < start_idx:
            raise WorkflowConfigError('scene_end occurs before scene_start; please provide a valid range')

        allowed: Set[str] = set(scenes_iterable)
        sliced = [name for name in ordered[start_idx : end_idx + 1] if name in allowed]
        if not sliced:
            raise WorkflowConfigError(
                'Scene range selection produced an empty set; adjust scene_start or scene_end'
            )
        scenes_iterable = sliced

    return scenes_iterable


def resolve_levels(
    stage_cfg: Dict[str, Any],
    manifest: Optional[Dict[str, Any]],
    fallback: Optional[List[int]],
) -> List[int]:
    """根據 stage config/manifest 取得需要追蹤的層級。"""
    if 'levels' in stage_cfg and stage_cfg['levels'] is not None:
        try:
            return [int(x) for x in stage_cfg['levels']]
        except (TypeError, ValueError):
            raise WorkflowConfigError(f"Invalid levels in config: {stage_cfg['levels']}")
    if manifest and isinstance(manifest.get('levels'), list):
        try:
            return [int(x) for x in manifest['levels']]
        except (TypeError, ValueError):
            pass
    if fallback:
        try:
            return [int(x) for x in fallback]
        except (TypeError, ValueError) as exc:
            raise WorkflowConfigError(f'Invalid fallback levels: {fallback}') from exc
    raise WorkflowConfigError('Unable to determine levels for stage')


def stage_frames_string(stage_cfg: Dict[str, Any]) -> str:
    """將 frame 設定統一格式化為 'start:end:step'。"""
    frames_cfg = stage_cfg.get('frames', {}) or {}
    start_raw = frames_cfg.get('start', frames_cfg.get('from'))
    end_raw = frames_cfg.get('end', frames_cfg.get('to'))
    step = int(
        frames_cfg.get('step')
        or frames_cfg.get('freq')
        or frames_cfg.get('stride')
        or stage_cfg.get('freq')
        or 1
    )
    start = int(start_raw) if start_raw is not None else 0
    # Use -1 as sentinel to indicate "until end" when end not provided.
    end = int(end_raw) if end_raw is not None else -1
    return f"{start}:{end}:{step}"


def resolve_stage_gpu(stage_cfg: Optional[Dict[str, Any]], default_gpu: Optional[Any]) -> Optional[Any]:
    if isinstance(stage_cfg, dict) and 'gpu' in stage_cfg:
        return stage_cfg.get('gpu')
    return default_gpu


def derive_scene_metadata(data_path: str) -> Dict[str, Optional[str]]:
    """從資料路徑反推出場景基本資訊。"""
    path = Path(str(data_path)).expanduser()
    try:
        resolved = path.resolve()
    except FileNotFoundError:
        resolved = path.absolute()

    scene_name: Optional[str] = None
    scene_root: Optional[Path] = None
    dataset_root: Optional[Path] = None
    for candidate in [resolved] + list(resolved.parents):
        if candidate.name.startswith('scene_'):
            scene_name = candidate.name
            scene_root = candidate
            dataset_root = candidate.parent
            break

    return {
        'scene': scene_name,
        'scene_root': str(scene_root) if scene_root else None,
        'dataset_root': str(dataset_root) if dataset_root else None,
    }


__all__ = [
    'expand_output_path_template',
    'discover_scene_names',
    'normalize_scene_list',
    'resolve_levels',
    'stage_frames_string',
    'resolve_stage_gpu',
    'derive_scene_metadata',
]
