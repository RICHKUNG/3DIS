"""多場景工作流程聚合執行。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .errors import WorkflowConfigError
from .scene_workflow import run_scene_workflow
from .scenes import expand_output_path_template, normalize_scene_list
from .utils import now_local_iso, now_local_stamp


def execute_workflow(
    config: Dict[str, Any],
    *,
    override_output: Optional[str] = None,
    config_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """根據設定檔執行場景或多場景工作流程。"""
    experiment_cfg = config.get('experiment', {})
    if not isinstance(experiment_cfg, dict):
        raise WorkflowConfigError('`experiment` section must be a mapping')

    stages_cfg = config.get('stages', {})
    if not isinstance(stages_cfg, dict):
        raise WorkflowConfigError('`stages` section must be a mapping')
    default_stage_gpu = stages_cfg.get('gpu')

    scenes_cfg = experiment_cfg.get('scenes')
    scene_start_cfg = experiment_cfg.get('scene_start')
    scene_end_cfg = experiment_cfg.get('scene_end')
    scenes_list: Optional[List[str]] = None
    dataset_root_raw = experiment_cfg.get('dataset_root')
    if scenes_cfg is not None or scene_start_cfg is not None or scene_end_cfg is not None:
        if not dataset_root_raw:
            raise WorkflowConfigError('experiment.dataset_root is required when selecting scenes')
        dataset_root = Path(dataset_root_raw).expanduser()
        scenes_list = normalize_scene_list(
            scenes_cfg,
            dataset_root,
            scene_start=scene_start_cfg,
            scene_end=scene_end_cfg,
        )
    else:
        dataset_root = Path(dataset_root_raw).expanduser() if dataset_root_raw else None

    if scenes_list:
        output_root_base_raw = override_output or experiment_cfg.get('output_root')
        output_root_base_resolved = expand_output_path_template(output_root_base_raw, experiment_cfg)
        output_root_base = Path(output_root_base_resolved).expanduser()

        aggregate_output = bool(experiment_cfg.get('aggregate_output'))
        run_timestamp = experiment_cfg.get('run_timestamp')
        if aggregate_output:
            experiment_stamp = str(run_timestamp) if run_timestamp else now_local_stamp()
            experiment_root = output_root_base / experiment_stamp
        else:
            experiment_root = output_root_base
            experiment_stamp = str(run_timestamp) if run_timestamp else None

        experiment_root.mkdir(parents=True, exist_ok=True)

        parent_meta = {
            'name': experiment_cfg.get('name'),
            'scenes': scenes_list,
            'experiment_root': str(experiment_root),
            'timestamp': experiment_stamp,
            'aggregate_output': aggregate_output,
            'scene_start': scene_start_cfg,
            'scene_end': scene_end_cfg,
        }

        summaries: List[Dict[str, Any]] = []
        experiment_scene_records: List[Dict[str, Any]] = []

        for index, scene_name in enumerate(scenes_list):
            if dataset_root is None:
                raise WorkflowConfigError('dataset_root must be provided when scenes are enumerated')
            scene_data_path = dataset_root / scene_name / 'outputs' / 'color'
            if not scene_data_path.exists():
                raise WorkflowConfigError(f'data path not found for scene {scene_name}: {scene_data_path}')
            if aggregate_output:
                scene_output_root = experiment_root / scene_name
            else:
                scene_output_root = output_root_base / scene_name
            scene_output_root.mkdir(parents=True, exist_ok=True)

            scene_experiment_cfg = dict(experiment_cfg)
            scene_experiment_cfg['name'] = experiment_cfg.get('name')
            scene_experiment_cfg['data_path'] = str(scene_data_path)
            scene_experiment_cfg['output_root'] = str(scene_output_root)
            scene_experiment_cfg['aggregate_output'] = aggregate_output
            scene_experiment_cfg['run_timestamp'] = experiment_stamp
            scene_experiment_cfg.pop('scenes', None)

            parent_meta_scene = dict(parent_meta)
            parent_meta_scene['index'] = index
            parent_meta_scene['scene'] = scene_name

            summary = run_scene_workflow(
                config=config,
                experiment_cfg=scene_experiment_cfg,
                stages_cfg=stages_cfg,
                default_stage_gpu=default_stage_gpu,
                data_path=str(scene_data_path),
                output_root=str(scene_output_root),
                config_path=config_path,
                parent_meta=parent_meta_scene,
            )
            summaries.append(summary)

            summary_json_path = summary.get('summary_json')
            summary_json_rel = None
            if summary_json_path:
                try:
                    summary_json_rel = str(Path(summary_json_path).relative_to(experiment_root))
                except ValueError:
                    summary_json_rel = summary_json_path

            run_dir_path = summary.get('run_dir')
            run_dir_rel = None
            if run_dir_path:
                try:
                    run_dir_rel = str(Path(run_dir_path).relative_to(experiment_root))
                except ValueError:
                    run_dir_rel = run_dir_path

            scene_record = {
                'scene': scene_name,
                'run_dir': run_dir_rel,
                'summary_json': summary_json_rel,
                'levels': summary.get('scene_level_summary', {}).get('levels')
                if summary.get('scene_level_summary')
                else None,
            }
            experiment_scene_records.append(scene_record)

        if aggregate_output:
            experiment_summary = {
                'generated_at': now_local_iso(),
                'experiment': {
                    'name': experiment_cfg.get('name'),
                    'timestamp': experiment_stamp,
                    'scenes': scenes_list,
                    'output_root': str(experiment_root),
                },
                'runs': experiment_scene_records,
            }
            experiment_summary_path = experiment_root / 'experiment_summary.json'
            try:
                with experiment_summary_path.open('w') as handle:
                    json.dump(experiment_summary, handle, indent=2)
            except OSError:
                pass

        return summaries

    data_path_raw = experiment_cfg.get('data_path')
    if not data_path_raw:
        raise WorkflowConfigError('experiment.data_path is required')
    output_root_raw = expand_output_path_template(override_output or experiment_cfg.get('output_root'), experiment_cfg)

    summary = run_scene_workflow(
        config=config,
        experiment_cfg=experiment_cfg,
        stages_cfg=stages_cfg,
        default_stage_gpu=default_stage_gpu,
        data_path=data_path_raw,
        output_root=output_root_raw,
        config_path=config_path,
        parent_meta=None,
    )
    return [summary]


__all__ = ['execute_workflow']
