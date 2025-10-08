"""多場景工作流程聚合執行。"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .errors import WorkflowConfigError, WorkflowRuntimeError
from .scene_workflow import run_scene_workflow
from .scenes import expand_output_path_template, normalize_scene_list
from .utils import now_local_iso, now_local_stamp

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SceneJob:
    index: int
    scene: str
    kwargs: Dict[str, Any]


def _run_scene_job(job: _SceneJob) -> Dict[str, Any]:
    return run_scene_workflow(**job.kwargs)


def _resolve_path_override(env_var: str, configured: Optional[str]) -> Optional[str]:
    """Return the configured path, allowing the environment to override it."""

    override = os.environ.get(env_var)
    return override if override else configured


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
    dataset_root_raw = _resolve_path_override('MY3DIS_DATASET_ROOT', experiment_cfg.get('dataset_root'))
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
        output_root_base_raw = override_output or _resolve_path_override(
            'MY3DIS_OUTPUT_ROOT',
            experiment_cfg.get('output_root'),
        )
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

        parallel_cfg = experiment_cfg.get('parallel_scenes')
        try:
            parallel_scenes = int(parallel_cfg) if parallel_cfg is not None else 1
        except (TypeError, ValueError):
            LOGGER.warning(
                "Invalid experiment.parallel_scenes=%r, falling back to sequential execution",
                parallel_cfg,
            )
            parallel_scenes = 1
        parallel_scenes = max(1, parallel_scenes)

        jobs: List[_SceneJob] = []
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

            job_kwargs = {
                'config': config,
                'experiment_cfg': scene_experiment_cfg,
                'stages_cfg': stages_cfg,
                'default_stage_gpu': default_stage_gpu,
                'data_path': str(scene_data_path),
                'output_root': str(scene_output_root),
                'config_path': config_path,
                'parent_meta': parent_meta_scene,
            }
            jobs.append(_SceneJob(index=index, scene=scene_name, kwargs=job_kwargs))

        summaries_ordered: List[Optional[Dict[str, Any]]] = [None] * len(jobs)
        scene_errors: List[Tuple[_SceneJob, BaseException]] = []

        if parallel_scenes > 1 and len(jobs) > 1:
            LOGGER.info(
                "Executing %d scenes with up to %d parallel workers",
                len(jobs),
                parallel_scenes,
            )
            with concurrent.futures.ProcessPoolExecutor(max_workers=parallel_scenes) as executor:
                future_map = {executor.submit(_run_scene_job, job): job for job in jobs}
                for future in concurrent.futures.as_completed(future_map):
                    job = future_map[future]
                    try:
                        summary = future.result()
                    except BaseException as exc:  # pragma: no cover - propagate detailed logs
                        LOGGER.error("Scene %s failed during execution", job.scene, exc_info=True)
                        scene_errors.append((job, exc))
                    else:
                        summaries_ordered[job.index] = summary
        else:
            for job in jobs:
                try:
                    summaries_ordered[job.index] = _run_scene_job(job)
                except BaseException as exc:
                    LOGGER.error("Scene %s failed during execution", job.scene, exc_info=True)
                    raise

        if scene_errors:
            failed_names = ', '.join(job.scene for job, _ in scene_errors)
            primary_exc = scene_errors[0][1]
            raise WorkflowRuntimeError(
                f'One or more scene workflows failed: {failed_names}'
            ) from primary_exc

        summaries: List[Dict[str, Any]] = [summary for summary in summaries_ordered if summary is not None]
        experiment_scene_records: List[Dict[str, Any]] = []

        for job in jobs:
            summary = summaries_ordered[job.index]
            if summary is None:
                continue

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
                'scene': job.scene,
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
            except OSError as exc:
                LOGGER.warning(
                    "Failed to write experiment summary to %s: %s",
                    experiment_summary_path,
                    exc,
                    exc_info=True,
                )

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
