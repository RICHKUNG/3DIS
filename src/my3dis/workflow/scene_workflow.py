"""單場景工作流程實作。"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from my3dis.common_utils import RAW_DIR_NAME, list_to_csv
from my3dis.filter_candidates import run_filtering
from my3dis.generate_report import build_report

from .errors import WorkflowConfigError, WorkflowRuntimeError
from .scenes import derive_scene_metadata, resolve_levels, stage_frames_string
from .summary import (
    StageRecorder,
    append_run_history,
    apply_scene_level_layout,
    collect_environment_snapshot,
    export_stage_timings,
    load_manifest,
    update_summary_config,
)
from .utils import now_local_iso, serialise_gpu_spec, using_gpu


@dataclass
class SceneContext:
    config: Dict[str, Any]
    experiment_cfg: Dict[str, Any]
    stages_cfg: Dict[str, Any]
    default_stage_gpu: Optional[Any]
    data_path: str
    output_root: str
    config_path: Optional[Path]
    parent_meta: Optional[Dict[str, Any]] = None


class SceneWorkflow:
    """負責單一場景的工作流程執行。"""

    def __init__(self, context: SceneContext) -> None:
        self.config = context.config
        self.experiment_cfg = context.experiment_cfg
        self.stages_cfg = context.stages_cfg
        self.default_stage_gpu = context.default_stage_gpu
        self.data_path = str(Path(context.data_path).expanduser())
        self.output_root = str(Path(context.output_root).expanduser())
        self.config_path = context.config_path
        self.parent_meta = context.parent_meta

        self.summary: Dict[str, Any] = {
            'config_path': str(self.config_path) if self.config_path else None,
            'invoked_at': now_local_iso(),
        }
        update_summary_config(self.summary, self.config)

        self.output_layout_mode = self._determine_layout_mode()
        self.run_dir: Optional[Path] = None
        self.manifest: Optional[Dict[str, Any]] = None

        self._populate_experiment_metadata()
        self._stage_gpu_env: Optional[str] = None

    @staticmethod
    def _resolve_bool_flag(value: Any, default: bool) -> bool:
        """將可能的布林輸入正規化。"""
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {'1', 'true', 'yes', 'on'}:
                return True
            if lowered in {'0', 'false', 'no', 'off'}:
                return False
        return bool(value)

    def run(self) -> Dict[str, Any]:
        with using_gpu(self.default_stage_gpu):
            self._stage_gpu_env = serialise_gpu_spec(os.environ.get('CUDA_VISIBLE_DEVICES'))
            self._run_ssam_stage()
            self._run_filter_stage()
            self._run_tracker_stage()
            self._run_report_stage()
            self._finalize()
        return self.summary

    def _stage_cfg(self, name: str) -> Dict[str, Any]:
        raw = self.stages_cfg.get(name)
        return raw if isinstance(raw, dict) else {}

    def _stage_summary(self, name: str) -> Dict[str, Any]:
        return self.summary.setdefault('stages', {}).setdefault(name, {})

    def _determine_layout_mode(self) -> Optional[str]:
        raw = self.experiment_cfg.get('output_layout')
        if isinstance(raw, str):
            value = raw.strip().lower()
            return value or None
        return None

    def _populate_experiment_metadata(self) -> None:
        scene_meta = derive_scene_metadata(self.data_path)
        experiment_name = (
            self.experiment_cfg.get('name')
            or (self.parent_meta.get('name') if self.parent_meta else None)
            or scene_meta.get('scene')
        )
        experiment_summary = {
            'name': experiment_name,
            'scene': scene_meta.get('scene'),
            'scene_root': scene_meta.get('scene_root'),
            'dataset_root': scene_meta.get('dataset_root'),
            'data_path': self.data_path,
            'output_root': self.output_root,
            'levels': self.experiment_cfg.get('levels'),
            'tag': self.experiment_cfg.get('tag'),
            'scene_output_root': self.output_root,
        }
        if self.parent_meta:
            experiment_summary['parent_experiment'] = self.parent_meta.get('name')
            experiment_summary['experiment_root'] = self.parent_meta.get('experiment_root')
            experiment_summary['scene_index'] = self.parent_meta.get('index')
            if self.parent_meta.get('scenes') is not None:
                experiment_summary['scene_list'] = self.parent_meta.get('scenes')
            if self.parent_meta.get('scene_start') is not None:
                experiment_summary['scene_start'] = self.parent_meta.get('scene_start')
            if self.parent_meta.get('scene_end') is not None:
                experiment_summary['scene_end'] = self.parent_meta.get('scene_end')
        self.summary['experiment'] = experiment_summary

    def _ensure_run_dir(self) -> Path:
        if self.run_dir is None:
            raise WorkflowRuntimeError('Run directory not resolved; ensure SSAM stage executed properly.')
        return self.run_dir

    def _ensure_manifest(self) -> Optional[Dict[str, Any]]:
        if self.manifest is None and self.run_dir is not None:
            self.manifest = load_manifest(self.run_dir)
        return self.manifest

    def _run_ssam_stage(self) -> None:
        stage_cfg = self._stage_cfg('ssam')
        if not stage_cfg.get('enabled', True):
            reuse_run_dir = self.experiment_cfg.get('run_dir')
            if not reuse_run_dir:
                raise WorkflowConfigError('SSAM stage disabled but experiment.run_dir not provided')
            run_dir = Path(str(reuse_run_dir)).expanduser()
            if not run_dir.exists():
                raise WorkflowConfigError(f'Provided run_dir {run_dir} does not exist')
            self.run_dir = run_dir
            self.manifest = load_manifest(run_dir)
            return

        from my3dis.generate_candidates import run_generation as run_candidate_generation
        from .stage_config import SSAMStageConfig

        # Optimized: Centralized configuration validation (replaces 50+ lines of scattered validation)
        config = SSAMStageConfig.from_yaml_config(
            stage_cfg=stage_cfg,
            experiment_cfg=self.experiment_cfg,
            data_path=self.data_path,
            output_root=self.output_root,
        )

        # Extract values for summary logging
        levels = config.levels
        frames_str = f'{config.frames_start}:{config.frames_end}:{config.frames_step}'
        ssam_freq = config.ssam_freq
        min_area = config.min_area
        fill_area = config.fill_area
        stability = config.stability_threshold
        persist_raw = config.persist_raw
        skip_filtering = config.skip_filtering
        ssam_downscale_enabled = config.downscale_masks
        ssam_downscale_ratio = config.mask_scale_ratio

        print('Stage SSAM: Semantic-SAM 採樣與候選輸出')
        with StageRecorder(self.summary, 'ssam', self._stage_gpu_env):
            # Use legacy kwargs for backward compatibility during migration
            run_root, manifest = run_candidate_generation(**config.to_legacy_kwargs())
        self.run_dir = Path(run_root)
        self.manifest = manifest if isinstance(manifest, dict) else manifest

        mask_meta = self.manifest.get('mask_downscale', {}) if isinstance(self.manifest, dict) else {}
        try:
            actual_ratio = (
                float(self.manifest.get('mask_scale_ratio', 1.0)) if isinstance(self.manifest, dict) else 1.0
            )
        except (TypeError, ValueError):
            actual_ratio = 1.0
        actual_enabled = bool(mask_meta.get('enabled', actual_ratio < 1.0))

        stage_summary = self._stage_summary('ssam')
        stage_summary.update(
            {
                'params': {
                    'levels': levels,
                    'frames': frames_str,
                    'ssam_freq': ssam_freq,
                    'min_area': min_area,
                    'fill_area': fill_area,
                    'stability_threshold': stability,
                    'persist_raw': persist_raw,
                    'skip_filtering': skip_filtering,
                    'downscale_masks': actual_enabled,
                    'downscale_ratio': actual_ratio,
                }
            }
        )

    def _run_filter_stage(self) -> None:
        stage_cfg = self._stage_cfg('filter')
        if not stage_cfg.get('enabled', True):
            return

        manifest = self._ensure_manifest()
        levels = resolve_levels(stage_cfg, manifest, self.experiment_cfg.get('levels'))
        ssam_cfg = self._stage_cfg('ssam')
        min_area = int(stage_cfg.get('min_area', ssam_cfg.get('min_area', 300)))
        stability = float(stage_cfg.get('stability_threshold', ssam_cfg.get('stability_threshold', 0.9)))
        update_manifest_flag = stage_cfg.get('update_manifest', True)
        filter_quiet = bool(stage_cfg.get('quiet', False))

        run_dir = self._ensure_run_dir()
        raw_available = any((run_dir / f'level_{lvl}' / RAW_DIR_NAME).exists() for lvl in levels)
        if not raw_available:
            print('Stage Filter: 找不到 raw 候選資料，略過此階段')
            self.summary.setdefault('order', []).append('filter')
            self._stage_summary('filter')['skipped'] = 'missing_raw'
            return

        print('Stage Filter: 重新套用候選遮罩篩選條件')
        with StageRecorder(self.summary, 'filter', self._stage_gpu_env):
            run_filtering(
                root=str(run_dir),
                levels=levels,
                min_area=min_area,
                stability_threshold=stability,
                update_manifest=update_manifest_flag,
                quiet=filter_quiet,
            )

        self.manifest = load_manifest(run_dir)
        stage_summary = self._stage_summary('filter')
        stage_summary.update(
            {
                'params': {
                    'levels': levels,
                    'min_area': min_area,
                    'stability_threshold': stability,
                    'update_manifest': update_manifest_flag,
                }
            }
        )

    def _run_tracker_stage(self) -> None:
        """Execute SAM2 tracking stage using TrackingStageConfig (refactored for clarity)."""
        stage_cfg = self._stage_cfg('tracker')
        if not stage_cfg.get('enabled', True):
            return

        from my3dis.track_from_candidates import run_tracking as run_candidate_tracking
        from my3dis.workflow.stage_config import TrackingStageConfig

        run_dir = self._ensure_run_dir()

        # Use TrackingStageConfig for validation and parameter preparation
        manifest = self._ensure_manifest() or {}
        tracking_config = TrackingStageConfig.from_yaml_config(
            stage_cfg=stage_cfg,
            experiment_cfg=self.experiment_cfg,
            data_path=self.data_path,
            candidates_root=str(run_dir),
            output_root=str(run_dir),
            manifest=manifest,
        )

        print('Stage Tracker: SAM2 追蹤與遮罩匯出')
        with StageRecorder(self.summary, 'tracker', self._stage_gpu_env):
            # Use to_legacy_kwargs() for backward compatibility
            run_candidate_tracking(**tracking_config.to_legacy_kwargs())

        stage_summary = self._stage_summary('tracker')
        stage_summary.update(
            {
                'params': {
                    'levels': tracking_config.levels,
                    'max_propagate': tracking_config.sam2_max_propagate,
                    'iou_threshold': tracking_config.iou_threshold,
                    'long_tail_box_prompt': tracking_config.long_tail_box_prompt,
                    'all_box_prompt': tracking_config.all_box_prompt,
                    'downscale_ratio': tracking_config.mask_scale_ratio,
                    'render_viz': tracking_config.render_viz,
                    'comparison_sampling': {
                        'stride': tracking_config.comparison_sample_stride,
                        'max_frames': tracking_config.comparison_max_samples,
                    },
                }
            }
        )
        self.manifest = load_manifest(run_dir)
        manifest_snapshot = self.manifest or {}
        artifacts_entry = stage_summary.setdefault('artifacts', {})
        tracking_artifacts = manifest_snapshot.get('tracking_artifacts')
        if tracking_artifacts:
            artifacts_entry['tracking'] = tracking_artifacts
        comparison_summary = manifest_snapshot.get('comparison_summary')
        if comparison_summary:
            artifacts_entry['comparison'] = comparison_summary
        tracker_warnings = [
            warning
            for warning in manifest_snapshot.get('warnings', [])
            if isinstance(warning, dict) and warning.get('stage') == 'tracker'
        ]
        if tracker_warnings:
            stage_summary.setdefault('warnings', []).extend(tracker_warnings)

    def _run_report_stage(self) -> None:
        stage_cfg = self._stage_cfg('report')
        if not stage_cfg.get('enabled', True):
            return

        report_name = stage_cfg.get('name', 'report.md')
        max_width = int(stage_cfg.get('max_width', 960))
        run_dir = self._ensure_run_dir()

        print('Stage Report: 生成 Markdown 紀錄')
        with StageRecorder(self.summary, 'report', self._stage_gpu_env):
            build_report(run_dir, report_name=report_name, max_preview_width=max_width)
            report_summary = self._stage_summary('report').setdefault('params', {})
            report_summary['max_width'] = max_width
            report_summary['report_name'] = report_name

        record_timings_flag = bool(stage_cfg.get('record_timings'))
        timing_output_name = stage_cfg.get('timing_output')
        if record_timings_flag or timing_output_name:
            output_name = timing_output_name or 'stage_timings.json'
            timings_path = run_dir / output_name
            export_stage_timings(self.summary, timings_path)
            artifacts_entry = self._stage_summary('report').setdefault('artifacts', {})
            artifacts_entry['timings'] = str(timings_path)

        # Generate check.md summary if parent experiment directory exists
        generate_check_report = bool(stage_cfg.get('generate_check_report', False))
        if generate_check_report and run_dir.parent.is_dir():
            try:
                from .reporting import generate_experiment_check_report
                check_path = generate_experiment_check_report(run_dir.parent)
                artifacts_entry = self._stage_summary('report').setdefault('artifacts', {})
                artifacts_entry['experiment_check'] = str(check_path.relative_to(run_dir.parent))
                print(f'Generated experiment check report: {check_path}')
            except Exception as exc:
                import logging
                LOGGER = logging.getLogger(__name__)
                LOGGER.warning("Failed to generate experiment check report: %s", exc)

    def _finalize(self) -> None:
        run_dir = self._ensure_run_dir()
        manifest = self._ensure_manifest()

        # Build cross-level relations if all stages completed successfully
        levels = self.experiment_cfg.get('levels', [])
        if levels and manifest:
            try:
                from my3dis.relation_index import build_cross_level_relations
                relations_path = build_cross_level_relations(run_dir, levels)
                artifacts_entry = self.summary.setdefault('artifacts', {})
                artifacts_entry['relations'] = str(relations_path.relative_to(run_dir))
            except Exception as exc:
                import logging
                LOGGER = logging.getLogger(__name__)
                LOGGER.warning("Failed to build cross-level relations: %s", exc, exc_info=True)

        self.summary['generated_at'] = now_local_iso()
        self.summary['run_dir'] = str(run_dir)
        env_snapshot = collect_environment_snapshot()
        self.summary['environment'] = env_snapshot
        env_path = run_dir / 'environment_snapshot.json'
        try:
            with env_path.open('w', encoding='utf-8') as handle:
                json.dump(env_snapshot, handle, indent=2)
        except OSError:
            env_path = None
        artifacts_entry = self.summary.setdefault('artifacts', {})
        if env_path is not None:
            artifacts_entry['environment_snapshot'] = str(env_path)

        if self.output_layout_mode == 'scene_level':
            aggregated_payload = apply_scene_level_layout(run_dir, self.summary, manifest)
            if aggregated_payload is not None:
                self.summary['scene_level_summary'] = aggregated_payload

        with (run_dir / 'workflow_summary.json').open('w') as handle:
            json.dump(self.summary, handle, indent=2)

        append_run_history(self.summary, manifest)

        print(f'Workflow finished. 輸出路徑：{run_dir}')


def run_scene_workflow(
    *,
    config: Dict[str, Any],
    experiment_cfg: Dict[str, Any],
    stages_cfg: Dict[str, Any],
    default_stage_gpu: Optional[Any],
    data_path: str,
    output_root: str,
    config_path: Optional[Path],
    parent_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """建立 SceneWorkflow 並執行，回傳摘要。"""
    context = SceneContext(
        config=config,
        experiment_cfg=experiment_cfg,
        stages_cfg=stages_cfg,
        default_stage_gpu=default_stage_gpu,
        data_path=data_path,
        output_root=output_root,
        config_path=config_path,
        parent_meta=parent_meta,
    )
    return SceneWorkflow(context).run()


__all__ = ['SceneContext', 'SceneWorkflow', 'run_scene_workflow']
