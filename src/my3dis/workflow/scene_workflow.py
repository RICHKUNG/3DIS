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

        from my3dis.generate_candidates import (
            DEFAULT_SEMANTIC_SAM_CKPT as _DEFAULT_SEMANTIC_SAM_CKPT,
            run_generation as run_candidate_generation,
        )

        levels = resolve_levels(stage_cfg, None, self.experiment_cfg.get('levels'))
        frames_str = stage_frames_string(stage_cfg, self.experiment_cfg)
        try:
            ssam_freq = int(stage_cfg.get('ssam_freq', 1))
        except (TypeError, ValueError) as exc:
            raise WorkflowConfigError(f"invalid stages.ssam.ssam_freq: {stage_cfg.get('ssam_freq')!r}") from exc
        min_area = int(stage_cfg.get('min_area', 300))
        fill_area_cfg = stage_cfg.get('fill_area')
        if fill_area_cfg is None:
            fill_area = min_area
        else:
            try:
                fill_area = int(fill_area_cfg)
            except (TypeError, ValueError) as exc:
                raise WorkflowConfigError(f'invalid stages.ssam.fill_area: {fill_area_cfg!r}') from exc
        fill_area = max(0, fill_area)
        stability = float(stage_cfg.get('stability_threshold', 0.9))
        persist_raw = bool(stage_cfg.get('persist_raw', True))
        skip_filtering = bool(stage_cfg.get('skip_filtering', False))
        add_gaps = bool(stage_cfg.get('add_gaps', False))
        append_timestamp = stage_cfg.get('append_timestamp', True)
        experiment_tag = stage_cfg.get('experiment_tag') or self.experiment_cfg.get('tag')
        tag_in_path_raw = stage_cfg.get('tag_in_path')
        if tag_in_path_raw is None:
            tag_in_path_raw = self.experiment_cfg.get('tag_in_path')
        tag_in_path = self._resolve_bool_flag(tag_in_path_raw, True)
        ssam_downscale_enabled = bool(stage_cfg.get('downscale_masks', False))
        ssam_downscale_ratio = stage_cfg.get('downscale_ratio', stage_cfg.get('mask_scale_ratio', 1.0))
        try:
            ssam_downscale_ratio = float(ssam_downscale_ratio)
        except (TypeError, ValueError) as exc:
            raise WorkflowConfigError(
                f'invalid stages.ssam.downscale_ratio: {ssam_downscale_ratio!r}'
            ) from exc

        sam_ckpt_cfg = stage_cfg.get('sam_ckpt') or self.experiment_cfg.get('sam_ckpt')
        if sam_ckpt_cfg:
            sam_ckpt_path = Path(str(sam_ckpt_cfg)).expanduser()
        else:
            sam_ckpt_path = Path(_DEFAULT_SEMANTIC_SAM_CKPT)
        if not sam_ckpt_path.exists():
            raise WorkflowConfigError(
                f'Semantic-SAM checkpoint not found at {sam_ckpt_path}. '
                'Set stages.ssam.sam_ckpt or experiment.sam_ckpt to a valid file path.'
            )
        sam_ckpt = str(sam_ckpt_path)

        print('Stage SSAM: Semantic-SAM 採樣與候選輸出')
        with StageRecorder(self.summary, 'ssam', self._stage_gpu_env):
            run_root, manifest = run_candidate_generation(
                data_path=self.data_path,
                levels=list_to_csv(levels),
                frames=frames_str,
                sam_ckpt=sam_ckpt,
                output=self.output_root,
                min_area=min_area,
                fill_area=fill_area,
                stability_threshold=stability,
                add_gaps=add_gaps,
                no_timestamp=not append_timestamp,
                ssam_freq=ssam_freq,
                sam2_max_propagate=stage_cfg.get('sam2_max_propagate'),
                experiment_tag=experiment_tag,
                persist_raw=persist_raw,
                skip_filtering=skip_filtering,
                downscale_masks=ssam_downscale_enabled,
                mask_scale_ratio=ssam_downscale_ratio,
                tag_in_path=tag_in_path,
            )
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
        stage_cfg = self._stage_cfg('tracker')
        if not stage_cfg.get('enabled', True):
            return

        from my3dis.track_from_candidates import run_tracking as run_candidate_tracking

        manifest = self._ensure_manifest() or {}
        levels = resolve_levels(stage_cfg, manifest, self.experiment_cfg.get('levels'))
        max_propagate = stage_cfg.get('max_propagate')
        iou_threshold = float(stage_cfg.get('iou_threshold', 0.6))
        prompt_mode_raw = str(stage_cfg.get('prompt_mode', 'all_mask')).lower()
        prompt_aliases = {
            'none': 'all_mask',
            'all_mask': 'all_mask',
            'long_tail': 'lt_bbox',
            'lt_bbox': 'lt_bbox',
            'all': 'all_bbox',
            'all_bbox': 'all_bbox',
        }
        if prompt_mode_raw not in prompt_aliases:
            raise WorkflowConfigError(f'Unknown tracker.prompt_mode={prompt_mode_raw}')
        prompt_mode = prompt_aliases[prompt_mode_raw]
        all_box = prompt_mode == 'all_bbox'
        long_tail_box = prompt_mode == 'lt_bbox'

        try:
            mask_ratio_default = float(manifest.get('mask_scale_ratio', 1.0)) if manifest else 1.0
        except (TypeError, ValueError):
            mask_ratio_default = 1.0
        downscale_enabled = bool(stage_cfg.get('downscale_masks', mask_ratio_default < 1.0))
        downscale_ratio = stage_cfg.get(
            'downscale_ratio', mask_ratio_default if mask_ratio_default < 1.0 else 0.3
        )
        try:
            downscale_ratio = float(downscale_ratio)
        except (TypeError, ValueError) as exc:
            raise WorkflowConfigError(f'Invalid tracker.downscale_ratio={downscale_ratio!r}') from exc
        mask_scale_ratio = downscale_ratio if downscale_enabled else 1.0
        if mask_scale_ratio <= 0.0 or mask_scale_ratio > 1.0:
            raise WorkflowConfigError('tracker.downscale_ratio must be in (0, 1] when downscale_masks is true')

        sam2_cfg = stage_cfg.get('sam2_cfg') or self.experiment_cfg.get('sam2_cfg')
        sam2_ckpt = stage_cfg.get('sam2_ckpt') or self.experiment_cfg.get('sam2_ckpt')
        render_viz = bool(stage_cfg.get('render_viz', True))

        comparison_sample_stride: Optional[int] = None
        comparison_max_samples: Optional[int] = None
        comparison_cfg = stage_cfg.get('comparison_sampling')
        if comparison_cfg is not None:
            if not isinstance(comparison_cfg, dict):
                raise WorkflowConfigError('tracker.comparison_sampling must be a mapping')
            if 'stride' in comparison_cfg:
                try:
                    comparison_sample_stride = int(comparison_cfg['stride'])
                except (TypeError, ValueError):
                    raise WorkflowConfigError(
                        f"Invalid tracker.comparison_sampling.stride={comparison_cfg['stride']!r}"
                    )
                if comparison_sample_stride <= 0:
                    raise WorkflowConfigError('tracker.comparison_sampling.stride must be > 0')
            if 'max_frames' in comparison_cfg:
                try:
                    max_frames_val = int(comparison_cfg['max_frames'])
                except (TypeError, ValueError):
                    raise WorkflowConfigError(
                        f"Invalid tracker.comparison_sampling.max_frames={comparison_cfg['max_frames']!r}"
                    )
                if max_frames_val < 0:
                    raise WorkflowConfigError('tracker.comparison_sampling.max_frames must be >= 0')
                comparison_max_samples = max_frames_val if max_frames_val > 0 else None

        run_dir = self._ensure_run_dir()
        print('Stage Tracker: SAM2 追蹤與遮罩匯出')
        with StageRecorder(self.summary, 'tracker', self._stage_gpu_env):
            run_candidate_tracking(
                data_path=self.data_path,
                candidates_root=str(run_dir),
                output=str(run_dir),
                levels=list_to_csv(levels),
                sam2_cfg=sam2_cfg,
                sam2_ckpt=sam2_ckpt,
                sam2_max_propagate=max_propagate,
                iou_threshold=iou_threshold,
                long_tail_box_prompt=long_tail_box,
                all_box_prompt=all_box,
                mask_scale_ratio=mask_scale_ratio,
                comparison_sample_stride=comparison_sample_stride,
                comparison_max_samples=comparison_max_samples,
                render_viz=render_viz,
            )

        stage_summary = self._stage_summary('tracker')
        stage_summary.update(
            {
                'params': {
                    'levels': levels,
                    'max_propagate': max_propagate,
                    'iou_threshold': iou_threshold,
                    'prompt_mode': prompt_mode,
                    'downscale_masks': downscale_enabled,
                    'downscale_ratio': mask_scale_ratio,
                    'render_viz': render_viz,
                    'comparison_sampling': {
                        'stride': comparison_sample_stride,
                        'max_frames': comparison_max_samples,
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

    def _finalize(self) -> None:
        run_dir = self._ensure_run_dir()
        manifest = self._ensure_manifest()

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
