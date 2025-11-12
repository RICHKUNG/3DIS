"""Tracker (SAM2) stage runner."""

from __future__ import annotations

from pathlib import Path

from my3dis.workflow.stage_runner import StageRunner
from my3dis.workflow.summary import StageRecorder, load_manifest


class TrackerStageRunner(StageRunner):
    """Stage runner for SAM2 tracking."""

    @property
    def name(self) -> str:
        return "tracker"

    def should_run(self) -> bool:
        stage_cfg = self._stage_cfg()
        return stage_cfg.get('enabled', True)

    def execute(self) -> None:
        """Execute SAM2 tracking stage."""
        from my3dis.track_from_candidates import run_tracking as run_candidate_tracking
        from my3dis.workflow.stage_config import TrackingStageConfig

        run_dir = getattr(self.context, '_run_dir', None)
        if run_dir is None:
            print('Stage Tracker: No run directory available, skipping')
            return

        manifest = getattr(self.context, '_manifest', None) or {}
        tracking_config = TrackingStageConfig.from_yaml_config(
            stage_cfg=self._stage_cfg(),
            experiment_cfg=self.context.experiment_cfg,
            data_path=self.context.data_path,
            candidates_root=str(run_dir),
            output_root=str(run_dir),
            manifest=manifest,
        )

        print('Stage Tracker: SAM2 追蹤與遮罩匯出')
        with StageRecorder(self.summary, 'tracker', self._stage_gpu_env):
            run_candidate_tracking(**tracking_config.to_legacy_kwargs())

        stage_summary = self._stage_summary()
        stage_summary.update({
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
        })

        # Update manifest
        self.context._manifest = load_manifest(Path(run_dir))  # type: ignore
        manifest_snapshot = self.context._manifest or {}  # type: ignore

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
