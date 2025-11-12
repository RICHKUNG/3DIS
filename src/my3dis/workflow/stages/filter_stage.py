"""Filter stage runner."""

from __future__ import annotations

from pathlib import Path

from my3dis.common_utils import RAW_DIR_NAME
from my3dis.filter_candidates import run_filtering
from my3dis.workflow.scenes import resolve_levels
from my3dis.workflow.stage_runner import StageRunner
from my3dis.workflow.summary import StageRecorder, load_manifest


class FilterStageRunner(StageRunner):
    """Stage runner for candidate filtering."""

    @property
    def name(self) -> str:
        return "filter"

    def should_run(self) -> bool:
        stage_cfg = self._stage_cfg()
        return stage_cfg.get('enabled', True)

    def execute(self) -> None:
        """Execute filtering stage."""
        stage_cfg = self._stage_cfg()

        # Get run_dir and manifest from context (set by SSAM stage)
        run_dir = getattr(self.context, '_run_dir', None)
        manifest = getattr(self.context, '_manifest', None)

        if run_dir is None or not Path(run_dir).exists():
            print('Stage Filter: No run directory available, skipping')
            self.summary.setdefault('order', []).append('filter')
            self._stage_summary()['skipped'] = 'no_run_dir'
            return

        levels = resolve_levels(stage_cfg, manifest, self.context.experiment_cfg.get('levels'))
        ssam_cfg = self.context.stages_cfg.get('ssam', {})
        min_area = int(stage_cfg.get('min_area', ssam_cfg.get('min_area', 300)))
        stability = float(stage_cfg.get('stability_threshold', ssam_cfg.get('stability_threshold', 0.9)))
        update_manifest_flag = stage_cfg.get('update_manifest', True)
        filter_quiet = bool(stage_cfg.get('quiet', False))

        run_dir_path = Path(run_dir)
        raw_available = any((run_dir_path / f'level_{lvl}' / RAW_DIR_NAME).exists() for lvl in levels)
        if not raw_available:
            print('Stage Filter: 找不到 raw 候選資料，略過此階段')
            self.summary.setdefault('order', []).append('filter')
            self._stage_summary()['skipped'] = 'missing_raw'
            return

        print('Stage Filter: 重新套用候選遮罩篩選條件')
        with StageRecorder(self.summary, 'filter', self._stage_gpu_env):
            run_filtering(
                root=str(run_dir_path),
                levels=levels,
                min_area=min_area,
                stability_threshold=stability,
                update_manifest=update_manifest_flag,
                quiet=filter_quiet,
            )

        # Update context with new manifest
        self.context._manifest = load_manifest(run_dir_path)  # type: ignore

        stage_summary = self._stage_summary()
        stage_summary.update({
            'params': {
                'levels': levels,
                'min_area': min_area,
                'stability_threshold': stability,
                'update_manifest': update_manifest_flag,
            }
        })
