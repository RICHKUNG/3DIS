"""SSAM (Semantic-SAM) stage runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from my3dis.workflow.errors import WorkflowConfigError, WorkflowRuntimeError
from my3dis.workflow.stage_runner import StageRunner
from my3dis.workflow.summary import StageRecorder, load_manifest


class SSAMStageRunner(StageRunner):
    """Stage runner for Semantic-SAM candidate generation."""

    @property
    def name(self) -> str:
        return "ssam"

    def should_run(self) -> bool:
        """SSAM stage runs unless explicitly disabled or reusing existing run_dir."""
        stage_cfg = self._stage_cfg()
        return stage_cfg.get('enabled', True)

    def execute(self) -> None:
        """Execute SSAM stage or reuse existing outputs."""
        stage_cfg = self._stage_cfg()

        if not stage_cfg.get('enabled', True):
            self._handle_reuse_existing()
            return

        self._run_candidate_generation()

    def _handle_reuse_existing(self) -> None:
        """Handle reusing existing SSAM outputs when stage is disabled."""
        reuse_run_dir = self.context.experiment_cfg.get('run_dir')
        if not reuse_run_dir:
            raise WorkflowConfigError('SSAM stage disabled but experiment.run_dir not provided')

        run_dir_candidate = Path(str(reuse_run_dir)).expanduser()
        if not run_dir_candidate.exists():
            raise WorkflowConfigError(f'Provided run_dir {run_dir_candidate} does not exist')

        # Try multiple candidate paths
        def _add_candidate(path: Optional[Path], *, seen: Set[str], targets: List[Path]) -> None:
            if path is None:
                return
            try:
                path_str = str(path)
            except Exception:
                return
            if path_str in seen:
                return
            seen.add(path_str)
            targets.append(path)

        candidates: List[Path] = []
        seen_paths: Set[str] = set()
        _add_candidate(run_dir_candidate, seen=seen_paths, targets=candidates)

        if not (run_dir_candidate.is_dir() and (run_dir_candidate / 'manifest.json').exists()):
            experiment_meta_raw = self.summary.get('experiment')
            experiment_meta = experiment_meta_raw if isinstance(experiment_meta_raw, dict) else {}
            scene_raw = experiment_meta.get('scene')
            scene_name = scene_raw.strip() if isinstance(scene_raw, str) and scene_raw.strip() else None
            tag_raw = self.context.experiment_cfg.get('tag')
            tag = tag_raw.strip() if isinstance(tag_raw, str) and tag_raw.strip() else None
            output_root_path = Path(self.context.output_root)

            if scene_name:
                _add_candidate(run_dir_candidate / scene_name, seen=seen_paths, targets=candidates)
                if tag:
                    _add_candidate(run_dir_candidate / scene_name / tag, seen=seen_paths, targets=candidates)

            _add_candidate(output_root_path, seen=seen_paths, targets=candidates)

            if scene_name:
                _add_candidate(output_root_path / scene_name, seen=seen_paths, targets=candidates)

            if tag:
                _add_candidate(output_root_path / tag, seen=seen_paths, targets=candidates)
                if scene_name:
                    _add_candidate(output_root_path / scene_name / tag, seen=seen_paths, targets=candidates)

        resolved_run_dir: Optional[Path] = None
        for candidate in candidates:
            if candidate.is_dir() and (candidate / 'manifest.json').exists():
                resolved_run_dir = candidate
                break

        if resolved_run_dir is None:
            raise WorkflowConfigError(
                'Unable to reuse SSAM outputs: manifest.json not found under '
                f'{run_dir_candidate}'
            )

        manifest = load_manifest(resolved_run_dir)
        if manifest is None:
            raise WorkflowConfigError(
                'Unable to reuse SSAM outputs: failed to load manifest.json under '
                f'{resolved_run_dir}'
            )

        # Store in context (accessed by other stages)
        # NOTE: This modifies context state - consider refactoring
        self.context._run_dir = resolved_run_dir  # type: ignore
        self.context._manifest = manifest  # type: ignore

    def _run_candidate_generation(self) -> None:
        """Run Semantic-SAM candidate generation."""
        from my3dis.generate_candidates import run_generation as run_candidate_generation
        from my3dis.workflow.stage_config import SSAMStageConfig

        stage_cfg = self._stage_cfg()

        # Use centralized configuration validation
        config = SSAMStageConfig.from_yaml_config(
            stage_cfg=stage_cfg,
            experiment_cfg=self.context.experiment_cfg,
            data_path=self.context.data_path,
            output_root=self.context.output_root,
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
            run_root, manifest = run_candidate_generation(**config.to_legacy_kwargs())

        # Store results in context
        self.context._run_dir = Path(run_root)  # type: ignore
        self.context._manifest = manifest if isinstance(manifest, dict) else manifest  # type: ignore

        mask_meta = manifest.get('mask_downscale', {}) if isinstance(manifest, dict) else {}
        try:
            actual_ratio = (
                float(manifest.get('mask_scale_ratio', 1.0)) if isinstance(manifest, dict) else 1.0
            )
        except (TypeError, ValueError):
            actual_ratio = 1.0
        actual_enabled = bool(mask_meta.get('enabled', actual_ratio < 1.0))

        stage_summary = self._stage_summary()
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
