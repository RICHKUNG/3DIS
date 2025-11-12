"""Refactored single-scene workflow implementation (v2).

This module provides the main SceneWorkflow class that orchestrates the execution
of all pipeline stages using the StageRunner pattern for better modularity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from my3dis.workflow.errors import WorkflowRuntimeError
from my3dis.workflow.scenes import derive_scene_metadata
from my3dis.workflow.stage_runner import StageRunner
from my3dis.workflow.stages import (
    SSAMStageRunner,
    FilterStageRunner,
    TrackerStageRunner,
    FamilyTreeStageRunner,
    FamilyVizStageRunner,
    ReportStageRunner,
)
from my3dis.workflow.summary import (
    update_summary_config,
    collect_environment_snapshot,
    apply_scene_level_layout,
    append_run_history,
)
from my3dis.workflow.utils import now_local_iso, serialise_gpu_spec, using_gpu


@dataclass
class SceneContext:
    """Context for scene workflow execution.

    Attributes:
        config: Complete workflow configuration
        experiment_cfg: Experiment-specific configuration
        stages_cfg: Per-stage configuration
        default_stage_gpu: Default GPU specification
        data_path: Path to scene color frames
        output_root: Output root directory
        config_path: Path to source config file
        parent_meta: Parent experiment metadata (for multi-scene runs)
    """
    config: Dict[str, Any]
    experiment_cfg: Dict[str, Any]
    stages_cfg: Dict[str, Any]
    default_stage_gpu: Optional[Any]
    data_path: str
    output_root: str
    config_path: Optional[Path]
    parent_meta: Optional[Dict[str, Any]] = None

    # Internal state (set by stages)
    _run_dir: Optional[Path] = None
    _manifest: Optional[Dict[str, Any]] = None


class SceneWorkflow:
    """Orchestrates single-scene workflow execution using stage runners.

    This refactored version delegates stage-specific logic to individual
    StageRunner classes, making the workflow more modular and testable.
    """

    def __init__(self, context: SceneContext) -> None:
        """Initialize workflow with context.

        Args:
            context: Scene execution context
        """
        self.context = context
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
        self._stage_gpu_env: Optional[str] = None

        self._populate_experiment_metadata()

        # Initialize stage runners
        self.stages: List[StageRunner] = [
            SSAMStageRunner(self.context, self.summary),
            FilterStageRunner(self.context, self.summary),
            TrackerStageRunner(self.context, self.summary),
            FamilyTreeStageRunner(self.context, self.summary),
            FamilyVizStageRunner(self.context, self.summary),
            ReportStageRunner(self.context, self.summary),
        ]

    @property
    def run_dir(self) -> Optional[Path]:
        """Get the run directory (set by SSAM stage)."""
        return self.context._run_dir

    @property
    def manifest(self) -> Optional[Dict[str, Any]]:
        """Get the workflow manifest (set by SSAM stage)."""
        return self.context._manifest

    def run(self) -> Dict[str, Any]:
        """Execute all workflow stages.

        Returns:
            Workflow summary dictionary

        Raises:
            WorkflowRuntimeError: If critical stage fails
        """
        with using_gpu(self.default_stage_gpu):
            self._stage_gpu_env = serialise_gpu_spec(os.environ.get('CUDA_VISIBLE_DEVICES'))  # type: ignore

            # Run all stages
            for stage in self.stages:
                stage.set_gpu_env(self._stage_gpu_env)
                if stage.should_run():
                    stage.execute()

            self._finalize()

        return self.summary

    def _determine_layout_mode(self) -> Optional[str]:
        """Determine output layout mode from config."""
        raw = self.experiment_cfg.get('output_layout')
        if isinstance(raw, str):
            value = raw.strip().lower()
            return value or None
        return None

    def _populate_experiment_metadata(self) -> None:
        """Populate experiment metadata in summary."""
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

    def _finalize(self) -> None:
        """Finalize workflow execution and save summary."""
        run_dir = self.run_dir
        if run_dir is None:
            raise WorkflowRuntimeError('Run directory not set; SSAM stage may have failed')

        manifest = self.manifest

        self.summary['generated_at'] = now_local_iso()
        self.summary['run_dir'] = str(run_dir)

        # Collect environment snapshot
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

        # Apply scene-level layout if configured
        if self.output_layout_mode == 'scene_level':
            aggregated_payload = apply_scene_level_layout(run_dir, self.summary, manifest)
            if aggregated_payload is not None:
                self.summary['scene_level_summary'] = aggregated_payload

        # Save workflow summary
        with (run_dir / 'workflow_summary.json').open('w') as handle:
            json.dump(self.summary, handle, indent=2)

        # Append to run history
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
    """Create and execute SceneWorkflow (compatibility function).

    Args:
        config: Complete workflow configuration
        experiment_cfg: Experiment-specific configuration
        stages_cfg: Per-stage configuration
        default_stage_gpu: Default GPU specification
        data_path: Path to scene color frames
        output_root: Output root directory
        config_path: Path to source config file
        parent_meta: Parent experiment metadata

    Returns:
        Workflow summary dictionary
    """
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


# Import os for GPU environment variable
import os

__all__ = ['SceneContext', 'SceneWorkflow', 'run_scene_workflow']
