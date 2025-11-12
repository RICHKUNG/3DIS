"""Abstract base class for workflow stage execution.

This module defines the interface that all stage runners must implement,
providing a consistent pattern for stage execution across the workflow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .scene_workflow import SceneContext


class StageRunner(ABC):
    """Abstract base class for workflow stage execution.

    Each stage runner encapsulates the logic for a single pipeline stage,
    including:
    - Determining if the stage should run
    - Executing the stage logic
    - Recording stage-specific metrics and outputs

    Attributes:
        context: The scene execution context (config, paths, etc.)
        summary: Dictionary to record stage execution details
    """

    def __init__(self, context: 'SceneContext', summary: Dict[str, Any]):
        """Initialize the stage runner.

        Args:
            context: Scene execution context containing configuration and paths
            summary: Workflow summary dictionary to record stage details
        """
        self.context = context
        self.summary = summary
        self._stage_gpu_env: Optional[str] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stage name for logging and summary recording.

        Returns:
            Stage name (e.g., 'ssam', 'filter', 'tracker')
        """
        pass

    @abstractmethod
    def should_run(self) -> bool:
        """Determine if this stage should execute.

        Returns:
            True if stage should execute, False otherwise
        """
        pass

    @abstractmethod
    def execute(self) -> None:
        """Execute the stage logic.

        This method should:
        1. Perform the stage-specific operations
        2. Update self.summary with stage results
        3. Handle any stage-specific error conditions

        Raises:
            WorkflowRuntimeError: If stage execution fails
        """
        pass

    def _stage_cfg(self) -> Dict[str, Any]:
        """Get configuration for this stage.

        Returns:
            Stage configuration dictionary
        """
        raw = self.context.stages_cfg.get(self.name)
        return raw if isinstance(raw, dict) else {}

    def _stage_summary(self) -> Dict[str, Any]:
        """Get or create summary entry for this stage.

        Returns:
            Stage summary dictionary (creates if doesn't exist)
        """
        return self.summary.setdefault('stages', {}).setdefault(self.name, {})

    def set_gpu_env(self, gpu_env: Optional[str]) -> None:
        """Set the GPU environment string for this stage.

        Args:
            gpu_env: Serialized GPU specification (e.g., "0,1" or None)
        """
        self._stage_gpu_env = gpu_env
