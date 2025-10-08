"""Workflow 模組的例外類別定義。"""


class WorkflowError(Exception):
    """Base class for workflow related failures."""


class WorkflowConfigError(WorkflowError):
    """Raised when the provided configuration is invalid."""


class WorkflowRuntimeError(WorkflowError):
    """Raised when runtime side-effects (files, IO, etc.) fail."""


__all__ = [
    'WorkflowError',
    'WorkflowConfigError',
    'WorkflowRuntimeError',
]
