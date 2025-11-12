"""Logging configuration utilities for My3DIS."""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence


ENTRY_LOG_FORMAT = "%(asctime)s [pid=%(process)d] %(levelname)s %(message)s"


def setup_logging(
    *,
    explicit_level: Optional[int] = None,
    env_var: str = "MY3DIS_LOG_LEVEL",
    logger_names_to_quiet: Optional[Sequence[str]] = None,
) -> int:
    """Configure root logging once and return the effective level.

    Args:
        explicit_level: Logging level (e.g., logging.INFO). If None, reads from env_var
        env_var: Environment variable name for log level (default: MY3DIS_LOG_LEVEL)
        logger_names_to_quiet: List of logger names to set to WARNING level

    Returns:
        Effective logging level

    Examples:
        >>> setup_logging(explicit_level=logging.DEBUG)
        10
        >>> setup_logging(logger_names_to_quiet=["PIL", "matplotlib"])
        20
    """
    if explicit_level is None:
        level_name = os.environ.get(env_var, "INFO").upper()
        explicit_level = getattr(logging, level_name, logging.INFO)

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=explicit_level, format="%(message)s")
    root_logger.setLevel(explicit_level)

    if logger_names_to_quiet:
        for name in logger_names_to_quiet:
            logging.getLogger(name).setLevel(logging.WARNING)

    return explicit_level


def configure_entry_log_format(*, explicit_level: Optional[int] = None) -> int:
    """Ensure root handlers emit timestamps and PIDs like the CLI entrypoint.

    Args:
        explicit_level: Logging level (e.g., logging.INFO)

    Returns:
        Effective logging level
    """
    level = setup_logging(explicit_level=explicit_level)
    formatter = logging.Formatter(ENTRY_LOG_FORMAT)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)
    return level
